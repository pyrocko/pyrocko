import { ref, computed } from '../vue.esm-browser.js'

import { strToTime } from './common.js'

import { squirrelConnection } from './connection.js'

export const squirrelGate = () => {
    const codes = ref([])
    const connection = squirrelConnection()

    const fetchCodes = async () => {
        const codes = new Set()
        for (const kind of ['waveform', 'channel', 'response']) {
            for (const c of await connection.request('raw/get_codes', {
                kind: kind,
            })) {
                codes.add(c)
            }
        }
        return Array.from(codes)
    }

    const update = async () => {
        const new_codes = await fetchCodes()
        codes.value = new_codes
    }

    return { codes, update }
}

export const squirrelBlock = (block) => {
    const my = { ...block }
    const connection = squirrelConnection()
    let lastTouched = -1
    let coverages = null

    const fetchCoverage = async (kind) => {
        const coverages = await connection.request('raw/get_coverage', {
            kind: kind,
        })

        for (let coverage of coverages) {
            coverage.id = [
                coverage.kind,
                coverage.tmin,
                coverage.tmax,
                coverage.codes,
            ].join('+++')
            coverage.codes = coverage.codes + '.'
            coverage.tmin = strToTime(coverage.tmin)
            coverage.tmax = strToTime(coverage.tmax)
        }
        return coverages
    }

    my.update = async () => {
        if (coverages === null) {
            coverages = await fetchCoverage('channel')
        }
    }
    my.touch = (counter) => {
        lastTouched = counter
    }

    my.getLastTouched = () => {
        return lastTouched
    }

    my.getCoverages = () => {
        return coverages || []
    }

    my.overlaps = (tmin, tmax) => {
        return my.timeMin < tmax && my.timeMax > tmin
    }

    my.ready = () => {
        return coverages !== null
    }

    return my
}
export const squirrelGates = () => {
    const gates = ref([])
    const timeMin = ref(strToTime('1900-01-01 00:00:00'))
    const timeMax = ref(strToTime('2030-01-01 00:00:00'))
    const blockFactor = 4
    const blocks = new Map()
    let counter = 0

    const makeTimeBlock = (tmin, tmax) => {
        const iscale = Math.ceil(Math.log2(blockFactor * (tmax - tmin)))
        const tstep = Math.pow(2, iscale)
        const itime = Math.round((tmin + tmax) / tstep)
        return squirrelBlock({
            iScale: iscale,
            iTime: itime,
            timeStep: tstep,
            timeMin: (itime - 1) * tstep * 0.5,
            timeMax: (itime + 1) * tstep * 0.5,
        })
    }

    const blockKey = (block) => block.iScale + ',' + block.iTime

    const updateTimeWindow = () => {
        const block = makeTimeBlock(timeMin.value, timeMax.value)
        const k = blockKey(block)
        if (!blocks.has(k)) {
            blocks.set(k, block)
            block.update()
        }
        blocks.get(k).touch(counter)
        counter++
    }

    const setTimeSpan = (tmin, tmax) => {
        timeMin.value = tmin
        timeMax.value = tmax
        updateTimeWindow()
    }

    const addGate = () => {
        const gate = squirrelGate()
        gates.value.push(gate)
        gate.update()
    }

    const getRelevantBlocks = () => {
        return Array.from(blocks.values())
            .toSorted((a, b) => b.getLastTouched() - a.getLastTouched())
            .filter(
                (block) =>
                    block.overlaps(timeMin.value, timeMax.value) &&
                    block.ready()
            )
    }

    const getRelevantBlock = () => {
        const relevant = getRelevantBlocks()
        return relevant.length > 0 ? relevant[0] : null
    }

    const getCoverages = () => {
        const block = getRelevantBlock()
        console.log(block)
        if (block === null) {
            return []
        }
        return block.getCoverages()
    }

    const codes = computed(() => {
        const codes = new Set()
        for (const gate of gates.value) {
            for (const c of gate.codes) {
                codes.add(c)
            }
        }
        return Array.from(codes)
    })

    return { timeMin, timeMax, setTimeSpan, addGate, codes, getCoverages }
}
