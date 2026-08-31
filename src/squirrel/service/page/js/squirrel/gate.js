const { ref, shallowRef, computed, watch } = Vue

import { strToTime, timeToStr, tomorrow } from './common.js'

import { squirrelConnection } from './connection.js'

const TIME_MIN = strToTime('1900-01-01 00:00:00')
const TIME_MAX = tomorrow() + 5 * 365 * 24 * 60 * 60
const RE_2COMMA = new RegExp('([^,]+),([^,]+),', 'g')

// Runs `fn` for the latest of possibly many arguments passed to the
// returned function, allowing at most one call to `fn` to be in flight
// at a time. A call that arrives while one is already running just
// replaces the single pending argument (the queue is depth 1, latest
// wins) instead of piling up; whichever argument is current the moment
// the running call finishes is the only one run next. This is what
// keeps fast repeated calls (e.g. from mouse movement, or from a user
// panning quickly), or a slow round trip, from turning into a pile of
// concurrent or stale requests.
//
// With `debounceMs` set, a fresh call made while idle waits that long
// for the input to settle before the first `fn` call of a burst fires
// (it keeps getting pushed back by further calls); once a call is
// running, further updates take effect immediately, without waiting
// out the debounce again -- there is no benefit in delaying a value
// that is already just waiting for its turn.
//
// Errors from `fn` are logged and do not stop the runner.
const makeLatestWinsRunner = (fn, { debounceMs = 0 } = {}) => {
    let running = false
    let hasWanted = false
    let wanted
    let debounceTimeoutId = null

    const runLoop = async () => {
        running = true
        while (hasWanted) {
            const arg = wanted
            hasWanted = false
            wanted = undefined
            try {
                await fn(arg)
            } catch (error) {
                console.log(error)
            }
        }
        running = false
    }

    return (arg) => {
        wanted = arg
        hasWanted = true

        if (running) {
            return
        }

        if (debounceTimeoutId !== null) {
            clearTimeout(debounceTimeoutId)
        }
        debounceTimeoutId = setTimeout(() => {
            debounceTimeoutId = null
            runLoop()
        }, debounceMs)
    }
}

export const squirrelGate = (gate_id_) => {
    const gate_id = gate_id_
    const counter = ref(0)
    const filter = ref('')
    const codes = shallowRef([])
    const channels = shallowRef([])
    const sensors = shallowRef([])
    const responses = shallowRef([])
    const events = shallowRef([])
    const timeSpans = shallowRef({
        waveform: null,
        channel: null,
        response: null,
        carpet: null,
    })

    const contextInfos = shallowRef([])

    const connection = squirrelConnection()

    const gateRequest = (method, data) => {
        return connection.value.request('gate/' + gate_id + '/' + method, data)
    }

    const fetchCodes = async () => {
        const codes = new Set()
        for (const kind of ['waveform', 'channel', 'response', 'carpet']) {
            for (const c of await gateRequest('get_codes', {
                kind: kind,
            })) {
                codes.add(c)
            }
        }
        return Array.from(codes)
    }

    const fetchChannels = async () => {
        return gateRequest('get_channels')
    }

    const fetchSensors = async () => {
        return gateRequest('get_sensors')
    }

    //const fetchResponses = async () => {
    //    return gateRequest('get_responses')
    //}
    //
    //const fetchEvents = async () => {
    //    const events = await gateRequest('get_events')
    //    for (const ev of events) {
    //        ev.time = strToTime(ev.time)
    //    }
    //    return events
    //}

    const fetchTimeSpans = async () => {
        const newTimeSpans = {}
        for (const kind of ['waveform', 'channel', 'response', 'carpet']) {
            const span = await gateRequest('get_time_span', { kind: kind })
            span.tmin = span.tmin != null ? strToTime(span.tmin) : null
            span.tmax = span.tmax != null ? Math.min(strToTime(span.tmax), tomorrow()) : null
            newTimeSpans[kind] = span
        }
        return newTimeSpans
    }

    const update = async () => {
        codes.value = await fetchCodes()
        timeSpans.value = await fetchTimeSpans()
        channels.value = await fetchChannels()
        sensors.value = await fetchSensors()
        //responses.value = await fetchResponses()
        //events.value = await fetchEvents()
    }

    const fetchContextInfos = async (request) => {
        return await gateRequest('get_context', request)
    }

    const updateContext = makeLatestWinsRunner(async (request) => {
        contextInfos.value = await fetchContextInfos(request)
    })

    return {
        codes,
        timeSpans,
        channels,
        sensors,
        responses,
        events,
        update,
        counter,
        filter,
        updateContext,
        contextInfos,
    }
}

export const squirrelBlock = (block) => {
    const counter = ref(0)
    const my = { ...block }
    const connection = squirrelConnection()
    let lastTouched = -1
    let coverages = null
    let waveviews = null
    let carpets = null
    let oldCarpets = []

    const fetchCoverage = async () => {
        const coverages = await connection.value.request('gate/default/get_rich_coverage', {
            tmin: timeToStr(my.timeMin),
            tmax: timeToStr(my.timeMax),
        })

        for (const coverage of coverages) {
            coverage.id = [coverage.kind, coverage.tmin, coverage.tmax, coverage.codes].join('+++')
            coverage.tmin = strToTime(coverage.tmin)
            coverage.tmax = strToTime(coverage.tmax)
        }
        return coverages
    }

    const fetchWaveviews = async (params) => {
        const waveviews = await connection.value.request('gate/default/get_waveviews', {
            tmin: timeToStr(my.timeMin),
            tmax: timeToStr(my.timeMax),
            codes: params.codes,
            fmin: params.ymin,
            fmax: params.ymax,
            nx: params.nx,
            ny: params.ny,
        })
        for (const waveview of waveviews) {
            waveview.id = [
                waveview.kind,
                waveview.tmin,
                waveview.tmax,
                waveview.codes,
                waveview.fmin,
                waveview.fmax,
            ].join('+++')
            waveview.tmin = strToTime(waveview.tmin)
            waveview.tmax = strToTime(waveview.tmax)

            const data_uint8 = Uint8Array.fromBase64(waveview.polygon_data_base64)
            const data_float32 = new Float32Array(data_uint8.buffer)
            waveview.points_string = data_float32.join(',').replaceAll(RE_2COMMA, '$1,$2 ')
        }
        return waveviews
    }

    const fetchCarpets = async (params) => {
        const carpets = await connection.value.request('gate/default/get_carpets', {
            tmin: timeToStr(my.timeMin),
            tmax: timeToStr(my.timeMax),
            ...params,
        })
        for (const carpet of carpets) {
            carpet.id = [
                carpet.tmin,
                carpet.tmax,
                carpet.ymin,
                carpet.ymax,
                carpet.shape[0],
                carpet.shape[1],
                carpet.codes,
                carpet.overview_method,
            ].join('+++')
            carpet.zombie1 = false
            carpet.tmin = strToTime(carpet.tmin)
            carpet.tmax = strToTime(carpet.tmax)
        }
        return carpets
    }

    my.key = () => my.iScale + ',' + my.iTime

    my.cleanup = () => {
        const now = Date.now()
        oldCarpets = oldCarpets.filter((carpet) => carpet.zombie1Timestamp > now - 1000)
    }

    // Runs one fetch cycle for this block. Concurrency (making sure only
    // one such cycle is ever in flight, and that a burst of calls
    // collapses to just the latest one) is the caller's responsibility
    // -- see the shared scheduler in `setupGates`.
    my.fetch = async (params) => {
        if (coverages === null) {
            coverages = await fetchCoverage()
        }
        waveviews = await fetchWaveviews(params)
        const newCarpets = await fetchCarpets(params)
        for (const carpet of carpets || []) {
            carpet.zombie1 = true
            carpet.zombie1Timestamp = Date.now()
            oldCarpets.push(carpet)
        }
        carpets = newCarpets
        setTimeout(my.cleanup, 1100)

        counter.value++
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

    my.getWaveviews = () => {
        return waveviews || []
    }

    my.getCarpets = () => {
        return (carpets || []).concat(oldCarpets || [])
    }

    my.overlaps = (tmin, tmax) => {
        return my.timeMin < tmax && my.timeMax > tmin
    }

    my.ready = () => {
        return coverages !== null && carpets !== null && waveviews !== null
    }

    my.unwatch = null

    my.destroy = () => {
        if (my.unwatch !== null) {
            my.unwatch()
        }
        my.unwatch = null
    }

    my.counter = counter

    my.active = false
    my.activeChanged = 0

    my.setActive = (active) => {
        if (my.active != active) {
            my.active = active
            my.activeChanged = Date.now()
        }
    }

    my.durationSinceChange = () => {
        return Date.now() - my.activeChanged
    }

    my.isActiveOrZombie = () => {
        return my.active || my.durationSinceChange() < 1000
    }

    return my
}

export const setupGates = () => {
    const gates = ref([])
    const timeMin = ref(TIME_MIN)
    const timeMax = ref(TIME_MAX)
    const hover = shallowRef(null)
    const imageHeight = ref(100)
    const imageWidth = ref(100)
    const codesVisible = shallowRef(null)
    const yMin = ref(null)
    const yMax = ref(null)
    const overviewMethod = ref('mean')
    const blockFactor = 2
    const resolutionFactor = 1.0
    const blocks = new Map()
    let counter = ref(0)
    let initialTimeSpanSet = false
    let _relevantBlocks = []

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

    const dropOldBlocks = () => {
        const kDelete = Array.from(blocks.values())
            .toSorted((a, b) => b.getLastTouched() - a.getLastTouched())
            .slice(5)
            .map((block) => block.key())

        for (const k of kDelete) {
            blocks.delete(k)
        }
    }

    // Across all blocks of this gate, allow only one fetch cycle
    // (coverage + waveviews + carpets) to be in flight at a time, with
    // a short leading debounce so a block only briefly passed through
    // while panning quickly doesn't get fetched at all. This is what
    // keeps a fast click-and-drag pan, or a slow server, from spawning
    // a burst of concurrent requests: whichever block/params were
    // current the moment the in-flight cycle completes are the only
    // ones fetched next, and everything visited only fleetingly in
    // between is skipped for good.
    const scheduleBlockFetch = makeLatestWinsRunner(async ({ block, params }) => {
        if (!blocks.has(block.key())) {
            // Superseded and evicted before its turn came up.
            return
        }
        await block.fetch(params)
    }, { debounceMs: 100 })

    const updateBlocks = () => {
        const sorted = Array.from(blocks.values()).toSorted(
            (a, b) => b.getLastTouched() - a.getLastTouched()
        )

        if (sorted.size == 0) {
            return
        }

        const kNewest = sorted[0].key()

        for (const k of blocks.keys()) {
            if (k == kNewest) {
                scheduleBlockFetch({
                    block: blocks.get(k),
                    params: {
                        ymin: yMin.value,
                        ymax: yMax.value,
                        nx: imageWidth.value,
                        ny: imageHeight.value,
                        codes: codesVisible.value,
                        overview_method: overviewMethod.value,
                    },
                })
            } else {
                blocks.delete(k)
            }
        }
    }

    watch([yMin, yMax, imageWidth, imageHeight, codesVisible, overviewMethod], updateBlocks)

    const update = () => {
        const block = makeTimeBlock(timeMin.value, timeMax.value)
        const k = block.key()
        if (!blocks.has(k)) {
            blocks.set(k, block)
            watch([block.counter], () => counter.value++)
            scheduleBlockFetch({
                block,
                params: {
                    ymin: yMin.value,
                    ymax: yMax.value,
                    nx: blockFactor * imageWidth.value * resolutionFactor,
                    ny: imageHeight.value,
                    codes: codesVisible.value,
                    overview_method: overviewMethod.value,
                },
            })
        }
        blocks.get(k).touch(counter.value)
        counter.value++
        dropOldBlocks()
    }

    const setTimeSpan = (tmin, tmax) => {
        timeMin.value = Math.max(tmin, TIME_MIN)
        timeMax.value = Math.min(tmax, TIME_MAX)
        update()
    }

    const setHover = (t) => {
        hover.value = t
    }

    const setImageWidth = (nx) => {
        imageWidth.value = Math.max(1, Math.round(nx))
    }

    const setImageHeight = (ny) => {
        imageHeight.value = Math.max(1, Math.round(ny))
    }

    const setCodesVisible = (codes) => {
        codesVisible.value = codes
    }

    const makePageMove = (amount) => {
        return () => {
            const tmin = timeMin.value
            const tmax = timeMax.value
            const dt = tmax - tmin
            setTimeSpan(tmin + amount * dt, tmax + amount * dt)
        }
    }

    const halfPageForward = makePageMove(0.5)
    const halfPageBackward = makePageMove(-0.5)
    const pageForward = makePageMove(1)
    const pageBackward = makePageMove(-1)

    const addGate = () => {
        const gate = squirrelGate('default')
        gates.value.push(gate)
        gate.update()
    }

    const getRelevantBlocks = () => {
        const relevant = Array.from(blocks.values())
            .toSorted((a, b) => b.getLastTouched() - a.getLastTouched())
            .filter((block) => block.overlaps(timeMin.value, timeMax.value) && block.ready())
        if (
            relevant.length > 0 &&
            (_relevantBlocks.length == 0 || relevant[0] !== _relevantBlocks[0])
        ) {
            const i = _relevantBlocks.indexOf(relevant[0])
            if (i > -1) {
                _relevantBlocks.splice(i, 1)
            }
            _relevantBlocks.unshift(relevant[0])
            if (_relevantBlocks.length > 4) {
                _relevantBlocks.length = 4
            }
        }
        _relevantBlocks = _relevantBlocks
            .map((block, iblock) => (block.setActive(iblock == 0), block))
            .filter((block) => block.isActiveOrZombie())

        return _relevantBlocks
    }

    const getCoverages = () => {
        return getRelevantBlocks()
            .slice(0, 1)
            .flatMap((block) => block.getCoverages())
    }

    const getWaveviews = () => {
        return getRelevantBlocks()
            .slice(0, 1)
            .flatMap((block) => block.getWaveviews())
    }

    const getCarpets = () => {
        const carpets = []
        for (const block of getRelevantBlocks()) {
            for (const carpet of block.getCarpets()) {
                carpet.zombie = !block.active || carpet.zombie1
                carpets.push(carpet)
            }
        }
        return carpets
    }

    const getDataRanges = () => {
        const ranges = new Map()
        Map.groupBy(getCarpets(), (carpet) => carpet.codes).forEach((carpets, codes) => {
            ranges.set(codes, [
                yMin.value !== null
                    ? yMin.value
                    : Math.min(...carpets.map((carpet) => carpet.ymin)),
                yMax.value !== null
                    ? yMax.value
                    : Math.max(...carpets.map((carpet) => carpet.ymax)),
            ])
        })
        return ranges
    }

    const getDataScales = () => {
        const scales = new Map()
        for (const carpet of getCarpets()) {
            const scale = carpet.yscale
            scales.set(carpet.codes, scale == (scales.get(carpet.codes) ?? scale) ? scale : 'lin')
        }
        return scales
    }

    const channels = computed(() => {
        const channels = []
        for (const gate of gates.value) {
            for (const channel of gate.channels.value) {
                channels.push(channel)
            }
        }
        return channels
    })

    const sensors = computed(() => {
        const sensors = []
        for (const gate of gates.value) {
            for (const sensor of gate.sensors) {
                sensors.push(sensor)
            }
        }
        return sensors
    })

    const codes = computed(() => {
        const codes = new Set()
        for (const gate of gates.value) {
            for (const c of gate.codes) {
                codes.add(c)
            }
        }
        return Array.from(codes)
    })

    //responses
    const responses = computed(() => {
        const responses = []
        for (const gate of gates.value) {
            for (const r of gate.responses) {
                responses.push(r)
            }
        }
        return responses
    })

    const events = computed(() => {
        const events = []
        for (const gate of gates.value) {
            for (const ev of gate.events) {
                events.push(ev)
            }
        }
        return events
    })

    const eventGroups = computed(() => {
        const groups = Array.from(Map.groupBy(events.value, (ev) => ev.extras.group_id).values())
        groups.sort((a, b) => a[0].time - b[0].time)
        return groups
    })

    const timeSpans = computed(() => {
        const spans = {
            channel: null,
            response: null,
            waveform: null,
            carpet: null,
        }
        for (const gate of gates.value) {
            for (const kind of ['channel', 'response', 'waveform', 'carpet']) {
                const span = gate.timeSpans[kind]
                if (span != null && span.tmin != null && span.tmax != null) {
                    if (spans[kind] === null) {
                        spans[kind] = span
                    } else {
                        const [tmin1, tmax1] = [spans[kind].tmin, spans[kind].tmax]
                        const [tmin2, tmax2] = [span.tmin, span.tmax]
                        spans[kind] = {
                            tmin: Math.min(tmin1, tmin2),
                            tmax: Math.max(tmax1, tmax2),
                        }
                    }
                }
            }
        }
        return spans
    })

    const contextInfos = computed(() => {
        const contextInfos = []
        for (const gate of gates.value) {
            for (const contextInfo of gate.contextInfos) {
              contextInfos.push(contextInfo)
            }
        }
        return contextInfos
    })

    watch([timeSpans], () => {
        if (!initialTimeSpanSet) {
            let tmin = null
            let tmax = null
            for (const kind of ['carpet', 'waveform']) {
                const span = timeSpans.value[kind]
                if (span != null) {
                    tmin = tmin === null ? span.tmin : Math.min(tmin, span.tmin)
                    tmax = tmax === null ? span.tmax : Math.max(tmax, span.tmax)
                }
            }
            if (tmin !== null) {
                const duration = tmax - tmin
                setTimeSpan(tmin - duration * 0.025, tmax + duration * 0.025)
                initialTimeSpanSet = true
            }
        }
    })

    const contextRequest = () => {
        return {
            time: hover.value !== null ? timeToStr(hover.value.time) : null,
            tmin: timeToStr(timeMin.value),
            tmax: timeToStr(timeMax.value),
            codes:
                hover.value !== null && hover.value.track !== null ? hover.value.track.codes : [],
            codes_visible:
                codesVisible.value,
            fmin: yMin.value,
            fmax: yMax.value,
        }
    }

    watch([hover, timeMin, timeMax, yMin, yMax], () => {
        for (const gate of gates.value) {
            gate.updateContext(contextRequest())
        }
    })

    update()

    return {
        timeMin,
        timeMax,
        hover,
        yMin,
        yMax,
        overviewMethod,
        counter,
        setTimeSpan,
        setHover,
        setImageWidth,
        setImageHeight,
        setCodesVisible,
        pageForward,
        pageBackward,
        halfPageForward,
        halfPageBackward,
        addGate,
        codes,
        codesVisible,
        channels,
        sensors,
        responses,
        events,
        eventGroups,
        timeSpans,
        getCoverages,
        getWaveviews,
        getCarpets,
        getDataRanges,
        getDataScales,
        contextInfos,
    }
}

let gates = null

export const squirrelGates = () => {
    if (gates === null) {
        gates = setupGates()
    }
    return gates
}
