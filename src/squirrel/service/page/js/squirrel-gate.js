import { ref, computed } from './vue.esm-browser.js'

import { strToTime } from './squirrel-common.js'

import { squirrelConnection } from './squirrel-connection.js'

export const squirrelGate = () => {
    const codes = ref(Set())
    const connection = squirrelConnection()

    const fetchCodes = async () => {
        for (const kind of ['waveform', 'channel', 'response']) {
            for (const c of await connection.request('raw/get_codes', {
                kind: kind,
            })) {
                codes.add(c)
            }
        }
        return codes
    }

    const update = async () => {
        const codes_new = await fetchCodes()
        codes.value = codes_new
    }
}

export const squirrelGates = () => {
    const gates = ref([])
    const timeMin = ref(strToTime('1900-01-01 00:00:00'))
    const timeMax = ref(strToTime('2030-01-01 00:00:00'))
    const setTimeSpan = (min, max) => {
        timeMin.value = min
        timeMax.value = max
    }

    const addGate = () => {
        gates.value.push(squirrelGate())
    }

    const codes = computed({
        const codes = Set()
        for (const gate of gates) {
            for (const c of gate.codes) {
                codes.add(c)
            }
        }
        return codes
    })

    return { timeMin, timeMax, setTimeSpan, addGate }
}
