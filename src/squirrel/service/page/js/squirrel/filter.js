const { ref, watch } = Vue
import { squirrelGates } from '../squirrel/gate.js'
import { parse_codes_filter } from './codes_expression_parser.js'

const searchInput = ref('')  // state of the text box
const searchActive = ref('')
const searchValid = ref(true)

const selectedOption = ref('Sensor')
const filteredSensors = ref([])

const sensors = squirrelGates().sensors

function makeCodesMatcher(query) {
    const trimmed = (query ?? '').trim()
    return trimmed === '' ? () => true : parse_codes_filter(trimmed)
}

// Tries to compile `query` into a matcher. Returns the matcher on
// success, or null if `query` is not (yet) a valid codes expression.
function tryCompile(query) {
    try {
        return makeCodesMatcher(query)
    } catch {
        return null
    }
}

watch(searchInput, (value) => {
    // Normalize null/undefined to '' here so searchActive -- which
    // other code (e.g. quick_filter.js's saveSearchHistory) also
    // calls .trim() on directly -- is always a string.
    const text = value ?? ''
    const matcher = tryCompile(text)
    searchValid.value = matcher !== null
    if (matcher !== null) {
        searchActive.value = text
    }
})

watch(searchActive, filterSensors)
watch(selectedOption, filterSensors)
watch(sensors, filterSensors, { immediate: true })

function filterSensors() {
    const matches = makeCodesMatcher(searchActive.value)

    if (selectedOption.value === 'Sensor') {
        filteredSensors.value = sensors.value.filter((sensor) =>
            matches(sensor.codes)
        )
    } else if (selectedOption.value === 'Channel') {
        filteredSensors.value = sensors.value
            .map((sensor) => {
                const matchingChannels = sensor.channels.filter((channel) =>
                    matches(channel.codes)
                )
                if (matchingChannels.length > 0) {
                    return { ...sensor, channels: matchingChannels }
                }
                return null
            })
            .filter((sensor) => sensor !== null)
    }
}

export function useFilters() {
    return {
        searchInput,
        searchActive,
        searchValid,
        selectedOption,
        filteredSensors,
        filterSensors,
        makeCodesMatcher,
    }
}
