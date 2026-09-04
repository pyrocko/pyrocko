const { ref, watch } = Vue
import { squirrelGates } from '../squirrel/gate.js'
import { parse_codes_filter } from './codes_expression_parser.js'

// `searchInput` is the raw text as typed, updated on every keystroke
// and possibly an incomplete/invalid codes expression. `activeQuery`
// is the last text that compiled successfully -- it lags behind
// `searchInput` while the user is mid-edit, and is what filtering
// elsewhere (here and in the timeline) actually runs against.
// `searchValid` reflects whether `searchInput` itself compiles right
// now, for UI feedback (e.g. highlighting the search box).
const searchInput = ref('')
const activeQuery = ref('')
const searchValid = ref(true)

const selectedOption = ref('Sensor')
const filteredSensors = ref([])

const sensors = squirrelGates().sensors

// Builds a `codes => bool` matcher for `query`. `query` must be empty
// or a valid codes expression -- pass `activeQuery.value`, which is
// kept in that state; for possibly-invalid, in-progress text (e.g.
// straight from a text field), use `tryCompile` instead.
function makeCodesMatcher(query) {
    const trimmed = query.trim()
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
    const matcher = tryCompile(value)
    searchValid.value = matcher !== null
    if (matcher !== null) {
        activeQuery.value = value
    }
})

watch(activeQuery, filterSensors)
watch(selectedOption, filterSensors)
watch(sensors, filterSensors, { immediate: true })

function filterSensors() {
    const matches = makeCodesMatcher(activeQuery.value)

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
        activeQuery,
        searchValid,
        selectedOption,
        filteredSensors,
        filterSensors,
        makeCodesMatcher,
    }
}
