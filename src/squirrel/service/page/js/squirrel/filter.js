import { ref, computed, watch } from '../vue.esm-browser.js'
import { squirrelGates } from '../squirrel/gate.js'

const searchQuery = ref('')
const selectedOption = ref('Sensor')
const filteredSensors = ref([])

const sensors = squirrelGates().sensors

watch(searchQuery, filterSensors)
watch(selectedOption, filterSensors)
watch(sensors, filterSensors, { immediate: true })

function filterSensors() {
    const query = searchQuery.value.trim()

    //check if regex
    let regex = null
    try {
        regex = new RegExp(query, 'i')
    } catch (error) {
        //no regex
    }

    if (selectedOption.value === 'Sensor') {
        filteredSensors.value = sensors.value.filter(sensor =>
            regex ? regex.test(sensor.codes) : sensor.codes.toLowerCase().includes(query.toLowerCase())
        )
    } else if (selectedOption.value === 'Channel') {
        filteredSensors.value = sensors.value.map(sensor => {
            const matchingChannels = sensor.channels.filter(channel =>
                regex ? regex.test(channel.codes) : channel.codes.toLowerCase().includes(query.toLowerCase())
            )
            if (matchingChannels.length > 0) {
                return { ...sensor, channels: matchingChannels }
            }
            return null
        }).filter(sensor => sensor !== null)
    }
}

export function useFilters() {
    return {
        searchQuery,
        selectedOption,
        filteredSensors,
        filterSensors
    }
}
