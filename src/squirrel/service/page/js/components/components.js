import {
    ref,
    shallowRef,
    computed,
    onMounted,
    onActivated,
    watch,
} from '../vue.esm-browser.js'
import { squirrelMap } from '../squirrel/map.js'
import { squirrelTimeline } from '../squirrel/timeline.js'
import { squirrelRangeSelect } from '../squirrel/range_select.js'
import { squirrelGates } from '../squirrel/gate.js'
import { squirrelConnection } from '../squirrel/connection.js'
import { useFilters } from '../squirrel/filter.js'
import { timeToStr } from '../squirrel/common.js'


const positiveOrNull = (s) => {
    if (s.trim() == '') {
        return null
    }
    const x = Number(s)
    if (!Number.isFinite(x)) {
        throw new Error('Invalid number: ' + s)
    }
    if (x <= 0) {
        throw new Error('Number is zero or negative: ' + x)
    }
    return x
}

export const ComponentRangeSelect = {
    props: ['min', 'max'],

    setup(props) {
        let rangeSelect = squirrelRangeSelect()

        const updateRange = (range) => {
            props.min.value = range[0]
            props.max.value = range[1]
        }

        rangeSelect.on('brushed', updateRange)

        watch([props.min, props.max], rangeSelect.setRange)

        onMounted(() => {
            d3.select('#rangeSelect').call(rangeSelect)
        })
        return {}
    },
    template: `
      <div id="rangeSelect"></div>
    `,
}

export const componentTimeline = {
    label: '≋', // ⩫
    components: {
        ComponentRangeSelect,
    },
    setup() {
        const gates = squirrelGates()
        const timeline = squirrelTimeline()

        onMounted(() => {
            d3.select('#timeline').call(timeline)
        })

        onActivated(() => timeline.activate)

        const yMinInput = ref('')
        const yMaxInput = ref('')
        const yMinError = ref(null)
        const yMaxError = ref(null)

        let muteIn = false
        let muteOut = false

        const propagate = () => {
            if (muteOut) {
                return
            }
            let yMin
            let yMax
            try {
                yMin = positiveOrNull(yMinInput.value)
                yMinError.value = null
            } catch (e) {
                yMinError.value = e
            }
            try {
                yMax = positiveOrNull(yMaxInput.value)
                yMaxError.value = null
            } catch (e) {
                yMaxError.value = e
            }

            if (yMinError.value !== null || yMaxError.value !== null) {
                return
            }
            try {
                if (yMin !== null && yMax !== null && yMax < yMin) {
                    throw new Error('Invalid entries: yMax < yMin')
                }
                yMinError.value = null
                yMaxError.value = null
                muteIn = true
                gates.yMin.value = yMin
                gates.yMax.value = yMax
                muteIn = false
            } catch (e) {
                yMinError.value = e
                yMaxError.value = e
            }
        }

        watch([yMinInput, yMaxInput], propagate)

        const propagateIn = () => {
            if (muteIn) {
                return
            }
            const fmt = d3.format('.4g')
            muteOut = true
            if (gates.yMin.value === null) {
                yMinInput.value = ''
            } else {
                yMinInput.value = fmt(gates.yMin.value)
            }
            if (gates.yMax.value === null) {
                yMaxInput.value = ''
            } else {
                yMaxInput.value = fmt(gates.yMax.value)
            }
            muteOut = false
            yMinError.value = null
            yMaxError.value = null
        }

        watch([gates.yMin, gates.yMax], propagateIn, { flush: 'sync' })

        const blur = (e) => {
            e.target.blur()
        }

        return { yMinInput, yMaxInput, yMinError, yMaxError, gates, overviewMethod: gates.overviewMethod, blur}
    },
    template: `
        <div id="timeline" tabindex="0" class="vbox-main tab-pane"></div>
        <div class="container-fluid bg-light pt-2 pb-2" style="border-top: 1px solid #eee;">
            <div class="form-group row">
                <div class="col-4 col-md-2">
                    <input placeholder="fₘᵢₙ" type="text" class="form-control" @keyup.enter="blur" :class="{ 'input-error': yMinError }" v-model="yMinInput" />
                </div>
                <div class="d-none d-md-block col-4 col-md-6">
                    <component-range-select :min="gates.yMin" :max="gates.yMax" style="height: 3em;"></component-range-select>
                </div>
                <div class="col-4 col-md-2">
                    <input placeholder="fₘₐₓ" type="text" class="form-control" @keyup.enter="blur" :class="{ 'input-error': yMaxError }" v-model="yMaxInput" />
                </div>
                <div class="col-4 col-md-2">
                    <select v-model="overviewMethod" class="form-select">
                        <option value="mean">Mean</option>
                        <option value="min">Min</option>
                        <option value="max">Max</option>
                    </select>
                </div>
            </div>
        </div>
    `,
}

export const componentMap = {
    label: '⦾', // ⦾ ⦿ ⊙

    setup() {
        let map = squirrelMap()
        onMounted(() => {
            d3.select('#map').call(map)
            map.addBasemap()
        })

        return {}
    },
    template: `
      <div id="map" class="map-container vbox-main tab-pane"></div>
    `,
}

export const componentFilter = {
    label: '🔍',
    setup() {
        const { searchQuery, selectedOption, filterSensors } = useFilters()

        const options = ['Filter 1', 'Filter 2']

        const searchHistory = ref([])
        const typingTimer = ref(null)
        const searchDelay = 1000

        const saveSearchHistory = () => {
            if (
                searchQuery.value.trim() &&
                !searchHistory.value.includes(searchQuery.value)
            ) {
                searchHistory.value.unshift(searchQuery.value.trim())
                searchHistory.value = searchHistory.value.slice(0, 3)
                sessionStorage.setItem(
                    'searchHistory',
                    JSON.stringify(searchHistory.value)
                )
            }
        }

        const onSearchInput = () => {
            clearTimeout(typingTimer.value)

            typingTimer.value = setTimeout(() => {
                saveSearchHistory()
            }, searchDelay)
        }

        const onSearchFinalize = (e) => {
            e.target.blur()
            filterSensors()

        }

        onMounted(() => {
            const storedHistory = sessionStorage.getItem('searchHistory')
            if (storedHistory) {
                searchHistory.value = JSON.parse(storedHistory)
            }
        })


        return {
            searchQuery,
            selectedOption,
            searchHistory,
            saveSearchHistory,
            onSearchInput,
            onSearchFinalize,
        }
    },
    template: `<div class="d-flex justify-content-end">
            <input list="filters" type="search" class="form-control" placeholder="" v-model="searchQuery" @input="onSearchInput" @keyup.enter="onSearchFinalize"/>
            <datalist id="filters">
                <option v-for="(historyItem, index) in searchHistory" :key="index" :value="historyItem"></option>
                <option value="..Z"></option>
                <option value="..[EN]"></option>
            </datalist>
    </div>`,
}

export const componentTabs = {
    components: {
        componentFilter
    },
    setup() {
        const selectedTab = ref('timeline')

        const tabs = shallowRef({
            timeline: componentTimeline,
            map: componentMap,
            info: componentTable,
        })

        const selectTab = (tabName) => {
            selectedTab.value = tabName
        }

        const addTab = () => {
            const newTabName = `tab_${Date.now()}`
            tabs.value[newTabName] = componentTable
        }

        const closeTab = (tabName, event) => {
            event.stopPropagation()
            delete tabs.value[tabName]

            if (selectedTab.value === tabName) {
                selectedTab.value = 'info'
            }
        }

        const isDefaultTab = (name) => {
            return ['timeline', 'map', 'info'].includes(name)
        }

        const openInfoTab = ({sensor,channel,sortedSensors}) => {
            const tabName = channel.codes
            const componentInfo = {
                setup() {
                    onMounted(() => {
                        const defaultIcon = L.icon({
                        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
                        iconSize: [25, 41],
                        iconAnchor: [12, 41],
                        popupAnchor: [1, -34],
                        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.3/images/marker-shadow.png',
                        shadowSize: [41, 41]
                    });

                    const inactiveIcon = L.icon({
                        iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-grey.png',
                        iconSize: [25, 41],
                        iconAnchor: [12, 41],
                        popupAnchor: [1, -34],
                        shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.3/images/marker-shadow.png',
                        shadowSize: [41, 41]
                    });


                        let map = L.map('leaflet-map').setView([channel.lat,channel.lon], 13)
                        // let marker = L.marker([channel.lat, channel.lon]).addTo(map)
                        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
                            maxZoom: 19,
                            attribution: '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                            }).addTo(map)

                        sortedSensors.forEach(s => {
                            const sensorSelected = s.codes == sensor.codes
                            L.marker([s.lat, s.lon], {
                                icon: sensorSelected ? defaultIcon : inactiveIcon
                            }).addTo(map).bindPopup(s.codes)
                        })
                    })
                return {sensor,channel,sortedSensors}
                },
            template: `
            <div class="row h-100 w-100">
                <div class="col-12 col-md-6 p-5" style="max-height: 80vh; overflow-y: auto;">
                    <h5>Sensor Details</h5>
                    <table class="table table-sm table-bordered">
                        <tbody>
                            <tr><th>Codes</th><td>{{ sensor.codes }}</td></tr>
                            <tr v-if="sensor.lat != null"><th>Latitude</th><td>{{ sensor.lat }}</td></tr>
                            <tr v-if="sensor.lon != null"><th>Longitude</th><td>{{ sensor.lon }}</td></tr>
                            <tr v-if="sensor.depth != null"><th>Depth</th><td>{{ sensor.depth }}</td></tr>
                            <tr v-if="sensor.elevation != null"><th>Elevation</th><td>{{ sensor.elevation }}</td></tr>
                            <tr v-if="sensor.north_shift != null"><th>North Shift</th><td>{{ sensor.north_shift }}</td></tr>
                            <tr v-if="sensor.east_shift != null"><th>East Shift</th><td>{{ sensor.east_shift }}</td></tr>
                            <tr v-if="sensor.tmin != null"><th>Start Time</th><td>{{ sensor.tmin }}</td></tr>
                            <tr v-if="sensor.tmax != null"><th>End Time</th><td>{{ sensor.tmax }}</td></tr>
                            <tr v-if="sensor.deltat != null"><th>Delta T</th><td>{{ sensor.deltat }}</td></tr>

                        </tbody>
                    </table>

                    <h5>Channel Details</h5>
                    <table class="table table-sm table-bordered">
                        <tbody>
                            <tr><th>Codes</th><td>{{ channel.codes }}</td></tr>
                            <tr v-if="channel.lat != sensor.lat && channel.lat != null"><th>Latitude</th><td>{{ channel.lat }}</td></tr>
                            <tr v-if="channel.lon != sensor.lon && channel.lon != null"><th>Longitude</th><td>{{ channel.lon }}</td></tr>
                            <tr v-if="channel.depth != sensor.depth && channel.depth != null"><th>Depth</th><td>{{ channel.depth }}</td></tr>
                            <tr v-if="channel.elevation != sensor.elevation && channel.elevation != null"><th>Elevation</th><td>{{ channel.elevation }}</td></tr>
                            <tr v-if="channel.north_shift != sensor.north_shift && channel.north_shift != null"><th>North shift</th><td>{{ channel.north_shift }}</td></tr>
                            <tr v-if="channel.east_shift != sensor.east_shift && channel.east_shift != null"><th>East shift</th><td>{{ channel.east_shift }}</td></tr>
                            <tr v-if="channel.tmin != sensor.tmin && channel.tmin != null"><th>Start Time</th><td>{{ channel.tmin }}</td></tr>
                            <tr v-if="channel.tmax != sensor.tmax && channel.tmax != null"><th>End Time</th><td>{{ channel.tmax }}</td></tr>
                            <tr v-if="channel.deltat != sensor.deltat && channel.deltat != null"><th>Delta T</th><td>{{ channel.deltat }}</td></tr>
                            <tr v-if="channel.dip != null"><th>Dip</th><td>{{ channel.dip }}</td></tr>
                            <tr v-if="channel.azimuth != null"><th>Azimuth</th><td>{{ channel.azimuth }}</td></tr>
                        </tbody>
                    </table>
                </div>

                <div id="leaflet-map" class="col-12 col-md-6"></div>
            </div>
        `,
            }
        tabs.value[tabName] = componentInfo
        selectedTab.value = tabName
    }
    return {selectedTab, selectTab, tabs, addTab, closeTab, isDefaultTab, openInfoTab}
    },

    template: `
            <nav class="navbar navbar-expand-md navbar-light bg-light">
                <div class="container-fluid">
                    <button
                        class="navbar-toggler"
                        type="button"
                        data-bs-toggle="collapse"
                        data-bs-target="#navbarNav"
                    >
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarNav">
                        <ul class="navbar-nav me-auto">
                            <li
                                class="nav-item has-close-btn"
                                v-for="(tab, name) in tabs"
                                :key="name"
                            >
                                <a
                                    class="nav-link"
                                    :class="{ active: selectedTab === name }"
                                    @click="selectTab(name)"
                                    style="
                                        margin-top: -1rem;
                                        margin-bottom: -1rem;
                                        font-size: 2rem;
                                    "
                                >
                                    <span v-if="tab.label">{{ tab.label }}</span>
                                    <span v-if="!tab.label">{{ name.charAt(0).toUpperCase() + name.slice(1) }}</span>
                                    <span class="close-btn" v-if="!isDefaultTab(name)" @click="closeTab(name, $event)">&times;</span>
                                </a>
                            </li>
                        </ul>
                    </div>
                    <component-filter class="ms-auto"></component-filter>
                </div>
            </nav>

            <!--
           <ul class="nav nav-tabs mt-1" role="tablist">
                <li class="nav-item" v-for="(tab, name) in tabs" :key="name">
                    <a
                        class="nav-link"
                        :class="{ active: selectedTab === name }"
                        @click="selectTab(name)"
                        >{{ name.charAt(0).toUpperCase() + name.slice(1) }}
                        <button
                            v-if="!isDefaultTab(name)"
                            @click="closeTab(name, $event)"
                            class="close-btn"
                        >
                            &times;
                        </button>
                    </a>
                </li>
                <button class="add-tab-btn" @click="addTab()">+</button>
                <component-filter class="ms-auto"></component-filter>
            </ul>
            -->

            <keep-alive>
                <component :is="tabs[selectedTab]" @open-tab="openInfoTab"></component>
            </keep-alive>`
}




export const componentTable = {
    label: '⟁',
    emits: ['open-tab'],

    setup() {
        const { filteredSensors, selectedOption } = useFilters()
        const sortTable = (sortValue) => {
            if (sortValue === currentSort.value) {
                currentSortDir.value =
                    currentSortDir.value === 'asc' ? 'desc' : 'asc'
            } else {
                currentSort.value = sortValue
                currentSortDir.value = 'asc'
            }
        }

        const setOption = (option) => {
            selectedOption.value = option
        }

        const gates = squirrelGates()

        const sensors = gates.sensors
        const responses = gates.responses
        const channels = gates.channels
        const currentSort = ref('codes')
        const currentSortDir = ref('asc')
        const responsesMap = ref({})
        const noResults = computed(() => filteredSensors.value.length === 0)

        // watch(sensors, () => {
        //     filteredSensors.value = [...sensors.value]
        // })

        const sortedSensors = computed(() => {
            return [...filteredSensors.value].sort((a, b) => {
                let modifier = 1
                if (currentSortDir.value === 'desc') modifier = -1

                if (a[currentSort.value] < b[currentSort.value])
                    return -1 * modifier
                if (a[currentSort.value] > b[currentSort.value])
                    return 1 * modifier
                return 0
            })
        })

        const mapResponses = () => {
            responsesMap.value = responses.value.reduce((map, response) => {
                map[response.codes] = response
                return map
            }, {})
        }

        watch(responses, mapResponses, { immediate: true })

        const formatResponse = (response) => {
            const stage = response.stages[0]
            return `${stage.input_quantity} -> ${stage.output_quantity}`
        }

        return {
            sortTable,
            setOption,
            currentSort,
            currentSortDir,
            selectedOption,
            sortedSensors,
            responses,
            responsesMap,
            formatResponse,
            noResults,
        }
    },
    template: `



    <div class="vbox-main tab-pane sensor-table">
                    <table class="table">
                        <thead>
                            <tr>
                                <th scope="col">
                                    <div class="dropdown">
                                        <button
                                            class="btn btn-outline-primary dropdown-toggle"
                                            type="button"
                                            data-bs-toggle="dropdown"

                                        >
                                            {{ selectedOption }}
                                        </button>
                                        <ul class="dropdown-menu">
                                            <li><a class="dropdown-item" href="#" @click="setOption('Sensor')">Sensor</a></li>
                                            <li><a class="dropdown-item" href="#" @click="setOption('Channel')">Channel</a></li>
                                        </ul>
                                    </div>
                                </th>


                                <th scope="col" @click="sortTable('lat')" style="text-align: right">Latitude     <span v-if="currentSort === 'lat'">
                                    {{ currentSortDir === 'asc' ? '▲' : '▼' }}
                                </span></th>
                                <th scope="col" @click="sortTable('lon')" style="text-align: right">Longitude     <span v-if="currentSort === 'lon'">
                                    {{ currentSortDir === 'asc' ? '▲' : '▼' }}
                                </span></th>
                                <th scope="col" @click="sortTable('tmin')" style="text-align: right">Start<span v-if="currentSort === 'tmin'">
                                    {{ currentSortDir === 'asc' ? '▲' : '▼' }}
                                </span></th>
                                <th scope="col" @click="sortTable('tmin')" style="text-align: right">End<span v-if="currentSort === 'tmin'">
                                    {{ currentSortDir === 'asc' ? '▲' : '▼' }}
                                </span></th>
                                <th scope="col" @click="sortTable('deltat')" style="text-align: right">ΔT [Hz]<span v-if="currentSort === 'deltat'">
                                    {{ currentSortDir === 'asc' ? '▲' : '▼' }}
                                </span></th>
                                <th scope="col" @click="sortTable('flag')">Flag<span v-if="currentSort === 'flag'">
                                    {{ currentSortDir === 'asc' ? '▲' : '▼' }}
                                </span></th>
                                <th v-if="selectedOption === 'Channel'" scope="col" @click="sortTable('responses')">Response<span v-if="currentSort === 'response'">
                                    {{ currentSortDir === 'asc' ? '▲' : '▼' }}
                                </span></th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-if="noResults">
                                    <td colspan="8" class="text-center text-muted">No sensors found.</td>
                                </tr>
                            <template v-for="sensor in sortedSensors">

                                <tr v-for="channel in sensor.channels">

                                    <td class="table-code" @click="$emit('open-tab', {sensor,channel,sortedSensors})">

                                        <template v-if="selectedOption === 'Sensor'">
                                            {{sensor.codes}}
                                        </template>
                                        <template v-if="selectedOption === 'Channel'">
                                            {{channel.codes}}
                                        </template>
                                        <template v-if="selectedOption === 'Response'">
                                            {{}}
                                        </template>

                                    </td>
                                      <td style="text-align: right">{{ channel.lat }}</td>
                                      <td style="text-align: right">{{ channel.lon }}</td>
                                      <td style="text-align: right">{{channel.tmin}}</td>
                                      <td style="text-align: right">{{channel.tmax ? channel.tmax : ""}}</td>
                                      <td style="text-align: right">{{1/sensor.deltat}}</td>
                                      <td>
                                        <span v-if="channel.azimuth !== 0 && channel.azimuth !== 90 || channel.dip !== 0 && channel.dip !== -90" data-bs-toggle="tooltip" :title="'Unusual Orientation: Azimuth ' + channel.azimuth + ', Dip ' + channel.dip">&#x2221;</span>
                                      </td>
                                    <td v-if="selectedOption === 'Channel'">
                                        <template v-if="responsesMap[channel.codes]">
                                            {{ formatResponse(responsesMap[channel.codes]) }}
                                        </template>
                                    </td>


                                    </tr>
                                      </template>



                        </tbody>
                    </table>
                </div>
    `,
}

export const componentCatalog = {
    label: '✩', // ★
    setup() {
        const gates = squirrelGates()

        const event_groups = gates.eventGroups

        const fmt = (format, value) => {
            return d3.format(format)(value)
        }

        const toParams = (o) => {
            return new URLSearchParams(o).toString()
        }

        const depth_scale = d3
            .scaleThreshold()
            .domain([15e3, 30e3, 60e3, 120e3, 240e3, 480e3])
            .range([
                'brick',
                'sienna',
                'ochre',
                'foliage',
                'ocean',
                'sky',
                'plum',
            ])

        const get_m6 = ({ mnn, mee, mdd, mne, mnd, med }) => ({
            mnn,
            mee,
            mdd,
            mne,
            mnd,
            med,
        })

        const beachball_link = (ev) => {
            if (ev.moment_tensor) {
                return (
                    'beachball?' +
                    toParams(get_m6(ev.moment_tensor)) +
                    '&' +
                    toParams({ theme: depth_scale(ev.depth) })
                )
            } else {
                return ''
            }
        }

        const get_region = (group) => {
            for (let i = 0; i < group.length; i++) {
                if (group[i].region) {
                    return group[i].region
                }
            }
        }

        return { event_groups, timeToStr, fmt, beachball_link, get_region }
    },
    template: `

            <div class="mb-4" v-for="event_group in event_groups">
                <div
                    class="clickable text-nowrap"
                    @click="event_group.details = !event_group.details;"
                >
                    {{ get_region(event_group) }}
                </div>
                <div
                    v-if="event_group.details"
                    class="text-nowrap"
                    style="font-size: small"
                >
                    {{ event_group[0].extras.group_id }}
                </div>
                <div
                    class="row align-items-center"
                    v-for="event in event_group"
                >
                    <!--<div class="col-3" style="color: gray; font-size: 70%">{{ event.name }} {{event.extras.catalog_id }}</div>-->
                    <div class="col-9 col-md-10">
                        <div
                            class="row align-items-center clickable"
                            @click="event_group.details = !event_group.details"
                        >
                            <!--<div class="col-6 col-md-3" style="color: gray; font-size: 70%">{{event.extras.catalog_id }}</div>-->
                            <div class="col-3 col-md-2 varsize text-nowrap">
                                <span class="badge">{{ event.catalog }}</span>
                            </div>
                            <div class="col-4 col-md-3 varsize text-nowrap">
                                {{ timeToStr(event.time, '%Y-%m-%d') }}
                            </div>
                            <div
                                class="col-5 col-md-3 varsize text-nowrap text-end"
                                style="color: gray"
                            >
                                {{ timeToStr(event.time, '%H:%M:%S.3FRAC') }}
                            </div>
                            <div class="col-6 col-md-2 text-nowrap">
                                {{ event.magnitude_type || 'M' }} {{ fmt('3.1f',
                                event.magnitude) }}
                            </div>
                            <div class="col-6 col-md-2 text-end text-nowrap">
                                {{ fmt('4.0f', event.depth / 1000.) }} km
                            </div>
                        </div>
                        <div
                            v-if="event_group.details"
                            style="font-size: small"
                        >
                            <div>{{ event.lat }}, {{ event.lon }}</div>
                            <div style="font-size: small">
                                {{ event.name }}, {{ event.extras.catalog_id }}
                            </div>
                        </div>
                    </div>
                    <div
                        class="col-3 col-md-2 text-center"
                        style="padding: 0em"
                    >
                        <img
                            style="height: 100%"
                            :src="beachball_link(event)"
                        />
                    </div>
                </div>
            </div>

    `,
}
