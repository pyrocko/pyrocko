const { ref, watch, useTemplateRef, onMounted } = Vue

import { squirrelGates } from './../squirrel/gate.js'
import { squirrelMap } from './../squirrel/map.js'
import { timeToStr, format, onResizeDebounced } from './../squirrel/common.js'

export default {
    setup: () => {
        const gates = squirrelGates()

        const inspectors = ['coordinates', 'response', 'map', 'mini-map']

        const scout = ref('coordinates')

        const mapContainer = useTemplateRef('map-container')

        const stationIcon = leaflet.divIcon({
            className: 'station-icon',
            html:
                '<svg width="20" height="18" viewBox="0 0 20 18">' +
                '<polygon points="10,1 19,17 1,17" stroke-width="1.5" ' +
                'stroke-linejoin="round" /></svg>',
            iconSize: [20, 18],
            iconAnchor: [10, 12],
            popupAnchor: [0, -12],
        })

        let map = null
        let mapResizeObserver = null

        const resizeMapToFit = () => {
            if (map === null || !mapContainer.value) {
                return
            }

            const scrollArea = mapContainer.value.closest('.q-scrollarea')
            if (!scrollArea) {
                return
            }

            const bottom = scrollArea.getBoundingClientRect().bottom
            const top = mapContainer.value.getBoundingClientRect().top
            mapContainer.value.style.height = Math.max(200, bottom - top) + 'px'

            map.invalidateSize()
        }

        const stationKey = (codes) => codes.split('.').slice(0, 3).join('.')

        const fitToVisibleStations = () => {
            if (map === null) {
                return
            }

            const visible = new Set(
                (gates.codesVisible.value ?? []).map(stationKey)
            )

            const latlons = gates.sensors.value
                .filter((sensor) => visible.has(stationKey(sensor.codes)))
                .map((sensor) => [sensor.lat, sensor.lon])

            if (latlons.length > 0) {
                map.fitBounds(latlons, { padding: [20, 20] })
            }
        }

        const markers = new Map()

        const resetMarkers = () => {
            markers.clear()
        }

        const updateMarkers = () => {
            if (map === null || !mapContainer.value) {
                return
            }

            const visible = new Set(
                (gates.codesVisible.value ?? []).map(stationKey)
            )
            const isVisible = (sensor) =>
                visible.has(stationKey(sensor.codes)) &&
                (sensor.tmin === null || sensor.tmin < gates.timeMax.value) &&
                (sensor.tmax === null || sensor.tmax > gates.timeMin.value)

            const markerKeysAll = new Set(
                gates.sensors.value.map((sensor) => sensor.markerKey)
            )

            const markerKeysVisible = new Set(
                gates.sensors.value
                    .filter(isVisible)
                    .map((sensor) => sensor.markerKey)
            )

            let markersAddedOrRemoved = false

            for (const sensor of gates.sensors.value) {
                let marker = null
                if (!markers.has(sensor.markerKey)) {
                    marker = leaflet.marker([sensor.lat, sensor.lon], {
                        icon: stationIcon,
                    })

                    marker.addTo(map).bindPopup(sensor.codes)
                    markers.set(sensor.markerKey, marker)
                    markersAddedOrRemoved = true
                } else {
                    marker = markers.get(sensor.markerKey)
                }

                if (markerKeysVisible.has(sensor.markerKey)) {
                    leaflet.DomUtil.removeClass(
                        marker._icon,
                        'station-icon-inactive'
                    )
                } else {
                    leaflet.DomUtil.addClass(
                        marker._icon,
                        'station-icon-inactive'
                    )
                }
            }

            for (const key of markers.keys()) {
                if (!markerKeysAll.has(key)) {
                    map.removeLayer(markers.get(key))
                    markers.delete(key)
                    markersAddedOrRemoved = true
                }
            }

            if (markersAddedOrRemoved) {
                fitToVisibleStations()
            }
        }

        watch([mapContainer], () => {
            if (mapContainer.value) {
                map = leaflet.map(mapContainer.value, {
                    center: [0, 0],
                    zoom: 2,
                })

                leaflet
                    .tileLayer(
                        'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                        {
                            maxZoom: 19,
                            attribution:
                                '&copy; <a href="http://www.openstreetmap.org/copyright">OSM</a>',
                        }
                    )
                    .addTo(map)

                resizeMapToFit()

                if (mapResizeObserver === null) {
                    const scrollArea =
                        mapContainer.value.closest('.q-scrollarea')
                    if (scrollArea) {
                        mapResizeObserver = onResizeDebounced(
                            scrollArea,
                            resizeMapToFit
                        )
                    }
                }

                resetMarkers()
                updateMarkers()
            }
        })

        watch(
            [gates.codesVisible, gates.sensors, gates.timeMin, gates.timeMax],
            updateMarkers
        )

        let miniMap = squirrelMap()
        onMounted(() => {
            d3.select('#mini-map').call(miniMap)
            miniMap.addBasemap()
        })

        return { gates, scout, inspectors, timeToStr, format }
    },

    template: `
        <div id="scouts-container">
            <div style="padding: 0.5rem">
                <q-select v-model="scout" :options="inspectors"> </q-select>
            </div>

            <div v-if="scout == 'coordinates'" style="padding: 0.5rem">
                {{ gates.hover.value ? timeToStr(gates.hover.value.time) : '-' }},
                {{
                    gates.hover.value && gates.hover.value.y !== null
                        ? format('.3g')(gates.hover.value.y)
                        : '-'
                }}
            </div>

            <div v-if="scout == 'response'">
                <div v-for="contextInfo in gates.contextInfos.value" :key="contextInfo.name">
                    <img :src="contextInfo.image_data_base64" style="max-width: 100%" />
                </div>
            </div>

            <KeepAlive>
                <div v-if="scout == 'map'">
                    <div id="map-container-x" ref="map-container" style="width: 100%; height: 50vh;"></div>
                </div>
            </KeepAlive>

            <div v-show="scout == 'mini-map'">
              <div id="mini-map" class="vbox-main tab-pane" style="width: 100%; height: 50vh;"></div>
            </div>
        </div>
    `,
}
