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

        // Leaflet has no resize-detection of its own -- it has to be
        // told explicitly whenever its container's on-screen size
        // changes, for any reason (window resize, drawer width drag,
        // etc.), regardless of what CSS is doing to size it.
        const mapResized = () => {
            if (map !== null) {
                map.invalidateSize()
            }
        }

        const fitToVisibleStations = () => {
            if (map === null) {
                return
            }

            const visible = gates.visibleStationKeys.value

            const latlons = gates.stations.value
                .filter((station) => visible.has(station.key))
                .map((station) => [station.lat, station.lon])

            if (latlons.length > 0) {
                map.fitBounds(latlons, { padding: [20, 20] })
            }
        }

        const markers = new Map()

        const resetMarkers = () => {
            markers.clear()
        }

        // Adds/removes markers to match `gates.stations` (the set of
        // known stations, not how many are currently visible). This
        // only runs when that set actually changes -- see gate.js --
        // so a pan/zoom drag never touches marker creation/removal at
        // all, only `updateActiveState` below.
        const reconcileMarkers = () => {
            if (map === null || !mapContainer.value) {
                return
            }

            const stationKeys = new Set(
                gates.stations.value.map((station) => station.key)
            )

            let markersAddedOrRemoved = false

            for (const station of gates.stations.value) {
                if (!markers.has(station.key)) {
                    const marker = leaflet
                        .marker([station.lat, station.lon], {
                            icon: stationIcon,
                        })
                        .addTo(map)
                        .bindPopup(station.codes)

                    markers.set(station.key, marker)
                    markersAddedOrRemoved = true
                }
            }

            for (const key of markers.keys()) {
                if (!stationKeys.has(key)) {
                    map.removeLayer(markers.get(key))
                    markers.delete(key)
                    markersAddedOrRemoved = true
                }
            }

            if (markersAddedOrRemoved) {
                updateActiveState()
                fitToVisibleStations()
            }
        }

        // Toggles the active/inactive styling of already-existing
        // markers to match `gates.visibleStationKeys`. Runs on every
        // pan/zoom, but only ever touches a CSS class on markers that
        // already exist -- no DOM creation/removal here.
        const updateActiveState = () => {
            if (map === null) {
                return
            }

            const visible = gates.visibleStationKeys.value

            for (const [key, marker] of markers) {
                if (visible.has(key)) {
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

                mapResized()

                // The container is a fresh DOM node each time this
                // tab is reopened (it's v-if'd, not kept alive), so
                // the observer from a previous opening is watching an
                // orphaned element by now -- drop it and watch the
                // new one instead.
                if (mapResizeObserver !== null) {
                    mapResizeObserver.disconnect()
                }
                mapResizeObserver = onResizeDebounced(
                    mapContainer.value,
                    mapResized
                )

                resetMarkers()
                reconcileMarkers()
                updateActiveState()
            }
        })

        watch([gates.stations], reconcileMarkers)
        watch([gates.visibleStationKeys], updateActiveState)

        let miniMap = squirrelMap()
        onMounted(() => {
            // squirrelMap() watches its own container's size directly
            // (see map.js), so as long as CSS gives #mini-map a real
            // height -- which it now does, no extra wiring needed here.
            d3.select('#mini-map').call(miniMap)
            miniMap.addBasemap()
        })

        return { gates, scout, inspectors, timeToStr, format }
    },

    template: `
        <div id="scouts-container" class="fit vbox-container">
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

            <q-scroll-area v-if="scout == 'response'" class="vbox-main">
                <div v-for="contextInfo in gates.contextInfos.value" :key="contextInfo.name">
                    <img :src="contextInfo.image_data_base64" style="max-width: 100%" />
                </div>
            </q-scroll-area>

            <KeepAlive>
                <div v-if="scout == 'map'" class="vbox-main">
                    <div id="map-container-x" ref="map-container" style="width: 100%; height: 100%;"></div>
                </div>
            </KeepAlive>

            <div v-show="scout == 'mini-map'" class="vbox-main">
              <div id="mini-map" style="width: 100%; height: 100%;"></div>
            </div>
        </div>
    `,
}
