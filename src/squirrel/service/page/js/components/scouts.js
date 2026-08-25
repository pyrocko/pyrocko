const { ref, watch, useTemplateRef, onMounted } = Vue

import { squirrelGates } from './../squirrel/gate.js'
import { squirrelMap } from './../squirrel/map.js'
import { timeToStr, format } from './../squirrel/common.js'

export default {
    setup: () => {
        const gates = squirrelGates()

        const inspectors = ['coordinates', 'response', 'map', 'mini-map']

        const scout = ref('coordinates')

        const mapContainer = useTemplateRef('map-container')

        const defaultIcon = leaflet.icon({
            iconUrl:
                'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowUrl:
                'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.3/images/marker-shadow.png',
            shadowSize: [41, 41],
        })

        const inactiveIcon = leaflet.icon({
            iconUrl:
                'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-grey.png',
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowUrl:
                'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.3/images/marker-shadow.png',
            shadowSize: [41, 41],
        })

        defaultIcon, inactiveIcon

        watch([mapContainer], () => {
            if (mapContainer.value) {
                const map = leaflet.map(mapContainer.value, {
                    center: [51.505, -0.09],
                    zoom: 13,
                })

                leaflet
                    .tileLayer(
                        'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
                        {
                            maxZoom: 19,
                            attribution:
                                '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>',
                        }
                    )
                    .addTo(map)

                const updateMarkers = () => {
                    for (const sensor of gates.sensors.value) {
                        leaflet
                            .marker([sensor.lat, sensor.lon], {
                                icon: defaultIcon,
                            })
                            .addTo(map)
                            .bindPopup(sensor.codes)
                    }
                }

                updateMarkers()
            }
        })

        let miniMap = squirrelMap()
        onMounted(() => {
            d3.select('#mini-map').call(miniMap)
            miniMap.addBasemap()
        })

        return {gates, scout, inspectors, timeToStr, format}
    },

    template: `
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
      <div id="mini-map" class="map-container vbox-main tab-pane" style="width: 100%; height: 50vh;"></div>
    </div>
    `,
}
