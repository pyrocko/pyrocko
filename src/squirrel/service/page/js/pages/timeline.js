const { onMounted, onActivated, ref, watch } = Vue
import { squirrelTimeline } from '../squirrel/timeline.js'
import { squirrelGates } from '../squirrel/gate.js'
import { positiveOrNull } from '../squirrel/common.js'
import ComponentRangeSelect from '../components/range_select.js'

export default {
    components: {
        ComponentRangeSelect,
    },
    setup: (props) => {
        const gates = squirrelGates()
        const timeline = squirrelTimeline()

        onMounted(() => {
            d3.select('#timeline').call(timeline)
        })

        onActivated(timeline.activate)

        const yMinInput = ref('')
        const yMaxInput = ref('')
        const yMinError = ref(null)
        const yMaxError = ref(null)
        const overviewMethod = gates.overviewMethod

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

        const resizeTweak = (offset) => {
            return {
                minHeight: '0px',
                height: offset ? `calc(100dvh - ${offset}px)` : '100dvh',
            }
        }

        const blur = (ev) => {
            ev.target.blur()
        }

        return {
            blur,
            gates,
            overviewMethod,
            resizeTweak,
            yMaxError,
            yMaxInput,
            yMinError,
            yMinInput,
        }
    },

    template: `
    <q-page :style-fn="resizeTweak" class="vbox-container">
        <div tabindex="0" class="vbox-main" id="timeline"></div>
        <div class="frequency-panel" style="display: flex; align-content: stretch; padding: 1em; gap: 1em">
            <q-input
                placeholder="fₘᵢₙ"
                type="text"
                @keyup.enter="blur"
                :class="{ 'input-error': yMinError }"
                v-model="yMinInput"
            />
            <component-range-select
                :min="gates.yMin"
                :max="gates.yMax"
                class="range-select"
                style="height: 5em; flex: 1 1 auto; min-width: 0px"
                @update:min="($event) => (gates.yMin.value = $event)"
                @update:max="($event) => (gates.yMax.value = $event)"
            ></component-range-select>
            <q-input
                placeholder="fₘₐₓ"
                type="text"
                @keyup.enter="blur"
                :class="{ 'input-error': yMaxError }"
                v-model="yMaxInput"
            />
            <q-select v-model="overviewMethod" :options="['mean', 'min', 'max']"> </q-select>
        </div>
    </q-page>
    `,
}
