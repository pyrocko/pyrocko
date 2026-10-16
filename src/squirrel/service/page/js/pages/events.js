const { ref, shallowRef, reactive, watch } = Vue
import { squirrelConnection } from '../squirrel/connection.js'
import {
    timeToStr,
    strToTime,
    format,
    depthThemeName,
} from '../squirrel/common.js'

const fmtMagnitude = format('3.1f')
const fmtDepth = format('4.0f')

// Number of event groups requested per page.
const PAGE_SIZE = 100

// Load the next page when the list is scrolled to within this many groups
// of its end.
const PREFETCH_MARGIN = 30

const FILTER_DEBOUNCE_MS = 400

const toNumberOrNull = (s) => {
    s = String(s ?? '').trim()
    if (s === '') {
        return null
    }
    const x = Number(s)
    if (isNaN(x)) {
        throw new Error('Invalid number: ' + s)
    }
    return x
}

const toStringOrNull = (s) => {
    s = String(s ?? '').trim()
    return s === '' ? null : s
}

export default {
    setup: () => {
        const connection = squirrelConnection()

        // Times are passed to the server as given, so that partial dates
        // like "2015" or "2015-03" can be used.
        const filter = reactive({
            tmin: '',
            tmax: '',
            magnitudeMin: '',
            magnitudeMax: '',
            depthMinKm: '',
            depthMaxKm: '',
        })

        const groups = shallowRef([])
        const next = ref(null)
        const loading = ref(false)
        const error = ref(null)

        // Incremented whenever the filter changes, so that responses to
        // requests made with an outdated filter can be dropped.
        let generation = 0

        const makeRequest = () => {
            const depthMin = toNumberOrNull(filter.depthMinKm)
            const depthMax = toNumberOrNull(filter.depthMaxKm)
            return {
                tmin: toStringOrNull(filter.tmin),
                tmax: toStringOrNull(filter.tmax),
                magnitude_min: toNumberOrNull(filter.magnitudeMin),
                magnitude_max: toNumberOrNull(filter.magnitudeMax),
                depth_min: depthMin != null ? depthMin * 1000 : null,
                depth_max: depthMax != null ? depthMax * 1000 : null,
                limit: PAGE_SIZE,
            }
        }

        const loadPage = async (reset) => {
            if (reset) {
                generation += 1
                groups.value = []
                next.value = null
            } else if (loading.value || next.value === null) {
                return
            }

            const myGeneration = generation
            let request
            try {
                request = makeRequest()
            } catch (e) {
                error.value = e.message
                return
            }
            if (!reset) {
                request.after = next.value
            }

            loading.value = true
            error.value = null
            try {
                const response = await connection.value.request(
                    'raw/get_event_groups',
                    request
                )
                if (myGeneration !== generation) {
                    return
                }
                for (const group of response.groups) {
                    for (const ev of group.events) {
                        ev.time = strToTime(ev.time)
                    }
                }
                groups.value = [...groups.value, ...response.groups]
                next.value = response.next
            } catch (e) {
                if (myGeneration === generation) {
                    error.value = e.reason || e.message
                }
            } finally {
                if (myGeneration === generation) {
                    loading.value = false
                }
            }
        }

        let debounceTimeoutId = null
        watch(filter, () => {
            clearTimeout(debounceTimeoutId)
            debounceTimeoutId = setTimeout(
                () => loadPage(true),
                FILTER_DEBOUNCE_MS
            )
        })

        loadPage(true)

        const onVirtualScroll = ({ to }) => {
            if (to >= groups.value.length - PREFETCH_MARGIN) {
                loadPage(false)
            }
        }

        const groupKey = (group) => group.key.join('@')

        const expanded = ref(new Set())

        const toggle = (group) => {
            const key = groupKey(group)
            if (expanded.value.has(key)) {
                expanded.value.delete(key)
            } else {
                expanded.value.add(key)
            }
        }

        const isExpanded = (group) => expanded.value.has(groupKey(group))

        const region = (group) => group.events.find((ev) => ev.region)?.region

        const magnitude = (ev) =>
            ev.magnitude != null
                ? (ev.magnitude_type || 'M') + ' ' + fmtMagnitude(ev.magnitude)
                : ''

        const depth = (ev) =>
            ev.depth != null ? fmtDepth(ev.depth / 1000) + ' km' : ''

        const beachballUrl = (ev) => {
            const mt = ev.moment_tensor
            if (!mt) {
                return null
            }
            const params = new URLSearchParams({
                mnn: mt.mnn,
                mee: mt.mee,
                mdd: mt.mdd,
                mne: mt.mne,
                mnd: mt.mnd,
                med: mt.med,
                theme: ev.depth != null ? depthThemeName(ev.depth) : 'black',
            })
            return 'beachball?' + params.toString()
        }

        return {
            filter,
            groups,
            next,
            loading,
            error,
            onVirtualScroll,
            groupKey,
            toggle,
            isExpanded,
            region,
            magnitude,
            depth,
            beachballUrl,
            timeToStr,
        }
    },
    template: `
<q-page class="column no-wrap" style="height: calc(100vh - 50px)">
<div class="q-px-md q-pt-md" style="max-width: calc(50rem + 32px); width: 100%; margin: 0 auto">
    <div class="row q-col-gutter-sm q-mb-sm">
        <q-input class="col-6 col-sm-3" dense clearable v-model="filter.tmin"
            label="From" placeholder="YYYY-MM-DD" />
        <q-input class="col-6 col-sm-3" dense clearable v-model="filter.tmax"
            label="Until" placeholder="YYYY-MM-DD" />
        <q-input class="col-3 col-sm" dense v-model="filter.magnitudeMin"
            label="M min" inputmode="decimal" />
        <q-input class="col-3 col-sm" dense v-model="filter.magnitudeMax"
            label="M max" inputmode="decimal" />
        <q-input class="col-3 col-sm" dense v-model="filter.depthMinKm"
            label="Depth min" suffix="km" inputmode="decimal" />
        <q-input class="col-3 col-sm" dense v-model="filter.depthMaxKm"
            label="Depth max" suffix="km" inputmode="decimal" />
    </div>

    <div class="text-caption text-grey q-mb-sm">
        <span v-if="error" class="text-negative">{{ error }}</span>
        <template v-else>
            {{ groups.length }} event group{{ groups.length === 1 ? '' : 's' }}
            {{ next !== null ? 'loaded, scroll for more' : '' }}
            <q-spinner v-if="loading" size="1em" class="q-ml-xs" />
        </template>
    </div>

</div>

    <div v-if="groups.length === 0 && !loading && !error" class="text-center text-grey q-pa-xl">
        No events found.
    </div>

    <q-virtual-scroll
        v-else
        :items="groups"
        @virtual-scroll="onVirtualScroll"
        v-slot="{ item: group }"
        style="flex: 1 1 0; min-height: 0; scrollbar-gutter: stable both-edges"
    >
        <div :key="groupKey(group)" class="q-px-md q-pb-md">
        <div class="cursor-pointer" style="max-width: 50rem; margin: 0 auto" @click="toggle(group)">
            <div class="text-subtitle1 ellipsis">{{ region(group) || '&nbsp;' }}</div>
            <div v-if="isExpanded(group)" class="text-caption text-grey">
                {{ group.key[1] }}
            </div>
            <div
                v-for="ev in group.events"
                :key="ev.name"
                class="row items-center no-wrap q-col-gutter-sm"
            >
                <div class="col">
                    <div class="row items-center no-wrap q-col-gutter-sm">
                        <div class="col-2 ellipsis">
                            <q-badge v-if="ev.catalog" outline color="primary" :label="ev.catalog" />
                        </div>
                        <div class="col-3 text-no-wrap">
                            {{ timeToStr(ev.time, '%Y-%m-%d') }}
                        </div>
                        <div class="col-3 text-no-wrap text-right text-grey">
                            {{ timeToStr(ev.time, '%H:%M:%S.3FRAC') }}
                        </div>
                        <div class="col-2 text-no-wrap">{{ magnitude(ev) }}</div>
                        <div class="col-2 text-no-wrap text-right">{{ depth(ev) }}</div>
                    </div>
                    <div v-if="isExpanded(group)" class="text-caption">
                        <div>{{ ev.lat }}, {{ ev.lon }}</div>
                        <div class="text-grey">
                            {{ ev.name }}<template v-if="ev.extras?.catalog_id">,
                            {{ ev.extras.catalog_id }}</template>
                        </div>
                    </div>
                </div>
                <div class="col-auto" style="width: 3em; height: 3em">
                    <img
                        v-if="beachballUrl(ev)"
                        :src="beachballUrl(ev)"
                        style="width: 100%; height: 100%"
                    />
                </div>
            </div>
        </div>
        </div>
    </q-virtual-scroll>
</q-page>
  `,
}
