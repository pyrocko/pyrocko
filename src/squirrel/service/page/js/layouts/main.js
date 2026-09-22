const { ref, watch } = Vue
import { squirrelConnection } from './../squirrel/connection.js'
import { squirrelGates } from './../squirrel/gate.js'
import { useScoutNav } from './../squirrel/scout_nav.js'
import { fmtDuration } from './../squirrel/common.js'
import ComponentQuickFilter from '../components/quick_filter.js'
import ComponentScouts from '../components/scouts.js'

export default {
    components: {
        ComponentScouts,
        ComponentQuickFilter,
    },
    setup: () => {
        const $q = Quasar

        const leftDrawerOpen = ref(false)
        const rightDrawerOpen = ref(false)
        const fullscreenMode = ref(false)
        const darkMode = ref($q.Dark.isActive)

        function toggleLeftDrawer() {
            leftDrawerOpen.value = !leftDrawerOpen.value
        }

        function toggleRightDrawer() {
            rightDrawerOpen.value = !rightDrawerOpen.value
        }

        const { scout, scoutList, pinnedScoutCount } = useScoutNav()

        function selectScout(value) {
            scout.value = value
            rightDrawerOpen.value = true
        }

        // Quasar's QDrawer has no built-in resize handle, so this is a
        // small drag handle on its left edge, driven by v-touch-pan.
        const RIGHT_DRAWER_WIDTH_MIN = 250
        const RIGHT_DRAWER_WIDTH_MAX = 800

        const rightDrawerWidth = ref(400)
        const rightDrawerResizing = ref(false)
        let rightDrawerWidthAtDragStart = rightDrawerWidth.value

        function resizeRightDrawer(ev) {
            if (ev.isFirst) {
                rightDrawerWidthAtDragStart = rightDrawerWidth.value
                // QDrawer's width is transitioned by Quasar's own CSS,
                // which would otherwise make it lag behind the pointer
                // while dragging.
                rightDrawerResizing.value = true
            }

            // Right-side drawer: dragging the handle left (negative
            // offset) widens the drawer, since its right edge stays
            // pinned to the window edge.
            rightDrawerWidth.value = Math.min(
                RIGHT_DRAWER_WIDTH_MAX,
                Math.max(
                    RIGHT_DRAWER_WIDTH_MIN,
                    rightDrawerWidthAtDragStart - ev.offset.x
                )
            )

            if (ev.isFinal) {
                rightDrawerResizing.value = false
            }
        }

        const connection = squirrelConnection()
        const gates = squirrelGates()
        gates.loadGates()

        // The inspector panel (context info) lives in the right drawer;
        // no point fetching it while nobody can see it.
        watch(rightDrawerOpen, gates.setContextEnabled, { immediate: true })

        const updateDarkMode = (darkMode) => {
            $q.Dark.set(darkMode)
        }

        watch(darkMode, updateDarkMode)

        const updateFullscreenMode = (active) => {
            const fullscreenModeActive = $q.AppFullscreen.isActive
            if (active) {
                if (!fullscreenModeActive) {
                    $q.AppFullscreen.request().catch(() => {
                        fullscreenMode.value = false
                    })
                }
            } else {
                $q.AppFullscreen.exit()
            }
        }

        watch(
            () => $q.AppFullscreen.isActive,
            (val) => {
                if (fullscreenMode.value != val) {
                    fullscreenMode.value = $q.AppFullscreen.isActive
                }
            }
        )

        watch(fullscreenMode, updateFullscreenMode)

        return {
            connection,
            dark: $q.Dark,
            fmtDuration,
            toggleRightDrawer,
            toggleLeftDrawer,
            leftDrawerOpen,
            rightDrawerOpen,
            rightDrawerWidth,
            rightDrawerResizing,
            resizeRightDrawer,
            scout,
            scoutList,
            pinnedScoutCount,
            selectScout,
            darkMode,
            fullscreenMode,
        }
    },

    template: `
        <q-layout view="hHh LpR fFf">
            <q-drawer v-model="leftDrawerOpen" bordered>
                <q-scroll-area class="fit" :horizontal-thumb-style="{ opacity: 0 }">
                    <q-list padding>
                        <q-item clickable v-ripple to="/timeline">
                            <q-item-section avatar>
                                <q-icon name="view_agenda" />
                            </q-item-section>

                            <q-item-section>Timeline</q-item-section>
                        </q-item>

                        <q-item clickable v-ripple to="/map">
                            <q-item-section avatar>
                                <q-icon name="map" />
                            </q-item-section>

                            <q-item-section>Map</q-item-section>
                        </q-item>

                        <q-item clickable v-ripple to="/mantras">
                            <q-item-section avatar>
                                <q-icon name="settings" />
                            </q-item-section>

                            <q-item-section>Mantras</q-item-section>
                        </q-item>

                        <q-item>
                            <q-toggle v-model="darkMode" label="Dark Mode" left-label />
                        </q-item>
                        <q-item>
                            <q-toggle v-model="fullscreenMode" label="Fullscreen" left-label />
                        </q-item>
                    </q-list>
                </q-scroll-area>
            </q-drawer>

            <q-drawer
                side="right"
                v-model="rightDrawerOpen"
                bordered
                :width="rightDrawerWidth"
                :breakpoint="500"
                :class="[
                    dark.isActive ? 'bg-grey-9' : 'bg-grey-3',
                    rightDrawerResizing ? 'no-drawer-transition' : '',
                ]"
            >
                <component-scouts></component-scouts>

                <div
                    class="drawer-resize-handle"
                    v-touch-pan.preserveCursor.prevent.mouse.horizontal="resizeRightDrawer"
                ></div>
            </q-drawer>

            <q-page-container>
                <router-view v-slot="{ Component }">
                    <keep-alive>
                        <component :is="Component" />
                    </keep-alive>
                </router-view>
            </q-page-container>

            <div @click="connection.connect()" v-if="!connection.connected" class="curtain">
                <div>DISCONNECTED</div>
                <div>
                    <small>Click to reconnect.</small>
                </div>
            </div>
            <div v-if="connection.connected && connection.connected.delay > 5.0" class="curtain">
                DELAY {{ fmtDuration(connection.connected.delay) }}
            </div>

            <q-footer class="bg-20">
                <q-toolbar style="justify-content: space-between">
                    <q-btn flat dense round icon="menu" aria-label="Menu" @click="toggleLeftDrawer" />
                    <q-toolbar-title>
                        <component-quick-filter></component-quick-filter>
                    </q-toolbar-title>

                    <q-chip
                        style="min-width: 5em"
                        icon="build"
                        :color="connection.activeRequests > 4 ? 'red' : ''"
                    >
                        {{ connection.activeRequests }}
                    </q-chip>

                    <q-btn
                        v-for="s in scoutList.slice(0, pinnedScoutCount)"
                        :key="s.value"
                        flat
                        dense
                        round
                        :icon="s.icon"
                        :color="scout === s.value ? 'primary' : ''"
                        @click="selectScout(s.value)"
                    >
                        <q-tooltip>{{ s.label }}</q-tooltip>
                    </q-btn>

                    <q-btn flat dense round icon="more_vert">
                        <q-tooltip>More</q-tooltip>
                        <q-menu>
                            <q-list>
                                <q-item
                                    v-for="s in scoutList.slice(pinnedScoutCount)"
                                    :key="s.value"
                                    clickable
                                    v-close-popup
                                    :active="scout === s.value"
                                    @click="selectScout(s.value)"
                                >
                                    <q-item-section avatar>
                                        <q-icon :name="s.icon" />
                                    </q-item-section>
                                    <q-item-section style="white-space: nowrap"
                                        >{{ s.label }}</q-item-section
                                    >
                                </q-item>
                            </q-list>
                        </q-menu>
                    </q-btn>

                    <q-btn flat dense round icon="menu" aria-label="Menu" @click="toggleRightDrawer" />
                </q-toolbar>
            </q-footer>
        </q-layout>
    `,
}
