const { ref, watch } = Vue
import { squirrelConnection } from './../squirrel/connection.js'
import { squirrelGates } from './../squirrel/gate.js'
import { fmtDuration } from './../squirrel/common.js'
import ComponentQuickFilter from '../components/quick_filter.js'
import ComponentScouts from '../components/scouts.js'

export default {
    components: {
        ComponentScouts,
        ComponentQuickFilter,
    },
    setup: (props) => {
        const $q = Quasar

        const leftDrawerOpen = ref(false)
        const rightDrawerOpen = ref(false)
        const fullscreen_mode = ref(false)
        const dark_mode = ref($q.Dark.isActive)

        function toggleLeftDrawer() {
            leftDrawerOpen.value = !leftDrawerOpen.value
        }

        function toggleRightDrawer() {
            rightDrawerOpen.value = !rightDrawerOpen.value
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
        gates.addGate()

        // The inspector panel (context info) lives in the right drawer;
        // no point fetching it while nobody can see it.
        watch(rightDrawerOpen, gates.setContextEnabled, { immediate: true })

        const update_dark_mode = (dark_mode) => {
            $q.Dark.set(dark_mode)
        }

        watch(dark_mode, update_dark_mode)

        const update_fullscreen_mode = (fullscreen_mode) => {
            const fullscreen_mode_active = $q.AppFullscreen.isActive
            if (fullscreen_mode) {
                if (!fullscreen_mode_active) {
                    $q.AppFullscreen.request().catch(() => {
                        fullscreen_mode.value = false
                    })
                }
            } else {
                $q.AppFullscreen.exit()
            }
        }

        watch(
            () => $q.AppFullscreen.isActive,
            (val) => {
                if (fullscreen_mode.value != val) {
                    fullscreen_mode.value = $q.AppFullscreen.isActive
                }
            }
        )

        watch(fullscreen_mode, update_fullscreen_mode)

        const disconnected_dialog = ref(true)

        const update_disconnected_dialog = (connected) => {
            if (!connected) {
                disconnected_dialog.value = true
            }
        }

        watch(() => connection.value.connected, update_disconnected_dialog)

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
            dark_mode,
            fullscreen_mode,
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
                            <q-toggle v-model="dark_mode" label="Dark Mode" left-label />
                        </q-item>
                        <q-item>
                            <q-toggle v-model="fullscreen_mode" label="Fullscreen" left-label />
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

                    <span v-if="connection.connected">
                        <q-chip
                            style="min-width: 8em"
                            icon="pause"
                            color="red"
                            v-if="connection.connected.delay > 2.0"
                        >
                            {{ fmtDuration(connection.connected.delay) }}
                        </q-chip>
                        <q-chip
                            style="min-width: 8em"
                            icon="build"
                            :color="connection.activeRequests > 4 ? 'red' : ''"
                        >
                            {{ connection.activeRequests }}
                        </q-chip>
                        <q-chip style="min-width: 8em" icon="functions">
                            {{ connection.serverInfo ? connection.serverInfo.n_requests : '' }}
                        </q-chip>
                        <q-chip style="min-width: 8em" icon="cloud">
                            {{ fmtDuration(connection.connected.duration) }}
                        </q-chip>
                        <q-chip style="min-width: 8em"
                            >v{{ connection.serverInfo.pyrocko_version }}
                        </q-chip>
                    </span>
                    <span v-else>
                        <q-chip icon="cloud_off" color="yellow"> Disconnected </q-chip>
                    </span>
                    <q-btn flat dense round icon="menu" aria-label="Menu" @click="toggleRightDrawer" />
                </q-toolbar>
            </q-footer>
        </q-layout>
    `,
}
