import {
    createRouter,
    createWebHashHistory,
} from 'vue-router'

import MainLayout from './layouts/main.js'
import TimelinePage from './pages/timeline.js'
import MapPage from './pages/map.js'
import MantrasPage from './pages/mantras.js'
import ErrorNotFound from './pages/error_not_found.js'


const routes = [
    {
        path: '/',
        component: MainLayout,
        children: [
            { path: 'timeline', component: TimelinePage },
            { path: 'map', component: MapPage },
            { path: 'mantras', component: MantrasPage },
            { path: '', redirect: '/timeline' },
        ],
    },
    {
        path: '/:catchAll(.*)*',
        component: ErrorNotFound,
    },
]

const router = createRouter({
    scrollBehavior: () => ({ left: 0, top: 0 }),
    history: createWebHashHistory(),
    routes,
})

export default router
