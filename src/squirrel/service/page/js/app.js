const { createApp, ref, onMounted } = Vue
import router from './router.js'

const app = createApp({
    setup() {

    },
})

app.use(router)
app.use(Quasar)

app.mount('#q-app')
