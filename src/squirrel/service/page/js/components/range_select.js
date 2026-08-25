const { onMounted, watch } = Vue
import { squirrelRangeSelect } from './../squirrel/range_select.js'

export default {
    props: ['min', 'max'],
    emits: ['update:min', 'update:max'],
    setup: (props, ctx) => {
        const rangeSelect = squirrelRangeSelect()

        const updateRange = (range) => {
            ctx.emit('update:min', range[0])
            ctx.emit('update:max', range[1])
        }

        rangeSelect.on('brushed', updateRange)

        watch([props.min, props.max], rangeSelect.setRange)

        onMounted(() => {
            d3.select('#rangeSelect').call(rangeSelect)
        })
    },

    template: `
    <div id="rangeSelect"></div>
    `,
}
