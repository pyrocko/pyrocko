import { Position, Handle } from '@vue-flow/core'

export default {
    components: {
        Handle,
    },
    props: {
        data: {
            type: Object,
            required: true,
        },
    },
    setup: (props) => {
        return { props, Position }
    },
    template: `
    <div>
  <handle v-if="props.data.subtype != 'input'" type="target" :position="Position.Top" />
  <img v-if="props.data.iconFile" style="width: 3em; height: 3em" :src="props.data.iconFile" />
  <q-icon v-if="props.data.icon" style="font-size: 3em" :name="props.data.icon" />
    <span style="margin-left: 0.5em;"> {{ props.data.label }}</span>
  <handle v-if="props.data.subtype != 'output'" type="source" :position="Position.Bottom" />
  </div>
`,
}
