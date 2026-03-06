const { ref, markRaw } = Vue

import { VueFlow } from '@vue-flow/core'

// these components are only shown as examples of how to use a custom node or edge
// you can find many examples of how to create these custom components in the examples page of the docs
import SpecialNode from '../components/special_node.js'

// import squirrelIcon from '../assets/squirrel.svg'

export default {
    components: {
        VueFlow,
    },
    setup: () => {
        const nodeTypes = {
            special: markRaw(SpecialNode),
        }
        const nodes = ref([
            {
                id: '1',
                type: 'special',
                position: { x: 100, y: 50 },
                data: {
                    //iconFile: squirrelIcon,
                    subtype: 'input',
                    //icon: 'egg',
                    label: 'Squirrel',
                },
            },

            {
                id: '2',
                type: 'special',
                position: { x: 100, y: 150 },
                data: { label: 'Restitution', icon: 'factory' },
            },

            {
                id: '3',
                type: 'special',
                position: { x: 100, y: 250 },
                data: { label: 'ToENZ', icon: 'factory' },
            },

            {
                id: '4',
                type: 'special',
                position: { x: 100, y: 350 },
                label: 'View',
                data: {
                    subtype: 'output',
                    icon: 'view_agenda',
                    label: 'Timeline',
                },
            },
        ])

        const edges = ref([
            {
                id: 'e1->2',
                source: '1',
                target: '2',
            },
            {
                id: 'e2->3',
                source: '2',
                target: '3',
            },
            {
                id: 'e3->4',
                source: '3',
                target: '4',
            },
        ])

        return { edges, nodes, nodeTypes }
    },
    template: `
  <q-page class="flex flex-center">
    <vue-flow
      class="bg-grey-3"
      :nodes="nodes"
      :edges="edges"
      :node-types="nodeTypes"
      style="width: 100%; height: calc(100vh - 50px)"
    >
    </vue-flow>
  </q-page>
  `,
}
