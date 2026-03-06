import { createIfNeeded } from './common.js'


export const squirrelRangeSelect = () => {
    let svg
    let container
    let bounds
    let group_axis
    let group_brush
    let brush
    let margin = {top: 0, right: 10, bottom: 25, left: 10}
    let spacing = 3
    let handlers = {}
    let muted = false

    let brushOverlayStroke = 'none'
    let brushOverlayFill = '#eeeeeb'

    let brushSelectionStroke = 'none'
    let brushSelectionFill = '#998'
    

    const scale = d3.scaleLog([0.00001, 1000], [0, 1])

    const resizeHandler = () => {
        bounds = container.node().getBoundingClientRect()
        svg.attr('width', bounds.width).attr('height', bounds.height)
        scale.range([margin.left, bounds.width - margin.right])
        group_axis.attr('transform', `translate(0, ${bounds.height - margin.bottom})`).call(d3.axisBottom(scale))
        group_axis.selectAll('.domain').attr('stroke', 'none')

        brush.extent([[margin.left, margin.top], [bounds.width - margin.right, bounds.height - margin.bottom - spacing]])
        group_brush.call(brush).selectAll('.overlay').attr('stroke', brushOverlayStroke).attr('fill', brushOverlayFill)
        group_brush.call(brush).selectAll('.selection').attr('stroke', brushSelectionStroke).attr('fill', brushSelectionFill)
        update()
    }

    const brushed = (event) => {
        if (muted) {
            return
        }
        if (event.selection === null) {
            handlers['brushed']([null, null])
        } else {
            const range = event.selection.map(scale.invert)
            if (handlers['brushed'] !== null) {
                handlers['brushed'](range)
            }
        }
    }

    const update = () => {
    }

    const my = (selection) => {
        container = selection
        svg = createIfNeeded(container, 'svg')
        group_axis = svg.append('g')
        group_brush = svg.append('g')

        brush = d3.brushX()
        brush.on('start brush end', brushed)

        window.addEventListener('resize', resizeHandler)
        resizeHandler()
    }

    my.setRange = (range) => {
        const [min, max] = range
        if (min === null && max === null) {
            group_brush.call(brush.clear)
        } else {
            const [cmin, cmax] = scale.domain()
            const srange = [min !== null ? min : cmin, max !== null ? max : cmax]
            muted = true
            group_brush.call(brush.move, srange.map(scale))
            muted = false
        }
    }

    my.on = (eventName, handler) => {
        handlers[eventName] = handler
    }

    return my
}
