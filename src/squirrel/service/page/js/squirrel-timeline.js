import { createIfNeeded, strToTime } from './squirrel-common.js'

export const squirrelTimeline = () => {
    let container
    let timeline
    let x = d3.scaleLinear()
    let timeMin = strToTime('1900-01-01 00:00:00')
    let timeMax = strToTime('2030-01-01 00:00:00')

    const bounds = () => {
        return container.node().getBoundingClientRect()
    }

    const getWidth = () => {
        return bounds().width
    }

    const getHeight = () => {
        return bounds().height
    }

    const setupProjection = () => {
        x.domain(timeMin, timeMax).range(0, getWidth())
    }

    const reProject = () => {
        setupProjection()
    }

    const resize = () => {
        timeline.attr('width', getWidth()).attr('height', getHeight())
        reProject()
    }
    let my = (selection) => {
        container = selection
        timeline = createIfNeeded(container, 'svg')
        window.onresize = resize
        resize()
        let boxes = [
            {
                tmin: strToTime('1977-02-20 00:00:00'),
                tmax: strToTime('2023-08-05 00:00:00'),
                ymin: 0.0,
                ymax: 1.0,
            },
        ]

        let boxWidth = (box) => {
            x(box.tmax) - x(box.tmin)
        }
        let boxHeight = (box) => {
            box.ymax - box.ymin
        }
        let boxLeft = (box) => {
            x(box.tmin)
        }
        let boxTop = (box) => {
            box.ymin
        }

        timeline.append('g').append('circle').attr('r', 5).attr('fill', '#000').attr('cx', 10).attr('cy', 20)
        console.log('xx')
        console.log(boxes)

        let xx = timeline
            .append('g')
            .data(boxes)
            .enter()
            .append('rect')
            //.attr('width', boxWidth)
            //.attr('height', boxHeight)
            //.attr('x', boxLeft)
            //.attr('y', boxTop)
            //.attr('fill', '#cde')
            //.attr('stroke', '#abc')

        console.log(xx)
    }

    my.timeMin = function (_) {
        arguments.length == 0 ? timeMin : ((timeMin = _), my)
    }

    my.timeMax = function (_) {
        arguments.length == 0 ? timeMax : ((timeMax = _), my)
    }

    return my
}
