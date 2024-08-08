import {
    createIfNeeded,
    strToTime,
    colors,
    niceTimeTickInc,
    timeTickLabels,
} from './squirrel-common.js'

export const squirrelTimeline = () => {
    // parameters
    let timeMin = strToTime('1900-01-01 00:00:00')
    let timeMax = strToTime('2030-01-01 00:00:00')
    let trackMin = 0
    let trackMax = 3
    let marginTop = 150 
    let marginBottom = 150
    let padding = 10
    let tickLength = padding
    let fontSize = 12

    // private
    let container
    let timeline
    let boxesGroup
    let axesGroup
    let x = d3.scaleLinear()
    let y = d3.scaleLinear()
    let trackStart = null

    const bounds = () => {
        return container.node().getBoundingClientRect()
    }

    const getWidth = () => {
        return bounds().width
    }

    const getHeight = () => {
        return bounds().height
    }

    const updateProjection = () => {
        x.domain([timeMin, timeMax]).range([0, getWidth()])
        y.domain([trackMin, trackMax]).range([
            marginTop,
            getHeight() - (marginTop + marginBottom),
        ])
    }

    const update = () => {
        updateProjection()
        updateBoxes()
        updateAxes()
    }

    const resize = () => {
        timeline.attr('width', getWidth()).attr('height', getHeight())
        update()
    }

    const pointerDownHandler = (ev) => {
        container.node().setPointerCapture(ev.pointerId)
        trackStart = {
            position: [ev.clientX, ev.clientY],
            domain: [...x.domain()],
            mode:
                ev.clientY > getHeight() - marginBottom
                    ? 'global_fixed'
                    : 'global',
        }
    }

    const pointerMoveHandler = (ev) => {
        if (trackStart) {
            let x1 = ev.clientX
            let y1 = ev.clientY
            let x0 = trackStart.position[0]
            let y0 = trackStart.position[1]
            let w = getWidth()
            let h = getHeight()
            let tmin0 = trackStart.domain[0]
            let tmax0 = trackStart.domain[1]
            let mode = trackStart.mode
            let scale, xfrac, dx, dy, dt, dtr

            dx = (x1 - x0) / w
            dy = (y1 - y0) / h
            xfrac = x0 / w
            scale = mode == 'global' ? (scale = Math.exp(-dy * 5)) : 1.0
            dtr = (tmax0 - tmin0) * (scale - 1.0)
            dt = dx * (tmax0 - tmin0) * scale
            timeMin = tmin0 - dt - dtr * xfrac
            timeMax = tmax0 - dt + dtr * (1 - xfrac)

            update()
        }
    }

    const pointerUpHandler = (ev) => {
        trackStart = null
    }

    let boxes = [
        {
            id: 0,
            tmin: strToTime('1977-02-20 00:00:00'),
            tmax: strToTime('2023-08-05 00:00:00'),
            ymin: 0.0,
            ymax: 1.0,
        },
        {
            id: 1,
            tmin: strToTime('1939-01-01 00:00:00'),
            tmax: strToTime('1945-08-05 00:00:00'),
            ymin: 1.0,
            ymax: 2.0,
        },
    ]

    const boxWidth = (box) => {
        return x(box.tmax) - x(box.tmin)
    }
    const boxHeight = (box) => {
        return y(box.ymax) - y(box.ymin) - padding * 2
    }
    const boxLeft = (box) => {
        return x(box.tmin)
    }
    const boxTop = (box) => {
        return y(box.ymin) + padding
    }

    const updateBoxes = () => {
        boxesGroup
            .selectAll('rect')
            .data(boxes, (box) => box.id)
            .join('rect')
            .attr('width', boxWidth)
            .attr('height', boxHeight)
            .attr('x', boxLeft)
            .attr('y', boxTop)
            .attr('fill', '#cde')
            .attr('stroke', '#abc')
    }

    const updateAxes = () => {
        const axisY = (i) => {
            return [getHeight() - marginBottom + padding, marginTop - padding][
                i
            ]
        }

        let napprox = 5
        let [tinc, tinc_units] = niceTimeTickInc((timeMax - timeMin) / napprox)
        let [times, labels] = timeTickLabels(timeMin, timeMax, tinc, tinc_units)
        let ticks = times.map((t, i) => ({
            t: t,
            labels: ('' + labels[i]).split('\n').reverse(),
        }))

        let axisGroups = axesGroup
            .selectAll('.axis-group')
            .data([0, 1], (axisId) => axisId)
            .join((enter) => enter.append('g').attr('class', 'axis-group'))

        axisGroups
            .selectAll('.axis-line')
            .data((axisId) => [axisId])
            .join((enter) =>
                enter
                    .append('line')
                    .attr('class', 'axis-line')
                    .attr('stroke', colors['aluminium4'])
                    .attr('stroke-width', 3.0)
            )
            .attr('x1', 0.0)
            .attr('x2', getWidth)
            .attr('y1', axisY)
            .attr('y2', axisY)

        axisGroups
            .selectAll('.axis-tick')
            .data(
                (axisId) =>
                    ticks
                        .filter((tick) => tick.t > timeMin)
                        .map((tick) => ({ axisId, tick })),
                (d) => d.tick.t
            )
            .join('line')
            .attr('class', 'axis-tick')
            .attr('stroke', colors['aluminium4'])
            .attr('stroke-width', 3.0)
            .attr('y1', (d) => axisY(d.axisId))
            .attr(
                'y2',
                (d) =>
                    axisY(d.axisId) + (d.axisId == 0 ? tickLength : -tickLength)
            )
            .attr('x1', (d) => x(d.tick.t))
            .attr('x2', (d) => x(d.tick.t))

        axisGroups
            .filter((d) => d == 0)
            .selectAll('.axis-tick-label-group')
            .data(
                (axisId) => ticks.map((tick) => ({ axisId, tick })),
                (d) => d.tick.t
            )
            .join('g')
            .attr('class', 'axis-tick-label-group')
            .selectAll('.axis-tick-label')
            .data(
                (etick) =>
                    etick.tick.labels.map((label, ilabel) => ({
                        ...etick,
                        label,
                        ilabel,
                    })),
                (d) => d.ilabel
            )
            .join('text')
            .attr('class', 'axis-tick-label')
            .attr(
                'y',
                (d) => axisY(d.axisId) + (d.axisId == 0 ? 1 : -1) * tickLength
            )
            .attr(
                'dy',
                (d) => (d.axisId == 0 ? 1 : -1) * (0.5 + 1.5 * d.ilabel) + 'em'
            )
            .attr('x', (d) => x(d.tick.t))
            .attr('dx', (d) => '0.5em')
            .attr('dx', (d) => d.tick.t == timeMin ? '0.5em' : '0')
            .style('font-size', fontSize + 'pt')
            .style('text-anchor', (d) => d.tick.t == timeMin ? 'left' : 'middle')
            .style('dominant-baseline', (d) =>
                d.axisId == 0 ? 'text-before-edge' : 'no-change'
            )
            .text((d) => d.label)

        axisGroups
    }

    let my = (selection) => {
        container = selection
        timeline = createIfNeeded(container, 'svg')

        //let group = timeline.append('g')
        //group
        //    .selectAll('circle')
        //    .data([10, 20, 30], (d) => d)
        //    .join(
        //        enter => {console.log('enter 1 '); console.log(enter); return enter.append('circle').attr('fill', 'red')},
        //        update => {console.log('update 1'); return update.attr('fill', 'black')})
        //    .attr('r', 5)
        //    .attr('cx', 20)
        //    .attr('cy', (d) => d)
        //group
        //    .selectAll('circle')
        //    .data([10, 20, 30, 40], (d) => d)
        //    .join(
        //        enter => {console.log('enter 2 '); console.log(enter); return enter.append('circle').attr('fill', 'red')},
        //        update => {console.log('update 2'); return update.attr('fill', 'black')})
        //    .attr('r', 5)
        //    .attr('cx', 40)
        //    .attr('cy', (d) => d)
        //
        boxesGroup = timeline.append('g')
        axesGroup = timeline.append('g')

        container.on('pointerdown', pointerDownHandler)
        container.on('pointerup', pointerUpHandler)
        container.on('pointermove', pointerMoveHandler)

        window.onresize = resize
        resize()
    }

    my.timeMin = function (_) {
        arguments.length == 0 ? timeMin : ((timeMin = _), my)
    }

    my.timeMax = function (_) {
        arguments.length == 0 ? timeMax : ((timeMax = _), my)
    }

    return my
}
