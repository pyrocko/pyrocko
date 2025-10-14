import { watch, shallowRef } from '../vue.esm-browser.js'
import { now, arraysEqual, onResizeDebounced } from './common.js'

import {
    createIfNeeded,
    strToTime,
    colors,
    niceTimeTickInc,
    timeTickLabels,
    tomorrow,
} from './common.js'
import { squirrelConnection } from './connection.js'
import { squirrelGates } from './gate.js'
import { useFilters } from './filter.js'

const projectionHelper = () => {
    let scale = d3.scaleLinear()
    let limits = [0, 1]
    let discrete = true
    let attribute = (obj) => obj.index
    let spanMin = 1
    const my = (obj) => {
        return scale(attribute(obj) + 0.5)
    }

    my.range = scale.range

    my.trackHeight = () => {
        return scale(scale.domain()[0] + 1) - scale(scale.domain()[0])
    }

    my.lower = (obj) => {
        return scale(attribute(obj))
    }

    my.upper = (obj) => {
        return scale(attribute(obj) + 1)
    }

    my.relative = (x) => {
        const [xmin, xmax] = scale.range()
        return (x - xmin) / (xmax - xmin)
    }

    my.domainSpan = (x) => {
        const [xmin, xmax] = scale.domain()
        return xmax - xmin
    }

    my.zoom = (anchor, delta) => {
        const domain = scale.domain()
        const spanMax = limits[1] - limits[0]
        let span = domain[1] - domain[0]

        if (
            (span <= spanMin && delta <= 0) ||
            (span >= spanMax && delta >= 0)
        ) {
            return
        }

        span += discrete ? Math.round(delta) : delta
        span = Math.min(Math.max(spanMin, span), spanMax)

        let xmin = scale.domain()[0]
        let xminNew = Math.max(0, xmin - anchor * delta)
        let xmaxNew = xminNew + span
        if (xmaxNew > spanMax) {
            xminNew -= xmaxNew - spanMax
            xmaxNew -= xmaxNew - spanMax
        }

        if (discrete) {
            xminNew = Math.round(xminNew)
            xmaxNew = Math.round(xmaxNew)
        }

        scale.domain([xminNew, xmaxNew])
    }


    my.scroll = (delta) => {
        if (discrete) {
            delta = Math.round(delta)
        }
        const domain = scale.domain()
        const deltaMin = Math.min(0, limits[0] - domain[0])
        const deltaMax = Math.max(0, limits[1] - domain[1])
        delta = Math.max(deltaMin, Math.min(delta, deltaMax))
        domain[0] += delta
        domain[1] += delta
        scale.domain(domain)
    }

    my.domain = function (_) {
        if (arguments.length == 0) {
            return scale.domain()
        }
        limits = [..._]
        scale.domain(limits)
        return my
    }
    my.min = function (_) {
        return arguments.length == 0 ? min : ((min = _), my)
    }

    my.max = function (_) {
        return arguments.length == 0 ? max : ((max = _), my)
    }

    my.attribute = function (_) {
        return arguments.length == 0 ? attribute : ((attribute = _), my)
    }

    my.scale = scale
    return my
}

export const squirrelTimeline = () => {
    // parameters

    let gates = squirrelGates()

    let marginTop = 100
    let marginBottom = 110
    let trackPadding = 5
    let tickLength = 10
    let axisTickStrokeWidth = 2
    let axisLineWidth = 2
    let fontSize = 12
    let labelPadding = [0.4, 0.2]
    let labelBackgroundFill = '#fff7'
    let labelBackgroundStroke = 'none'

    // private
    let container
    let timeline
    let defs
    let pageRect
    let boxesGroup
    let imageGroup
    let timeAxesGroup
    let tracksGroup
    let x = d3.scaleLinear()
    let y = d3.scaleLinear()
    let trackStart = null
    let pinchStart = null
    let pinchScaleStart = null
    let pinchFinalize = null
    let codes
    let codesToTracks = new Map()
    let tracks = []
    let trackProjection = projectionHelper()
    let effectiveTrackPadding = trackPadding
    let bounds
    let dataRanges = new Map()
    let scrollDeltaY = 0
    let scrollTime = 0
    let updateTimeoutId = null

    let deactivateScrollMarginsTimeoutId = null
    let needScrollMargins = false

    const pointerEvents = []
    const trackHeight = shallowRef(100)
    const trackWidth = shallowRef(800)
    const timeSpan = shallowRef([0, 1])
    const visibleCodes = shallowRef([])

    const visibleTracks = () => {
        return tracks.filter(trackVisible)
    }

    const containerBounds = () => {
        return container.node().getBoundingClientRect()
    }

    const updateTrackHeight = () => {
        trackHeight.value =
            trackProjection.trackHeight() - 2 * effectiveTrackPadding
    }

    const updateVisibleCodes = () => {
        const visible = visibleTracks()
            .map((track) => track.codes)
            .flat()
        visible.sort()
        if (!arraysEqual(visibleCodes.value, visible)) {
            visibleCodes.value = visible
        }
    }

    const updateDataRanges = () => {
        dataRanges = gates.getDataRanges()
    }

    const makeEffectiveTrackPadding = () => {
        return Math.max(
            1,
            Math.min(
                trackPadding,
                ((trackProjection.range()[1] - trackProjection.range()[0]) /
                    (trackProjection.domain()[1] -
                        trackProjection.domain()[0]) /
                    1.333) *
                    0.2
            )
        )
    }

    const updateProjection = () => {
        x.domain([gates.timeMin.value, gates.timeMax.value]).range([
            0,
            bounds.width,
        ])
        trackProjection.range([marginTop, bounds.height - marginBottom])
        effectiveTrackPadding = makeEffectiveTrackPadding()
        pageRect.attr('x', 0)
        pageRect.attr('y', marginTop)
        pageRect.attr('width', bounds.width)
        pageRect.attr('height', bounds.height - marginTop - marginBottom)
    }

    const update = () => {
        const t = d3.transition('update').duration(50).ease(d3.easeLinear)

        updateVisibleCodes()
        updateTrackHeight()

        updateDataRanges()
        updateProjection()
        updateBoxes(t)
        updateTimeAxes()
        updateTracks(t)
    }

    const updateSoon = () => {
        if (updateTimeoutId !== null) {
             clearTimeout(updateTimeoutId)
        }
        updateTimeoutId = setTimeout(() => {
            update()
            updateTimeoutId = null
        }, 100)
    }

    const resizeHandler = () => {
        bounds = containerBounds()
        if (bounds.width <= 0 || bounds.height <= 0) {
            return
        }
        timeline.attr('width', bounds.width).attr('height', bounds.height)
        trackWidth.value = bounds.width
        update()
    }

    const keyHandlers = {
        ' ': gates.halfPageForward,
        b: gates.halfPageBackward,
        PageUp: () => scrollTracks(-1.0),
        PageDown: () => scrollTracks(1.0),
    }

    const keyDownHandler = (ev) => {
        const handler = keyHandlers[ev.key]
        if (handler) {
            handler()
        }
    }

    const pointerDownHandler = (ev) => {
        pointerEvents.push(ev)
        container.node().setPointerCapture(ev.pointerId)
        if (pointerEvents.length == 1) {
            trackStart = {
                position: [ev.clientX, ev.clientY],
                domain: [...x.domain()],
                mode:
                    ev.clientY > bounds.height - marginBottom
                        ? 'global_fixed'
                        : 'global',
            }
        } else {
            trackStart = null
        }
    }

    const pointerMoveHandler = (ev) => {
        for (let i=0; i<pointerEvents.length; i++) {
            if (ev.pointerId == pointerEvents[i].pointerId) {
                pointerEvents[i] = ev;
                break
            }
        }

        if (pointerEvents.length == 1) {
            if (trackStart) {
                let x1 = ev.clientX
                let y1 = ev.clientY
                let x0 = trackStart.position[0]
                let y0 = trackStart.position[1]
                let w = bounds.width
                let h = bounds.height
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
                let timeMin = tmin0 - dt - dtr * xfrac
                let timeMax = tmax0 - dt + dtr * (1 - xfrac)
                timeSpan.value = [timeMin, timeMax]
            }

        } else if (pointerEvents.length == 2) {
            let pinch = [
                [pointerEvents[0].clientX, pointerEvents[0].clientY],
                [pointerEvents[1].clientX, pointerEvents[1].clientY],
            ];

            if (pinchStart == null) {
                pinchStart = pinch;
                pinchScaleStart = trackProjection.scale.copy()
            } else {
                const [[p0x, p0y], [p1x, p1y]] = pinch
                const [[s0x, s0y], [s1x, s1y]] = pinchStart
                let f = 1.0
                if (Math.abs(s1y-s0y) > 2 * Math.abs(s1x-s0x)) {
                    // vertical pinch
                    f = Math.abs(p1y-p0y) / Math.abs(s1y-s0y)
                }

                const csy = 0.5 * (s0y + s1y)
                const cpy = 0.5 * (p0y + p1y)

                const cs = pinchScaleStart.invert(csy)
                const cp = pinchScaleStart.invert(cpy)

                const [a, b] = pinchScaleStart.domain()

                let aNew = cs + 1.0 / f * (a - cs) - (cp - cs)
                let bNew = cs + 1.0 / f * (b - cs) - (cp - cs)

                let dNew = bNew - aNew
                dNew = Math.round(dNew)

                if (dNew < 1) {
                    dNew = 1
                }

                if (dNew > tracks.length) {
                    dNew = tracks.length
                }

                if (aNew + dNew > tracks.length) {
                    bNew = tracks.length
                    aNew = bNew - dNew
                }

                if (aNew < 0) {
                    aNew = 0
                    bNew = aNew + dNew
                }

                let aNewRounded = Math.round(aNew)
                if (Math.abs(aNewRounded - aNew) < 0.2) {
                    aNew = aNewRounded
                }
                bNew = aNew + dNew

                let aNewFinal = aNewRounded
                let bNewFinal = aNewFinal + dNew

                trackProjection.domain([aNew, bNew])
                update()

                pinchFinalize = () => {
                    trackProjection.domain([aNewFinal, bNewFinal])
                    update()
                }
            }
        }
    }

    const pointerUpHandler = (ev) => {
        for (let i=0; i<pointerEvents.length; i++) {
            if (pointerEvents[i].pointerId == ev.pointerId) {
                pointerEvents.splice(i, 1);
                break
            }
        }
        trackStart = null
        if (pointerEvents.length < 2) {
            pinchStart = null
        }
        if (pinchFinalize !== null) {
            pinchFinalize()
            pinchFinalize = null
        }
    }

    const scrollHandler = (ev) => {
        ev.preventDefault()

        const tnow = now()
        if (tnow - scrollTime > 500) {
            scrollDeltaY = 0
        }

        scrollTime = tnow

        scrollDeltaY += ev.deltaY

        const step = 200

        if (Math.abs(scrollDeltaY) < step) {
            return
        }

        const iamount = Math.round(scrollDeltaY / step)
        scrollDeltaY = scrollDeltaY % step

        let relPos = trackProjection.relative(ev.clientY)
        const amount =
            iamount * Math.max(1, Math.abs(trackProjection.domainSpan()) / 5)

        activateScrollMargins()

        if (ev.ctrlKey) {
            zoomTracks(relPos, -amount)
        } else {
            scrollTracks(amount)
        }
    }

    const zoomTracks = (anchor, delta) => {
        trackProjection.zoom(anchor, delta)
        update()
    }

    const scrollTracks = (delta) => {
        trackProjection.scroll(delta)
        update()
    }

    const coverageToBox = (coverage) => {
        let track = codesToTracks.get(coverage.codes)
        if (track == null || !trackVisible(track)) {
            return null
        }
        let box = {
            id: coverage.id,
            tmin: coverage.tmin,
            tmax: Math.min(coverage.tmax, tomorrow()),
            ymin: trackProjection.lower(track),
            ymax: trackProjection.upper(track),
        }
        return box
    }

    const aggregateRanges = (ranges) => {
        if (ranges.length == 0) {
            return [0.1, 1]
        } else {
            return [
                Math.min(...ranges.map((range) => range[0])),
                Math.max(...ranges.map((range) => range[1])),
            ]
        }
    }

    const getTrackScale = (track) => {
        const yMinTrack = trackProjection.lower(track) + effectiveTrackPadding
        const yMaxTrack = trackProjection.upper(track) - effectiveTrackPadding
        const [yMinData, yMaxData] = aggregateRanges(
            track.codes.map((codes) => dataRanges.get(codes) || [0.1, 1])
        )
        return d3.scaleLog([yMinData, yMaxData], [yMaxTrack, yMinTrack])
    }

    const carpetToImage = (carpet) => {
        let track = codesToTracks.get(carpet.codes)
        if (track == null || !trackVisible(track)) {
            return null
        }
        const yMinTrack = trackProjection.lower(track) + effectiveTrackPadding
        const yMaxTrack = trackProjection.upper(track) - effectiveTrackPadding

        const fproject = getTrackScale(track)

        return {
            id: carpet.id,
            xmin: x(carpet.tmin),
            xmax: x(carpet.tmax),
            ymin: fproject(carpet.ymax),
            ymax: fproject(carpet.ymin),
            clip: `url(#track-${track.index})`,
            xminClip: x(gates.timeMin.value),
            xmaxClip: x(gates.timeMax.value),
            yminClip: yMinTrack,
            ymaxClip: yMaxTrack,
            zombie: carpet.zombie,
            image_data_base64: carpet.image_data_base64,
        }
    }

    const getBoxes = () => {
        const coverages = gates.getCoverages()
        return coverages === null
            ? []
            : coverages.map(coverageToBox).filter((box) => box !== null)
    }
    const getImages = () => {
        return gates
            .getCarpets()
            .map(carpetToImage)
            .filter((img) => img !== null)
    }

    const updateBoxes = (t) => {
        boxesGroup
            .selectAll('rect')
            .data(getBoxes(), (box) => box.id)
            .join(
                (enter) =>
                    enter
                        .append('rect')
                        .attr('fill', colors['aluminium2'] + '32')
                        .attr('stroke', colors['aluminium4'] + '32')
                        .attr(
                            'height',
                            (box) =>
                                box.ymax - box.ymin - effectiveTrackPadding * 2
                        )
                        .attr('y', (box) => box.ymin + effectiveTrackPadding)
                        .style('opacity', 0)
                        .call((enter) =>
                            enter.transition().duration(100).style('opacity', 1)
                        ),
                (update) =>
                    update.call((update) =>
                        update
                            .transition(t)
                            .attr(
                                'height',
                                (box) =>
                                    box.ymax -
                                    box.ymin -
                                    effectiveTrackPadding * 2
                            )
                            .attr(
                                'y',
                                (box) => box.ymin + effectiveTrackPadding
                            )
                    )
            )
            .attr('x', (box) => x(box.tmin))
            .attr('width', (box) => x(box.tmax) - x(box.tmin))

        const images = getImages()

        const imageTransition = d3.transition('image')

        imageGroup
            .selectAll('image')
            .data(images, (img) => img.id)
            .join(
                (enter) =>
                    enter
                        .append('image')
                        .attr('draggable', 'false')
                        .attr('preserveAspectRatio', 'none')
                        .attr('href', (img) => img.image_data_base64)
                        .attr('y', (img) => img.ymin)
                        .attr('height', (img) => img.ymax - img.ymin)
                        .style('mix-blend-mode', 'plus-lighter')
                        .style('opacity', (img) => 0),
                (update) =>
                    update.call((update) =>
                        update
                            .transition(t)
                            .attr('y', (img) => img.ymin)
                            .attr('height', (img) => img.ymax - img.ymin)
                    )
            )
            .attr('clip-path', (img) => img.clip)
            .attr('x', (img) => img.xmin)
            .attr('width', (img) => img.xmax - img.xmin)
            .transition(imageTransition)
            .duration((img) => 500)
            .style('opacity', (img) => (img.zombie ? 0 : 1))
    }

    const updateTimeAxes = () => {
        const axisY = (i) => {
            return [
                bounds.height - marginBottom + effectiveTrackPadding,
                marginTop - effectiveTrackPadding,
            ][i]
        }

        let napprox = 5 * bounds.width / 1200.
        let [tinc, tinc_units] = niceTimeTickInc(
            (gates.timeMax.value - gates.timeMin.value) / napprox
        )
        let [times, labels] = timeTickLabels(
            gates.timeMin.value,
            gates.timeMax.value,
            tinc,
            tinc_units,
            napprox,
        )
        let ticks = times.map((t, i) => ({
            t: t,
            labels: ('' + labels[i]).split('\n').reverse(),
        }))

        let axisGroups = timeAxesGroup
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
                    .attr('stroke-width', axisLineWidth)
            )
            .attr('x1', 0.0)
            .attr('x2', bounds.width)
            .attr('y1', axisY)
            .attr('y2', axisY)

        axisGroups
            .selectAll('.axis-tick')
            .data(
                (axisId) =>
                    ticks
                        .filter((tick) => tick.t > gates.timeMin.value)
                        .map((tick) => ({ axisId, tick })),
                (d) => d.tick.t
            )
            .join((enter) =>
                enter
                    .append('line')
                    .attr('class', 'axis-tick')
                    .attr('stroke', colors['aluminium4'])
                    .attr('stroke-width', axisTickStrokeWidth)
            )
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
            .join((enter) =>
                enter.append('g').attr('class', 'axis-tick-label-group')
            )
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
            .join(
                (enter) =>
                    enter
                        .append('text')
                        .attr('class', 'axis-tick-label')
                        .style('font-size', fontSize + 'pt')
                        .call((enter) =>
                            enter
                                .filter((d) => d.tick.t != gates.timeMin.value)
                                .style('opacity', 0)
                                .transition()
                                .duration(500)
                                .style('opacity', 1)
                        ),
                (update) => update
            )
            .style('dominant-baseline', (d) =>
                d.axisId == 0 ? 'text-before-edge' : 'no-change'
            )
            .text((d) => d.label)
            .style('text-anchor', (d) =>
                d.tick.t == gates.timeMin.value ? 'left' : 'middle'
            )
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
            .attr('dx', (d) =>
                d.tick.t == gates.timeMin.value ? '0.5em' : '0'
            )

        axisGroups
    }

    const activateScrollMargins = () => {
        needScrollMargins = true
        if (deactivateScrollMarginsTimeoutId !== null) {
            clearTimeout(deactivateScrollMarginsTimeoutId)
        }
        deactivateScrollMarginsTimeoutId = setTimeout(() => {
            needScrollMargins = false
        }, 3000)
    }

    const visibilityMargins = () => {
        if (needScrollMargins) {
            return [
                trackProjection.trackHeight(),
                trackProjection.trackHeight(),
            ]
        } else {
            return [0, 0]
        }
    }

    const trackVisible = (track) => {
        const margins = visibilityMargins()
        return (
            trackProjection.range()[0] - margins[0] <=
                trackProjection.lower(track) &&
            trackProjection.upper(track) <=
                trackProjection.range()[1] + margins[1]
        )
    }

    const updateTracks = (t) => {
        const tracks = visibleTracks()
        const labels = tracksGroup
            .selectAll('.track-label')
            .data(tracks, (track) => track.id)
            .join((enter) => {
                const group = enter.append('g').attr('class', 'track-label')

                group
                    .append('rect')
                    .attr('fill', labelBackgroundFill)
                    .attr('stroke', labelBackgroundStroke)

                group
                    .append('text')
                    .attr('class', 'track-label-text')
                    .attr('x', 0)
                    .attr('dx', '1em')
                    .style('dominant-baseline', 'central')

                group
                    .append('text')
                    .attr('class', 'track-label-helper-text')
                    .attr('x', 0)
                    .attr('dx', '1em')
                    .style('dominant-baseline', 'central')

                return group
            })

        const labelFontSize = Math.min(
            fontSize,
            ((trackProjection.range()[1] - trackProjection.range()[0]) /
                (trackProjection.domain()[1] - trackProjection.domain()[0]) /
                1.333) *
                0.8
        )

        labels
            .select('.track-label-text')
            .text((track) => track.codes.join(', '))
            .style('font-size', labelFontSize + 'pt')
            .transition(t)
            .attr('y', (track) => trackProjection(track))

        labels
            .select('.track-label-helper-text')
            .text((track) => track.codes.join(', '))
            .style('font-size', labelFontSize + 'pt')
            .style('fill', 'none')
            .attr('y', (track) => trackProjection(track))

        const bboxes = labels
            .select('.track-label-helper-text')
            .nodes()
            .map((node) => node.getBBox())

        for (let i = 0; i < bboxes.length; i++) {
            tracks[i].textBBox = bboxes[i]
        }

        labels
            .select('rect')
            .attr(
                'x',
                (track) => track.textBBox.x - labelPadding[0] * labelFontSize
            )
            .attr(
                'width',
                (track) =>
                    track.textBBox.width + 2 * labelPadding[0] * labelFontSize
            )
            .attr(
                'height',
                (track) =>
                    track.textBBox.height + 2 * labelPadding[1] * labelFontSize
            )
            .transition(t)
            .attr(
                'y',
                (track) => track.textBBox.y - labelPadding[1] * labelFontSize
            )

        defs.selectAll('.track-clip-path')
            .data(tracks, (track) => track.id)
            .join((enter) =>
                enter
                    .append('clipPath')
                    .attr('class', 'track-clip-path')
                    .attr('id', (track) => `track-${track.index}`)
                    .call((enter) => enter.append('rect'))
            )
            .select('rect')
            .attr('x', x.range()[0])
            .attr('width', x.range()[1] - x.range()[0])
            .transition(t)
            .attr(
                'y',
                (track) => trackProjection.lower(track) + effectiveTrackPadding
            )
            .attr(
                'height',
                (track) =>
                    trackProjection.upper(track) -
                    trackProjection.lower(track) -
                    2.0 * effectiveTrackPadding
            )
        const d3format = d3.format('.4g')
        const format = (x) =>
            d3format(x)
                .replace(/(\.[0-9]*?)0+$/, '$1')
                .replace(/\.$/, '')

        const trackAxes = tracksGroup
            .selectAll('.track-axis')
            .data(tracks, (track) => track.id)
            .join(
                (enter) => enter.append('g').attr('class', 'track-axis'),
                (update) => update
            )

        trackAxes
            .attr('transform', `translate(${bounds.width - 10},0)`)
            .each((track, i, nodes) =>
                d3.axisLeft().scale(getTrackScale(track)).ticks(5, format)(
                    d3.select(nodes[i])
                )
            )
    }

    const updateCodes = async () => {
        let groupKeySensor = (c) => {
            const nslce = c.split('.')
            const sensor = nslce[3].substring(0, nslce[3].length - 1)
            return nslce.splice(0, 3) + [sensor]
        }
        let groupKeyChannel = (c) => {
            return c.split('.')
        }

        let groups = Map.groupBy(gates.codes.value, (c) => groupKeyChannel(c))
        tracks.length = 0
        let i = 0
        for (const [k, codes] of groups) {
            tracks.push({
                id: codes.join('+++'),
                index: i,
                codes: codes,
            })
            i++
        }
        codesToTracks.clear()
        for (const track of tracks) {
            for (const codes of track.codes) {
                codesToTracks.set(codes, track)
            }
        }
        trackProjection.domain([0, tracks.length])
        trackProjection.attribute((track) => track.index)
        update()
    }

    let my = (selection) => {
        container = selection
        timeline = createIfNeeded(container, 'svg')
        timeline.attr('draggable', 'false')
        timeline.style('user-select', 'none')

        defs = timeline.append('defs')

        pageRect = defs.append('clipPath').attr('id', 'page').append('rect')

        timeAxesGroup = timeline.append('g')
        boxesGroup = timeline.append('g').attr('clip-path', 'url(#page)')
        imageGroup = timeline
            .append('g')
            .attr('clip-path', 'url(#page)')
            .attr('id', 'images')
        tracksGroup = timeline.append('g').attr('clip-path', 'url(#page)')

        container.on('pointerdown', pointerDownHandler)
        container.on('pointerup', pointerUpHandler)
        container.on('pointercancel', pointerUpHandler)
        //container.on('pointerout', pointerUpHandler)
        container.on('pointerleave', pointerUpHandler)
        container.on('pointermove', pointerMoveHandler)
        container.on('keydown', keyDownHandler)
        container.on('wheel', scrollHandler)
        container.on('contextmenu', (ev) => {
            ev.preventDefault()
        })

        onResizeDebounced(container.node(), resizeHandler)
        resizeHandler()

        timeSpan.value = [gates.timeMin.value, gates.timeMax.value]

        watch(gates.counter, update)
        watch(gates.codes, updateCodes)
        watch(trackHeight, gates.setImageHeight)
        watch(trackWidth, gates.setImageWidth)

        gates.setImageHeight(trackHeight.value)
        gates.setImageWidth(trackWidth.value)

        watch(timeSpan, (newVal, oldVal) => {
            gates.setTimeSpan(newVal[0], newVal[1])
        })
        watch(visibleCodes, gates.setCodesVisible)
    }

    my.activate = () => {
        resizeHandler()
    }

    return my
}
