import { createIfNeeded, colors, onResizeDebounced } from './common.js'

// A framework-agnostic world-map widget: it neither imports Vue nor
// knows anything about `gates`/Squirrel. It has no outputs -- it's a
// pure renderer -- so unlike timeline.js/range_select.js it doesn't
// need an on()/emit() pair, just the two inputs below.
//
// Inputs (call whenever the corresponding state changes):
//   setStations(stations)          -- the known stations to plot, each
//                                      {key, lat, lon, ...}
//   setVisibleStationKeys(keySet)  -- a Set of station keys to draw as
//                                      "active" rather than dimmed
export const squirrelMap = () => {
    let map
    let basemapGroup
    let symbolGroup
    let scale = 1.0
    let projection
    let container
    let bounds

    let stations = []
    let visibleStationKeys = new Set()

    const containerBounds = () => {
        return container.node().getBoundingClientRect()
    }

    const setupProjection = () => {
        projection
            .scale((scale * bounds.height) / 1.5 / 1.3 / Math.PI)
            .translate([bounds.width / 2, bounds.height / 2])
    }

    const projectBasemap = () => {
        basemapGroup
            .selectAll('g')
            .selectAll('path')
            .attr('d', d3.geoPath().projection(projection))
    }

    const projectCircles = () => {
        symbolGroup.selectAll('circle').attr('transform', function (ev) {
            return 'translate(' + projection([ev.lon, ev.lat]) + ')'
        })
    }

    const project = () => {
        projectBasemap()
        projectCircles()
    }

    const reProject = () => {
        setupProjection()
        project()
    }

    const resizeHandler = () => {
        bounds = containerBounds()
        if (bounds.width <= 0 || bounds.height <= 0) {
            return
        }
        reconcileStations()
        map.attr('width', bounds.width).attr('height', bounds.height)
        reProject()
    }

    const rotate = (latlon) => {
        projection.rotate([-latlon[0], -latlon[1]])
        reProject()
    }

    const scaleDelta = (delta) => {
        scale *= 1.0 + delta * 0.2
        reProject()
    }

    const addBasemap = async () => {
        const data = await d3.json(
            'https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson'
        )

        basemapGroup
            .append('g')
            .selectAll('path')
            .data(data.features)
            .enter()
            .append('path')
            .attr('fill', colors['aluminium1'])
            .style('stroke', colors['aluminium2'])

        let graticule = d3.geoGraticule()

        basemapGroup
            .append('g')
            .append('path')
            .datum(graticule)
            .attr('fill', 'none')
            .attr('stroke', colors['aluminium2'])

        //map.append('g')
        //    .append('path')
        //    .datum(graticule.outline)
        //    .attr('fill', '#0002')
        //    .attr('stroke', colors['aluminium5'])
        //

        projectBasemap()
    }

    // Adds/removes circles to match `stations` (the set of known
    // stations, not how many are currently visible). Only runs when
    // that set actually changes -- via setStations() -- so a pan/zoom
    // drag never touches this, only `updateActiveState` below.
    const reconcileStations = () => {
        symbolGroup
            .selectAll('circle')
            .data(stations, (station) => station.key)
            .join('circle')
            .attr('r', 3)

        updateActiveState()
        projectCircles()
    }

    // Restyles already-existing circles to match
    // `visibleStationKeys`. Runs on every pan/zoom, but only ever
    // touches fill/stroke of circles that already exist -- no join, no
    // DOM creation/removal here.
    const updateActiveState = () => {
        symbolGroup
            .selectAll('circle')
            .attr(
                'fill',
                (station) =>
                    colors['scarletred2'] +
                    (visibleStationKeys.has(station.key) ? '' : '33')
            )
            .attr(
                'stroke',
                (station) =>
                    colors['scarletred3'] +
                    (visibleStationKeys.has(station.key) ? '' : '33')
            )
    }

    const my = (selection) => {
        container = selection
        map = createIfNeeded(container, 'svg')

        basemapGroup = map.append('g')
        symbolGroup = map.append('g')

        const projections = {
            ed: d3.geoAzimuthalEquidistant().clipAngle(180.0 - 1e-3),
            ea: d3.geoAzimuthalEqualArea().clipAngle(180.0 - 1),
            g1: d3.geoEqualEarth(),
            g2: d3.geoNaturalEarth1(),
        }

        projection = projections.ea

        onResizeDebounced(container.node(), resizeHandler)
        resizeHandler()

        map.on('click', (ev) => {
            rotate(projection.invert(d3.pointer(ev)))
        })
        map.on('wheel', (ev) => {
            scaleDelta(ev.wheelDeltaY / 120)
        })
    }

    my.setStations = (newStations) => {
        stations = newStations
        reconcileStations()
    }

    my.setVisibleStationKeys = (keys) => {
        visibleStationKeys = keys
        updateActiveState()
    }

    my.scale = function (_) {
        if (!arguments.length) {
            return scale
        }
        scale = +_
        reProject()
        return my
    }

    my.addBasemap = () => {
        addBasemap()
        return my
    }

    return my
}
