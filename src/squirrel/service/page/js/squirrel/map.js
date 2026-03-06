const { watch } = Vue
import { createIfNeeded, colors, onResizeDebounced } from './common.js'
import { squirrelGates } from './gate.js'

export const squirrelMap = () => {
    let gates = squirrelGates()

    let map
    let basemapGroup
    let symbolGroup
    let scale = 1.0
    let projection
    let container
    let bounds

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

    // Adds/removes circles to match `gates.stations` (the set of known
    // stations, not how many are currently visible). Only runs when
    // that set actually changes -- see gate.js -- so a pan/zoom drag
    // never touches this, only `updateActiveState` below.
    const reconcileStations = () => {
        symbolGroup
            .selectAll('circle')
            .data(gates.stations.value, (station) => station.key)
            .join('circle')
            .attr('r', 3)

        updateActiveState()
        projectCircles()
    }

    // Restyles already-existing circles to match
    // `gates.visibleStationKeys`. Runs on every pan/zoom, but only
    // ever touches fill/stroke of circles that already exist -- no
    // join, no DOM creation/removal here.
    const updateActiveState = () => {
        const visible = gates.visibleStationKeys.value

        symbolGroup
            .selectAll('circle')
            .attr(
                'fill',
                (station) =>
                    colors['scarletred2'] +
                    (visible.has(station.key) ? '' : '33')
            )
            .attr(
                'stroke',
                (station) =>
                    colors['scarletred3'] +
                    (visible.has(station.key) ? '' : '33')
            )
    }

    let my = async (selection) => {
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

        watch([gates.stations], reconcileStations)
        watch([gates.visibleStationKeys], updateActiveState)
    }

    my.scale = function (_) {
        return arguments.length ? ((scale = +_), reProject()) : scale
    }

    my.addBasemap = () => {
        addBasemap()
        return my
    }

    return my
}
