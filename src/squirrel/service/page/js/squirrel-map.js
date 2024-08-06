import { createIfNeeded } from './squirrel-common.js'
import { squirrelConnection } from './squirrel-connection.js'

const colors = {
    aluminium1: '#eeeeec',
    aluminium2: '#d3d7cf',
    aluminium3: '#babdb6',
    aluminium4: '#888a85',
    aluminium5: '#555753',
    aluminium6: '#2e3436',
    butter1: '#fce94f',
    butter2: '#edd400',
    butter3: '#c4a000',
    chameleon1: '#8ae234',
    chameleon2: '#73d216',
    chameleon3: '#4e9a06',
    chocolate1: '#e9b96e',
    chocolate2: '#c17d11',
    chocolate3: '#8f5902',
    orange1: '#fcaf3e',
    orange2: '#f57900',
    orange3: '#ce5c00',
    plum1: '#ad7fa8',
    plum2: '#75507b',
    plum3: '#5c3566',
    scarletred1: '#ef2929',
    scarletred2: '#cc0000',
    scarletred3: '#a40000',
    skyblue1: '#729fcf',
    skyblue2: '#3465a4',
    skyblue3: '#204a87',
}

export const squirrelMap = () => {
    let map
    let scale = 1.0
    let projection
    let container

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
        projection
            .scale((scale * getHeight()) / 1.5 / 1.3 / Math.PI)
            .translate([getWidth() / 2, getHeight() / 2])
    }

    const projectBasemap = () => {
        map.selectAll('g')
            .selectAll('path')
            .attr('d', d3.geoPath().projection(projection))
    }

    const projectCircles = () => {
        map.selectAll('circle').attr('transform', function (ev) {
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

    const resize = () => {
        map.attr('width', getWidth()).attr('height', getHeight())
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

        map.append('g')
            .selectAll('path')
            .data(data.features)
            .enter()
            .append('path')
            .attr('fill', colors['aluminium1'])
            .style('stroke', colors['aluminium2'])

        let graticule = d3.geoGraticule()

        map.append('g')
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

    const addSensors = async () => {
        const connection = squirrelConnection()
        const locations = await connection.request('get_sensors')

        map.selectAll('circle')
            .data(locations)
            .enter()
            .append('circle')
            .attr('r', 3)
            .attr('fill', colors['scarletred2'] + '33')
            .attr('stroke', colors['scarletred3'] + '33')

        projectCircles()
    }

    let my = async (selection) => {
        container = selection
        map = createIfNeeded(container, 'svg')

        const projections = {
            ed: d3.geoAzimuthalEquidistant().clipAngle(180.0 - 1e-3),
            ea: d3.geoAzimuthalEqualArea().clipAngle(180.0 - 1),
            g1: d3.geoEqualEarth(),
            g2: d3.geoNaturalEarth1(),
        }

        projection = projections.ed

        resize()

        window.onresize = resize
        map.on('click', (ev) => {
            rotate(projection.invert(d3.pointer(ev)))
        })
        map.on('wheel', (ev) => {
            scaleDelta(ev.wheelDeltaY / 120)
        })
    }

    my.scale = function (_) {
        return arguments.length ? ((scale = +_), reProject()) : scale
    }

    my.addBasemap = () => {
        addBasemap()
        return my
    }

    my.addSensors = () => {
        addSensors()
        return my
    }

    return my
}
