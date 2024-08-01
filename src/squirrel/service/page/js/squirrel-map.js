import { squirrelConnection } from './squirrel-connection.js'

const colors = {
    'aluminium1': '#eeeeec',
    'aluminium2': '#d3d7cf',
    'aluminium3': '#babdb6',
    'aluminium4': '#888a85',
    'aluminium5': '#555753',
    'aluminium6': '#2e3436',
    'butter1': '#fce94f',
    'butter2': '#edd400',
    'butter3': '#c4a000',
    'chameleon1': '#8ae234',
    'chameleon2': '#73d216',
    'chameleon3': '#4e9a06',
    'chocolate1': '#e9b96e',
    'chocolate2': '#c17d11',
    'chocolate3': '#8f5902',
    'orange1': '#fcaf3e',
    'orange2': '#f57900',
    'orange3': '#ce5c00',
    'plum1': '#ad7fa8',
    'plum2': '#75507b',
    'plum3': '#5c3566',
    'scarletred1': '#ef2929',
    'scarletred2': '#cc0000',
    'scarletred3': '#a40000',
    'skyblue1': '#729fcf',
    'skyblue2': '#3465a4',
    'skyblue3': '#204a87',
}

const createIfNeeded = (selection, type) => {
    return selection.selectAll(type).data([null]).enter().append(type)
}

export const squirrelMap = () => {
    let my = async (selection) => {
        const map = createIfNeeded(selection, 'svg')

        const bounds = () => {
            return selection.node().getBoundingClientRect()
        }

        const getWidth = () => {
            return bounds().width
        }

        const getHeight = () => {
            return bounds().height
        }

        const getScale = () => {
            return 1
        }

        const resizeProjection = (projection) => {
            projection
                .scale((getScale() * getHeight()) / 1.5 / 1.3 / Math.PI)
                .translate([getWidth() / 2, getHeight() / 2])
        }

        const projections = {
            ed: d3.geoAzimuthalEquidistant().clipAngle(180.0 - 1e-3),
            ea: d3.geoAzimuthalEqualArea().clipAngle(180.0 - 1),
        }

        const projection = projections.ed
        resizeProjection(projection)

        projection.rotate([-50, -10])

        // Load external data and boot
        const data = await d3.json(
            'https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson'
        )

        // Draw the map
        map.append('g')
            .selectAll('path')
            .data(data.features)
            .enter()
            .append('path')
            .attr('fill', colors['aluminium3'])
            .style('stroke', colors['aluminium4'])

        let graticule = d3.geoGraticule()

        map.append('g')
            .append('path')
            .datum(graticule)
            .attr('fill', 'none')
            .attr('stroke', colors['aluminium2'])

        const resize = () => {
            resizeProjection(projection)
            map.attr('width', getWidth()).attr('height', getHeight())

            map.selectAll('g')
                .selectAll('path')
                .attr('d', d3.geoPath().projection(projection))

            map.selectAll('circle').attr('transform', function (ev) {
                return 'translate(' + projection([ev.lon, ev.lat]) + ')'
            })
        }

        const addSensors = async () {
            const connection = squirrelConnection()
            const locations = await connection.request('get_sensors')

            map.selectAll('circle')
                .data(locations)
                .enter()
                .append('circle')
                .attr('r', 3)
                .attr('fill', colors['scarletred2']+"33")
                .attr('stroke', colors['scarletred3']+"33")

            resize()
        }

        resize()

        window.onresize = resize
    }

    return my
}
