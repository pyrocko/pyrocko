import { squirrelConnection } from "./squirrel-connection.js"

export const squirrelMap = async () => {
    const containerPosition = () => {
        return document.getElementById("mapcontainer").getBoundingClientRect();
    };

    const getWidth = () => {
        return containerPosition().width;
    };

    const getHeight = () => {
        let height = containerPosition().height;
        return height;
    };

    const getScale = () => {
        return 1;
    };

    const resizeProjection = (projection) => {
        projection
            .scale((getScale() * getHeight()) / 1.5 / 1.3 / Math.PI)
            .translate([getWidth() / 2, getHeight() / 2]);
    };

    const projections = {
        ed: d3.geoAzimuthalEquidistant().clipAngle(180.0 - 1e-3),
        ea: d3.geoAzimuthalEqualArea().clipAngle(180.0 - 1),
    };

    const projection = projections.ed;
    resizeProjection(projection);

    projection.rotate([-50, -10]);

    const svg = d3
        .select("svg")
        .attr("width", getWidth())
        .attr("height", getHeight());

    // Load external data and boot
    const data = await d3.json(
        "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson"
    );

    // Draw the map
    svg.append("g")
        .selectAll("path")
        .data(data.features)
        .enter()
        .append("path")
        .attr("fill", "#dddddd")
        .attr("d", d3.geoPath().projection(projection))
        .style("stroke", "#aaa");

    let graticule = d3.geoGraticule();

    svg.append("g")
        .append("path")
        .datum(graticule)
        .attr("d", d3.geoPath().projection(projection))
        .attr("fill", "none")
        .attr("stroke", "#aaa");

    const connection = squirrelConnection()
    const locations = await connection.request("get_sensors");

    svg.selectAll("circle")
        .data(locations)
        .enter()
        .append("circle")
        .attr("r", 2)
        .attr("fill", "#ff555577")
        .attr("transform", function (loc) {
            return "translate(" + projection([loc.lon, loc.lat]) + ")";
        });

    const resize = () => {
        resizeProjection(projection);
        svg.attr("width", getWidth()).attr("height", getHeight());

        svg.selectAll("g")
            .selectAll("path")
            .attr("d", d3.geoPath().projection(projection));

        svg.selectAll("circle").attr("transform", function (ev) {
            return "translate(" + projection([ev.lon, ev.lat]) + ")";
        });
    };

    window.onresize = resize;
};
