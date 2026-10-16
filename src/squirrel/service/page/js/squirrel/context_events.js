import { timeToStr, format, depthColor } from './common.js'

// Helpers for the presentation of context events (the largest events in
// the current time window, see gate.js) in timeline and maps. Vue-free, so
// that the widgets can use them as well.

const fmtMagnitude = format('.1f')
const fmtDepth = format('.0f')

// One marker per group of context events (e.g. one earthquake reported by
// several catalogs), placed at its largest event. `group` is sorted by
// descending magnitude (see gate.js).
export const eventGroupToMarker = (group) => {
    const ev = group[0]
    const lines = [
        group.find((e) => e.region)?.region ?? '',
        timeToStr(ev.time, '%Y-%m-%d %H:%M:%S.3FRAC'),
        ...group.map(
            (e) =>
                (e.catalog || '?') +
                ': ' +
                (e.magnitude != null
                    ? (e.magnitude_type || 'M') +
                      ' ' +
                      fmtMagnitude(e.magnitude)
                    : 'M ?') +
                (e.depth != null ? ', ' + fmtDepth(e.depth / 1000) + ' km' : '')
        ),
    ]
    return {
        key: ev.extras?.group_id ?? ev.name + '@' + ev.time,
        time: ev.time,
        lat: ev.lat,
        lon: ev.lon,
        magnitude: ev.magnitude,
        color: ev.depth != null ? depthColor(ev.depth) : null,
        title: lines.filter((line) => line).join('\n'),
    }
}

// Linear mapping of magnitude to marker size, relative to the given
// markers, spanning at least one magnitude unit. Markers without magnitude
// get the smallest size.
export const makeMagnitudeScale = (markers, [sizeMin, sizeMax]) => {
    const magnitudes = markers
        .map((m) => m.magnitude)
        .filter((m) => m != null)

    let [mmin, mmax] = magnitudes.length
        ? [Math.min(...magnitudes), Math.max(...magnitudes)]
        : [0, 1]

    if (mmax - mmin < 1.0) {
        const mmid = 0.5 * (mmin + mmax)
        mmin = mmid - 0.5
        mmax = mmid + 0.5
    }

    const scale = d3
        .scaleLinear()
        .domain([mmin, mmax])
        .range([sizeMin, sizeMax])
        .clamp(true)

    return (magnitude) => scale(magnitude ?? mmin)
}

// Draw order: small ones first, so that large ones end up on top.
export const byMagnitudeAscending = (a, b) =>
    (a.magnitude ?? -Infinity) - (b.magnitude ?? -Infinity)
