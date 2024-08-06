export const createIfNeeded = (selection, type) => {
    return selection.selectAll(type).data([null]).enter().append(type)
}

export const timeToStr = (time) => {
    let secs = Math.floor(time)
    let subsecs = time - secs
    assert(subsecs < 1.0)
    return (
        new Date(secs * 1000).toISOString().substr(0, 19) +
        subsecs.toFixed(6).substr(2)
    )
}

export const strToTime = (s) => {
    let reDateTime = /^(\d\d\d\d-\d\d-\d\d)([ T])(\d\d:\d\d:\d\d)(\.\d+)?Z?$/
    let m = s.match(reDateTime)
    let sdatetime = m[1] + 'T' + m[3] + 'Z'
    let msecs = Date.parse(sdatetime)
    if (msecs == null || isNaN(msecs)) {
        throw new Error('Invalid date: ' + sdatetime)
    }
    return msecs / 1000.0 + (m[4] ? +m[4] : 0.0)
}
