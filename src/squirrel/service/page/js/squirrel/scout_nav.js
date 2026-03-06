const { ref } = Vue

// The panels ("scouts") shown in the right drawer, ordered by how
// commonly each is used. The first `pinnedScoutCount` get a dedicated
// button in the main toolbar; the rest -- and anything added later --
// live in its "more" menu.
export const scoutList = [
    { value: 'info', label: 'Info', icon: 'info' },
    { value: 'response', label: 'Response', icon: 'show_chart' },
    { value: 'map', label: 'Map', icon: 'map' },
    { value: 'mini-map', label: 'Mini-map', icon: 'public' },
]

export const pinnedScoutCount = 3

const scout = ref(scoutList[0].value)

export function useScoutNav() {
    return { scout, scoutList, pinnedScoutCount }
}
