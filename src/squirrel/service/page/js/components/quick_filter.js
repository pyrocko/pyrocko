const { onMounted, ref } = Vue
import { useFilters } from './../squirrel/filter.js'

export default {
    setup: () => {
        const { searchQuery, filterSensors } = useFilters()

        const searchHistory = ref([])
        const typingTimer = ref(null)
        const searchDelay = 1000

        const saveSearchHistory = () => {
            if (
                searchQuery.value.trim() &&
                !searchHistory.value.includes(searchQuery.value)
            ) {
                searchHistory.value.unshift(searchQuery.value.trim())
                searchHistory.value = searchHistory.value.slice(0, 3)
                sessionStorage.setItem(
                    'searchHistory',
                    JSON.stringify(searchHistory.value)
                )
            }
        }

        const onSearchInput = () => {
            clearTimeout(typingTimer.value)

            typingTimer.value = setTimeout(() => {
                saveSearchHistory()
            }, searchDelay)
        }

        const onSearchFinalize = (e) => {
            e.target.blur()
            filterSensors()
        }

        onMounted(() => {
            const storedHistory = sessionStorage.getItem('searchHistory')
            if (storedHistory) {
                searchHistory.value = JSON.parse(storedHistory)
            }
        })
        return { searchQuery, searchHistory, onSearchInput, onSearchFinalize }
    },
    template: `
    <div>
        <q-input
            dark
            dense
            stdout
            list="filters"
            type="search"
            v-model="searchQuery"
            @input="onSearchInput"
            @keyup.enter="onSearchFinalize"
        >
            <template v-slot:prepend>
                <q-icon name="search" /> </template
        ></q-input>
        <datalist id="filters">
            <option
                v-for="(historyItem, index) in searchHistory"
                :key="index"
                :value="historyItem"
            ></option>
            <option value="HZ"></option>
            <option value="LH"></option>
            <option value="BH"></option>
            <option value="CH"></option>
            <option value="HH"></option>
        </datalist>
    </div>
    `,
}
