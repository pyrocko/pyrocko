const { onMounted, ref, computed } = Vue
import { useFilters } from './../squirrel/filter.js'

export default {
    setup: () => {
        const { searchInput, activeQuery, searchValid, filterSensors } =
            useFilters()

        const dark = computed(() => Quasar.Dark.isActive)

        const searchHistory = ref([])

        const searchHistoryDefaults = ['c *z', 'c *e *n']

        const saveSearchHistory = () => {
            if (
                searchValid.value &&
                activeQuery.value.trim() &&
                !searchHistory.value.includes(activeQuery.value) &&
                !searchHistoryDefaults.includes(activeQuery.value)
            ) {
                searchHistory.value.unshift(activeQuery.value.trim())
                searchHistory.value = searchHistory.value.slice(0, 10)
                sessionStorage.setItem(
                    'searchHistory',
                    JSON.stringify(searchHistory.value)
                )
            }
        }

        onMounted(() => {
            const storedHistory = sessionStorage.getItem('searchHistory')
            if (storedHistory) {
                searchHistory.value = JSON.parse(storedHistory)
            }
        })

        const onSearchFinalize = (e) => {
            e.target.blur()
            saveSearchHistory()
            filterSensors()
        }

        const searchHistoryWithDefaults = computed(() => {
            return [...searchHistoryDefaults, ...searchHistory.value]
        })

        // Options currently shown in the dropdown, narrowed to what's
        // typed so far. Kept as its own ref (rather than a computed
        // straight off searchInput) because q-select drives this
        // through its own @filter event instead of firing on every
        // model change.
        const filterOptions = ref(searchHistoryWithDefaults.value)

        const onFilter = (val, update) => {
            update(() => {
                filterOptions.value =
                    val === ''
                        ? searchHistoryWithDefaults.value
                        : searchHistoryWithDefaults.value.filter((option) =>
                              option.toLowerCase().includes(val.toLowerCase())
                          )
            })
        }

        // q-select's v-model, in use-input mode, only updates on
        // selection/confirm -- @input-value is what fires on every
        // keystroke, so it's what keeps searchInput (and from there
        // searchValid/activeQuery) live while typing.
        const onInputValue = (val) => {
            searchInput.value = val
        }

        return {
            dark,
            searchInput,
            searchValid,
            filterOptions,
            onFilter,
            onInputValue,
            onSearchFinalize,
        }
    },
    template: `
    <div>
        <q-select
            :dark="dark"
            dense
            stdout
            use-input
            fill-input
            hide-selected
            clearable
            input-debounce="0"
            new-value-mode="add-unique"
            :options="filterOptions"
            v-model="searchInput"
            :class="{ 'search-invalid': !searchValid }"
            @filter="onFilter"
            @input-value="onInputValue"
            @keyup.enter="onSearchFinalize"
            style="max-width: 20em"
        >
            <template v-slot:prepend>
                <q-icon name="search" />
            </template>
        </q-select>
    </div>
    `,
}
