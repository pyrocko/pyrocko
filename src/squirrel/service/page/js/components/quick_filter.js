const { onMounted, ref, computed } = Vue
import { useFilters } from './../squirrel/filter.js'

export default {
    setup: () => {
        const { searchInput, searchActive, searchValid, filterSensors } =
            useFilters()

        const dark = computed(() => Quasar.Dark.isActive)

        const searchHistory = ref([])

        const searchHistoryDefaults = ['c *z', 'c *e *n']

        const saveSearchHistory = () => {
            if (
                searchValid.value &&
                searchActive.value.trim() &&
                !searchHistory.value.includes(searchActive.value) &&
                !searchHistoryDefaults.includes(searchActive.value)
            ) {
                searchHistory.value.unshift(searchActive.value.trim())
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
        // searchValid/searchActive) live while typing.
        const onInputValue = (val) => {
            searchInput.value = val
        }

        const showHelp = ref(false)

        return {
            dark,
            searchInput,
            searchValid,
            filterOptions,
            onFilter,
            onInputValue,
            onSearchFinalize,
            showHelp,
        }
    }, template: `
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
            <template v-slot:before-options>
                <div class="quick-search-help q-px-md q-py-sm text-weight-bold text-grey-7">
                    <div class="text-right">
                    <q-btn size="xs" outline round color="primary" icon="question_mark" @click.stop="showHelp = !showHelp" />
                    </div>
                    <div v-if="showHelp">
                        <p></p>
                        <p>Matching operators:</p>
                        <table>
                            <tr><td><code>n</code></td><td>network</td></tr>
                            <tr><td><code>s</code></td><td>station</td></tr>
                            <tr><td><code>l</code></td><td>location</td></tr>
                            <tr><td><code>c</code></td><td>channel</td></tr>
                            <tr><td><code>e</code></td><td>extra</td></tr>
                            <tr><td><code>nslce</code></td><td>network.station.location.channel.extra</td></tr>
                            <tr><td><code>sc</code></td><td>station.channel</td></tr>
                            <tr><td></td><td>Any combination is supported.</td></tr>
                        </table>
                        <p></p>
                        <p>Glob patterns:</p>
                        <table>
                            <tr><td><code>*</code></td><td>Match zero or more arbitrary characters.</td></tr>
                            <tr><td><code>?</code></td><td>Match exactly one arbitrary character.</td></tr>
                        </table>
                        <p></p>
                        <p>Logical operators:</p>
                        <table>
                            <tr><td><code>&amp;&amp;</code></td><td>and</td></tr>
                            <tr><td><code>||</code></td><td>or</td></tr>
                            <tr><td><code>!</code></td><td>negation</td></tr>
                        </table>
                        <p></p>
                        <p>Examples:</p>
                        <table>
                            <tr><td><code>c *z</code></td><td>Show vertials, i.e. channels ending with Z.</td></tr>
                            <tr><td><code>s GRA1 GRA2</code></td><td>Show only stations GRA1 and GRA2.</td></tr>
                            <tr><td><code>c *z && ! n GR</code></td><td>Show vertical components but hide network GR.</td></tr>
                            <tr><td><code>nslc GR.GRA1..BHZ</code></td><td>Show that specific channel.</td></tr>
                        </table>
                    </div>
                </div>
            </template>
            <template v-slot:prepend>
                <q-icon name="search" />
            </template>
        </q-select>
    </div>
    `,
}
