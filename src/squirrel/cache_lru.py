import logging
from typing import NamedTuple

from lru import LRU

from pyrocko.guts import Float, Int, Object
from pyrocko.squirrel.model import WAVEFORM, Nut

logger = logging.getLogger('psq.squirrel')
MB = 1024 * 1024
MAX_SIZE = 4096
MAX_BYTES = 500*MB


class LRUCacheStats(Object):
    '''
    Information about cache state.
    '''
    nentries = Int.T(help='Number of items in the cache.')
    cache_hits = Int.T(help='Number of cache hits.')
    cache_misses = Int.T(help='Number of cache misses.')
    nbytes = Int.T(help='Number of bytes used by the cache.')
    percent = Float.T(help='Percentage of cache used (0-100).')


class CacheKey(NamedTuple):
    path: str
    segment: int
    mtime: float

    @classmethod
    def from_nut(cls, nut: Nut):
        path, segment, _, nut_mtime = nut.key
        return cls(path, segment, nut_mtime)


def get_size_bytes(nut: Nut) -> int:
    if nut.kind_id == WAVEFORM:
        return nut.content.ydata.nbytes
    return nut.file_size


class LRUCache:
    def __init__(self, size_bytes: int = MAX_BYTES):
        self._cache: LRU[CacheKey, Nut] = LRU(
            size=MAX_SIZE, callback=self._nut_removed)
        self._size_bytes = 0
        self._max_size_bytes = size_bytes

    def _nut_added(self, nut: Nut) -> None:
        self._size_bytes += get_size_bytes(nut)

        if self._size_bytes > self._max_size_bytes:
            new_size = max(1, self._cache.get_size() - 64)
            self._cache.set_size(new_size)
            logger.debug(
                'Cache size exceeded %d bytes, shrank cache to %d elements',
                MAX_BYTES, new_size)

    def _nut_removed(self, key: CacheKey, nut: Nut) -> None:
        if self._size_bytes < self._max_size_bytes * 0.9:
            new_size = self._cache.get_size() + 64
            self._cache.set_size(new_size)
            logger.debug('Nut removed early, grew cache size to %d elements',
                         new_size)

        self._size_bytes -= get_size_bytes(nut)

    def put(self, nut: Nut) -> None:
        '''
        Insert a new/updated item into cache.

        :param nut:
            Content item with attached data object.
        :type nut:
            :py:class:`~pyrocko.squirrel.model.Nut`
        '''
        self._cache[CacheKey.from_nut(nut)] = nut
        self._nut_added(nut)

    def get(self, nut: Nut,
            accessor: str = 'default', model: str = 'squirrel') -> object:
        '''
        Get a content item and track its access.

        :param nut:
            Content item.
        :type nut:
            :py:class:`~pyrocko.squirrel.model.Nut`

        :param accessor:
            Name of accessing consumer. Giving a new name initializes a new
            accessor.
        :type accessor:
            str

        :returns:
            Content data object
        '''
        cache_key = CacheKey.from_nut(nut)
        nut = self._cache.get(cache_key)
        if nut is None:
            raise KeyError(f'Nut {cache_key} not found in cache.')

        if model == 'squirrel':
            return nut.content
        elif model.endswith('+'):
            return nut.content, nut.raw_content[model[:-1]]
        else:
            return nut.raw_content[model]

    def has(self, nut: Nut) -> bool:
        '''
        Check if a nut is in cache.

        :param nut:
            Content item.
        :type nut:
            :py:class:`~pyrocko.squirrel.model.Nut`

        :returns:
            True if the item is in cache, False otherwise.
        '''
        return self._cache.has_key(CacheKey.from_nut(nut))

    def advance_accessor(self, accessor: str = 'default') -> None:
        '''
        Dummy implementation for the LRU cache.

        :param accessor:
            Name of the accessor.
        :type accessor:
            str
        '''
        ...

    def clear_accessor(self, accessor: str = 'default') -> None:
        '''
        Dummy implementation for the LRU cache.

        :param accessor:
            Name of the accessor.
        :type accessor:
            str
        '''
        ...

    def clear(self) -> None:
        '''
        Clear the cache.
        '''
        self._cache.clear()

    def get_stats(self) -> LRUCacheStats:
        '''
        Get cache statistics.

        :returns:
            Cache statistics.
        '''
        hits, misses = self._cache.get_stats()
        return LRUCacheStats(
            nentries=len(self._cache),
            cache_hits=hits,
            cache_misses=misses,
            nbytes=self._size_bytes,
            percent=(self._size_bytes / self._max_size_bytes) * 100.0,
        )
