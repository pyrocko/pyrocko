from pyrocko.model import Event, dump_events
from pyrocko.guts import Object, List
from pyrocko.util import stt

from typing import List as TList
from typing import Dict as TDict

import numpy as num


km = 1000.


class HypoDDDataset(Object):
    '''
    Reader for the output of hypodd relocated events.
    '''
    events = List.T(
        Event.T(), default=[Event.D()], help='List of Pyrocko events.')

    @classmethod
    def from_catalog(cls, filename: str):
        hdd = cls()
        hdd._data = num.loadtxt(filename, skiprows=1)
        hdd.events = hdd.get_pyrocko_events()
        return hdd

    @property
    def event_ids(self) -> num.ndarray:
        return self._data[:, 0].astype('int')

    @property
    def lats(self) -> num.ndarray:
        return self._data[:, 1]

    @property
    def lons(self) -> num.ndarray:
        return self._data[:, 2]

    @property
    def depths(self) -> num.ndarray:
        return self._data[:, 3] * km

    @property
    def east_shifts(self) -> num.ndarray:
        return self._data[:, 4]

    @property
    def north_shifts(self) -> num.ndarray:
        return self._data[:, 5]

    @property
    def depth_shifts(self) -> num.ndarray:
        return self._data[:, 6]

    @property
    def error_east_shifts(self) -> num.ndarray:
        return self._data[:, 7]

    @property
    def error_north_shifts(self) -> num.ndarray:
        return self._data[:, 8]

    @property
    def error_depth_shifts(self) -> num.ndarray:
        return self._data[:, 9]

    @property
    def years(self) -> num.ndarray:
        return self._data[:, 10].astype('int')

    @property
    def months(self) -> num.ndarray:
        return self._data[:, 11].astype('int')

    @property
    def days(self) -> num.ndarray:
        return self._data[:, 12].astype('int')

    @property
    def hours(self) -> num.ndarray:
        return self._data[:, 13].astype('int')

    @property
    def minutes(self) -> num.ndarray:
        return self._data[:, 14].astype('int')

    @property
    def seconds(self) -> num.ndarray:
        return self._data[:, 15]

    @property
    def magnitudes(self) -> num.ndarray:
        return self._data[:, 16]

    @property
    def cluster_ids(self) -> num.ndarray:
        return self._data[:, 17].astype('int')

    def get_unique_cluster_ids(self) -> set:
        return set(self.cluster_ids.tolist())

    @property
    def nevents(self) -> int:
        return self._data.shape[0]

    def get_pyrocko_events(self, cluster_id: int = None) -> TList[Event]:

        events = []
        idxs = num.arange(self.nevents, dtype="int")
        if cluster_id is not None:
            ucids = self.get_unique_cluster_ids()
            if cluster_id not in ucids:
                raise ValueError(
                    'Requested cluster ID is not contained in the dataset!')

            event_idxs_mask = self.cluster_ids == cluster_id
            idxs = idxs[event_idxs_mask]

        for idx in idxs:
            time_str = '%i-%i-%i %i:%i:%f' % (
                self.years[idx],
                self.months[idx],
                self.days[idx],
                self.hours[idx],
                self.minutes[idx],
                self.seconds[idx])

            events.append(Event(
                lat=self.lats[idx],
                lon=self.lons[idx],
                depth=self.depths[idx],
                time=stt(time_str),
                magnitude=self.magnitudes[idx],
                extras=dict(cluster_id=self.cluster_ids[idx])))

        return events

    def get_event_clusters(self) -> TDict[int, TList[Event]]:
        clustered_events = {}
        ucids = self.get_unique_cluster_ids()

        for ucid in ucids:
            clustered_events[ucid] = self.get_pyrocko_events(ucid)

        return clustered_events

    def dump_events(self, filename: str, format: str = 'basic'):
        dump_events(self.events, filename=filename, format=format)
