import numpy as num


def int_arr(*args):
    return num.array(args, dtype=num.int)


class Builder:
    def __init__(self, gf_set, block_size=None):
        if block_size is None:
            if len(gf_set.ns) == 3:
                block_size = (10, 1, 10)
            elif len(gf_set.ns) == 2:
                block_size = (1, 10)
            else:
                assert False

        self.gf_set = gf_set
        self._block_size = int_arr(*block_size)

    @property
    def nblocks(self):
        return num.prod(self.block_dims)

    @property
    def block_dims(self):
        return (self.gf_set.ns-1) / self._block_size + 1

    def all_block_indices(self):
        return num.arange(self.nblocks)

    def get_block(self, index):
        dims = self.block_dims
        iblock = num.unravel_index(index, dims)
        ibegins = iblock * self._block_size
        iends = num.minimum(ibegins + self._block_size, self.gf_set.ns)
        return ibegins, iends

    def get_block_extents(self, index):
        ibegins, iends = self.get_block(index)
        begins = self.gf_set.mins + ibegins * self.gf_set.deltas
        ends = self.gf_set.mins + (iends-1) * self.gf_set.deltas
        return begins, ends, iends - ibegins

__all__ = ['Builder']
