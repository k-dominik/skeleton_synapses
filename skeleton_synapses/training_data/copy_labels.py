import os
import logging

import numpy as np
import h5py

from skeleton_synapses.training_data.common import ILP_PATH, lane_paths


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('copy_labels')
dlogger = logging.getLogger('copy_labels.dry_run')

label_name_mapping = {
    'Synapse': 'Synapse',
    'Membrane': 'Membrane',
    'Cytoplasm': 'Other',
    'Dark cytoplasm': 'Other',
    'Defect': 'Other',
    'Mitochondria': 'Other',
    'Mito Boundary': 'Other',
    'Vesicles/Microtubes': 'Other'
}


def line_overlaps(start1, stop1, start2, stop2):
    are_distinct = max(start1, stop1) <= min(start2, stop2) or max(start2, stop2) <= min(start1, stop1)
    return not are_distinct


class Extent(object):
    def __init__(self, xmin, xmax, ymin, ymax, zmin, zmax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    @classmethod
    def from_str(cls, s):
        """Assumes XYZ order"""
        pairs_strs = s[1:-1].split(',')
        items = []
        for pair_str in pairs_strs:
            items.extend(int(n) for n in pair_str.split(':'))

        return cls(*items[:-2])

    def to_str(self):
        """Assumes XYZ order"""
        return '[{}:{},{}:{},{}:{},0:1]'.format(
            self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax
        )

    def overlaps(self, other):
        return all(
            [
                line_overlaps(self.xmin, self.xmax, other.xmin, other.xmax),
                line_overlaps(self.ymin, self.ymax, other.ymin, other.ymax),
                line_overlaps(self.zmin, self.zmax, other.zmin, other.zmax)
            ]
        )

    def overlaps_any(self, iterable):
        return any(self.overlaps(other) for other in iterable)

    def __hash__(self):
        return hash(self.to_str())

    def __eq__(self, other):
        return isinstance(other, Extent) and hash(self) == hash(other)


def remap_array(arr, mapping):
    """
    Given a numpy array and dict mapping, return a new array of the same shape and dtype with all values in the
    mapping's keys replaced by the mapping's values (all other cells will be 0).
    """
    new_arr = np.zeros(arr.shape, arr.dtype)
    for key, value in mapping.items():
        new_arr[arr == key] = value

    return new_arr


def block_name_to_idx(block_name):
    """Convert block name of form 'blockXXXX' into an integer ID"""
    return int(block_name[5:])


def block_idx_to_name(block_idx):
    """Convert an integer ID into a block name of form 'blockXXXX'"""
    return 'block{:04}'.format(block_idx)


def remap_blocks(src_group, tgt_group, label_mapping, dry_run=False):
    """
    Given a source and target HDF5 group, and a label mapping, add all blocks from the source group to the target
    block with its labels remapped, so long as a block does not already exist which overlaps with it.
    """
    tgt_blocks_extents = []
    block_max_id = -1
    for block_name in tgt_group:
        tgt_blocks_extents.append(Extent.from_str(tgt_group[block_name].attrs['blockSlice']))
        block_max_id = max(block_max_id, block_name_to_idx(block_name))

    for block_name in src_group:
        src_block = src_group[block_name]
        extent = Extent.from_str(src_block.attrs['blockSlice'])
        if not extent.overlaps_any(tgt_blocks_extents):
            remapped_data = remap_array(np.array(src_block), label_mapping)
            block_max_id += 1
            if not dry_run:
                tgt_block = tgt_group.create_dataset(block_idx_to_name(block_max_id), data=remapped_data)
                tgt_block.attrs['blockSlice'] = src_block.attrs['blockSlice']
            else:
                dlogger.info('Creating new block "%s" at slice %s',
                             block_idx_to_name(block_max_id), src_block.attrs['blockSlice'])


def main(dry_run=False):
    with h5py.File(ILP_PATH, 'r' if dry_run else 'r+') as f:
        src_labels = list(f[os.path.join(lane_paths[0], 'LabelNames')])
        src_label_idxs = {key: value for value, key in enumerate(src_labels, 1)}
        tgt_labels = list(f[os.path.join(lane_paths[1], 'LabelNames')])
        tgt_label_idxs = {key: value for value, key in enumerate(tgt_labels, 1)}

        label_idx_mapping = {src_label_idxs[key]: tgt_label_idxs[value] for key, value in label_name_mapping.items()}

        remap_blocks(
            f[os.path.join(lane_paths[0], 'LabelSets', 'labels000')],
            f[os.path.join(lane_paths[1], 'LabelSets', 'labels000')],
            label_idx_mapping,
            dry_run
        )

if __name__ == '__main__':
    DRY_RUN = False
    main(DRY_RUN)
