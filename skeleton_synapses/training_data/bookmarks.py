import re
import json
import logging

import h5py

from skeleton_synapses.training_data.common import ILP_PATH, lane_paths

logger = logging.getLogger('bookmarks')

# this_dir = os.path.dirname(os.path.realpath(__file__))
# rel_path_to_projects = '../../projects-2017/L1-CNS/projects'
# ilp_name = 'full-vol-autocontext.ilp'
#
# ILP_PATH = os.path.join(this_dir, rel_path_to_projects, ilp_name)


inner_leaf = '/Bookmarks/0000'

entry_form = '''I{z_px}
I{x_px}
I{y_px}
tp{idx}
S'{label}'
p{idx_plus_one}
tp{idx_plus_two}
a'''

entry_re = re.compile(
    entry_form.format(
        z_px='(?P<z_px>\d+)', x_px='(?P<x_px>\d+)', y_px='(?P<y_px>\d+)',
        idx='(?P<idx1>\d+)', idx_plus_one='(?P<idx2>\d+)', idx_plus_two='(?P<idx3>\d+)',
        label='(?P<label>.*)',
    ), re.MULTILINE
)

prefix = '''(lp1
(('''

suffix = '.'


class Bookmark(object):
    def __init__(self, x_px=0, y_px=0, z_px=0, label=''):
        self.x_px = x_px
        self.y_px = y_px
        self.z_px = z_px
        self.label = label

    @classmethod
    def from_ilp(cls, s):
        """Convert a string with no prefix or suffix into a Bookmark object"""
        d = entry_re.search(s).groupdict()
        return cls(int(d['x_px']), int(d['y_px']), int(d['z_px']), d['label'])

    def to_ilp(self, current_max=1, with_fixes=False):
        """Convert a Bookmark object into a string for use in an ILP file"""
        middle = entry_form.format(
            z_px=self.z_px, y_px=self.y_px, x_px=self.x_px,
            idx=current_max + 1, idx_plus_one=current_max + 2, idx_plus_two=current_max + 3,
            label=self.label
        )

        if with_fixes:
            return prefix + middle + suffix
        else:
            return middle

    @classmethod
    def deserialise(cls, s):
        """Convert a string from an ILP file into a list of Bookmark objects"""
        trimmed = s.lstrip(prefix).rstrip(suffix)
        return [Bookmark.from_ilp(item) for item in trimmed.split('((')]

    @staticmethod
    def serialise(*bookmarks):
        """Convert a list of Bookmark objects into a string to be inserted into an ILP file"""
        items = []
        current_max = 1
        for bookmark in bookmarks:
            items.append(bookmark.to_ilp(current_max))
            current_max += 3

        return prefix + '(('.join(items) + suffix

    def to_dict(self):
        return {
            'x_px': self.x_px,
            'y_px': self.y_px,
            'z_px': self.z_px,
            'label': self.label,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d['x_px'], d['y_px'], d['z_px'], d.get('label', ''))

    @classmethod
    def from_json_multi(cls, path):
        """Multiple Bookmarks from a json-serialised list of dicts"""
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [cls.from_dict(item) for item in data]
            else:
                return [cls.from_dict(data)]

    @classmethod
    def from_json(cls, path):
        """Bookmark from a json-serialised dict"""
        with open(path) as f:
            data = json.load(f)
            return cls.from_dict(data)

    def __repr__(self):
        return 'Bookmark(x_px={x_px}, y_px={y_px}, z_px={z_px}, label="{label}")'.format(**self.__dict__)

    def __eq__(self, other):
        try:
            other_attrs = (other.x_px, other.y_px, other.z_px)
        except KeyError:
            return False
        return (self.x_px, self.y_px, self.z_px) == other_attrs


def read_bookmarks(path, lane):
    """Read bookmarks directly from an ILP file"""
    assert lane in (0, 1)
    with h5py.File(path, 'r') as f:
        bookmark_str = f[lane_paths[lane] + inner_leaf].value

    return Bookmark.deserialise(bookmark_str)


def write_bookmarks(path, lane, bookmarks):
    """Write bookmarks directly to an ILP file"""
    assert lane in (0, 1)
    bookmark_str = Bookmark.serialise(*bookmarks)
    with h5py.File(path) as f:
        inner_path = lane_paths[lane] + inner_leaf
        try:
            del f[inner_path]
            logger.info('Deleted existing bookmark dataset')
        except KeyError:
            logger.info('No existing bookmark dataset')

        f.create_dataset(inner_path, data=bookmark_str)
        logger.info('Adding new bookmarks')


def append_bookmarks(path, lane, bookmarks, check=True):
    """Append bookmarks to an existing ILP file"""
    old_bookmarks = read_bookmarks(path, lane)

    if check:
        bookmarks_to_add = []
        for bookmark in bookmarks:
            if bookmark in old_bookmarks:
                logger.warning('{} is already in the project (possibly with a different label), skipping')
            else:
                bookmarks_to_add.append(bookmark)
    else:
        bookmarks_to_add = bookmarks

    all_bookmarks = old_bookmarks + bookmarks_to_add
    write_bookmarks(path, lane, all_bookmarks)


if __name__ == '__main__':
    bookmarks = read_bookmarks(ILP_PATH, 0)
    print(bookmarks)
