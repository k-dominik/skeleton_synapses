import os

from bookmarks import Bookmark, append_bookmarks, ILP_PATH

root = 'bookmarks'
fnames = [
    'treenode_bookmarks.json',
    'connector_bookmarks.json',
    'noise_bookmarks.json',
    'not_synapse_bookmarks.json',
    'vesicle_bookmarks.json'
]

if __name__ == '__main__':
    bookmarks_to_add = []
    for fname in fnames:
        bookmarks_to_add.extend(Bookmark.from_json_multi(os.path.join(root, fname)))

    for lane in [0, 1]:
        append_bookmarks(ILP_PATH, lane, bookmarks_to_add)
