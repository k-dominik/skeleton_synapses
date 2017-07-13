from __future__ import absolute_import
import os
import argparse
import sys
import random
from datetime import datetime, timedelta
import json
from catpy import CatmaidClient
from skeleton_synapses.catmaid_interface import CatmaidSynapseSuggestionAPI
from skeleton_synapses.training_data.bookmarks import Bookmark
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

TIME_WINDOW_YEARS = 3
DEFAULT_SAMPLE_SIZE = 200
XY_PADDING = 10
Z_PADDING = 1
TRUSTED_USERS = ['caseysm', 'albert']
STACK_ID = 1
SEED = 1


def copy_dict(d):
    return json.loads(json.dumps(d))


class TrainingDataBookmarker(object):
    def __init__(self, credentials_path, seed=None, output_root='.'):
        self._seed = seed
        self.catmaid = CatmaidSynapseSuggestionAPI(CatmaidClient.from_json(credentials_path))
        self.coord_transformer = self.catmaid.get_coord_transformer(STACK_ID)
        self.output_root = output_root

    @property
    def seed(self):
        if self._seed is None:
            return random.randint(0, sys.maxsize)
        else:
            return self._seed

    def _dump_bookmarks(self, bookmarks, rel_output_path):
        if rel_output_path is None:
            return
        print('Writing output')
        with open(os.path.join(self.output_root, rel_output_path), 'w') as f:
            json.dump([bookmark.to_dict() for bookmark in bookmarks], f, indent=2, sort_keys=True)

    def get_connector_bookmarks(self, sample_size=DEFAULT_SAMPLE_SIZE, output_path=None):
        """
        Randomly sample connectors recently annotated by trusted experts and return a list of Bookmark objects for
        insertion into Ilastik.

        Parameters
        ----------
        credentials_path : str
            Path to CATMAID credentials JSON
        sample_size : int
            How many connectors to sample
        output_path : str
            Where to save the JSON output

        Returns
        -------
        list of Bookmark
        """
        responses = dict()
        date_to = datetime.now()
        date_from = date_to - timedelta(days=365.25*TIME_WINDOW_YEARS)

        # todo: speed this up with threading?
        for username in tqdm(TRUSTED_USERS, desc='Getting connectors by trusted annotators', file=sys.stdout):
            responses.update(self.catmaid.get_connectors(username, date_from, date_to))

        coord_transformer = self.catmaid.get_coord_transformer(STACK_ID)
        print('Sampling responses')
        rand = random.Random(SEED)
        sample = {
            key: coord_transformer.project_to_stack(responses[key])
            for key in rand.sample(responses, sample_size)
        }

        bookmarks = []
        sample_items = sorted(sample.items())
        rand.shuffle(sample_items)
        for idx, (conn_id, coords) in tqdm(enumerate(sample_items), desc='Converting connectors to bookmarks',
                                           file=sys.stdout):
            bookmarks.append(Bookmark(
                x_px=int(coords['x']), y_px=int(coords['y']), z_px=int(coords['z']),
                label='Synapse #{} (connector {})'.format(idx, conn_id)
            ))

        self._dump_bookmarks(bookmarks, output_path)

        return bookmarks

    def get_treenode_bookmarks(self, sample_size=DEFAULT_SAMPLE_SIZE, output_path=None):
        response = self.catmaid.sample_treenodes(sample_size, seed=self.seed)
        bookmarks = []
        for idx, row in tqdm(enumerate(response['data']), desc='Converting treenodes to bookmarks', file=sys.stdout):
            bookmarks.append(Bookmark(
                x_px=int(self.coord_transformer.project_to_stack_coord('x', row[1])),
                y_px=int(self.coord_transformer.project_to_stack_coord('y', row[2])),
                z_px=int(self.coord_transformer.project_to_stack_coord('z', row[3])),
                label='Location #{} (treenode {})'.format(idx, row[0])
            ))

        self._dump_bookmarks(bookmarks, output_path)

        return bookmarks

    def get_tag_bookmarks(self, tags, output_path=None):
        response = self.catmaid.treenodes_by_tag(*tags)
        bookmarks = []
        for idx, row in tqdm(enumerate(response['data']), desc='Converting tags to bookmarks', file=sys.stdout):
            bookmarks.append(Bookmark(
                x_px=int(self.coord_transformer.project_to_stack_coord('x', row[2])),
                y_px=int(self.coord_transformer.project_to_stack_coord('y', row[3])),
                z_px=int(self.coord_transformer.project_to_stack_coord('z', row[4])),
                label='{} #{} (treenode {})'.format(row[0], idx, row[1])
            ))

        self._dump_bookmarks(bookmarks, output_path)

        return bookmarks


if __name__ == "__main__":
    DEBUGGING = True
    if DEBUGGING:
        print("USING DEBUG ARGUMENTS")

        CREDENTIALS_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        credentials_path = os.path.join(CREDENTIALS_ROOT, 'credentials_catsop.json')
        output = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'bookmarks')
        random_seed = 1
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('credentials_path',
                            help='Path to a JSON file containing CATMAID credentials (see credentials.jsonEXAMPLE)')
        parser.add_argument('-o', '--output', default='',
                            help="Optional path to folder containing output json files.")
        parser.add_argument('-r', '--random-seed', default=None,
                            help='Random seed to use for samplers.')

        args = parser.parse_args()
        credentials_path = args.credentials_path
        output = args.output
        sample_size = args.sample_size
        random_seed = None if args.random_seed is None else int(args.random_seed)

    bookmarker = TrainingDataBookmarker(credentials_path, random_seed, output)

    print('Getting connectors')
    bookmarker.get_connector_bookmarks(output_path='connector_bookmarks.json')
    print('Getting treenodes')
    bookmarker.get_treenode_bookmarks(output_path='treenode_bookmarks.json')
    print('Getting not-synapses')
    bookmarker.get_tag_bookmarks(('not a synapse', ), output_path='not_synapse_bookmarks.json')
    print('Getting vesicles')
    bookmarker.get_tag_bookmarks(('example of clathrin-coated vesicle', ), output_path='vesicle_bookmarks.json')
    print('Getting noise')
    bookmarker.get_tag_bookmarks(('noise', ), output_path='noise_bookmarks.json')
