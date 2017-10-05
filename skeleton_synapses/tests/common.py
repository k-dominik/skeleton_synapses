import os
import json

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def get_fixture_path(*args):
    return os.path.join(FIXTURE_DIR, *args)


def get_fixture_data(*args):
    path = get_fixture_path(*args)
    ext = os.path.splitext(path)[1]

    if ext == '.json':
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError('Unknown extension {} for fixture at path {}'.format(ext, path))
