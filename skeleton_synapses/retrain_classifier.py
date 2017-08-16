import os
from argparse import ArgumentParser
import subprocess
import logging

from skeleton_synapses.locate_synapses import ensure_description_file


VALID_PROJECTS = ('autocontext', 'multicut')
ILASTIK_PATH = os.path.join(os.getenv('CONDA_PREFIX'), 'run_ilastik.sh')

logger = logging.getLogger('retrain_classifier')


def check_description_exists(project_dir):
    assert os.path.isfile(os.path.join(project_dir, 'L1-CNS-description-NO-OFFSET.json'))


def retrain_autocontext(project_dir):
    logging.info('Retraining autocontext')
    autocontext_path = os.path.join(project_dir, 'projects', 'full-vol-autocontext.ilp')
    subprocess.check_call([ILASTIK_PATH, '--headless', '--retrain', '--project="{}"'.format(autocontext_path)])

if __name__ == '__main__':
    retrain_fns = {
        'autocontext': retrain_autocontext
    }

    parser = ArgumentParser()
    parser.add_argument('project_dir')
    parser.add_argument('projects', nargs='+')

    parsed_args = parser.parse_args()

    assert set(parsed_args.projects).issubset(VALID_PROJECTS), 'Valid projects are {}'.format(VALID_PROJECTS)

    for project_name in parsed_args.projects:
        if project_name == 'multicut':
            logger.warning('Multicut retraining has not been implemented, skipping')
        else:
            retrain_fns[project_name](parsed_args.project_dir)
