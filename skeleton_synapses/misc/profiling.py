import timeit
from datetime import datetime
from functools import partial

import dill
# from copy import deepcopy
from pathos import multiprocessing as mp

from ilastik_utils.projects import setup_classifier as lib_setup_classifier

DESCRIPTION_FILE = '/home/cbarnes/work/synapse_detection/skeleton_synapses/projects-2017/L1-CNS/L1-CNS-description-NO-OFFSET.json'
AUTOCONTEXT_PATH = '/home/cbarnes/work/synapse_detection/skeleton_synapses/projects-2017/L1-CNS/projects/full-vol-autocontext.ilp'
MULTICUT_PATH = '/home/cbarnes/work/synapse_detection/skeleton_synapses/projects-2017/L1-CNS/projects/multicut/L1-CNS-multicut.ilp'


clock = timeit.default_timer


class Timer(object):
    def __init__(self):
        self.start_time = None
        self.times = []

    def lap_start(self, name):
        if self.start_time is not None:
            self.lap_stop()

        self.times.append([name, None])
        self.start_time = datetime.utcnow()

    def lap_stop(self):
        stop_time = datetime.utcnow()
        self.times[-1][1] = stop_time - self.start_time

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lap_stop()


def setup_classifier(count=0, description_file=DESCRIPTION_FILE, autocontext_project_path=AUTOCONTEXT_PATH,
                     logging=False):
    with Timer() as t:
        t.lap_start('open_project')
        autocontext_shell = open_project(autocontext_project_path, init_logging=logging)

        t.lap_start('type_checks')
        assert isinstance(autocontext_shell, HeadlessShell)
        assert isinstance(autocontext_shell.workflow, NewAutocontextWorkflowBase)

        t.lap_start('append_lane')
        append_lane(autocontext_shell.workflow, description_file, 'xyt')

        t.lap_start('get_op')
        # We only use the final stage predictions
        opPixelClassification = autocontext_shell.workflow.pcApplets[-1].topLevelOperator

        t.lap_start('sanity_checks')
        # Sanity checks
        assert isinstance(opPixelClassification, OpPixelClassification)
        assert opPixelClassification.Classifier.ready()
        assert opPixelClassification.HeadlessPredictionProbabilities[-1].meta.drange == (0.0, 1.0)

    type(opPixelClassification)
    print('Finished rep {}'.format(count))

    return [(name, delta.total_seconds()) for name, delta in t.times]


def deepcopy(obj):
    return dill.loads(dill.dumps(obj, dill.HIGHEST_PROTOCOL))


def yield_classifier(description_file=DESCRIPTION_FILE, autocontext_project_path=AUTOCONTEXT_PATH,
                     logging=False):
    autocontext_shell = open_project(autocontext_project_path, init_logging=logging)

    assert isinstance(autocontext_shell, HeadlessShell)
    assert isinstance(autocontext_shell.workflow, NewAutocontextWorkflowBase)

    append_lane(autocontext_shell.workflow, description_file, 'xyt')

    count = 0

    while True:
        with Timer() as t:
            t.lap_start('copy_shell')
            this_shell = deepcopy(autocontext_shell)

            t.lap_start('get_op')
            # We only use the final stage predictions
            opPixelClassification = this_shell.workflow.pcApplets[-1].topLevelOperator

            t.lap_start('sanity_checks')
            # Sanity checks
            assert isinstance(opPixelClassification, OpPixelClassification)
            assert opPixelClassification.Classifier.ready()
            assert opPixelClassification.HeadlessPredictionProbabilities[-1].meta.drange == (0.0, 1.0)

        type(opPixelClassification)
        count += 1
        print('Finished rep {}'.format(count))
        yield [(name, delta.total_seconds()) for name, delta in t.times]


def setup_multicut(count=0, multicut_project=MULTICUT_PATH, logging=False):
    with Timer() as t:
        t.lap_start('open_project')
        multicut_shell = open_project(multicut_project, init_logging=logging)

        t.lap_start('type_checks')
        assert isinstance(multicut_shell, HeadlessShell)
        assert isinstance(multicut_shell.workflow, EdgeTrainingWithMulticutWorkflow)

    type(multicut_shell)
    print('Finished rep {}'.format(count))

    return [(name, delta.total_seconds()) for name, delta in t.times]


def mean_times(time_lists):
    output = []
    for eq_laps in zip(*time_lists):
        assert all(eq_lap[0] == eq_laps[0][0] for eq_lap in eq_laps), 'mixture of lap names'
        output.append((eq_laps[0][0], np.mean([eq_lap[1] for eq_lap in eq_laps])))

    return output


def serial_tests(fn, logging=False, reps=10):
    time_lists = [fn(i+1, logging=logging) for i in range(reps)]
    return mean_times(time_lists)


def parallel_tests(fn, logging=False, reps=10):
    p = mp.Pool(mp.cpu_count()-1)
    results = p.map(partial(fn, logging=logging), list(range(1, reps+1)))

    return mean_times(results)


if __name__ == '__main__':
    # it = yield_classifier()
    # c1 = next(it)
    # print(c1)
    # c2 = next(it)
    # print(c2)

    lib_setup_classifier(DESCRIPTION_FILE, AUTOCONTEXT_PATH)