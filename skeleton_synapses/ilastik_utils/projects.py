import os
import tempfile
import logging
from warnings import warn

import vigra

import ilastik_main
from ilastik.applets.dataSelection import DataSelectionApplet, DatasetInfo
from ilastik.applets.pixelClassification import OpPixelClassification
from ilastik.shell.headless.headlessShell import HeadlessShell
from ilastik.workflows.edgeTrainingWithMulticut import EdgeTrainingWithMulticutWorkflow
from ilastik.workflows.newAutocontext.newAutocontextWorkflow import NewAutocontextWorkflowBase
from lazyflow.utility import PathComponents, isUrl

from skeleton_synapses.constants import ILP_RETRAIN, ILP_READONLY


logger = logging.getLogger(__name__)


def setup_classifier(description_file, autocontext_project_path, retrain=ILP_RETRAIN, readonly=ILP_READONLY):
    logger.debug('Setting up opPixelClassification')
    autocontext_shell = _open_project(autocontext_project_path, retrain, readonly, init_logging=False)
    assert isinstance(autocontext_shell, HeadlessShell)
    assert isinstance(autocontext_shell.workflow, NewAutocontextWorkflowBase)

    _append_lane(autocontext_shell.workflow, description_file, 'xyt')

    # We only use the final stage predictions
    opPixelClassification = autocontext_shell.workflow.pcApplets[-1].topLevelOperator

    # Sanity checks
    assert isinstance(opPixelClassification, OpPixelClassification)
    assert opPixelClassification.Classifier.ready()
    assert opPixelClassification.HeadlessPredictionProbabilities[-1].meta.drange == (0.0, 1.0)

    return opPixelClassification


def setup_multicut(multicut_project, retrain=ILP_RETRAIN, readonly=ILP_READONLY):
    logger.debug('Setting up multicut_shell')
    multicut_shell = _open_project(multicut_project, retrain, readonly, init_logging=False)
    assert isinstance(multicut_shell, HeadlessShell)
    assert isinstance(multicut_shell.workflow, EdgeTrainingWithMulticutWorkflow)

    return multicut_shell


def setup_classifier_and_multicut(
        description_file, autocontext_project_path, multicut_project, retrain=ILP_RETRAIN, readonly=ILP_READONLY
):
    """
    Boilerplate for getting the requisite ilastik interface objects and sanity-checking them

    Parameters
    ----------
    description_file : str
        Path to JSON stack description file
    autocontext_project_path : str
    multicut_project : str
        path

    Returns
    -------
    (OpPixelClassification, HeadlessShell)
        opPixelClassification, multicut_shell
    """
    logger.debug('Setting up opPixelClassification and multicut_shell')
    opPixelClassification = setup_classifier(description_file, autocontext_project_path, retrain, readonly)
    multicut_shell = setup_multicut(multicut_project, retrain, readonly)

    return opPixelClassification, multicut_shell


def _open_project(project_path, retrain=ILP_RETRAIN, readonly=ILP_READONLY, init_logging=False):
    """
    Open a project file and return the HeadlessShell instance.
    """
    parsed_args = ilastik_main.parser.parse_args([])
    parsed_args.headless = True
    parsed_args.project = project_path
    # parsed_args.readonly = True
    if retrain and not readonly:
        warn('ILP must be writable if it is to retrain. Disabling read-only mode')
        readonly = True
    parsed_args.readonly = readonly
    parsed_args.debug = True  # possibly delete this?

    if retrain:
        shell = ilastik_main.main(parsed_args, workflow_cmdline_args=['--retrain'], init_logging=init_logging)
    else:
        shell = ilastik_main.main(parsed_args, init_logging=init_logging)
    return shell


def _append_lane(workflow, input_filepath, axisorder=None):
    """
    Add a lane to the project file for the given input file.

    If axisorder is given, override the default axisorder for
    the file and force the project to use the given one.

    Globstrings are supported, in which case the files are converted to HDF5 first.
    """
    # If the filepath is a globstring, convert the stack to h5  # todo: skip this?
    tmp_dir = tempfile.mkdtemp()
    input_filepath = DataSelectionApplet.convertStacksToH5( [input_filepath], tmp_dir )[0]

    try:
        os.rmdir(tmp_dir)
    except OSError as e:
        if e.errno == 39:
            logger.warning('Temporary directory {} was populated: should be deleted')
        else:
            raise

    info = DatasetInfo()
    info.location = DatasetInfo.Location.FileSystem
    info.filePath = input_filepath

    comp = PathComponents(input_filepath)

    # Convert all (non-url) paths to absolute
    # (otherwise they are relative to the project file, which probably isn't what the user meant)
    if not isUrl(input_filepath):
        comp.externalPath = os.path.abspath(comp.externalPath)
        info.filePath = comp.totalPath()
    info.nickname = comp.filenameBase
    if axisorder:
        info.axistags = vigra.defaultAxistags(axisorder)

    logger.debug( "adding lane: {}".format( info ) )

    opDataSelection = workflow.dataSelectionApplet.topLevelOperator

    # Add a lane
    num_lanes = len( opDataSelection.DatasetGroup )+1
    logger.debug( "num_lanes: {}".format( num_lanes ) )
    opDataSelection.DatasetGroup.resize( num_lanes )

    # Configure it.
    role_index = 0 # raw data
    opDataSelection.DatasetGroup[-1][role_index].setValue( info )
