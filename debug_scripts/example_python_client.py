"""
This script provides an example of how a pre-trained ilastik
PixelClassification project can be used to generate predictions
from within Python, without the need to read/write data from disk.
Once the project is loaded, this script doesn't touch the hard-disk.
"""
from collections import OrderedDict
import os

import numpy
import vigra

from matplotlib import pyplot as plt

import ilastik_main
from ilastik.applets.dataSelection import DatasetInfo
from ilastik.workflows.pixelClassification import PixelClassificationWorkflow

INPUT_DIR = os.path.expanduser('~/work/synapse_detection/projects-2018')
autocontext_ilp_path = os.path.join(INPUT_DIR, "full-vol-autocontext.ilp")


# Before we start ilastik, optionally prepare these environment variable settings.
os.environ["LAZYFLOW_THREADS"] = "2"
os.environ["LAZYFLOW_TOTAL_RAM_MB"] = "2000"

# Programmatically set the command-line arguments directly into the argparse.Namespace object
# Provide your project file, and don't forget to specify headless.
args = ilastik_main.parser.parse_args([])
args.headless = True
# args.retrain = False
args.project = os.path.join(INPUT_DIR, "full-vol-autocontext.ilp")

# Instantiate the 'shell', (in this case, an instance of ilastik.shell.HeadlessShell)
# This also loads the project file into shell.projectManager
shell = ilastik_main.main( args )
# assert isinstance(shell.workflow, PixelClassificationWorkflow)

# Obtain the training operator
opPixelClassification = shell.workflow.pcApplets[-1].topLevelOperator

# Sanity checks
assert len(opPixelClassification.InputImages) > 0
assert opPixelClassification.Classifier.ready()

# For this example, we'll use random input data to "batch process"
input_data1 = numpy.random.randint(0,255, (200,200,1) ).astype(numpy.uint8)
input_data2 = numpy.random.randint(0,255, (300,300, 1) ).astype(numpy.uint8)
print(input_data1.shape)

# In this example, we're using 2D data (with an extra dimension for  channel).
# Tagging the data this way ensures that ilastik interprets the axes correctly.
input_data1 = vigra.taggedView( input_data1, 'yxc' )
input_data2 = vigra.taggedView( input_data2, 'yxc' )

# In case you're curious about which label class is which,
# let's read the label names from the project file.
label_names = opPixelClassification.LabelNames.value
label_colors = opPixelClassification.LabelColors.value
probability_colors = opPixelClassification.PmapColors.value

print(label_names, label_colors, probability_colors)

# Construct an OrderedDict of role-names -> DatasetInfos
# (See PixelClassificationWorkflow.ROLE_NAMES)
role_data_dict = OrderedDict([ ("Raw Data", [ DatasetInfo(preloaded_array=input_data1),
                                              DatasetInfo(preloaded_array=input_data2) ]) ])

## Note: If you want to pull your data from disk instead of in-memory, just provide filepaths like so:
# role_data_dict = OrderedDict([ ("Raw Data", [ '/path/to/input-file-1.png',
#                                               '/path/to/input-file-2.h5/mydata' ]) ])

# Run the export via the BatchProcessingApplet
# Note: If you don't provide export_to_array, then the results will
#       be exported to disk accordering to your project's DataExport settings.
#       In that case, run_export() returns None.
predictions = shell.workflow.batchProcessingApplet.run_export(role_data_dict, export_to_array=True)

print("Computed {} result arrays:".format( len(predictions) ))
for result in predictions:
    print(result.dtype, result.shape, f"{result.min()} to {result.max()}")

fig, ax_arr = plt.subplots(2, 2)
raw_ax1, pred_ax1, raw_ax2, pred_ax2 = ax_arr.flatten()

for raw_data, pred_data, raw_ax, pred_ax in zip(
        [input_data1, input_data2], predictions, [raw_ax1, raw_ax2], [pred_ax1, pred_ax2]
):
    raw_ax.imshow(raw_data)
    pred_ax.imshow(pred_data)

plt.show()

print("DONE.")
