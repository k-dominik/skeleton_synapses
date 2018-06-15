#!/usr/bin/env python
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

# hack to make skeleton_synapses importable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# imports need to be in this order
from skeleton_synapses.ilastik_utils import projects
from skeleton_synapses.ilastik_utils import analyse

# should contain:
# L1-CNS-description-NO-OFFSET.json
# full-vol-autocontext.ilp
INPUT_DIR = os.path.expanduser('~/work/synapse_detection/projects-2018')

# ROI
OFFSET_XYZ = np.array([13489, 20513, 2215])
SHAPE_XYZ = np.array([512, 512, 1])


roi_xyz = np.array([OFFSET_XYZ, OFFSET_XYZ + SHAPE_XYZ])

description_json_path = os.path.join(INPUT_DIR, "L1-CNS-description-NO-OFFSET.json")
autocontext_ilp_path = os.path.join(INPUT_DIR, "full-vol-autocontext.ilp")

opPixelClassification = projects.setup_classifier(description_json_path, autocontext_ilp_path)
raw_xy, predictions_xyc = analyse.fetch_and_predict(roi_xyz, opPixelClassification)

fig, ax_arr = plt.subplots(1, 2)
raw_ax, pred_ax = ax_arr.flatten()

raw_ax.imshow(raw_xy, cmap="gray")
pred_ax.imshow(predictions_xyc)

plt.show()
