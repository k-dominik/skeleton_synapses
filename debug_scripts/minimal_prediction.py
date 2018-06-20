#!/usr/bin/env python
import os
import sys
import logging
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import psutil

# setup logging
logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

# enforce ram limit:
from lazyflow.utility import Memory
RAM_LIMIT_MB = 2000
ram = RAM_LIMIT_MB * 1024**2
fmt = Memory.format(ram)
logger.info("Configuring lazyflow RAM limit to {}".format(fmt))
Memory.setAvailableRam(ram)

# Trigger refresh of MemoryManager more often
from lazyflow.operators.cacheMemoryManager import CacheMemoryManager
memory_logger = logging.getLogger('lazyflow.operators.cacheMemoryManager')
memory_logger.setLevel(logging.DEBUG)
status_interval_secs = 1
CacheMemoryManager().setRefreshInterval(status_interval_secs)

# hack to make skeleton_synapses importable
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# imports need to be in this order
from skeleton_synapses.ilastik_utils import projects
from skeleton_synapses.ilastik_utils import analyse

# should contain:
# L1-CNS-description-NO-OFFSET.json
# full-vol-autocontext.ilp
# INPUT_DIR = os.path.expanduser('/export/home/dkutra_local/sources/chris_barnes_debug/autocontext')
INPUT_DIR = os.path.abspath('../../autocontext-files')

# ROI
OFFSET_XYZ = np.array([13489, 20513, 2215])
SHAPE_XYZ = np.array([512, 512, 1])
COUNT_Z = 20

description_json_path = os.path.join(INPUT_DIR, "L1-CNS-description-NO-OFFSET.json")
autocontext_ilp_path = os.path.join(INPUT_DIR, "full-vol-autocontext.ilp")

opPixelClassification = projects.setup_classifier(description_json_path, autocontext_ilp_path)

proc = psutil.Process()
RAM = [proc.memory_info().rss / 1e6]
print("Latest memory use: {}MB".format(int(RAM[-1])))
out = []
for idx in tqdm(range(COUNT_Z)):
    roi_xyz = np.array([OFFSET_XYZ + [0, 0, idx], OFFSET_XYZ + SHAPE_XYZ + [0, 0, idx]])
    raw_xy, pred_xyc = analyse.fetch_and_predict(roi_xyz, opPixelClassification)
    RAM.append(proc.memory_info().rss / 1e6)
    tqdm.write("Latest memory use: {}MB".format(int(RAM[-1])))


# fig, ax_arr = plt.subplots(1, 3)
# raw_ax, pred_ax, ram_ax = ax_arr.flatten()
#
# raw_ax.imshow(out[-1][0], cmap="gray")
# pred_ax.imshow(out[-1][1])

fig, ram_ax = plt.subplots(1, 1)
ram_ax.plot(RAM, label="RAM usage")
ram_ax.set_xlabel("iteration")
ram_ax.set_ylabel("RAM usage (MB)")

if RAM_LIMIT_MB:
   ram_ax.plot([0, COUNT_Z], [RAM_LIMIT_MB, RAM_LIMIT_MB], label="RAM limit")

plt.show()
