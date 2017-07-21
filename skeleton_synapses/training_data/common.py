import os

this_dir = os.path.dirname(os.path.realpath(__file__))
rel_path_to_projects = '../../projects-2017/L1-CNS/projects'
ilp_name = 'full-vol-autocontext.ilp'

ILP_PATH = os.path.join(this_dir, rel_path_to_projects, ilp_name)

lane_paths = ['PixelClassification', 'PixelClassification01']