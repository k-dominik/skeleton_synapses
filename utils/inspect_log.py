#!/usr/bin/env python
import re

path = '/home/cbarnes/work/synapse_detection/skeleton_synapses/projects-2017/L1-CNS/logs/2017-05-22_19:59:14/locate_synapses.txt'

input_lst = []
output_lst = []

input_node_set = set()
output_node_set = set()

input_re = re.compile('Addressing node (\d+);')
output_re = re.compile('PROGRESS: segmented area around node (\d+);')

with open(path) as f:
    for line in f:
        if 'Addressing node' in line:
            input_lst.append(line)
            node = input_re.search(line).groups()[0]
            assert node
            input_node_set.add(node)
        elif 'PROGRESS: segmented' in line:
            output_lst.append(line)
            node = output_re.search(line).groups()[0]
            assert node
            output_node_set.add(node)

assert len(input_node_set) == len(output_node_set)
