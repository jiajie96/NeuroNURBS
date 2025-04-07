#!/bin/bash

# Adjust the timeout time accordingly, memory could explode when loading few very large cad models


### Process DeepCAD data ###
for i in $(seq 0 18)
do
    # Call python script with different interval values
    timeout 500 python process_brep.py --input ../../../deepcad_export/cad_step --interval $i --option 'deepcad'
    pkill -f '^python process_brep.py' # cleanup after each run
done


### Process ABC data ###
# for i in $(seq 0 99)
# do
#     # Call python script with different interval values
#     timeout 1000 python process_brep.py --input ../../../datasets/abc_step --interval $i --option 'abc'
#     pkill -f '^python process_brep.py' # cleanup after each run
# done


### Process Furniture data ###
# python process_brep.py --input ../../datasets/DeepCAD --option 'deepcad'
# python process_brep.py --input ../Datasets/furniture/sofa/assembly --option 'furniture'
# python process_brep.py --input ../Datasets/furniture/chair/assembly --option 'furniture'