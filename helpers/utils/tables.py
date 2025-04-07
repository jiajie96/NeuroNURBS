# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:57:15 2019
@author: 2624224
"""
import numpy as np
from OCC.Display.OCCViewer import rgb_color

colors = []
rgb_list = np.array(np.meshgrid([0.9, 0.6, 0.3], [0.9, 0.6, 0.3], [0.9, 0.6, 0.3])).T.reshape(-1,3)
for rgb in rgb_list:
    colors.append(rgb_color(rgb[0], rgb[1], rgb[2]))