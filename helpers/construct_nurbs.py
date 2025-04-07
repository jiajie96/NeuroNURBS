from helpers.utils.occ_utils import *
from helpers.utils import occ_utils, occ_face, occ_construct, occ_topology
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array1OfInteger, TColStd_Array2OfReal
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.gp import gp_Pnt
from OCC.Core.Geom import Geom_BSplineSurface
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.StlAPI import StlAPI_Reader, StlAPI_Writer
from OCC.Core.BRepCheck import BRepCheck_Analyzer
import matplotlib.pyplot as plt
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepAlgoAPI import *
from OCC.Core.TopAbs import *
import itertools as it
from OCC.Core.gp import gp_Vec, gp_Trsf
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
import numpy as np
from OCC.Core.BRepBuilderAPI import *
from OCC.Core.StlAPI import StlAPI_Reader, StlAPI_Writer
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
import torch
import torch.nn as nn
from OCC.Core.ShapeFix import ShapeFix_Face, ShapeFix_Wire, ShapeFix_Edge

def fix_face(face):
    fixer = ShapeFix_Face(face)
    fixer.SetPrecision(0.01)
    fixer.SetMaxTolerance(0.1)
    ok = fixer.Perform()
    # assert ok
    fixer.FixOrientation()
    face = fixer.Face()
    return face

def uv_points_from_face(face, _show = True):
    face_util = occ_face.Face(face)

    u_min, u_max,v_min,v_max = face_util.domain()

    sampling_int = 32
    f_s = np.zeros((sampling_int, sampling_int, 3))
    us = np.linspace(u_min, u_max, sampling_int)
    vs = np.linspace(v_min, v_max, sampling_int)

    for _u_nb in range(sampling_int):
        for _v_nb in range(sampling_int):
            xyz_coord = face_util.surface.Value(us[_u_nb], vs[_v_nb])
            f_s[_u_nb, _v_nb] = xyz_coord.Coord()

    if _show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(f_s[:, :, 0], f_s[:,:, 1], f_s[:,:,2], c='b',alpha=0.5)
        plt.show()
        plt.ioff()
    return np.asarray(f_s)

def get_kv_from_raw(raw_kv):
    kv_rounded = raw_kv
    kv_rounded = np.round(raw_kv, 1)
    last_one_idx = np.where(kv_rounded>0.5)[0][-1]
    kv_geomdl = kv_rounded[:last_one_idx + 1]
    kv_set = set(kv_geomdl)
    kv_set = sorted(kv_set)
    kv_occ = TColStd_Array1OfReal(1, len(kv_set))
    for i, knot in enumerate(kv_set):
        kv_occ.SetValue(i + 1, float(knot))
    mult = [np.count_nonzero(kv_geomdl ==_) for _ in kv_set]
    mult_occ = TColStd_Array1OfInteger(1, len(mult))
    for i, _mult in enumerate(mult):
        mult_occ.SetValue(i + 1, _mult)
    
    return kv_occ, mult_occ, mult

# Convert geomdl control points to OpenCASCADE control points format
def construct_Nurbs(ctrlpts_raw, ukv_raw, vkv_raw, tolerance = True, save_path=None):
    """Creates a non-rational b-spline surface (weights default value is 1.). //
    ! the following conditions must be verified. 
    ! 0 < udegree <= maxdegree. //
    ! uknots.length() == umults.length() >= 2 //
    ! uknots(i) < uknots(i+1) (knots are increasing) 1 <= umults(i) <= udegree //
    ! on a non uperiodic surface the first and last umultiplicities may be udegree+1 
    (this is even recommanded if you want the curve to start and finish on the first and last pole). //
    ! on a uperiodic surface the first and the last umultiplicities must be the same. //
    ! on non-uperiodic surfaces //
    ! poles.collength() == sum(umults(i)) - udegree - 1 >= 2 //
    """
    ctrlpts_raw = np.round(ctrlpts_raw, 4)
    non_zero_idx = np.where(ctrlpts_raw[:,:,3]>0.1)
    size_u = len(list(set(non_zero_idx[0])))
    size_v = len(list(set(non_zero_idx[1])))
    ctrlpts = ctrlpts_raw[:size_u,:,:]
    ctrlpts = ctrlpts[:,:size_v,:]
    
    ctrlpts_occ = TColgp_Array2OfPnt(1, int(size_u), 1, int(size_v))
    for i in range(size_u):
        for j in range(size_v):
            pt = ctrlpts[i][j]
            ctrlpts_occ.SetValue(i + 1, j + 1, gp_Pnt(float(pt[0]), float(pt[1]), float(pt[2])))
            
    weights_occ = TColStd_Array2OfReal(1, int(size_u), 1, int(size_v))
    for i in range(size_u):
        for j in range(size_v):
            weight = ctrlpts[i][j][3]
            weights_occ.SetValue(i + 1, j + 1, float(weight))
            
    ukv_occ, mult_u_occ, umults = get_kv_from_raw(ukv_raw) 
    vkv_occ, mult_v_occ, vmults = get_kv_from_raw(vkv_raw) 
    degree_u = int(sum(umults) - size_u - 1)
    degree_v = int(sum(vmults) - size_v - 1)
    try:
        surf = Geom_BSplineSurface(
        ctrlpts_occ, weights_occ,
        ukv_occ, vkv_occ,
        mult_u_occ, mult_v_occ,
        degree_u, degree_v,
        False, False  # Not periodic in u or v direction
        )
        return surf
        
    except Exception as e:
        if tolerance:
            return None
        else:
            print(f"Error: {e}", umults, vmults) 
            
