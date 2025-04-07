# -*- coding: utf-8 -*-

import os
import pickle
import glob

from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.STEPConstruct import stepconstruct_FindEntity
from OCC.Core.TCollection import TCollection_HAsciiString
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Display.SimpleGui import init_display

import helpers.utils.occ_utils as occ_utils
import helpers.utils.tables as tables


def shape_with_fid_to_step(filename, shape, id_map):
    '''
    input
        filename
        shape
        id_map： {TopoDS_Face: int}
    output
    '''
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)

    finderp = writer.WS().TransferWriter().FinderProcess()

    fset = occ_utils.list_face(shape)

    loc = TopLoc_Location()
    for face in fset:
        item = stepconstruct_FindEntity(finderp, face, loc)
        if item is None:
            print(face)
            continue
        item.SetName(TCollection_HAsciiString(str(id_map[face])))

    writer.Write(filename)


def shape_with_fid_from_step(filename):
    """
    input
    output
        shape:      TopoDS_Shape
        id_map:  {TopoDS_Face: int}
    """
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    id_map = {}
    fset = occ_utils.list_face(shape)
    # read the face names
    for face in fset:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            nameid = int(name)
            id_map[face] = nameid

    return shape, id_map


class LabeledShape:
    def __init__(self):
        self.shape_name = ''

    def load(self, shape_path, shape_name):
        filename = os.path.join(shape_path, shape_name + '.step')
        self.shape, self.face_ids = shape_with_fid_from_step(filename)

        filename = os.path.join(shape_path, shape_name + '.face_truth')

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                self.face_truth = pickle.load(file)
        else:
            self.face_truth = [100] * len(self.face_ids)

        self.shape_name = shape_name

    def save(self, shape_path):
        filename = os.path.join(shape_path, self.shape_name + '.step')
        shape_with_fid_to_step(filename, self.shape, self.face_ids)

        filename = os.path.join(shape_path, self.shape_name + '.face_truth')
        with open(filename, 'wb') as file:
            pickle.dump(self.face_truth, file)

    def display(self, occ_display):
        occ_display.EraseAll()
        AIS = AIS_ColoredShape(self.shape)
        face_label_map = {f:self.face_truth[self.face_ids[f]] for f in self.face_ids}
        for a_face in face_label_map:
            AIS.SetCustomColor(a_face, tables.colors[face_label_map[a_face]])

        occ_display.Context.Display(AIS)
        occ_display.View_Iso()
        occ_display.FitAll()
        print(self.shape_name)
