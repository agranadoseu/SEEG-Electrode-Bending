"""
This class generates data while an electrode trajectory is inferred.
This mimics what the FullPipeline script does for entire trajectories

inputs: pred trajetory
outputs data dictionary

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import numpy as np
import pandas as pd

from Framework.Patient import MedicalImages
from Framework.Preprocessing import ElasticRodFilter
from Framework.Preprocessing import TrajectoryFilter
from Framework.Preprocessing import SurfaceCollisionFilter
from Framework.Preprocessing import AssemblyFilter

from Framework.Tools import XMLParser
from Framework.Tools import FileSystem


class DataGeneration:
    def __init__(self, case=None):
        self.directory = 'E:\\UCL\\Backup\\medic-biomechanics\\Datasets\\BendingJournal\\datasets'

        # init medical images
        self.case = os.path.join(self.directory, case)
        self.acpc = os.path.join(self.case, 'acpc')
        self.mni_aff = os.path.join(self.case, 'mniE')
        self.mni_f3d = os.path.join(self.case, 'f3dE')
        self.medicalImages = MedicalImages.MedicalImages(main=self.case, acpc=self.acpc, mni_aff=self.mni_aff, mni_f3d=self.mni_f3d)
        self.xmlparser = XMLParser.XMLParser()
        self.filesystem = FileSystem.FileSystem()

        # load images
        self.medicalImages.open(type='T1', space='mni_aff', stripped=True)
        self.medicalImages.open(type='GIF', space='mni_aff')

        # load stylet file
        self.stylet_df = None
        stylet_file = os.path.join(self.case, 'stylet.txt')
        stylet_exists = os.path.exists(stylet_file)
        if stylet_exists:
            self.stylet_df = pd.read_csv(stylet_file, sep=",", header=0)
        print('stylet_df', self.stylet_df)

        # collision
        self.surface_collision = SurfaceCollisionFilter.SurfaceCollisionFilter(images=self.medicalImages,
                                                                          electrodes=None,
                                                                          space='mni_aff')
        self.surface_collision.load_polydata()

        # create prediction folder
        self.pred_folder = os.path.join(self.mni_aff, 'pred')
        if not os.path.exists(self.pred_folder):
            self.filesystem.create_dir(self.pred_folder)

    def create_electrode(self, name=None, ep=None, tp=None, contacts=None, points=None):
        ''' points go from TP to EP '''
        # dictionary to save data
        data = {'name': [], 'id': [], 'num_contacts': [], 'ep': [], 'tp': [], 'stylet': [], 'points': []}

        # stylet information
        stylet_val = -1
        if self.stylet_df is not None:
            elec_id = int(''.join([i for i in name if i.isdigit()]))
            for index, row in self.stylet_df.iterrows():
                stylet_id = int(''.join([i for i in row['name'] if i.isdigit()]))
                if elec_id == stylet_id:
                    stylet_val = row['stylet']
                    break

        # update dictionary
        data['name'].append(name)
        data['id'].append(int(''.join([i for i in name if i.isdigit()])))
        data['num_contacts'].append(contacts)
        data['ep'].append(ep)
        data['tp'].append(tp)
        data['stylet'].append(stylet_val)
        # data['contacts'].append(np.flip(contacts, 0))
        data['points'].append(np.flip(points, 0))

        return data

    def create_rod(self, elec=None):
        ''' points go from TP to EP '''
        # dictionary to save data
        data = {'name': [], 'id': [], 'ep': [], 'tp': [], 'points': []}

        # create rods
        elastic_rod = ElasticRodFilter.ElasticRodFilter(images=self.medicalImages, electrodes=None)
        # elastic_rod.execute()

        name = elec['name'][0]
        id = elec['id'][0]
        num_contacts = elec['num_contacts'][0]
        ep = elec['ep'][0]
        tp = elec['tp'][0]
        points = elec['points'][0]

        # compute ghost points
        gEp, mEp, gPoints, gMids = elastic_rod.compute_ghost_points(ep=ep, points=points, frame=elastic_rod.frame['mni_aff'])
        # print('     gEp={} mEp{} gPoints={} gMids={}'.format(gEp, mEp, gPoints, gMids))

        # update dictionary
        data['name'].append('G'+name[1:])
        data['id'].append(id)
        data['ep'].append(gEp)
        data['tp'].append(gPoints[0])
        data['points'].append(gPoints)

        return data

    def compute_displacements(self, plan=None, pred=None, ghost=None):
        # compute displacements
        trajectories = TrajectoryFilter.TrajectoryFilter(images=self.medicalImages, electrodes=None)
        trajectories.execute_infer(plan=plan, pred=pred, ghost=ghost)

        return trajectories

    def compute_collision(self, impl=None):

        self.surface_collision.execute_infer(impl=impl)

        return self.surface_collision.collision

    def assemble(self, case=None, plan=None, pred=None, features=None, depth=None):
        # assembly of features
        assembly = AssemblyFilter.AssemblyFilter(images=self.medicalImages, electrodes=None,
                                                 space='mni_aff', features=features)
        case_features_df = assembly.execute(case=case, plan=plan, impl=pred, depth=depth)

        return case_features_df

    def save_electrode(self, name=None, ep=None, points=None, colour=None):
        xml_file = os.path.join(self.pred_folder, name + '.xmlE')

        self.xmlparser.save_points_as_xml(name=name, ep=ep, points=np.flip(points, 0), colour=colour, xml_file=xml_file)