"""
This class loads necessary data for pre-processing

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
from Framework.Patient import Electrodes
from Framework.Preprocessing import TemplateImageLoader


class PatientDataLoader:

    def __init__(self, directory=None, case=None):
        self.directory = directory
        self.case = os.path.join(self.directory, case)
        self.acpc = os.path.join(self.case, 'acpc')
        self.mni_aff = os.path.join(self.case, 'mniE')
        self.mni_f3d = os.path.join(self.case, 'f3dE')
        self.medicalImages = MedicalImages.MedicalImages(main=self.case, acpc=self.acpc, mni_aff=self.mni_aff, mni_f3d=self.mni_f3d)
        self.electrodes = Electrodes.Electrodes(main=self.case, acpc=self.acpc, mni_aff=self.mni_aff, mni_f3d=self.mni_f3d)
        self.templates = TemplateImageLoader.TemplateImageLoader(datasets=self.directory)

        # path = os.getcwd()
        # print('The current working directory is {}'.format(path))

    def load_image(self, type=None, space=None, stripped=False):
        # open image
        ct_image, ct_props = self.medicalImages.open(type=type, space=space, stripped=stripped)

    def load_electrodes(self, type=None, space=None):
        # interpolated electrode suffix filename
        suffix = 'i'
        # if space == 'mni_aff':
        #     suffix = 'imniAffine'
        # elif space == 'mni_f3d':
        #     suffix = 'imniF3d'

        # open electrodes
        elec = self.electrodes.load(type=type, suffix=suffix, space=space)
        ghost = self.electrodes.load_ghost(space=space)
        # print(type, space, elec)

        # open features
        # self.electrodes.load_features()

    def load_features(self):
        self.features = None
        df_file = os.path.join(self.case, 'features.pkl')
        if os.path.exists(df_file):
            self.features = pd.read_pickle(df_file)
            print('Dataframe of cases does exist and is loaded for analysis: ', self.features.shape)

    def execute(self):
        # open images
        self.load_image(type='T1', space='patient')
        self.load_image(type='T1', space='acpc')
        self.load_image(type='T1', space='mni_aff', stripped=True)
        self.load_image(type='T1', space='mni_f3d')
        self.load_image(type='CT', space='patient')
        self.load_image(type='GIF', space='patient')
        self.load_image(type='GIF', space='acpc')
        self.load_image(type='GIF', space='mni_aff')

        # load electrodes
        self.load_electrodes(type='plan', space='patient')
        self.load_electrodes(type='plan', space='acpc')
        self.load_electrodes(type='plan', space='mni_aff')
        self.load_electrodes(type='plan', space='mni_f3d')
        self.load_electrodes(type='impl', space='patient')
        self.load_electrodes(type='impl', space='acpc')
        self.load_electrodes(type='impl', space='mni_aff')
        self.load_electrodes(type='impl', space='mni_f3d')

        # load features
        self.load_features()
