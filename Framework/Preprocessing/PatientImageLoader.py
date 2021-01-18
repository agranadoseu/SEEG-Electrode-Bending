"""
This class loads images from patient or acpc spaces

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import subprocess

import numpy as np
import pandas as pd

import SimpleITK as sitk


class PatientImageLoader:

    def __init__(self, datasets=None, case=None, space=None):
        # templates
        self.case_dir = os.path.join(datasets, case)
        if space == 'patient':
            self.space_dir = self.case_dir
            self.t1 = os.path.join(self.space_dir, 'T1.nii.gz')
            self.gif = os.path.join(self.space_dir, 'GIF.nii.gz')
        elif space == 'acpc':
            self.space_dir = os.path.join(self.case_dir, 'acpc')
            self.t1 = os.path.join(self.space_dir, 'T1-acpc.nii.gz')
            self.gif = os.path.join(self.space_dir, 'GIF-acpc.nii.gz')
        elif space == 'mni_aff':
            self.space_dir = os.path.join(self.case_dir, 'mniE')
            self.t1 = os.path.join(self.space_dir, 'T1-mni.nii.gz')
            self.gif = os.path.join(self.space_dir, 'GIF-mni.nii.gz')

        self.image = {'T1': None, 'GIF': None}

    def open_patient_images(self):
        self.image['T1'] = sitk.ReadImage(self.t1)
        self.image['GIF'] = sitk.ReadImage(self.gif)