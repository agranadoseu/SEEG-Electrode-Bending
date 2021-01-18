"""
This class handles all template related functions

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


class TemplateImageLoader:

    def __init__(self, datasets=None):
        # templates
        self.templates_dir = os.path.join(datasets[0:datasets.rfind('\\')], 'template')
        self.mni_template = os.path.join(self.templates_dir, 'MNI_152_mri.nii.gz')
        self.mni_template_stripped = os.path.join(self.templates_dir, 'MNI_152_mri-stripped.nii.gz')
        self.mni_template_gif = os.path.join(self.templates_dir, 'MNI_152_gif.nii.gz')
        self.mni = {'T1': None, 'GIF': None}

    def open_mni_images(self):
        self.mni['T1'] = sitk.ReadImage(self.mni_template)
        self.mni['GIF'] = sitk.ReadImage(self.mni_template_gif)