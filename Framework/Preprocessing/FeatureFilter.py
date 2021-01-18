"""
This class generates a hand-crafted features of electrode trajectories

Steps:
-

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

import SimpleITK as sitk


class FeatureFilter:

    def __init__(self, images=None, electrodes=None):
        self.medicalImages = images
        self.electrodes = electrodes

        self.mni_frame = np.eye(3, dtype=np.float)

    def execute(self, space=None):
        self.space = space
        self.plan = self.electrodes.plan[space]
        self.impl = self.electrodes.impl[space]

        # compute
        # self.local_delta = self.compute_local_displacement(impl=self.impl)
        # self.delta = self.compute_displacement(plan=self.plan, impl=self.impl)
