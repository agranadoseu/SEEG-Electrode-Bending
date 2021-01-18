"""
Script to create synthetic data and be able to evaluate methods
This will be based on the MNI template and similar to FullPipeline to generate a npy data file

Features:
- 5 main trajectories each side along the temporal lobe
    * no bending
    * bending upward/downward
    * bending left/right
- introduce noise at EP and bolt direction


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

from Framework.Preprocessing import TemplateImageLoader


class SyntheticDataGenerator:
    
    def __init__(self, directory=None, case=None):
        self.directory = directory
        self.case = os.path.join(self.directory, case)
        self.templates = TemplateImageLoader.TemplateImageLoader(datasets=self.directory)
        
    def execute(self):
        # plan
        # impl
        return
