"""
This class generates necessary data for pre-processing

Requires (patient space):
    - co-registered medical images: T1.nii.gz, CT.nii.gz, GIF.nii.gz
    - segmented electrodes from CT (E*.xmlE)
    - stylet information (stylet.txt)
    - AC-PC coordinates (acpcih.mps)

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

from Framework.Patient import MedicalImages
from Framework.Patient import Electrodes
from Framework.Patient import Surfaces
from Framework.Preprocessing import TemplateImageLoader
from Framework.Preprocessing import ImageTransformFilter
from Framework.Preprocessing import ImageProcessingFilter
from Framework.Preprocessing import ElectrodeTransformFilter
from Framework.Preprocessing import ElasticRodFilter

from Framework.Tools import FileSystem

class PatientDataGenerator:

    def __init__(self, directory=None, case=None):
        self.directory = directory
        self.case = os.path.join(self.directory, case)
        self.acpc = os.path.join(self.case, 'acpc')
        self.mni_aff = os.path.join(self.case, 'mniE')
        self.mni_f3d = os.path.join(self.case, 'f3dE')
        self.medicalImages = MedicalImages.MedicalImages(main=self.case, acpc=self.acpc, mni_aff=self.mni_aff, mni_f3d=self.mni_f3d)
        self.electrodes = Electrodes.Electrodes(main=self.case, acpc=self.acpc, mni_aff=self.mni_aff, mni_f3d=self.mni_f3d)
        self.templates = TemplateImageLoader.TemplateImageLoader(datasets=self.directory)
        self.surfaces = Surfaces.Surfaces()
        self.filesystem = FileSystem.FileSystem()

        # load electrodes
        plan = self.electrodes.load(type='plan', suffix='i', space='patient')
        impl = self.electrodes.load(type='impl', suffix='i', space='patient')

    def execute(self):
        # @TODO create rigid electrodes
        # @TODO interpolate all electrodes electrodes
        # @TODO generate features

        # transform images
        self.transform_images()

        # pre-process images
        self.process_images()

        # transform electrodes to ACPC and MNI (affine and f3d)
        self.transform_electrodes()

        # create rods (mni and mni_aff)
        self.create_rods()

        # transform surfaces
        self.transform_surfaces()

    def transform_images(self):
        imageTransform = ImageTransformFilter.ImageTransformFilter(images=self.medicalImages, templates=self.templates)
        imageTransform.execute()

    def process_images(self):
        imageProcessing = ImageProcessingFilter.ImageProcessingFilter(images=self.medicalImages)
        imageProcessing.execute()

    def transform_electrodes(self):
        electrodeTransform = ElectrodeTransformFilter.ElectrodeTransformFilter(images=self.medicalImages, electrodes=self.electrodes, templates=self.templates)
        electrodeTransform.execute()

    def create_rods(self):
        # load electrodes
        plan = self.electrodes.load(type='plan', suffix='i', space='acpc')
        impl = self.electrodes.load(type='impl', suffix='i', space='acpc')
        plan = self.electrodes.load(type='plan', suffix='i', space='mni_aff')
        impl = self.electrodes.load(type='impl', suffix='i', space='mni_aff')

        # create rods
        elastic_rod = ElasticRodFilter.ElasticRodFilter(images=self.medicalImages, electrodes=self.electrodes)
        elastic_rod.execute()

    def transform_surfaces(self):
        # load surfaces from patient space
        directory = self.medicalImages.get_directory(space='patient')
        filenames = ['s_scalp.stl', 's_cortex.stl', 's_white.stl', 's_deep.stl']
        polydata = {'scalp': self.surfaces.load_scalp(stl_folder=directory, image=self.medicalImages.mri_img['patient']),
                    'cortex': self.surfaces.load_structure(stl_folder=directory, image=self.medicalImages.gif_img['patient'], structure='cortex'),
                    'white': self.surfaces.load_structure(stl_folder=directory, image=self.medicalImages.gif_img['patient'], structure='white'),
                    'deep': self.surfaces.load_structure(stl_folder=directory, image=self.medicalImages.gif_img['patient'], structure='deep')}

        # create stl directory
        acpc_stl_folder = os.path.join(self.medicalImages.get_directory(space='acpc'), 'stl')
        mni_stl_folder = os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'stl')
        if not os.path.exists(acpc_stl_folder):
            self.filesystem.create_dir(acpc_stl_folder)
        if not os.path.exists(mni_stl_folder):
            self.filesystem.create_dir(mni_stl_folder)

        # load transformations
        ref_file = os.path.join(self.medicalImages.dir, 'T1.nii.gz')
        acpc_dir = self.medicalImages.get_directory(space='acpc')
        mni_dir = self.medicalImages.get_directory(space='mni_aff')
        acpc_trans_inv_file = os.path.join(acpc_dir, 'T1-acpc-inv.txt')
        mni_trans_inv_file = os.path.join(mni_dir, 'T1-mni.txt')

        # acpc_M = self.filesystem.open_registration_matrix(file=os.path.join(self.medicalImages.get_directory(space='acpc'), 'T1-acpc.txt'))
        # mni_M = self.filesystem.open_registration_matrix(file=os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'T1-mni.txt'))

        # transform into acpc and mni
        for s in polydata.keys():
            # file exists?
            acpc_stl_file = os.path.join(acpc_stl_folder, 's_'+s+'.stl')
            mni_stl_file = os.path.join(mni_stl_folder, 's_'+s+'.stl')

            # transform and save
            if not os.path.exists(acpc_stl_file):
                stl_surface = self.surfaces.transform_polydata(polydata=polydata[s], ref_image=ref_file, M=acpc_trans_inv_file, stl_folder=acpc_stl_folder, filename='s_'+s)
            if not os.path.exists(mni_stl_file):
                stl_surface = self.surfaces.transform_polydata(polydata=polydata[s], ref_image=ref_file, M=mni_trans_inv_file, stl_folder=mni_stl_folder, filename='s_'+s)
