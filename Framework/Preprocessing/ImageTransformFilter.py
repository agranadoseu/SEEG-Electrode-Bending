"""
This class transform medical images into ACPC and MNI (affine and f3d) space

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
from Framework.Preprocessing import TemplateImageLoader
from Framework.Tools import FileSystem
from Framework.Tools import XMLParser
from Framework.Tools import NiftyReg
from Framework.Tools import Robex


class ImageTransformFilter:

    def __init__(self, images=None, templates=None):
        self.medicalImages = images
        self.templates = templates

        self.registration = NiftyReg.NiftyReg()
        self.filesystem = FileSystem.FileSystem()
        self.xmlparser = XMLParser.XMLParser()
        self.robex = Robex.Robex()

    def execute(self):
        # load acpc coordinates
        acpc_points = self.load_acpc_coord()

        # T1 to ACPC
        self.transform_rigid_acpc(acpcih=acpc_points)
        self.resample_image_acpc(flo='T1', inter=3)
        self.resample_image_acpc(flo='CT', inter=3)
        self.resample_image_acpc(flo='GIF', inter=0)

        # T1 -> MNI affine transform
        self.transform_affine_mni(flo='T1')
        self.resample_image_mni(flo='CT', inter=3)
        self.resample_image_mni(flo='GIF', inter=0)

        # Skull stripping and d) F3D transform
        self.transform_f3d_mni()

    def load_acpc_coord(self):
        acpc_points = []
        acpc_file = os.path.join(self.medicalImages.get_directory(space='patient'), 'acpcih.mps')
        if os.path.exists(acpc_file):
            acpc_points = self.xmlparser.open_mps(file=acpc_file)
            if len(acpc_points) >= 3:
                # get points
                ac, pc, ih = acpc_points[0], acpc_points[1], acpc_points[2]
                print('ac={}, pc={}, ih={}'.format(ac, pc, ih))
        else:
            print('ERROR: acpc file = {} do not exist'.format(acpc_file))

        return acpc_points

    def transform_rigid_acpc(self, acpcih=None):
        # create directory and transformation matrix
        acpc_dir = self.medicalImages.get_directory(space='acpc')
        if not os.path.exists(acpc_dir):
            self.filesystem.create_dir(acpc_dir)
        if not os.path.exists(os.path.join(acpc_dir, 'T1-acpc.txt')):
            # images
            M, Minv = self.build_acpc_transformation_matrix(acpcih=acpcih)
            self.filesystem.save_registration_matrix(matrix=M, file=os.path.join(acpc_dir, 'T1-acpc.txt'))
            self.filesystem.save_registration_matrix(matrix=Minv, file=os.path.join(acpc_dir, 'T1-acpc-inv.txt'))

            # # points negate 1st and 2nd rows
            # M[0, :] *= -1
            # M[1, :] *= -1
            # Minv[0, :] *= -1
            # Minv[1, :] *= -1
            # self.filesystem.save_registration_matrix(matrix=M, file=os.path.join(acpc_dir, 'T1-acpc-xyz.txt'))
            # self.filesystem.save_registration_matrix(matrix=Minv, file=os.path.join(acpc_dir, 'T1-acpc-inv-xyz.txt'))

    def build_acpc_transformation_matrix(self, acpcih=None):
        M = np.eye(4)
        [ac, pc, ih] = acpcih

        # build affine matrix: RAS
        frame_y = ac - pc
        frame_y /= np.linalg.norm(frame_y)
        frame_z = ih - ac
        frame_z /= np.linalg.norm(frame_z)
        frame_x = np.cross(frame_y, frame_z)
        frame_x /= np.linalg.norm(frame_x)
        frame_z = np.cross(frame_x, frame_y)
        frame_z /= np.linalg.norm(frame_z)
        R = np.vstack([frame_x, frame_y, frame_z])
        t = -np.dot(R, ac)
        # t = -ac
        M[:3, :3] = R
        M[:3, 3] = t
        Minv = np.linalg.inv(M)

        return M, Minv

    def resample_image_acpc(self, flo=None, inter=None):
        # directory
        to_dir = self.medicalImages.get_directory(space='acpc')

        # files
        flo_file = os.path.join(self.medicalImages.dir, flo+'.nii.gz')
        to_file = os.path.join(to_dir, flo+'-acpc.nii.gz')
        inv_trans_file = os.path.join(to_dir, 'T1-acpc-inv.txt')

        # affine transformation
        if not os.path.exists(to_file):
            if os.path.exists(flo_file) and os.path.exists(inv_trans_file):
                self.registration.resample(ref_img=self.templates.mni_template,
                                           flo_img=flo_file, res_img=to_file,
                                           trans_matrix=inv_trans_file, inter=inter)

    def resample_image_mni(self, flo=None, inter=None):
        # directory
        to_dir = self.medicalImages.get_directory(space='mni_aff')

        # files
        flo_file = os.path.join(self.medicalImages.dir, flo+'.nii.gz')
        to_file = os.path.join(to_dir, flo+'-mni.nii.gz')
        inv_trans_file = os.path.join(to_dir, 'T1-mni.txt')

        # affine transformation
        if not os.path.exists(to_file):
            if os.path.exists(flo_file) and os.path.exists(inv_trans_file):
                self.registration.resample(ref_img=self.templates.mni_template,
                                           flo_img=flo_file, res_img=to_file,
                                           trans_matrix=inv_trans_file, inter=inter)

    def transform_affine_mni(self, flo=None):
        # directory
        to_dir = self.medicalImages.get_directory(space='mni_aff')
        if not os.path.exists(to_dir):
            self.filesystem.create_dir(to_dir)

        # files
        flo_file = os.path.join(self.medicalImages.dir, flo+'.nii.gz')
        to_file = os.path.join(to_dir, flo+'-mni.nii.gz')
        trans_file = os.path.join(to_dir, flo+'-mni.txt')
        trans_inv_file = os.path.join(to_dir, flo + '-mni-inv.txt')

        # affine transformation
        if not os.path.exists(to_file) and not os.path.exists(trans_file):
            if os.path.exists(flo_file):
                self.registration.register_affine(ref_img=self.templates.mni_template,
                                                  flo_img=flo_file, res_img=to_file,
                                                  trans_matrix=trans_file)
            else:
                print('Reference file does not exist to perform affine transformation to MNI space')

        # invert transform
        if not os.path.exists(trans_inv_file):
            self.registration.invert_transform_matrix(trans_matrix=trans_file, inv_matrix=trans_inv_file)

    def transform_f3d_mni(self):
        # strip skull from T1 MNI image
        mni_dir = self.medicalImages.get_directory(space='mni_aff')
        skullstrip_file = os.path.join(mni_dir, 'T1-mni-stripped.nii.gz')
        if not os.path.exists(skullstrip_file):
            input_file = os.path.join(mni_dir, 'T1-mni.nii.gz')
            self.robex.skull_stripping(in_img=input_file, out_img=skullstrip_file)

        # F3D transform
        f3d_dir = self.medicalImages.get_directory(space='mni_f3d')
        f3d_file = os.path.join(f3d_dir, 'T1-mni-stripped-f3d.nii.gz')
        cpp_file = os.path.join(f3d_dir, 'T1-mni-stripped-cpp.nii.gz')
        cpp_inv_file = os.path.join(f3d_dir, 'T1-mni-stripped-cpp-inv.nii.gz')
        if not os.path.exists(f3d_dir):
            self.filesystem.create_dir(f3d_dir)
        if not os.path.exists(cpp_file) or not os.path.exists(f3d_file):
            self.registration.register_f3d(ref_img=self.templates.mni_template_stripped,
                                           flo_img=skullstrip_file,
                                           cpp_img=cpp_file,
                                           res_img=f3d_file)

        # invert transform
        if not os.path.exists(cpp_inv_file):
            self.registration.invert_transform_cpp(ref_img=self.templates.mni_template_stripped,
                                                   flo_img=skullstrip_file,
                                                   cpp_img=cpp_file,
                                                   out_img=cpp_inv_file)