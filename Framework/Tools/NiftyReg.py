"""
This class handles all NiftyReg related functions

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


class NiftyReg:

    def __init__(self):
        return

    def register_affine(self, ref_img='', flo_img='', res_img='', trans_matrix=''):
        # reg_aladin.exe -ref "icCT.nii" -flo "MNI_152_mri.nii.gz" -res "mni-t1.nii" -aff "reg.txt -interp 3"

        command = 'C:\\UCL\\EpiNav\\CUDA\\NiftyReg\\install\\bin\\reg_aladin.exe'
        params10, params11 = '-ref', ref_img
        params20, params21 = '-flo', flo_img
        params30, params31 = '-res', res_img
        params40, params41 = '-aff', trans_matrix
        params50, params51 = '-interp', '3'

        print('Executing:\n{} {} {} {} {} {} {} {} {} {} {}'.format(command, params10, params11, params20, params21,
                                                                    params30, params31, params40, params41, params50,
                                                                    params51))
        subprocess.check_call(
            ['cmd.exe', '/c', command, params10, params11, params20, params21, params30, params31, params40, params41,
             params50, params51], shell=True)

    def resample(self, ref_img='', flo_img='', res_img='', trans_matrix='', inter=3):
        # reg_resample.exe -ref "mni.nii" -flo "T1.nii.gz" -res "T1-mni.nii" -trans "inv_reg.txt -inter 3"

        command = 'C:\\UCL\\EpiNav\\CUDA\\NiftyReg\\install\\bin\\reg_resample.exe'
        params10, params11 = '-ref', ref_img
        params20, params21 = '-flo', flo_img
        params30, params31 = '-res', res_img
        params40, params41 = '-trans', trans_matrix
        params50, params51 = '-inter', str(inter)

        print('Executing:\n{} {} {} {} {} {} {} {} {} {} {}'.format(command, params10, params11, params20, params21,
                                                                    params30, params31, params40, params41, params50,
                                                                    params51))
        subprocess.check_call(
            ['cmd.exe', '/c', command, params10, params11, params20, params21, params30, params31, params40, params41,
             params50, params51], shell=True)

    def invert_transform_matrix(self, trans_matrix='', inv_matrix=''):
        # reg_transform.exe -invAff ./R01/mniE/T1-mni.txt ./R01/mniE/T1-mni-inv.txt

        command = 'C:\\UCL\\EpiNav\\CUDA\\NiftyReg\\install\\bin\\reg_transform.exe'
        params10 = '-invAff'
        params20, params21 = trans_matrix, inv_matrix

        print('Executing:\n{} {} {} {}'.format(command, params10, params20, params21))
        subprocess.check_call(['cmd.exe', '/c', command, params10, params20, params21], shell=True)

    def transform_points(self, ref_img='', trans_matrix='', in_points='', out_points=''):
        # reg_transform.exe -land ./R01/mniE/T1-mni.txt ./R01/mniE/E1i.txt ./R01/mniE/out_E1i.txt -ref ./R01/T1.nii.gz

        command = 'C:\\UCL\\EpiNav\\CUDA\\NiftyReg\\install\\bin\\reg_transform.exe'
        params10, params11 = '-land', trans_matrix
        params20, params21 = in_points, out_points
        params30, params31 = '-ref', ref_img

        print('Executing:\n{} {} {} {} {} {} {}'.format(command, params10, params11, params20, params21, params30, params31))
        subprocess.check_call(['cmd.exe', '/c', command, params10, params11, params20, params21, params30, params31], shell=True)

    def register_f3d(self, ref_img='', flo_img='', cpp_img='', res_img=''):
        # reg_f3d.exe -ref ../template/MNI_152_mri.nii.gz -flo ./R01/T1-mni.nii.gz -cpp ./R01/f3dE/T1-mni-cpp.nii.gz -res ./R01/f3dE/T1-mni-f3d.nii.gz

        command = 'C:\\UCL\\EpiNav\\CUDA\\NiftyReg\\install\\bin\\reg_f3d.exe'
        params10, params11 = '-ref', ref_img
        params20, params21 = '-flo', flo_img
        params30, params31 = '-cpp', cpp_img
        params40, params41 = '-res', res_img

        print('Executing:\n{} {} {} {} {} {} {} {} {}'.format(command, params10, params11, params20, params21, params30,
                                                              params31, params40, params41))
        subprocess.check_call(
            ['cmd.exe', '/c', command, params10, params11, params20, params21, params30, params31, params40, params41],
            shell=True)

    def invert_transform_cpp(self, ref_img='', flo_img='', cpp_img='', out_img=''):
        # reg_transform.exe -ref ./R01/mniE/T1-mni-stripped.nii.gz -invNrr ./R01/f3dE/T1-mni-stripped-cpp.nii.gz ./R01/mniE/T1-mni-stripped.nii.gz ./R01/f3dE/T1-mni-stripped-cpp-inv.nii.gz

        command = 'C:\\UCL\\EpiNav\\CUDA\\NiftyReg\\install\\bin\\reg_transform.exe'
        params10, params11 = '-ref', ref_img
        params20 = '-invNrr'
        params30, params31, params32 = cpp_img, flo_img, out_img

        print('Executing:\n{} {} {} {} {} {} {}'.format(command, params10, params11, params20, params30, params31, params32))
        subprocess.check_call(['cmd.exe', '/c', command, params10, params11, params20, params30, params31, params32], shell=True)
