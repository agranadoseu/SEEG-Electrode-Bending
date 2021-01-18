"""
This class processes medical images

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

from Framework.Patient import MedicalImages


class ImageProcessingFilter:

    def __init__(self, images=None):
        self.medicalImages = images

    def execute(self):
        # files
        t1_acpc_norm_filename = os.path.join(self.medicalImages.get_directory(space='acpc'), 'T1-acpc-norm.nii.gz')
        t1_mni_norm_filename = os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'T1-mni-norm.nii.gz')
        t1_mni_stripped_norm_filename = os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'T1-mni-stripped-norm.nii.gz')
        t1_mni_stripped_bc_filename = os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'T1-mni-stripped-bc.nii.gz')
        gif_acpc_cwd_filename = os.path.join(self.medicalImages.get_directory(space='acpc'), 'GIF-acpc-cwd.nii.gz')
        gif_mni_cwd_filename = os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'GIF-mni-cwd.nii.gz')

        # normalise
        if not os.path.exists(t1_acpc_norm_filename):
            t1_acpc, t1_acpc_props = self.medicalImages.open(type='T1', space='acpc')
            t1_acpc_norm = self.normalise(image=t1_acpc)
            sitk.WriteImage(t1_acpc_norm, t1_acpc_norm_filename)

        if not os.path.exists(t1_mni_norm_filename):
            t1_mni, t1_mni_props = self.medicalImages.open(type='T1', space='mni_aff')
            t1_mni_norm = self.normalise(image=t1_mni)
            sitk.WriteImage(t1_mni_norm, t1_mni_norm_filename)

        if not os.path.exists(t1_mni_stripped_norm_filename):
            t1_mni_stripped, t1_mni_stripped_props = self.medicalImages.open(type='T1', space='mni_aff', stripped=True)
            t1_mni_stripped_norm = self.normalise(image=t1_mni_stripped)
            # rescale, lower, upper, upper_binary = self.cut_threshold(image=t1_mni_stripped, value=0.5)
            # t1_mni_stripped_norm = self.normalise(image=lower)

            # t1_mni_stripped_rs_filename = os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'T1-mni-stripped-rs.nii.gz')
            # sitk.WriteImage(rescale, t1_mni_stripped_rs_filename)
            # t1_mni_stripped_lower_filename = os.path.join(self.medicalImages.get_directory(space='mni_aff'), 'T1-mni-stripped-lower.nii.gz')
            # sitk.WriteImage(lower, t1_mni_stripped_lower_filename)
            sitk.WriteImage(t1_mni_stripped_norm, t1_mni_stripped_norm_filename)
            # sitk.WriteImage(upper_binary, t1_mni_stripped_bc_filename)

        # create CWD
        if not os.path.exists(gif_acpc_cwd_filename):
            gif_acpc, gif_acpc_props = self.medicalImages.open(type='GIF', space='acpc')
            gif_acpc_cwd = self.create_cwd(image=gif_acpc)
            sitk.WriteImage(gif_acpc_cwd, gif_acpc_cwd_filename)

        if not os.path.exists(gif_mni_cwd_filename):
            gif_mni, gif_mni_props = self.medicalImages.open(type='GIF', space='mni_aff')
            gif_mni_cwd = self.create_cwd(image=gif_mni)
            sitk.WriteImage(gif_mni_cwd, gif_mni_cwd_filename)

    def normalise(self, image=None):
        # normalise image
        rescaleIntensityFilter = sitk.RescaleIntensityImageFilter()
        rescaleIntensityFilter.SetOutputMinimum(0.0)
        rescaleIntensityFilter.SetOutputMaximum(1.0)
        rescale_image = rescaleIntensityFilter.Execute(sitk.Cast(image, sitk.sitkFloat32))

        return rescale_image

    def cut_threshold(self, image=None, value=0.5):
        # normalise image
        rescaleIntensityFilter = sitk.RescaleIntensityImageFilter()
        rescaleIntensityFilter.SetOutputMinimum(0.0)
        rescaleIntensityFilter.SetOutputMaximum(1.0)
        rescale_image = rescaleIntensityFilter.Execute(sitk.Cast(image, sitk.sitkFloat32))

        # percentile of non-zero values
        # image_data = sitk.GetArrayFromImage(rescale_image)
        # threshold = np.percentile(image_data[image_data != 0], quantile)
        threshold = value

        # upper image threshold
        upperThresholdImageFilter = sitk.ThresholdImageFilter()
        upperThresholdImageFilter.SetLower(threshold)
        upperThresholdImageFilter.SetUpper(1.0)
        upperThresholdImageFilter.SetOutsideValue(0)
        upper_threshold_image = upperThresholdImageFilter.Execute(rescale_image)

        # lower image threshold
        lowerThresholdImageFilter = sitk.ThresholdImageFilter()
        lowerThresholdImageFilter.SetLower(0.01)
        lowerThresholdImageFilter.SetUpper(threshold)
        lowerThresholdImageFilter.SetOutsideValue(0)
        lower_threshold_image = lowerThresholdImageFilter.Execute(rescale_image)

        # binary threshold
        upperBinaryThresholdFilter = sitk.BinaryThresholdImageFilter()
        upperBinaryThresholdFilter.SetLowerThreshold(threshold)
        upperBinaryThresholdFilter.SetUpperThreshold(1.0)
        upperBinaryThresholdFilter.SetInsideValue(1)
        upperBinaryThresholdFilter.SetOutsideValue(0)
        upper_binary_image = upperBinaryThresholdFilter.Execute(rescale_image)
        upper_binary_image = sitk.Cast(upper_binary_image, image.GetPixelID())

        return rescale_image, lower_threshold_image, upper_threshold_image, upper_binary_image

    def create_cwd(self, image=None):
        # copy from GIF
        sData = sitk.GetArrayFromImage(image)
        sOrigin = image.GetOrigin()
        sSpacing = image.GetSpacing()
        sDirection = image.GetDirection()

        # convert regions to cwd
        cwd_data = np.zeros_like(sData)
        for i in list(np.unique(sData)):
            cwd = 0
            if np.isin(i, MedicalImages.gmGif):
                cwd = 1
            elif np.isin(i, MedicalImages.wmGif):
                cwd = 2
            elif np.isin(i, MedicalImages.dmGif):
                cwd = 3
            cwd_data[sData == i] = cwd

        # create output image
        cwd_img = sitk.GetImageFromArray(cwd_data)
        cwd_img.SetOrigin(sOrigin)
        cwd_img.SetSpacing(sSpacing)
        cwd_img.SetDirection(sDirection)

        return cwd_img
