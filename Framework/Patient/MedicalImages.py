"""
This class handles medical images from a specific patient.
The idea is to have a single class that has access to all image related information

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

# GIF v3
gmGif = np.array([101, 102, 105, 106, 107, 108, 109, 110, 113, 114, 115, 116, 119, 120, 121, 122, 123, 124, 125, 126, 129, 130, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208], dtype=np.int16)
wmGif = np.array([45, 46, 66, 67, 70, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94], dtype=np.int16)
dmGif = np.array([24, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 48, 49, 56, 57, 58, 59, 60, 61, 62, 63, 72, 73, 74, 76, 77, 96, 97, 103, 104, 117, 118, 171, 172, 173, 174], dtype=np.int16)


class MedicalImages:

    def __init__(self, main='', acpc='', mni_aff='', mni_f3d=''):
        # medical images
        self.ct_img = {'patient': None, 'acpc': None, 'mni_aff': None, 'mni_f3d': None}
        self.mri_img = {'patient': None, 'acpc': None, 'mni_aff': None, 'mni_f3d': None}
        self.gif_img = {'patient': None, 'acpc': None, 'mni_aff': None, 'mni_f3d': None}

        # sub folders
        self.dir = main
        self.acpc = acpc
        self.mni_aff = mni_aff
        self.mni_f3d = mni_f3d

    def get_directory(self, space=None):
        # set directory depending on space
        directory = None
        if space == 'patient':
            directory = self.dir
        elif space == 'acpc':
            directory = os.path.join(self.dir, self.acpc)
        elif space == 'mni_aff':
            directory = os.path.join(self.dir, self.mni_aff)
        elif space == 'mni_f3d':
            directory = os.path.join(self.dir, self.mni_f3d)
        return directory

    def open(self, type=None, space=None, stripped=False):
        """
        Opens a medical image
        :param type: type of image to load ['CT', 'T1', 'GIF']
        :param space: space of image ['patient', 'acpc', 'mni_aff', 'mni_f3d']
        :return: sitk object and image properties
        """
        # set directory depending on space
        directory = self.get_directory(space=space)

        # set filename
        filename = type
        if space == 'acpc':
            filename += '-acpc'
        elif space == 'mni_aff':
            # if type == 'T1':
            if stripped:
                filename += '-mni-stripped-norm'  # skull-stripped image
            else:
                filename += '-mni'    # full image
        elif space == 'mni_f3d':
            filename += '-mni-stripped-f3d'

        # open file
        filename += '.nii.gz'
        name_pos = filename.find('.')
        name = filename[0:name_pos]
        image = sitk.ReadImage(os.path.join(directory, filename))
        props = self.get_properties(image=image, name=name)

        # save reference
        if type == 'CT':
            self.ct_img[space] = image
        elif type == 'T1':
            self.mri_img[space] = image
        elif type == 'GIF':
            self.gif_img[space] = image

        return image, props

    def get_properties(self, image=None, name=None):
        """
        Get image properties from a file
        :param image: sitk image
        :param name: string of name
        :return: distionary with image properties
        """
        properties = {'name': name, 'origin': image.GetOrigin(), 'size': image.GetSize(), 'spacing': image.GetSpacing(),
                      'direction': image.GetDirection(), 'type': image.GetPixelIDTypeAsString(), 'components': image.GetNumberOfComponentsPerPixel()}

        # print('SITK image', name)
        # print('   origin: ', image.GetOrigin())
        # print('   size: ', image.GetSize())
        # print('   spacing: ', image.GetSpacing())
        # print('   direction: ', image.GetDirection())
        # print('   pixel type: ', image.GetPixelIDTypeAsString())
        # print('   pixel components: ', image.GetNumberOfComponentsPerPixel())

        return properties

    def copy(self, src_image=None):
        sData = sitk.GetArrayFromImage(src_image)
        sOrigin = src_image.GetOrigin()
        sSpacing = src_image.GetSpacing()
        sDirection = src_image.GetDirection()

        dst_img = sitk.GetImageFromArray(np.copy(sData))
        dst_img.SetOrigin(sOrigin)
        dst_img.SetSpacing(sSpacing)
        dst_img.SetDirection(sDirection)
        return dst_img

    def crop(self, space=None, type=None, point=None, kernel=None):
        w_array = np.zeros((2*int(kernel[0]/2)+1, 2*int(kernel[1]/2)+1, 2*int(kernel[2]/2)+1), dtype=np.int16)
        knl_idx = [int(kernel[0]/2), int(kernel[1]/2), int(kernel[2]/2)]
        # print('knl_idx={}'.format(knl_idx))

        # which image
        image = self.mri_img[space]
        if type == 'GIF':
            image = self.gif_img[space]

        # get index at position
        img_idx = image.TransformPhysicalPointToIndex((point[0], point[1], point[2]))
        # print('point={} img_idx={} img_val={}'.format(point, img_idx, image[int(img_idx[0]), int(img_idx[1]), int(img_idx[2])]))

        # fill window with neighbouring values
        for k in range(-knl_idx[2], knl_idx[2]+1, 1):
            for j in range(-knl_idx[1], knl_idx[1]+1, 1):
                for i in range(-knl_idx[0], knl_idx[0]+1, 1):
                    # print('ijk=[{},{},{}] idx=[{},{},{}] val={}'.format(i,j,k, img_idx[0]+i, img_idx[1]+j, img_idx[2]+k,
                    #                                                     image[int(img_idx[0]+i), int(img_idx[1]+j), int(img_idx[2]+k)]))
                    w_array[knl_idx[0]+i, knl_idx[1]+j, knl_idx[2]+k] = image[int(img_idx[0]+i), int(img_idx[1]+j), int(img_idx[2]+k)]
        # print('w_array: shape={}\n{}'.format(w_array.shape, w_array))

        # TODO image (delete)
        # print(self.get_properties(image=image, name='src_image'))
        # crop_img = self.copy(src_image=image)
        # print(self.get_properties(image=crop_img, name='crop_img'))
        # crop_img = crop_img[int(img_idx[0]-knl_idx[0]):int(img_idx[0]+knl_idx[0]), int(img_idx[1]-knl_idx[1]):int(img_idx[1]+knl_idx[1]), int(img_idx[2]-knl_idx[2]):int(img_idx[2]+knl_idx[2])]
        # return w_array, crop_img
        return w_array

    def crop_image(self, image=None, point=None, kernel=None):
        w_array = np.zeros((2*int(kernel[0]/2)+1, 2*int(kernel[1]/2)+1, 2*int(kernel[2]/2)+1), dtype=np.float)
        knl_idx = [int(kernel[0]/2), int(kernel[1]/2), int(kernel[2]/2)]

        # get index at position
        img_idx = image.TransformPhysicalPointToIndex((point[0], point[1], point[2]))
        # print('point={} img_idx={} img_val={}'.format(point, img_idx, image[int(img_idx[0]), int(img_idx[1]), int(img_idx[2])]))

        # fill window with neighbouring values
        for k in range(-knl_idx[2], knl_idx[2]+1, 1):
            for j in range(-knl_idx[1], knl_idx[1]+1, 1):
                for i in range(-knl_idx[0], knl_idx[0]+1, 1):
                    w_array[knl_idx[0]+i, knl_idx[1]+j, knl_idx[2]+k] = image[int(img_idx[0]+i), int(img_idx[1]+j), int(img_idx[2]+k)]

        # print('w_array: shape={}\n{}'.format(w_array.shape, w_array))
        return w_array

    def create_from(self, space=None, components=None, dtype=None):
        # https://simpleitk-prototype.readthedocs.io/en/latest/user_guide/plot_image.html
        # get image (by default copy from MRI image)
        image = self.mri_img[space]
        origin = image.GetOrigin()
        size = image.GetSize()
        spacing = image.GetSpacing()
        direction = image.GetDirection()

        # data type
        if dtype is None:
            dtype = image.GetPixelID()

        # components
        if components is None:
            components = image.GetNumberOfComponentsPerPixel()

        if components == 1:
            dim = len(size)
            direction_dim = np.asarray(direction).reshape(dim, dim)
            # direction3d = np.eye(3)
            direction3d = direction_dim[0:3, 0:3]
            direction = tuple(direction3d.flatten())
            new_image = sitk.Image([size[0], size[1], size[2]], dtype, components)
            new_image.SetOrigin((origin[0], origin[1], origin[2]))
            new_image.SetSpacing([spacing[0], spacing[1], spacing[2]])
            new_image.SetDirection(direction)
        else:
            # dtype += 12     # vector type
            dim = len(size)
            direction_dim = np.asarray(direction).reshape(dim, dim)
            direction4d = np.eye(4)
            direction4d[0:3, 0:3] = direction_dim[0:3, 0:3]
            direction = tuple(direction4d.flatten())
            new_image = sitk.Image([size[0], size[1], size[2], components], dtype)
            new_image.SetOrigin((origin[0], origin[1], origin[2], 0.0))
            new_image.SetSpacing([spacing[0], spacing[1], spacing[2], 1.0])
            new_image.SetDirection(direction)

        props = self.get_properties(image=new_image, name='new')

        return new_image

    def save(self, image=None, data=None, name=None, space=None):
        directory = None
        if space == 'patient':
            directory = self.dir
        elif space == 'acpc':
            directory = os.path.join(self.dir, self.acpc)
        elif space == 'mni_aff':
            directory = os.path.join(self.dir, self.mni_aff)
        elif space == 'mni_f3d':
            directory = os.path.join(self.dir, self.mni_f3d)

        if data is not None:
            new_image = None
            if len(data.shape) == 3:
                new_image = sitk.GetImageFromArray(data)
            elif len(data.shape) == 4:
                new_image = sitk.GetImageFromArray(data, isVector=False)
            origin = image.GetOrigin()
            spacing = image.GetSpacing()
            direction = image.GetDirection()

            new_image.SetOrigin(origin)
            new_image.SetSpacing(spacing)
            new_image.SetDirection(direction)

            self.get_properties(image=new_image, name=name)

            sitk.WriteImage(new_image, os.path.join(directory, name+'.nii.gz'))
        else:
            sitk.WriteImage(image, os.path.join(directory, name+'.nii.gz'))

    def sample_points(self, image=None, points=None):
        ''' Sample values from image given a set of points '''
        n = len(points)
        indices = np.zeros((n, 3), dtype=np.float64)
        values = np.zeros(n, dtype=np.float64)
        dim = image.GetSize()

        for i in range(n):
            # p = np.array([-_points[i][0], -_points[i][1], _points[i][2]])
            p = np.array([points[i][0], points[i][1], points[i][2]])
            idx = image.TransformPhysicalPointToIndex(p)
            # print('dim={} idx={}'.format(dim, idx))
            # val = _img[int(idx[2]), int(idx[1]), int(idx[0])]

            # validate index is within dimensions
            val = -1.0
            if idx[0] < dim[0] and idx[1] < dim[1] and idx[2] < dim[2] and idx[0] >= 0 and idx[1] >= 0 and idx[2] >= 0:
                val = image[int(idx[0]), int(idx[1]), int(idx[2])]

            # print('c{}: p[{}], phy=[{}], idx=<{}>, val={}'.format(i, _points[i], p, idx, val))

            indices[i] = idx
            values[i] = val

        return indices, values

    def compute_segments(self, image=None, ep=None, points=None):
        """
        :return segments: [id, id+1, t, region, s_fraction, s_length, acc_depth, acc_wmlength]
        """
        # output:
        S = []  # from TP indicating where region finishes
        T = 10  # query times between points

        points = np.append(points, [[ep[0], ep[1], ep[2]]], axis=0)

        # iterate through points
        u = [0., 0., 0.]
        prevGif = 0
        d2prevGif = 0.0
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            u = p2 - p1

            # from most proximal contact to EP, increase T
            if (i + 1) == (len(points) - 1):
                T = 100

            # traverse between two points
            sample = []
            for dt in range(T):
                t = dt / T
                p = p1 + t * u
                sample.append([p[0], p[1], p[2]])
            sample = np.asarray(sample, dtype=np.float64)
            # print('sample[{},{}]={} d={}'.format(i, i+1, sample, np.linalg.norm(u)))

            # query
            indicesGif, valuesGif = self.sample_points(image=image, points=sample)

            # init for TP
            startT = 0
            if i == 0:
                prevGif = valuesGif[0]
                startT = 1

            for dt in range(startT, T, 1):
                t = dt / T
                d2prevGif += round(1 / T, 2)
                # print('t={} prevGif={} valuesGif[{}]={} d2prevGif={}'.format(t, prevGif, dt, valuesGif[dt], d2prevGif))
                if valuesGif[dt] != prevGif:
                    S.append([i, i + 1, t, prevGif, round(d2prevGif, 2), 0., 0., 0.])
                    prevGif = valuesGif[dt]
                    d2prevGif = 0.0

        # add last segment up to EP
        d2prevGif += round(1 / T, 2)
        S.append([i, i + 1, 1, prevGif, round(d2prevGif, 2), 0., 0., 0.])

        # compute segments length from TP
        start_i = 0
        prev_j = 0
        prev_t = 0
        for i in range(len(S)):
            seg = S[i]
            s_i = int(seg[0])
            s_j = int(seg[1])
            s_t = seg[2]

            # add distance of current segment
            s_l = 0.0
            if prev_j != s_j:
                s_l = np.linalg.norm(points[s_j] - points[s_i]) * s_t
                prev_j = s_j
                prev_t = s_t
            else:
                s_l = np.linalg.norm(points[s_j] - points[s_i]) * (s_t - prev_t)
                prev_t = s_t

            # add distances since start of region
            for j in range(start_i, s_i, 1):
                s_l += np.linalg.norm(points[j] - points[j + 1])

            # save
            S[i][5] = round(s_l, 2)
            start_i = s_j

        # detect intracerebral EP
        brainEP = [0., 0., 0.]
        iEP = -1
        for i in range(len(S) - 1, -1, -1):
            seg = S[i]
            if seg[3] != 0 and seg[3] != 1 and seg[3] != 2 and seg[3] != 3 and seg[3] != 4:
                p1 = points[int(seg[0])]
                p2 = points[int(seg[1])]
                t = seg[2]
                u = p2 - p1
                p = p1 + t * u
                brainEP = [p[0], p[1], p[2]]
                iEP = i
                break

        # intracerebral
        iDepth = 0.0
        wmLength = 0.0
        wmRatio = 0.0
        if iEP != -1:
            for i in range(iEP, -1, -1):
                seg = S[i]

                # compute intracerebral depth
                iDepth += seg[5]

                # compute wmRatio
                if seg[3] in wmGif:
                    wmLength += seg[5]

                # how much WM has the electrode has traversed so far
                S[i][6] = round(iDepth, 2)
                S[i][7] = round(wmLength, 2)

        # overall wm ratio
        if iDepth != 0.0:
            wmRatio = wmLength / iDepth

        # print('S = ', S)
        # print('iEP={} brainEP={} depth={} wmLength={} wmRatio={}'.format(iEP, brainEP, iDepth, wmLength, wmRatio))
        return S, iEP, iDepth, wmRatio