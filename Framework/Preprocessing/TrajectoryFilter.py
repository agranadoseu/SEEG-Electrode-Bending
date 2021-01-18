"""
This class generates a 3D displacement vector of electrode trajectories
Displacement vector is computed locally and globally wrt to planning

Steps:
- Open CT image
- Load electrodes (plan, implementation)
- Create an image with same dimensions as CT but with a vector field
- Save displacement vector and image
- Visualise trajectories and displacement vector in VTK and MRViewer
- Repeat everything with CT image registered to MNI space

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
from Framework.Preprocessing import ElasticRodFilter


class TrajectoryFilter:

    def __init__(self, images=None, electrodes=None):
        self.medicalImages = images
        self.electrodes = electrodes

        self.elastic_rod = ElasticRodFilter.ElasticRodFilter(images=self.medicalImages, electrodes=self.electrodes)

        self.mni_frame = np.eye(3, dtype=np.float)

    def execute(self, space=None):
        self.space = space
        self.plan = self.electrodes.plan[space]
        self.impl = self.electrodes.impl[space]
        self.ghost = self.electrodes.ghost[space]

        # compute displacements
        self.local_delta = self.compute_local_displacement(impl=self.impl)
        self.delta = self.compute_displacement(plan=self.plan, impl=self.impl)

        # compute bending
        self.bending = self.compute_bending(impl=self.impl, ghost=self.ghost)

        # sample images along trajectory
        self.structure = self.sample_structural_image(mri=self.medicalImages.mri_img[self.space], gif=self.medicalImages.gif_img[self.space], impl=self.impl)

        # compute windows
        self.window3 = self.compute_image_window(impl=self.impl, kernel=[3, 3, 3])
        self.window5 = self.compute_image_window(impl=self.impl, kernel=[5, 5, 5])
        self.window9 = self.compute_image_window(impl=self.impl, kernel=[9, 9, 9])
        self.window11 = self.compute_image_window(impl=self.impl, kernel=[11, 11, 11])

        # create image
        self.local_dir_image = self.medicalImages.create_from(space=space, components=3, dtype=sitk.sitkFloat32)
        self.local_delta_image = self.medicalImages.create_from(space=space, components=3, dtype=sitk.sitkFloat32)
        self.delta_image = self.medicalImages.create_from(space=space, components=3, dtype=sitk.sitkFloat32)
        self.compute_local_displacement_images()
        self.compute_displacement_image()

    def execute_infer(self, plan=None, pred=None, ghost=None):
        self.space = 'mni_aff'
        self.plan = plan
        self.impl = pred
        self.ghost = ghost

        # compute displacements
        self.local_delta = self.compute_local_displacement(impl=self.impl)
        self.delta = self.compute_displacement(plan=self.plan, impl=self.impl)

        # compute bending
        self.bending = self.compute_bending(impl=self.impl, ghost=self.ghost)

        # sample images along trajectory
        self.structure = self.sample_structural_image(mri=self.medicalImages.mri_img[self.space],
                                                      gif=self.medicalImages.gif_img[self.space], impl=self.impl)

        # compute windows
        self.window9 = self.compute_image_window(impl=self.impl, kernel=[9, 9, 9])

    def compute_image_window(self, impl=None, kernel=None):
        window = {'name': [], 'id': [], 'mri': [], 'gif': [], 'cwd': []}

        if self.space == 'acpc' or self.space == 'mni_aff' or self.space == 'mni_f3d':
            # open additional images
            directory = self.medicalImages.get_directory(space=self.space)
            space_txt = 'acpc'
            if self.space == 'mni_aff':
                space_txt = 'mni'
            mri_nrm_img = sitk.ReadImage(os.path.join(directory, 'T1-' + space_txt + '-stripped-norm.nii.gz'))
            gif_cwd_img = sitk.ReadImage(os.path.join(directory, 'GIF-' + space_txt + '-cwd.nii.gz'))

            # iterate through electrodes
            for i in range(len(impl['name'])):
                x0 = impl['ep'][i]
                points = impl['points'][i]
                num_points = len(points)

                if num_points == 0:
                    continue

                # print(impl['name'][i])
                mri_w_array = np.zeros((num_points, 2*int(kernel[0]/2)+1, 2*int(kernel[1]/2)+1, 2*int(kernel[2]/2)+1), dtype=np.float)
                gif_w_array = np.zeros((num_points, 2*int(kernel[0]/2)+1, 2*int(kernel[1]/2)+1, 2*int(kernel[2]/2)+1), dtype=np.float)
                cwd_w_array = np.zeros((num_points, 2*int(kernel[0]/2)+1, 2*int(kernel[1]/2)+1, 2*int(kernel[2]/2)+1), dtype=np.float)

                # iterate from EP to TP (list is from TP to EP)
                for p in range(num_points-1, 0, -1):
                    x = points[p]

                    # compute window at point
                    # mri_w_array[p] = self.medicalImages.crop(space=self.space, type='T1', point=x, kernel=kernel)
                    mri_w_array[p] = self.medicalImages.crop_image(image=mri_nrm_img, point=x, kernel=kernel)
                    gif_w_array[p] = self.medicalImages.crop(space=self.space, type='GIF', point=x, kernel=kernel)
                    cwd_w_array[p] = self.medicalImages.crop_image(image=gif_cwd_img, point=x, kernel=kernel)

                    # TODO delete
                    # mri_w_array[p], mri_w_image = self.medicalImages.crop(space=self.space, type='T1', point=x, kernel=kernel)
                    # gif_w_array[p], gif_w_image = self.medicalImages.crop(space=self.space, type='GIF', point=x, kernel=kernel)
                    # mri_filename = os.path.join(self.medicalImages.get_directory(space=self.space), 'mri_' + impl['name'][i] + '_p' + str(p) + '.nii.gz')
                    # gif_filename = os.path.join(self.medicalImages.get_directory(space=self.space), 'gif_' + impl['name'][i] + '_p' + str(p) + '.nii.gz')
                    # sitk.WriteImage(mri_w_image, mri_filename)
                    # sitk.WriteImage(gif_w_image, gif_filename)

                # save
                window['name'].append(impl['name'][i])
                window['id'].append(impl['id'][i])
                window['mri'].append(mri_w_array)
                window['gif'].append(gif_w_array)
                window['cwd'].append(cwd_w_array)

        return window

    def compute_local_displacement(self, impl=None):
        delta = {'name': [], 'id': [], 'du': [], 'u': [], 'from': [], 'to': [], 'dir': [], 'from_dir': [], 'v_dir': []}

        if self.space == 'mni_aff' or self.space == 'mni_f3d':
            # MNI material frame
            xdir = self.mni_frame[:, 0]
            ydir = self.mni_frame[:, 1]
            zdir = self.mni_frame[:, 2]

            # iterate through electrodes
            for i in range(len(impl['name'])):
                x0 = impl['ep'][i]
                points = impl['points'][i]
                num_points = len(points)

                if num_points == 0:
                    continue

                # variables
                du_array = np.zeros(num_points, dtype=np.float64)
                u_array = np.zeros((num_points, 3), dtype=np.float64)
                from_array = np.zeros((num_points, 3), dtype=np.float64)
                to_array = np.zeros((num_points, 3), dtype=np.float64)
                dir_array = np.zeros((num_points, 3), dtype=np.float64)
                from_dir_array = np.zeros((num_points, 3), dtype=np.float64)
                v_dir_array = np.zeros((num_points, 3), dtype=np.float64)

                # iterate from EP to TP (list is from TP to EP)
                for p in range(num_points-1, 0, -1):
                    x1 = points[p]
                    x2 = points[p-1]
                    v_dir = x2 - x1
                    v_dir /= np.linalg.norm(v_dir)  # unit direction of bending
                    dir = x1 - x0
                    dir /= np.linalg.norm(dir)
                    proj = x1 + dir # assuming 1mm interpolation distance
                    u = x2 - proj
                    du = np.linalg.norm(u)
                    u /= du         # local displacement vector as unit vector

                    # save
                    du_array[p] = du
                    u_array[p,:] = u
                    from_array[p, :] = proj
                    to_array[p, :] = x2
                    dir_array[p, :] = dir
                    from_dir_array[p, :] = x1
                    v_dir_array[p, :] = v_dir

                    # move point
                    x0 = x1

                # save
                delta['name'].append(impl['name'][i])
                delta['id'].append(impl['id'][i])
                delta['du'].append(du_array)
                delta['u'].append(u_array)
                delta['from'].append(from_array)
                delta['to'].append(to_array)
                delta['dir'].append(dir_array)
                delta['from_dir'].append(from_dir_array)
                delta['v_dir'].append(v_dir_array)

        return delta

    def compute_displacement(self, plan=None, impl=None):
        # dictionary
        delta = {'name': [], 'id':[], 'du': [], 'u': [], 'from': [], 'to': [], 'dir': []}

        # print('plan[name] = ', plan['name'])
        # print('impl[name] = ', impl['name'])
        for i in range(len(plan['name'])):
            if len(plan['points'][i]) == 0:
                continue

            # displacements
            u = impl['points'][i] - plan['points'][i]
            du = np.linalg.norm(u, axis=1)

            # direction is the same for all points
            # print('name', plan['name'][i], 'ep', plan['ep'][i], 'num_points', len(plan['points'][i]))
            dir = np.zeros_like(u)
            plan_dir = plan['points'][i][0] - plan['ep'][i]
            plan_dir /= np.linalg.norm(plan_dir)
            dir[:,:] = plan_dir

            # save
            delta['name'].append(impl['name'][i])
            delta['id'].append(plan['id'][i])
            delta['du'].append(du)
            delta['u'].append(u)
            delta['from'].append(plan['points'][i])
            delta['to'].append(impl['points'][i])
            delta['dir'].append(dir)

            # print
            # print(plan['name'][i], impl['name'][i], plan['points'][i].shape, impl['points'][i].shape)
            # print(u)
            # print(du)

        return delta

    def compute_displacement_image(self):
        # save displacement vectors
        self.data = sitk.GetArrayFromImage(self.delta_image)
        self.data.fill(0)
        print('compute_displacement_image() of data.shape: ', self.data.shape)

        # compute image indices at physical points
        index_delta = {'index': [], 'delta': [], 'du': []}
        for e in range(len(self.delta['from'])):
            trajectory = self.delta['from'][e]
            # print(self.delta['name'][e], len(trajectory))
            for i in range(len(trajectory)):
                point = trajectory[i]
                index = self.delta_image.TransformPhysicalPointToIndex((point[0], point[1], point[2], 0.0))
                # print('     i={} point={} index={}'.format(i, point, index))

                if index in index_delta['index']:
                    # print('     appending to already existing index ')
                    for j in range(len(index_delta['index'])):
                        if index == index_delta['index'][j]:
                            index_delta['delta'].append(self.delta['u'][e][i])
                            index_delta['du'].append(self.delta['du'][e][i])
                            break
                else:
                    # print('     adding index={} with u={} delta={}'.format(index, self.delta['magnitude'][e][i], self.delta['vector'][e][i]))
                    index_delta['index'].append(index)
                    index_delta['delta'].append(self.delta['u'][e][i])
                    index_delta['du'].append(self.delta['du'][e][i])

        # print('     index_delta', index_delta)

        # update image data (invert axis)
        for i in range(len(index_delta['index'])):
            index = index_delta['index'][i]
            idx_max_value = np.argmax(index_delta['du'][i]).astype(np.uint8)
            self.data[0, index[2], index[1], index[0]] = index_delta['delta'][idx_max_value][0]
            self.data[1, index[2], index[1], index[0]] = index_delta['delta'][idx_max_value][1]
            self.data[2, index[2], index[1], index[0]] = index_delta['delta'][idx_max_value][2]

        # save image
        self.medicalImages.save(image=self.delta_image, data=self.data, name='delta', space=self.space)

    def compute_local_displacement_images(self):
        # save direction and local displacement vectors
        self.data_dir = sitk.GetArrayFromImage(self.local_dir_image)
        self.data_delta = sitk.GetArrayFromImage(self.local_delta_image)
        self.data_dir.fill(0)
        self.data_delta.fill(0)
        print('compute_displacement_image() of data.shape: ', self.data_dir.shape)

        # compute image indices at physical points
        index_delta = {'index': [], 'delta': [], 'du': [], 'dir': []}
        for e in range(len(self.local_delta['from_dir'])):
            trajectory = self.local_delta['from_dir'][e]
            # print(self.delta['name'][e], len(trajectory))
            for i in range(len(trajectory)):
                point = trajectory[i]
                index = self.local_delta_image.TransformPhysicalPointToIndex((point[0], point[1], point[2], 0.0))
                # print('     i={} point={} index={}'.format(i, point, index))

                if index in index_delta['index']:
                    # print('     appending to already existing index ')
                    for j in range(len(index_delta['index'])):
                        if index == index_delta['index'][j]:
                            index_delta['delta'].append(self.local_delta['u'][e][i])
                            index_delta['du'].append(self.local_delta['du'][e][i])
                            index_delta['dir'].append(self.local_delta['dir'][e][i])
                            break
                else:
                    # print('     adding index={} with u={} delta={}'.format(index, self.delta['magnitude'][e][i], self.delta['vector'][e][i]))
                    index_delta['index'].append(index)
                    index_delta['delta'].append(self.local_delta['u'][e][i])
                    index_delta['du'].append(self.local_delta['du'][e][i])
                    index_delta['dir'].append(self.local_delta['dir'][e][i])

        # print('     index_delta', index_delta)

        # update image data (invert axis)
        for i in range(len(index_delta['index'])):
            index = index_delta['index'][i]
            idx_max_value = np.argmax(index_delta['du'][i]).astype(np.uint8)

            # direction (regression: input)
            self.data_dir[0, index[2], index[1], index[0]] = index_delta['dir'][idx_max_value][0]
            self.data_dir[1, index[2], index[1], index[0]] = index_delta['dir'][idx_max_value][1]
            self.data_dir[2, index[2], index[1], index[0]] = index_delta['dir'][idx_max_value][2]

            # displacement vector (regression: label)
            self.data_delta[0, index[2], index[1], index[0]] = index_delta['delta'][idx_max_value][0]
            self.data_delta[1, index[2], index[1], index[0]] = index_delta['delta'][idx_max_value][1]
            self.data_delta[2, index[2], index[1], index[0]] = index_delta['delta'][idx_max_value][2]

        # save images
        self.medicalImages.save(image=self.local_dir_image, data=self.data_dir, name='delta_dir', space=self.space)
        self.medicalImages.save(image=self.local_delta_image, data=self.data_delta, name='delta_local', space=self.space)

    def compute_bending(self, impl=None, ghost=None):
        # dictionary
        bending = {'name': [], 'id': [],
                   'fep': [], 'frames': [],
                   'l_omega_s': [], 'l_omega_v': [], 'g_omega_s': [], 'g_omega_v': [],
                   'velocity': [], 'speed': [], 'tangent': [], 'normal': [], 'acceleration': [], 'sacceleration': [], 'curvature': []}

        # iterate through electrodes
        for i in range(len(impl['name'])):
            if len(impl['points'][i]) < 2:
                continue

            # compute frames
            frames = self.compute_material_frames(ep=impl['ep'][i], gep=ghost['ep'][i], impl=impl['points'][i], ghost=ghost['points'][i])

            # compute omegas
            l_omega_s, l_omega_v, g_omega_s, g_omega_v = self.compute_omega(frames=frames)

            # compute curvature
            v, s, t, n, va, sa, k = self.compute_curvature(impl=impl['points'][i])

            bending['name'].append(impl['name'][i])
            bending['id'].append(impl['id'][i])
            bending['frames'].append(frames)
            bending['l_omega_s'].append(l_omega_s)
            bending['l_omega_v'].append(l_omega_v)
            bending['g_omega_s'].append(g_omega_s)
            bending['g_omega_v'].append(g_omega_v)
            bending['velocity'].append(v)
            bending['speed'].append(s)
            bending['tangent'].append(t)
            bending['normal'].append(n)
            bending['acceleration'].append(va)
            bending['sacceleration'].append(sa)
            bending['curvature'].append(k)

        return bending

    def compute_material_frames(self, ep=None, gep=None, impl=None, ghost=None):
        # material frames have 3 directions (3x3)
        frames = np.zeros((len(impl), 3, 3), dtype=np.float64)

        # points go from TP to EP
        for i in range(len(impl) - 1):
            ci = impl[i+1]
            cj = impl[i]
            gi = ghost[i]

            frames[i] = self.elastic_rod.compute_material_frame([ci, cj, gi])

        # EP frame
        ci = ep
        cj = impl[-1]
        gi = gep
        frames[len(impl) - 1] = self.elastic_rod.compute_material_frame([ci, cj, gi])

        return frames

    def compute_omega(self, frames=None):
        # Compute 3DOF local and global bending
        l_omegas_s = np.zeros(len(frames), dtype=np.float64)
        l_omegas_v = np.zeros((len(frames), 3), dtype=np.float64)
        g_omegas_s = np.zeros(len(frames), dtype=np.float64)
        g_omegas_v = np.zeros((len(frames), 3), dtype=np.float64)

        # frame of EP is last one
        frame_ep = frames[-1]

        # frames go from TP to EP (inclusive)
        # first value is zero
        for i in range(len(frames) - 1):
            # local
            fi = frames[i+1]
            fj = frames[i]
            l_omegas_v[i+1] = self.elastic_rod.compute_omega(fi=fi, fj=fj)
            l_omegas_s[i+1] = np.linalg.norm(l_omegas_v[i+1])

            # global
            fi = frame_ep
            fj = frames[i]
            g_omegas_v[i+1] = self.elastic_rod.compute_omega(fi=fi, fj=fj)
            g_omegas_s[i+1] = np.linalg.norm(g_omegas_v[i+1])

        return l_omegas_s, l_omegas_v, g_omegas_s, g_omegas_v

    def compute_curvature(self, impl=None):
        # input: 3D points
        # output: velocity (nx3), speed (nx1), tangent (nx3), normal (nx3), acceleration (nx3), k (nx1)

        v, s, t, n, a, sa, k = self.elastic_rod.compute_curvature(points=np.flip(impl, 0))

        return np.flip(v,0), np.flip(s,0), np.flip(t,0), np.flip(n,0), np.flip(a,0), np.flip(sa,0), np.flip(k,0)

    def sample_structural_image(self, mri=None, gif=None, impl=None):
        ''' Sample image values along trajectory '''
        sample = {'name': [],
                  'indices': [], 'mri_values': [], 'gif_values': [],
                  'segments': [], 'i_ep': [], 'i_depth': [], 'wm_ratio': []}

        # iterate through electrodes
        for i in range(len(impl['name'])):
            name = impl['name'][i]
            ep = impl['ep'][i]
            points = impl['points'][i]

            # sample values at points
            indicesMri, valuesMri = self.medicalImages.sample_points(image=mri, points=points)
            indicesGif, valuesGif = self.medicalImages.sample_points(image=gif, points=points)
            # print('sample_structure_trajectories:   ep={} points=<{}>  MRI=<{},{}> GIF=<{},{}>'.format(ep, points.shape, indicesMri.shape, valuesMri.shape, indicesGif.shape, valuesGif.shape))

            # print('Compute segments ', name)
            segments, iEP, iDepth, wmRatio = self.medicalImages.compute_segments(image=gif, ep=ep, points=points)

            sample['name'].append(name)
            sample['indices'].append(indicesMri)
            sample['mri_values'].append(valuesMri)
            sample['gif_values'].append(valuesGif)
            sample['segments'].append(segments)
            sample['i_ep'].append(iEP)
            sample['i_depth'].append(iDepth)
            sample['wm_ratio'].append(wmRatio)

        return sample