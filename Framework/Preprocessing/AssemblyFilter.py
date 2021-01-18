"""
This class assemblies features of electrode trajectories

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
import pandas as pd

from Framework.Patient import MedicalImages
from Framework.Preprocessing import SurfaceCollisionFilter


class AssemblyFilter:

    def __init__(self, images=None, electrodes=None, space=None, features=None):
        self.medicalImages = images
        self.electrodes = electrodes
        self.space = space
        self.features = features
        self.directory = self.medicalImages.get_directory(space=self.space)

        self.surface_collision = SurfaceCollisionFilter.SurfaceCollisionFilter(images=self.medicalImages, electrodes=self.electrodes, space=self.space)

    def execute(self, case=None, plan=None, impl=None, depth=None):
        print('Assembly')
        general_df = pd.DataFrame()
        bending_df = pd.DataFrame()
        curvature_df = pd.DataFrame()
        displacement_df = pd.DataFrame()
        structural_df = pd.DataFrame()
        parcellation_df = pd.DataFrame()
        collision_df = pd.DataFrame()

        # find index of this case
        index = self.features['case'].index(case)

        # iterate through electrodes
        # plan = self.electrodes.plan[self.space]
        # impl = self.electrodes.impl[self.space]
        # print('plan = {} impl = {}'.format(plan))
        for i in range(len(impl['name'])):
            # general
            # case = impl['case'][i]
            name = impl['name'][i]
            stylet = impl['stylet'][i]
            points = impl['points'][i]
            plan_points = plan['points'][i]
            num_contacts = impl['num_contacts'][i]
            if depth is None:
                depth = len(plan_points)
            print('name = {}'.format(name))
            print('num_contacts = {}'.format(num_contacts))
            print('stylet = {}'.format(stylet))
            print('points[{}] = {}'.format(len(points), points))
            print('plan[{}] = {}'.format(len(plan_points), plan_points))

            general_features = self.assemble_general_features(case=case, name=name, points=points, plan=plan_points,
                                                              num_contacts=num_contacts, stylet=stylet, depth=depth)
            general_df = general_df.append(general_features, ignore_index=True)
            print('general_df', general_df)

            # bending
            frames = self.features['bending'][index]['frames'][i]
            l_omega_s = self.features['bending'][index]['l_omega_s'][i]
            l_omega_v = self.features['bending'][index]['l_omega_v'][i]
            g_omega_s = self.features['bending'][index]['g_omega_s'][i]
            g_omega_v = self.features['bending'][index]['g_omega_v'][i]
            print('frames[{}] = {}'.format(len(frames), frames))
            print('l_omega_s[{}] = {}'.format(len(l_omega_s), l_omega_s))
            print('l_omega_v[{}] = {}'.format(len(l_omega_v), l_omega_v))
            print('g_omega_s[{}] = {}'.format(len(g_omega_s), g_omega_s))
            print('g_omega_v[{}] = {}'.format(len(g_omega_v), g_omega_v))

            bending_features = self.assemble_bending_features(points=points, frames=frames,
                                                              los=l_omega_s, lov=l_omega_v, gos=g_omega_s, gov=g_omega_v)
            bending_df = bending_df.append(bending_features, ignore_index=True)
            print('bending_df', bending_df)

            # curvature
            curvature = self.features['bending'][index]['curvature'][i]
            velocity = self.features['bending'][index]['velocity'][i]
            speed = self.features['bending'][index]['speed'][i]
            acceleration = self.features['bending'][index]['acceleration'][i]
            sacceleration = self.features['bending'][index]['sacceleration'][i]
            print('curvature[{}] = {}'.format(len(curvature), curvature))
            print('velocity[{}] = {}'.format(len(velocity), velocity))
            print('speed[{}] = {}'.format(len(speed), speed))
            print('acceleration[{}] = {}'.format(len(acceleration), acceleration))
            print('sacceleration[{}] = {}'.format(len(sacceleration), sacceleration))

            curvature_features = self.assemble_curvature_features(points=points,
                                                                  curvature=curvature,
                                                                  velocity=velocity, speed=speed,
                                                                  acceleration=acceleration, sacceleration=sacceleration)
            curvature_df = curvature_df.append(curvature_features, ignore_index=True)
            print('curvature_df', curvature_df)

            # displacements
            lfrom = self.features['local_delta'][index]['from'][i]
            ldir = self.features['local_delta'][index]['dir'][i]
            lu_s = self.features['local_delta'][index]['du'][i]
            lu_v = self.features['local_delta'][index]['u'][i]
            gu_s = self.features['delta'][index]['du'][i]
            gu_v = self.features['delta'][index]['u'][i]
            print('lfrom[{}] = {}'.format(len(lfrom), lfrom))
            print('ldir[{}] = {}'.format(len(ldir), ldir))
            print('lu_s[{}] = {}'.format(len(lu_s), lu_s))
            print('lu_v[{}] = {}'.format(len(lu_v), lu_v))
            print('gu_s[{}] = {}'.format(len(gu_s), gu_s))
            print('gu_v[{}] = {}'.format(len(gu_v), gu_v))

            displacement_features = self.assemble_displacement_features(points=points,
                                                                        lproj=lfrom, ldir=ldir,
                                                                        lu_s=lu_s, lu_v=lu_v,
                                                                        gu_s=gu_s, gu_v=gu_v)
            displacement_df = displacement_df.append(displacement_features, ignore_index=True)
            print('displacement_df', displacement_df)

            # structure
            indices = self.features['structure'][index]['indices'][i]
            intensities = self.features['structure'][index]['mri_values'][i]
            # intracranial_ep = self.features['structure'][index]['i_ep'][i]
            intracranial_depth = self.features['structure'][index]['i_depth'][i]
            print('indices[{}] = {}'.format(len(indices), indices))
            print('intensities[{}] = {}'.format(len(intensities), intensities))
            # print('intracranial_ep = {}'.format(intracranial_ep))
            print('intracranial_depth = {}'.format(intracranial_depth))

            structural_features = self.assemble_structural_features(points=points,
                                                                    indices=indices, intensities=intensities,
                                                                    i_depth=intracranial_depth)
            structural_df = structural_df.append(structural_features, ignore_index=True)
            print('structural_df', structural_df)

            # parcellation
            regions = self.features['structure'][index]['gif_values'][i]
            segments = self.features['structure'][index]['segments'][i]
            wm_ratio = self.features['structure'][index]['wm_ratio'][i]
            print('regions[{}] = {}'.format(len(regions), regions))
            print('segments[{}] = {}'.format(len(segments), segments))
            print('wm_ratio = {}'.format(wm_ratio))

            parcellation_features = self.assemble_parcellation_features(points=points,
                                                                        regions=regions, segments=segments,
                                                                        wm_ratio=wm_ratio)
            parcellation_df = parcellation_df.append(parcellation_features, ignore_index=True)
            print('parcellation_df', parcellation_df)

            # collision
            acpcih = self.features['collision'][index]['acpcih'][i]
            col_scalp = self.features['collision'][index]['scalp'][i]
            col_cortex = self.features['collision'][index]['cortex'][i]
            col_white = self.features['collision'][index]['white'][i]
            col_deep = self.features['collision'][index]['deep'][i]
            print('acpcih[{}] = {}'.format(len(acpcih), acpcih))
            print('col_scalp[{}] = {}'.format(len(col_scalp), col_scalp))
            print('col_cortex[{}] = {}'.format(len(col_cortex), col_cortex))
            print('col_white[{}] = {}'.format(len(col_white), col_white))
            print('col_deep[{}] = {}'.format(len(col_deep), col_deep))

            collision_features = self.assemble_collision_features(points=points, acpcih=acpcih,
                                                                  col_scalp=col_scalp, col_cortex=col_cortex, col_white=col_white, col_deep=col_deep)
            collision_df = collision_df.append(collision_features, ignore_index=True)
            print('collision_df', collision_df)

            # assembly
            all_df = general_df
            all_df = all_df.join(bending_df)
            all_df = all_df.join(curvature_df)
            all_df = all_df.join(displacement_df)
            all_df = all_df.join(structural_df)
            all_df = all_df.join(parcellation_df)
            all_df = all_df.join(collision_df)

        return all_df

    def save(self, df=None):
        # save
        df.to_csv(os.path.join(self.directory, 'features.csv'), header=True, index=False)
        df.to_pickle(os.path.join(self.directory, 'features.pkl'))

    def assemble_general_features(self, case=None, name=None, points=None, plan=None, num_contacts=None, stylet=None, depth=None):
        features = []

        # account for assembling data during inference
        if stylet != -1.0 and len(points) < depth:
            if len(points) > (depth-stylet):
                stylet = len(points) - (depth-stylet)

        for i in range(len(points)):
            # stylet
            stylet_val = -1.0
            if stylet != -1.0:
                if stylet > i:
                    stylet_val = 0.0
                else:
                    stylet_val = 1.0

            features.append([case, name, num_contacts, i,
                             stylet_val,
                             points[i][0], points[i][1], points[i][2],
                             plan[i][0], plan[i][1], plan[i][2]])

        # create dataframe
        df = pd.DataFrame(features,
                          columns=['case', 'electrode', 'num_contacts', 'interpolation',
                                   'stylet',
                                   'x', 'y', 'z',
                                   'plan_x', 'plan_y', 'plan_z'])
        df = df.sort_index(ascending=0)

        return df

    def assemble_bending_features(self, points=None, frames=None,
                                  los=None, lov=None, gos=None, gov=None):
        features = []
        for i in range(len(points)):
            frame_ep = frames[-1]
            bolt_dir = frame_ep[:, 2]
            elec_dir = frames[i][:, 2]

            features.append([bolt_dir[0], bolt_dir[1], bolt_dir[2],
                             elec_dir[0], elec_dir[1], elec_dir[2],
                             los[i],
                             lov[i][0], lov[i][1], lov[i][2],
                             gos[i],
                             gov[i][0], gov[i][1], gov[i][2]])

            # create dataframe
            df = pd.DataFrame(features,
                              columns=['bolt_dir_x', 'bolt_dir_y', 'bolt_dir_z',
                                       'elec_dir_x', 'elec_dir_y', 'elec_dir_z',
                                       'l_omega',
                                       'l_omega_x', 'l_omega_y', 'l_omega_z',
                                       'g_omega',
                                       'g_omega_x', 'g_omega_y', 'g_omega_z'])
            df = df.sort_index(ascending=0)

        return df

    def assemble_curvature_features(self, points=None,
                                    curvature=None,
                                    velocity=None, speed=None,
                                    acceleration=None, sacceleration=None):
        features = []
        for i in range(len(points)):
            features.append([curvature[i],
                             speed[i], velocity[i][0], velocity[i][1], velocity[i][2],
                             sacceleration[i], acceleration[i][0], acceleration[i][1], acceleration[i][2]])

        # create dataframe
        df = pd.DataFrame(features,
                          columns=['curvature',
                                   'velocity', 'velocity_x', 'velocity_y', 'velocity_z',
                                   'acceleration', 'acceleration_x', 'acceleration_y', 'acceleration_z'])
        df = df.sort_index(ascending=0)

        return df

    def assemble_displacement_features(self, points=None,
                                       lproj=None, ldir=None,
                                       lu_s=None, lu_v=None,
                                       gu_s=None, gu_v=None):
        features = []
        for i in range(len(points)):
            features.append([lproj[i][0], lproj[i][1], lproj[i][2],
                             ldir[i][0], ldir[i][1], ldir[i][2],
                             lu_s[i], lu_v[i][0], lu_v[i][1], lu_v[i][2],
                             gu_s[i], gu_v[i][0], gu_v[i][1], gu_v[i][2]])

        # create dataframe
        df = pd.DataFrame(features,
                          columns=['lu_proj_x', 'lu_proj_y', 'lu_proj_z',
                                   'lu_dir_x', 'lu_dir_y', 'lu_dir_z',
                                   'lu', 'lu_x', 'lu_y', 'lu_z',
                                   'gu', 'gu_x', 'gu_y', 'gu_z'])
        df = df.sort_index(ascending=0)

        return df

    def assemble_structural_features(self, points=None,
                                     indices=None, intensities=None,
                                     i_depth=None):
        features = []
        point_depth = len(points)
        for i in range(len(points)):
            features.append([indices[i][0], indices[i][1], indices[i][2],
                             intensities[i],
                             i_depth, point_depth])
            point_depth -= 1

        # create dataframe
        df = pd.DataFrame(features,
                          columns=['voxel_x', 'voxel_y', 'voxel_z',
                                   'mri_intensity',
                                   'intracranial_depth', 'point_depth'])
        df = df.sort_index(ascending=0)

        return df

    def assemble_parcellation_features(self, points=None,
                                       regions=None, segments=None,
                                       wm_ratio=None):
        features = []

        # find EP and TP regions
        region_EP = 0
        for i in range(len(segments) - 1, -1, -1):
            seg = segments[i]
            if seg[3] != -1 and seg[3] != 0 and seg[3] != 1 and seg[3] != 2 and seg[3] != 3 and seg[3] != 4:
                region_EP = seg[3]
                break
        region_TP = regions[0]

        # find region at interpolated point
        for i in range(len(points)):
            # gif value
            gifval = regions[i]

            # [EP_region, TP_region, point_region, regions_traversed,
            #  cwd, cortex_traversed, white_traversed, deep_traversed,
            #  seg_length, seg_depth, seg_wmratio, elec_wmratio]
            features.append([region_EP, region_TP, gifval, 0,
                             0, 0, 0, 0,
                             0., 0., 0., wm_ratio])

        # number of regions traversed
        prev_reg = 0.0
        regions_traversed = 0
        cortex_traversed, white_traversed, deep_traversed = 0, 0, 0
        for i in range(len(points) - 1, -1, -1):
            gifval = regions[i]
            cwd = 0
            if gifval != 0 and gifval != 1 and gifval != 2 and gifval != 3 and gifval != 4:
                # regions_traversed
                if gifval != prev_reg:
                    regions_traversed += 1
                    prev_reg = gifval

                # cwd
                if gifval in MedicalImages.gmGif:
                    cwd = 1
                    cortex_traversed += 1
                elif gifval in MedicalImages.wmGif:
                    cwd = 2
                    white_traversed += 1
                elif gifval in MedicalImages.dmGif:
                    cwd = 3
                    deep_traversed += 1

            features[i][3] = regions_traversed
            features[i][4] = cwd
            features[i][5] = cortex_traversed
            features[i][6] = white_traversed
            features[i][7] = deep_traversed

            # segments: include details of the regions that are traversed
        ''' segments: [id, id+1, t, region, s_fraction, s_length, acc_depth, acc_wmlength] '''
        d_prev_r = 0.0
        d_next_r = 0.0
        for i in range(len(points)):
            for s in range(len(segments)):
                s_i = segments[s][0]
                s_l = segments[s][5]
                s_d = segments[s][6]
                s_wm = segments[s][7]
                if i <= s_i:
                    features[i][8] = s_l
                    features[i][9] = s_d
                    features[i][10] = s_wm
                    break

        # create dataframe
            # create dataframe
            df = pd.DataFrame(features,
                              columns=['EP_region', 'TP_region', 'point_region', 'regions_traversed',
                                       'cwd', 'cortex_traversed', 'white_traversed', 'deep_traversed',
                                       'segment_length', 'segment_depth', 'segment_wmratio',
                                       'elec_wmratio'])
        df = df.sort_index(ascending=0)

        return df

    def assemble_collision_features(self, points=None, acpcih=None,
                                    col_scalp=None, col_cortex=None, col_white=None, col_deep=None):
        features = []

        [scalp_point, scalp_normal, scalp_angle] = [np.zeros(3), np.zeros(3), 0.0]
        [cortex_j, cortex_i, cortex_x, cortex_n, cortex_a] = [-1, -1, np.zeros(3), np.zeros(3), 0.0]
        if len(col_scalp) is not 0:
            [scalp_point, scalp_normal, scalp_angle] = col_scalp
        if len(col_cortex) is not 0:
            [cortex_j, cortex_i, cortex_x, cortex_n, cortex_a] = col_cortex[0]

        # print('col_scalp[{}] = {}'.format(len(col_scalp), col_scalp))
        # print('col_cortex[{}] = {}'.format(len(col_cortex), col_cortex))
        # print('col_white[{}] = {}'.format(len(col_white), col_white))
        # print('col_deep[{}] = {}'.format(len(col_deep), col_deep))

        # iterate through points
        for i in range(len(points)):

            cortex_point, cortex_normal, cortex_angle = np.zeros(3), np.zeros(3), 0.0
            white_point, white_normal, white_angle = np.zeros(3), np.zeros(3), 0.0
            deep_point, deep_normal, deep_angle = np.zeros(3), np.zeros(3), 0.0

            # cortex angle at interpolation points
            if i <= cortex_i:
                cortex_point, cortex_normal, cortex_angle = cortex_x, cortex_n, cortex_a

            # white angle at interpolation points ([c_j, c_i, c_point, c_normal, c_angle])
            for j in range(len(col_white) - 1, -1, -1):
                if i <= col_white[j][1]:
                    white_point, white_normal, white_angle = col_white[j][2], col_white[j][3], col_white[j][4]
                    # if angle > 90.0:
                    #     white_angle = angle
                    # else:
                    #     white_angle = 180 - angle
                    break

            # deep angle at interpolation points ([c_j, c_i, c_point, c_normal, c_angle])
            for j in range(len(col_deep) - 1, -1, -1):
                if i <= col_deep[j][1]:
                    deep_point, deep_normal, deep_angle = col_deep[j][2], col_deep[j][3], col_deep[j][4]
                    # if angle > 90.0:
                    #     ideep_angle = angle
                    # else:
                    #     ideep_angle = 180 - angle
                    break

            # compute distance to acpc
            acpc_dist = np.abs(self.surface_collision.compute_distance_to_triangle(points[i], acpcih[0], acpcih[1], acpcih[2]))

            features.append([acpc_dist,
                             scalp_point[0], scalp_point[1], scalp_point[2], scalp_normal[0], scalp_normal[1], scalp_normal[2], scalp_angle,
                             cortex_point[0], cortex_point[1], cortex_point[2], cortex_normal[0], cortex_normal[1], cortex_normal[2], cortex_angle,
                             white_point[0], white_point[1], white_point[2], white_normal[0], white_normal[1], white_normal[2], white_angle,
                             deep_point[0], deep_point[1], deep_point[2], deep_normal[0], deep_normal[1], deep_normal[2], deep_angle])

        # create dataframe
        df = pd.DataFrame(features,
                          columns=['acpc_dist',
                                   'scalp_point_x', 'scalp_point_y', 'scalp_point_z', 'scalp_normal_x', 'scalp_normal_y', 'scalp_normal_z', 'scalp_angle',
                                   'cortex_point_x', 'cortex_point_y', 'cortex_point_z', 'cortex_normal_x', 'cortex_normal_y', 'cortex_normal_z', 'cortex_angle',
                                   'white_point_x', 'white_point_y', 'white_point_z', 'white_normal_x', 'white_normal_y', 'white_normal_z', 'white_angle',
                                   'deep_point_x', 'deep_point_y', 'deep_point_z', 'deep_normal_x', 'deep_normal_y', 'deep_normal_z', 'deep_angle'])
        df = df.sort_index(ascending=0)

        return df
