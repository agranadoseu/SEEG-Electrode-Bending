"""
Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import pandas as pd
import numpy as np
import pickle

from Framework.Data import CategoricalData


class InputFeatures:

    labels = []

    def __init__(self, directory=None, file=None, labels=None):
        self.directory = directory
        self.file = file
        self.labels = labels

        # labels for regression
        # self.labels = ['lu_x', 'lu_y', 'lu_z']
        # self.labels = ['gu_x', 'gu_y', 'gu_z']
        # self.labels = ['elec_dir_x', 'elec_dir_y', 'elec_dir_z']  # next

        # general features
        # 'stylet': [-1, 0, 1]
        # other: 'case', 'electrode',
        self.cols_general = ['num_contacts', 'interpolation', 'stylet',
                             'x', 'y', 'z',
                             'plan_x', 'plan_y', 'plan_z']
        self.cols_general_n = []

        # bending features
        # other:
        self.cols_bending = ['elec_dir_x', 'elec_dir_y', 'elec_dir_z',
                             'bolt_dir_x', 'bolt_dir_y', 'bolt_dir_z',
                             'l_omega',
                             'l_omega_x', 'l_omega_y', 'l_omega_z',
                             'g_omega',
                             'g_omega_x', 'g_omega_y', 'g_omega_z']
        self.cols_bending_n = ['l_omega',
                             'l_omega_x', 'l_omega_y', 'l_omega_z',
                             'g_omega',
                             'g_omega_x', 'g_omega_y', 'g_omega_z']

        # curvature
        self.cols_curvature = ['curvature',
                               'velocity', 'velocity_x', 'velocity_y', 'velocity_z',
                               'acceleration', 'acceleration_x', 'acceleration_y', 'acceleration_z']
        self.cols_curvature_n = ['curvature',
                               'velocity', 'velocity_x', 'velocity_y', 'velocity_z',
                               'acceleration', 'acceleration_x', 'acceleration_y', 'acceleration_z']

        # displacement
        # other: 'lu_proj_x', 'lu_proj_y', 'lu_proj_z'
        #        'lu_dir_x', 'lu_dir_y', 'lu_dir_z',
        #        'lu', 'lu_x', 'lu_y', 'lu_z',
        #        'gu', 'gu_x', 'gu_y', 'gu_z'
        self.cols_displacement = []
        self.cols_displacement_n = []

        # structural
        self.cols_structural = ['voxel_x', 'voxel_y', 'voxel_z',
                                'mri_intensity',
                                'intracranial_depth', 'point_depth']
        self.cols_structural_n = ['intracranial_depth', 'point_depth']

        # parcellation
        # 'EP_region', 'TP_region', 'point_region': GIF
        # 'cwd': [0, 1, 2]
        self.cols_parcellation = ['EP_region', 'TP_region', 'point_region', 'regions_traversed',
                                  'cwd', 'cortex_traversed', 'white_traversed', 'deep_traversed',
                                  'segment_length', 'segment_depth', 'segment_wmratio',
                                  'elec_wmratio']
        self.cols_parcellation_n = ['regions_traversed', 'cortex_traversed', 'white_traversed', 'deep_traversed',
                                    'segment_length', 'segment_depth', 'segment_wmratio',
                                    'elec_wmratio']

        # collision
        self.cols_collision = ['acpc_dist',
                               'scalp_point_x', 'scalp_point_y', 'scalp_point_z', 'scalp_normal_x', 'scalp_normal_y', 'scalp_normal_z', 'scalp_angle',
                               'cortex_point_x', 'cortex_point_y', 'cortex_point_z', 'cortex_normal_x', 'cortex_normal_y', 'cortex_normal_z', 'cortex_angle',
                               'white_point_x', 'white_point_y', 'white_point_z', 'white_normal_x', 'white_normal_y', 'white_normal_z', 'white_angle',
                               'deep_point_x', 'deep_point_y', 'deep_point_z', 'deep_normal_x', 'deep_normal_y', 'deep_normal_z', 'deep_angle']
        self.cols_collision_n = ['acpc_dist',
                                 'scalp_angle', 'cortex_angle', 'white_angle', 'deep_angle']

        self.columns = self.cols_general + self.cols_bending + self.cols_curvature + self.cols_displacement + self.cols_structural + self.cols_parcellation + self.cols_collision
        self.columns2normalise = self.cols_general_n + self.cols_bending_n + self.cols_curvature_n + self.cols_displacement_n + self.cols_structural_n + self.cols_parcellation_n + self.cols_collision_n

        self.df = self.load_data()
        self.fix_data()

        # categories (replace x by y)
        self.categorical = CategoricalData.CategoricalData()
        # self.update_categorical_features(self.categorical.stylet_cat, self.categorical.stylet_cols)
        # self.update_categorical_features(self.categorical.cwd_cat, self.categorical.cwd_cols)
        # self.update_categorical_features(self.categorical.ep_cat, self.categorical.ep_cols)
        # self.update_categorical_features(self.categorical.tp_cat, self.categorical.tp_cols)

    def get_num_features(self):
        return len(self.cols_general) + \
               len(self.cols_bending) + \
               len(self.cols_curvature) + \
               len(self.cols_displacement) + \
               len(self.cols_structural) + \
               len(self.cols_parcellation) + \
               len(self.cols_collision)

    def get_num_outputs(self):
        return len(self.labels)

    def update_categorical_features(self, category, category_columns):
        if any(np.isin(self.cols_general, category[0])):
            self.cols_general += category_columns
            self.cols_general.remove(category[0])
        elif any(np.isin(self.cols_bending, category[0])):
            self.cols_bending += category_columns
            self.cols_bending.remove(category[0])
        elif any(np.isin(self.cols_curvature, category[0])):
            self.cols_curvature += category_columns
            self.cols_curvature.remove(category[0])
        elif any(np.isin(self.cols_displacement, category[0])):
            self.cols_displacement += category_columns
            self.cols_displacement.remove(category[0])
        elif any(np.isin(self.cols_structural, category[0])):
            self.cols_structural += category_columns
            self.cols_structural.remove(category[0])
        elif any(np.isin(self.cols_parcellation, category[0])):
            self.cols_parcellation += category_columns
            self.cols_parcellation.remove(category[0])
        elif any(np.isin(self.cols_collision, category[0])):
            self.cols_collision += category_columns
            self.cols_collision.remove(category[0])

        self.columns = self.cols_general + self.cols_bending + self.cols_curvature + self.cols_displacement + self.cols_structural + self.cols_parcellation + self.cols_collision

    def load_data(self):
        # load cases from file
        pickle_file = open(os.path.join(self.directory, self.file), "rb")
        features_df = pickle.load(pickle_file)
        pickle_file.close()

        return features_df

    def fix_data(self):
        # TP_region
        self.replace_value(case='O05', elec='E0i', column='TP_region', value=172)
        self.replace_value(case='O05', elec='E2i', column='TP_region', value=172)
        self.replace_value(case='O10', elec='E10i', column='TP_region', value=139)
        self.replace_value(case='P01', elec='E5i', column='TP_region', value=102)
        self.replace_value(case='P01', elec='E9i', column='TP_region', value=194)
        self.replace_value(case='P05', elec='E7i', column='TP_region', value=106)
        self.replace_value(case='P06', elec='E10i', column='TP_region', value=153)
        self.replace_value(case='P06', elec='E12i', column='TP_region', value=137)
        self.replace_value(case='P08', elec='E1i', column='TP_region', value=138)
        self.replace_value(case='P08', elec='E4i', column='TP_region', value=194)
        self.replace_value(case='P10', elec='E0i', column='TP_region', value=118)
        self.replace_value(case='P10', elec='E8i', column='TP_region', value=154)
        self.replace_value(case='P11', elec='E2i', column='TP_region', value=118)
        self.replace_value(case='P11', elec='E8i', column='TP_region', value=148)
        self.replace_value(case='P12', elec='E1i', column='TP_region', value=137)
        self.replace_value(case='P17', elec='E0i', column='TP_region', value=169)
        self.replace_value(case='P17', elec='E1i', column='TP_region', value=101)
        self.replace_value(case='P17', elec='E2i', column='TP_region', value=139)
        self.replace_value(case='P17', elec='E3i', column='TP_region', value=119)
        self.replace_value(case='P17', elec='E4i', column='TP_region', value=193)
        self.replace_value(case='P17', elec='E6i', column='TP_region', value=193)
        self.replace_value(case='P17', elec='E7i', column='TP_region', value=151)
        self.replace_value(case='P17', elec='E9i', column='TP_region', value=179)

        self.replace_value(case='R02', elec='E3i', column='TP_region', value=154)
        self.replace_value(case='R03', elec='E6i', column='TP_region', value=101)
        self.replace_value(case='R03', elec='E10i', column='TP_region', value=193)
        self.replace_value(case='R06', elec='E10i', column='TP_region', value=48)
        self.replace_value(case='R07', elec='E10i', column='TP_region', value=194)
        self.replace_value(case='R07', elec='E8i', column='TP_region', value=102)
        self.replace_value(case='R08', elec='E3i', column='TP_region', value=141)
        self.replace_value(case='R08', elec='E6i', column='TP_region', value=153)
        self.replace_value(case='R08', elec='E8i', column='TP_region', value=139)
        self.replace_value(case='R12', elec='E6i', column='TP_region', value=102)
        self.replace_value(case='R13', elec='E3i', column='TP_region', value=172)
        self.replace_value(case='R13', elec='E13i', column='TP_region', value=152)
        self.replace_value(case='R14', elec='E7i', column='TP_region', value=110)
        self.replace_value(case='R14', elec='E14i', column='TP_region', value=170)
        self.replace_value(case='R15', elec='E6i', column='TP_region', value=208)
        self.replace_value(case='R17', elec='E5i', column='TP_region', value=206)
        self.replace_value(case='R20', elec='E6i', column='TP_region', value=154)
        self.replace_value(case='R20', elec='E10i', column='TP_region', value=154)
        self.replace_value(case='R22', elec='E4i', column='TP_region', value=182)
        self.replace_value(case='R24', elec='E11i', column='TP_region', value=194)
        self.replace_value(case='R24', elec='E12i', column='TP_region', value=152)
        self.replace_value(case='R31', elec='E3i', column='TP_region', value=135)
        self.replace_value(case='R31', elec='E7i', column='TP_region', value=169)
        self.replace_value(case='T01', elec='E0i', column='TP_region', value=147)
        self.replace_value(case='T05', elec='E4i', column='TP_region', value=139)
        self.replace_value(case='T09', elec='E6i', column='TP_region', value=181)
        self.replace_value(case='T17', elec='E6i', column='TP_region', value=181)
        self.replace_value(case='T20', elec='E4i', column='TP_region', value=204)
        self.replace_value(case='T22', elec='E1i', column='TP_region', value=104)
        self.replace_value(case='T22', elec='E2i', column='TP_region', value=102)
        self.replace_value(case='T22', elec='E3i', column='TP_region', value=154)
        self.replace_value(case='T23', elec='E10i', column='TP_region', value=152)
        self.replace_value(case='T24', elec='E12i', column='TP_region', value=153)
        self.replace_value(case='T27', elec='E2i', column='TP_region', value=153)
        self.replace_value(case='T33', elec='E0i', column='TP_region', value=153)
        self.replace_value(case='T33', elec='E2i', column='TP_region', value=101)
        self.replace_value(case='T33', elec='E4i', column='TP_region', value=117)
        self.replace_value(case='T33', elec='E6i', column='TP_region', value=48)
        self.replace_value(case='T34', elec='E10i', column='TP_region', value=194)
        self.replace_value(case='T34', elec='E2i', column='TP_region', value=102)
        self.replace_value(case='T34', elec='E4i', column='TP_region', value=180)
        self.replace_value(case='T36', elec='E5i', column='TP_region', value=167)

        # EP_region
        self.replace_value(case='O05', elec='E10i', column='EP_region', value=178)
        self.replace_value(case='P07', elec='E4i', column='EP_region', value=157)
        self.replace_value(case='R02', elec='E9i', column='EP_region', value=178)
        self.replace_value(case='R14', elec='E7i', column='EP_region', value=158)
        self.replace_value(case='R29', elec='E2i', column='EP_region', value=163)
        self.replace_value(case='T02', elec='E9i', column='EP_region', value=183)
        self.replace_value(case='T05', elec='E2i', column='EP_region', value=201)
        self.replace_value(case='T09', elec='E3i', column='EP_region', value=155)
        self.replace_value(case='T28', elec='E7i', column='EP_region', value=198)
        self.replace_value(case='T31', elec='E5i', column='EP_region', value=196)

        return

    def replace_value(self, case=None, elec=None, column=None, value=None):
        self.df.loc[(self.df.case == case) & (self.df.electrode == elec), column] = value

    # def compute_acceleration(self, case=None, elec=None):
    #     df_result = self.df[(self.df.case == case) & (self.df.electrode == elec)].reset_index(drop=True)
    #     points = df_result.loc[:, ['x', 'y', 'z']]
    #     points = points.to_numpy()

    # def get_cases(self, _df, _cases):
    #     """
    #     Get cases from an already loaded dataframe
    #     :param df:
    #     :param cases:
    #     :return:
    #     """
    #     all_df = pd.DataFrame()
    #     for i in range(len(_cases)):
    #         case_df = _df[_df.case==_cases[i]]
    #
    #         if i == 0:
    #             all_df = case_df
    #         else:
    #             all_df = all_df.append(case_df, sort=False)
    #
    #     return all_df

    def get_features(self, data_frame=None, type=None):
        if data_frame is None:
            data_frame = self.df

        X = []
        if type == 'general':
            # print('features: general[{}]={}'.format(len(self.cols_general), self.cols_general))
            X = data_frame[self.cols_general]
        elif type == 'bending':
            # print('features: bending[{}]={}'.format(len(self.cols_bending), self.cols_bending))
            X = data_frame[self.cols_bending]
        elif type == 'curvature':
            # print('features: curvature[{}]={}'.format(len(self.cols_curvature), self.cols_curvature))
            X = data_frame[self.cols_curvature]
        elif type == 'displacement':
            # print('features: displacement[{}]={}'.format(len(self.cols_displacement), self.cols_displacement))
            X = data_frame[self.cols_displacement]
        elif type == 'structural':
            # print('features: structural[{}]={}'.format(len(self.cols_structural), self.cols_structural))
            X = data_frame[self.cols_structural]
        elif type == 'parcellation':
            # print('features: parcellation[{}]={}'.format(len(self.cols_parcellation), self.cols_parcellation))
            X = data_frame[self.cols_parcellation]
        elif type == 'collision':
            # print('features: collision[{}]={}'.format(len(self.cols_collision), self.cols_collision))
            X = data_frame[self.cols_collision]

        return X

    def get_labels(self, data_frame=None):
        if data_frame is None:
            data_frame = self.df

        if self.labels == ['lu_x', 'lu_y', 'lu_z']:
            y = data_frame[self.labels] * data_frame[['lu']].values
            y = y.loc[0:len(y) - 2, :]
        elif self.labels == ['gu_x', 'gu_y', 'gu_z']:
            y = data_frame[self.labels]
            y = y.loc[0:len(y) - 2, :]
        elif self.labels == ['elec_dir_x', 'elec_dir_y', 'elec_dir_z']:
            y = data_frame[self.labels]
            y = y.loc[1:len(y) - 1, :]

        return y
