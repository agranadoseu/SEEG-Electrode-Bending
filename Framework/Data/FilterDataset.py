"""
Optimise data generation process to get the benefits of multiprocessing
This class returns the full trajectory of an electrode as an item

Based on:
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import numpy as np
import pandas as pd
import torch
from torch.utils import data


# leverage functionalities (e.g. multiprocessing) from Pytorch dataset class
class FilterDataset(data.Dataset):

    def __init__(self, input_features, categorical, mode, type, data):
        """
        :param input_features: class containing all the data
        :param mode: train, test, validation
        :param type: complete, stylet, fixels, all
        :param data: electrode, point
        """
        self.input_features = input_features
        self.categorical = categorical
        self.mode = mode
        self.type = type
        self.data = data

        # load data
        self.df = self.load(mode, type)
        self.df_orig = self.df.copy()

        # create list of cases: [['case','electrode']]
        self.cases = self.create_cases()

        # handle categorisation
        self.categorise()

        # total number of features
        self.update_num_features()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.cases)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample: ['case', 'electrode']
        ID = self.cases[index]
        ID_case = ID[0]
        ID_electrode = ID[1]
        if self.data == 'point':
            ID_interpolation = ID[2]

        # Get data
        df_result = self.df[(self.df.case == ID_case) & (self.df.electrode == ID_electrode)]
        if self.data == 'point':
            df_result = self.df[(self.df.case == ID_case) & (self.df.electrode == ID_electrode) & (self.df.interpolation == ID_interpolation)]

        # select features
        X, y = self.input_features.get_general_features(df_result)
        X_electrode, _ = self.input_features.get_electrode_features(df_result)
        X_structural, _ = self.input_features.get_structural_features(df_result)
        X_parcellation, _ = self.input_features.get_parcellation_features(df_result)
        if 'num_fibres' in df_result.columns:
            X_tractography, _ = self.input_features.get_tractography_features(df_result)

        X = X.join(X_electrode)
        X = X.join(X_structural)
        X = X.join(X_parcellation)
        if 'num_fibres' in df_result.columns:
            X = X.join(X_tractography)

        # convert to tensors
        # print('X={} size={}'.format(X.values, X.shape))
        X_tensor = torch.tensor(X.values.astype(np.float32))
        y_tensor = torch.tensor(y.values.astype(np.float32))

        if self.data == 'electrode':
            print('ID_case={} ID_electrode={} X_size={} Tensor size={}'.format(ID_case, ID_electrode, X.shape, X_tensor.size()))
        elif self.data == 'point':
            print('ID_case={} ID_electrode={} ID_interpolation={} X_size={} Tensor size={}'.format(ID_case, ID_electrode, ID_interpolation, X.shape, X_tensor.size()))

        return X_tensor, y_tensor, ID_case, ID_electrode

    def load(self, mode, type):

        df = []
        if type == 'complete':
            df = self.input_features.load_cases_complete(mode)
        elif type == 'stylet':
            df = self.input_features.load_cases_stylet(mode)
        elif type == 'fixels':
            df = self.input_features.load_cases_fixels(mode)
        elif type == 'all':
            df = self.input_features.load_cases_all(mode)

        return df

    def create_cases(self):
        ''' returns list of cases by key: case/electrode '''
        #   case electrode   0
        # 0  R02        E1  40
        # 1  R02       E10  55
        # 2  R02        E2  36
        # 3  R02        E3  20
        # 4  R02        E4  15
        # 5  R02        E5  15
        # 6  R02        E6  15
        # 7  R02        E7  30
        # 8  R02        E8  25
        # 9  R02        E9  30
        cases = []

        if self.data == 'electrode':
            df_group = self.df.groupby(['case','electrode']).size().reset_index()
            for index, row in df_group.iterrows():
                cases.append([row.case, row.electrode])

        elif self.data == 'point':
            df_group = self.df.groupby(['case','electrode','interpolation']).size().reset_index()
            for index, row in df_group.iterrows():
                cases.append([row.case, row.electrode, row.interpolation])

            # print('index={} case={} electrode={}'.format(index, row.case, row.electrode))

        return cases

    def normalise(self, scaler):
        # print('self.input_features.columns2normalise = ', self.input_features.columns2normalise)
        print('before normalisation', self.df.shape)
        norm_df = pd.DataFrame(scaler.transform(self.df[self.input_features.columns2normalise]),
                               columns=self.input_features.columns2normalise,
                               index=self.df.index)
        print('norm_df', norm_df.shape)
        # print('columns.difference', self.df.columns.difference(self.input_features.columns2normalise))
        original_df = self.df[self.df.columns.difference(self.input_features.columns2normalise)]
        # join_df = original_df.join(norm_df)   # did not work
        join_df = pd.concat([original_df, norm_df], axis=1)
        print('join_df', join_df.shape)
        self.df = join_df.reindex(columns=self.df.columns)
        print('after normalisation', self.df.shape)

    def categorise(self):
        # stylet
        self.df = self.categorical.stylet(self.df)
        print('[categorise] self.df updated', self.df.shape)

        # cwd
        self.df = self.categorical.cwd(self.df)
        print('[categorise] self.df updated', self.df.shape)

        # ep
        self.df = self.categorical.ep_by_lobes(self.df)
        print('[categorise] self.df updated', self.df.shape)

        # tp
        self.df = self.categorical.tp_by_lobes(self.df)
        print('[categorise] self.df updated', self.df.shape)

    def update_num_features(self):
        self.feature_sz = self.input_features.get_num_features()

    def search_by_plan(self, ep=None, tp=None):
        dataframe = []
        self.df = self.df_orig.copy()
        if ep is None and tp is None:
            return self.df_orig
        elif ep is None:
            self.df = self.df[self.df.TP_region.isin(tp)]
            dataframe = self.df_orig[self.df_orig.TP_region.isin(tp)]
        elif tp is None:
            self.df = self.df[self.df.EP_region.isin(ep)]
            dataframe = self.df_orig[self.df_orig.EP_region.isin(ep)]
        else:
            self.df = self.df[(self.df.EP_region.isin(ep)) & (self.df.TP_region.isin(tp))]
            dataframe = self.df_orig[(self.df_orig.EP_region.isin(ep)) & (self.df_orig.TP_region.isin(tp))]

        self.cases = self.create_cases()
        return dataframe

    def search_by_point_region(self, ep=None, tp=None):
        dataframe = []
        self.df = self.df_orig.copy()
        if ep is None and tp is None:
            return self.df_orig
        elif ep is None:
            self.df = self.df[self.df.point_region.isin(tp)]
            dataframe = self.df_orig[self.df_orig.point_region.isin(tp)]
        elif tp is None:
            self.df = self.df[self.df.EP_region.isin(ep)]
            dataframe = self.df_orig[self.df_orig.EP_region.isin(ep)]
        else:
            self.df = self.df[(self.df.EP_region.isin(ep)) & (self.df.point_region.isin(tp))]
            dataframe = self.df_orig[(self.df_orig.EP_region.isin(ep)) & (self.df_orig.point_region.isin(tp))]

        self.cases = self.create_cases()
        return dataframe
