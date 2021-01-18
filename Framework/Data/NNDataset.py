"""
Dataset including feature-based data

Optimise data generation process to get the benfits of multiprocessing
This class returns the batches of the trajectory of an electrode as an item

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

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
import joblib

import torch
from torch.utils import data


class NNDataset(data.Dataset):
    # parameters
    batch_time = 8
    t_scale = 0.01

    def __init__(self, data_dir=None, input_features=None, cases=None, filter_file=None, data_augment=False, batch_time=0):
        self.data_dir = data_dir
        self.input_features = input_features
        self.filter_file = filter_file
        self.data_augment = data_augment
        self.batch_time = batch_time

        # load data
        self.df = self.input_features.df
        self.feature_sz = self.input_features.get_num_features()
        self.output_sz = self.input_features.get_num_outputs()
        self.cases, self.df = self.select_cases(cases=cases)

        # scaler
        self.scaler = preprocessing.StandardScaler()

        # filter data
        if self.filter_file is not None:
            self.filter_dict, self.cases, self.df = self.filter()

        # handle categorisation
        self.categorise()

        # create dataset
        self.dataset = self.create()

    def __len__(self):
        """ Defines the total number of samples """
        return len(self.dataset)

    def __getitem__(self, index):
        # Select sample: ['case', 'electrode']
        ID = self.dataset[index]
        ID_case = ID[0]
        ID_electrode = ID[1]
        # print('case={} electrode={}'.format(ID_case, ID_electrode))

        # Get data
        df_result = self.df[(self.df.case == ID_case) & (self.df.electrode == ID_electrode)]

        df_result = df_result.reset_index(drop=True)
        # print('df_result = \n{}'.format(df_result))

        # get labels
        y = self.input_features.get_labels(data_frame=df_result)
        # y = y.loc[0:len(y)-2, :]
        # print('y = \n{}'.format(y.head()))

        # select features
        X = self.input_features.get_features(data_frame=df_result, type='general')
        X_bending = self.input_features.get_features(data_frame=df_result, type='bending')
        X_curvature = self.input_features.get_features(data_frame=df_result, type='curvature')
        X_displacement = self.input_features.get_features(data_frame=df_result, type='displacement')
        X_structural = self.input_features.get_features(data_frame=df_result, type='structural')
        X_parcellation = self.input_features.get_features(data_frame=df_result, type='parcellation')
        X_collision = self.input_features.get_features(data_frame=df_result, type='collision')

        # join columns
        X = X.join(X_bending)
        X = X.join(X_curvature)
        X = X.join(X_displacement)
        X = X.join(X_structural)
        X = X.join(X_parcellation)
        X = X.join(X_collision)
        X = X.loc[0:len(X)-2, :]
        # print('X = \n{}'.format(X.head()))

        # ground truth (without TP since we don't have ground truth of next point)
        impl = df_result.loc[0:len(df_result)-2, ['x', 'y', 'z']]
        impl_next = df_result.loc[1:len(df_result) - 1, ['x', 'y', 'z']]
        plan_next = df_result.loc[1:len(df_result) - 1, ['plan_x', 'plan_y', 'plan_z']]
        # print('impl = \n{}'.format(impl.head()))
        # print('impl_next = \n{}'.format(impl_next.head()))
        # print('plan_next = \n{}'.format(plan_next.head()))

        # convert to tensors
        X_tensor = torch.tensor(X.values.astype(np.float64))
        y_tensor = torch.tensor(y.values.astype(np.float64))
        impl_tensor = torch.tensor(impl.values.astype(np.float64))
        impl_next_tensor = torch.tensor(impl_next.values.astype(np.float64))
        plan_next_tensor = torch.tensor(plan_next.values.astype(np.float64))

        # print('     case={} elec={}: x[{}] y[{}] impl[{}] impl_next[{}] plan_next[{}]'.format(ID_case, ID_electrode, X_tensor.size(),
        #                                                                                       y_tensor.size(), impl_tensor.size(),
        #                                                                                       impl_next_tensor.size(),
        #                                                                                       plan_next_tensor.size()))

        return X_tensor, y_tensor, impl_tensor, impl_next_tensor, plan_next_tensor, ID_case, ID_electrode

    def filter(self):
        # filter_file example: 'filter_mtg.npy'
        pickle_file = open(os.path.join(self.data_dir, self.filter_file), "rb")
        filter_dict = pickle.load(pickle_file)
        pickle_file.close()

        filter_cases = self.cases[np.isin(self.cases, list(filter_dict.keys()))]

        filter_df = pd.DataFrame()
        for c in filter_cases:
            for e in filter_dict[c]:
                name = 'E' + str(e) + 'i'
                filter_df = filter_df.append(self.df[(self.df.case == c) & (self.df.electrode == name)], ignore_index=True)
                # print('case={} electrode={}'.format(c, name))

        print('NNDataset:: filter loaded with keys={} filter={} resulted in filter_cases={}'.format(list(filter_dict.keys()), filter_dict, filter_cases))

        return filter_dict, filter_cases, filter_df

    def select_cases(self, cases=None):
        selection_df = pd.DataFrame()
        for c in list(cases):
            selection_df = selection_df.append(self.df[self.df.case == c], ignore_index=True)

        return cases, selection_df

    def create_scaler(self, file=None):
        scaler_file = os.path.join(self.data_dir, file)
        if not os.path.exists(scaler_file):
            print('NNDataset::create_scaler() ... creating')
            self.scaler.fit(self.df[self.input_features.columns2normalise])
            joblib.dump(self.scaler, scaler_file)
        else:
            print('NNDataset::create_scaler() ... loading')
            self.scaler = joblib.load(scaler_file)

    def load_scaler(self, file=None):
        print('NNDataset::load_scaler()')
        scaler_file = os.path.join(self.data_dir, file)
        self.scaler = joblib.load(scaler_file)

    def normalise(self):
        norm_df = pd.DataFrame(self.scaler.transform(self.df[self.input_features.columns2normalise]),
                               columns=self.input_features.columns2normalise,
                               index=self.df.index)

        original_df = self.df[self.df.columns.difference(self.input_features.columns2normalise)]

        join_df = pd.concat([original_df, norm_df], axis=1)

        self.df = join_df.reindex(columns=self.df.columns)

    def save(self, checkpoint_dir=None, fold=None, type=None):
        data_csv_file = os.path.join(checkpoint_dir, 'df_' + type + '_f' + str(fold) + '.csv')
        data_pkl_file = os.path.join(checkpoint_dir, 'df_' + type + '_f' + str(fold) + '.pkl')
        self.df[self.input_features.columns].to_csv(data_csv_file, header=True, index=False)
        self.df[self.input_features.columns].to_pickle(data_pkl_file)
        data_csv_file = os.path.join(checkpoint_dir, 'df_all_' + type + '_f' + str(fold) + '.csv')
        data_pkl_file = os.path.join(checkpoint_dir, 'df_all_' + type + '_f' + str(fold) + '.pkl')
        self.df.to_csv(data_csv_file, header=True, index=False)
        self.df.to_pickle(data_pkl_file)

    def categorise(self):
        # stylet
        self.df = self.input_features.categorical.stylet(self.df)
        self.input_features.update_categorical_features(self.input_features.categorical.stylet_cat, self.input_features.categorical.stylet_cols)
        print('[categorise] self.df updated', self.df.shape)

        # cwd
        self.df = self.input_features.categorical.cwd(self.df)
        self.input_features.update_categorical_features(self.input_features.categorical.cwd_cat, self.input_features.categorical.cwd_cols)
        print('[categorise] self.df updated', self.df.shape)

        # ep
        self.df = self.input_features.categorical.ep_by_lobes(self.df)
        self.input_features.update_categorical_features(self.input_features.categorical.ep_cat, self.input_features.categorical.ep_cols)
        print('[categorise] self.df updated', self.df.shape)

        # tp
        self.df = self.input_features.categorical.tp_by_lobes(self.df)
        self.input_features.update_categorical_features(self.input_features.categorical.tp_cat, self.input_features.categorical.tp_cols)
        print('[categorise] self.df updated', self.df.shape)

        self.feature_sz = self.input_features.get_num_features()

    def update_num_features(self):
        self.feature_sz = self.input_features.get_num_features()

    def create(self):
        # returns list of cases by key: case/electrode
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
        dataset = []

        # per electrode
        df_group = self.df.groupby(['case','electrode']).size().reset_index()
        for index, row in df_group.iterrows():
            dataset.append([row.case, row.electrode])

        # per point
        # df_group = self.df.groupby(['case', 'electrode', 'interpolation']).size().reset_index()
        # for index, row in df_group.iterrows():
        #     dataset.append([row.case, row.electrode, row.interpolation])

        print('NNDataset:: {} cases / {} electrodes: '.format(len(np.unique(self.df.case)), len(dataset)))

        return dataset

    def database_backup(self):
        self.backup = self.dataset

    def database_restore(self):
        self.dataset = self.backup

    def get_names(self):
        names = []

        for i in range(len(self.dataset)):
            ID = self.dataset[i]
            ID_case = ID[0]
            ID_electrode = ID[1]
            names.append(ID_electrode)

        return names

    def database_by_electrode(self, name=None):
        ''' points go from EP to TP '''
        electrode = []

        # find index by value
        idx = 0
        for i in range(len(self.dataset)):
            ID = self.dataset[i]
            ID_case = ID[0]
            ID_electrode = ID[1]

            if ID_electrode == name:
                idx = i
                continue

        # add values to dictionary
        ID = self.dataset[idx]
        ID_case = ID[0]
        ID_electrode = ID[1]
        electrode.append([ID_case, ID_electrode])

        # replace dataset with electrode
        self.dataset = electrode

        # find values
        df_result = self.df[(self.df.case == ID_case) & (self.df.electrode == ID_electrode)]
        df_result = df_result.reset_index(drop=True)
        ep_gif = np.unique(df_result.EP_region.values)[0]
        tp_gif = np.unique(df_result.TP_region.values)[0]
        num_contacts = np.unique(df_result.num_contacts.values)[0]
        plan = df_result[['plan_x', 'plan_y', 'plan_z']].values

        N = len(df_result)
        cn = np.asarray([df_result['x'].loc[0], df_result['y'].loc[0], df_result['z'].loc[0]])
        cm = np.asarray([df_result['x'].loc[1], df_result['y'].loc[1], df_result['z'].loc[1]])
        tp = np.asarray([df_result['x'].loc[N-1], df_result['y'].loc[N-1], df_result['z'].loc[N-1]])
        dir = cn - cm
        ep = cn + 20.0*dir

        return ep_gif, tp_gif, ep, tp, num_contacts, plan

    def generate_gu_dataframe(self, region_name=None, plot=False):
        df = pd.DataFrame()

        sns.set_theme(style="white")
        sns.set(rc={'figure.figsize': (6, 2.5)})
        palette_x = sns.cubehelix_palette(as_cmap=True)
        palette_y = sns.cubehelix_palette(start=2, rot=0, dark=0, light=.95, as_cmap=True)
        palette_z = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)

        # iterate through dataset
        for index in range(len(self.dataset)):
            ID = self.dataset[index]
            ID_case = ID[0]
            ID_electrode = ID[1]
            # print('case={} electrode={}'.format(ID_case, ID_electrode))

            # Get data
            df_result = self.df[(self.df.case == ID_case) & (self.df.electrode == ID_electrode)]
            df_result = df_result.reset_index(drop=True)

            df_x = df_result[['point_depth', 'gu_x']]
            df_y = df_result[['point_depth', 'gu_y']]
            df_z = df_result[['point_depth', 'gu_z']]

            df_x.columns = ['i', 'gu']
            df_y.columns = ['i', 'gu']
            df_z.columns = ['i', 'gu']

            # invert interpolation
            # df_x['i'] = (df_x['i'] - (len(df_x) - 1)) * (-1)
            # df_y['i'] = (df_y['i'] - (len(df_y) - 1)) * (-1)
            # df_z['i'] = (df_z['i'] - (len(df_z) - 1)) * (-1)

            df_x['component'] = 'x'
            df_y['component'] = 'y'
            df_z['component'] = 'z'

            df = df.append(df_x)
            df = df.append(df_y)
            df = df.append(df_z)

            sns.set_theme(style="white")
            # sns.lineplot(x="i", y="gu", color='#e78ac3', linewidth=1.0, data=df_x, ci=None)
            sns.lineplot(x="i", y="gu", color='#a6d854', linewidth=1.0, data=df_y, ci=None)
            # sns.lineplot(x="i", y="gu", color='#8da0cb', linewidth=1, data=df_z, ci=None)

            if region_name is not None:
                df['region'] = region_name

        df = df.reset_index(drop=True)
        sns.set_theme(style="white")
        plt.xlabel('')
        plt.ylabel('')
        plt.xlim(0, 80)
        plt.ylim(-8, 8)
        plt.show()

        # plot
        if plot:
            sns.set_theme(style="white")
            sns.lineplot(x="i", y="gu",
                         hue="component",
                         data=df, ci=95)
            sns.set(rc={'figure.figsize':(10,6)})
            # plt.title(region_name, fontsize=30)
            plt.xlabel('depth (mm)', fontsize=20)
            plt.ylabel('global displacement (mm)', fontsize=20)
            plt.xlim(0,80)
            plt.ylim(-2, 2)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(loc='lower left')
            plt.show()

        return df