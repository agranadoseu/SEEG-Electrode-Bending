"""
Dataset for GPyTorch model

Optimise data generation process to get the benfits of multiprocessing
This class returns the batches of the trajectory of an electrode as an item

Based on:
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

TODO randomise training, testing, validation cases rather than having fixed values
TODO support cross validation

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

import torch
from torch.utils import data


class GPyDataset(data.Dataset):
    # parameters

    def __init__(self, data_dir=None, space=None, filter_file=None, mode=None, data_augment=False):
        self.data_dir = data_dir
        self.space = space
        self.filter_file = filter_file
        self.mode = mode
        self.data_augment = data_augment

        # load data
        self.data = self.load()
        self.cases = np.arange(len(self.data['case']))

        # split dataset based on mode
        self.cases = self.split()

        # filter data
        if self.filter_file is not None:
            self.filter_dict, self.cases = self.filter()

        # create dataset
        self.dataset, self.X, self.y = self.create()

        # points to extrapolate
        self.extrapolate = 0
        # self.extrapolate = 0 if self.mode == 'training' else 5

    def __len__(self):
        """ Defines the total number of samples """
        return len(self.dataset['case'])

    def __getitem__(self, index):
        """ Generates one sample of data """
        case = self.dataset['case'][index]
        name = self.dataset['name'][index]
        num_points = len(self.dataset['points'][index])

        x = np.zeros(shape=(num_points + self.extrapolate, 7), dtype=np.float32)
        y = np.zeros(shape=(num_points + self.extrapolate, 3), dtype=np.float32)
        for p in range(num_points + self.extrapolate):
            x[p][0] = p
            x[p][1] = self.dataset['points'][index][0][0]
            x[p][2] = self.dataset['points'][index][0][1]
            x[p][3] = self.dataset['points'][index][0][2]
            x[p][4] = self.dataset['plan_dir'][index][0][0].item()
            x[p][5] = self.dataset['plan_dir'][index][0][1].item()
            x[p][6] = self.dataset['plan_dir'][index][0][2].item()

            y[p][0] = self.dataset['gu'][index][p][0][0]
            y[p][1] = self.dataset['gu'][index][p][0][1]
            y[p][2] = self.dataset['gu'][index][p][0][2]

        return x, y, case, name

    def get_data(self):
        X = torch.from_numpy(self.X)
        y0 = torch.from_numpy(self.y[:, 0])  # 1dof for now
        y1 = torch.from_numpy(self.y[:, 1])  # 1dof for now
        y2 = torch.from_numpy(self.y[:, 2])  # 1dof for now
        print('X', X.shape)

        return X, [y0, y1, y2]

    def load(self):
        if not self.data_augment:
            data_file = 'data_' + self.space + '.npy'
        else:
            data_file = 'data_' + self.space + '_da.npy'

        pickle_file = open(os.path.join(self.data_dir, data_file), "rb")
        data = pickle.load(pickle_file)
        pickle_file.close()

        print('ODEDataset:: loaded data from {} with cases={}'.format(data_file, data['case']))

        return data

    def split(self):
        # data mode: all, train, test, validation
        cases_O = ['O{:02d}'.format(i) for i in [5, 10]]
        cases_P = ['P{:02d}'.format(i) for i in range(1, 18, 1) if not np.isin(i, [14])]
        cases_R = ['R{:02d}'.format(i) for i in range(1, 33, 1) if not np.isin(i, [1, 3])]
        cases_S = ['S{:02d}'.format(i) for i in [7, 13, 15]]
        cases_T = ['T{:02d}'.format(i) for i in range(1, 38, 1) if not np.isin(i, [4, 15, 19, 30])]
        if self.mode == 'training':
            cases_O = ['O{:02d}'.format(i) for i in [5, 10]]
            cases_P = ['P{:02d}'.format(i) for i in range(1, 16, 1) if not np.isin(i, [14])]
            cases_R = ['R{:02d}'.format(i) for i in range(1, 31, 1)]
            cases_S = ['S{:02d}'.format(i) for i in [7]]
            cases_T = ['T{:02d}'.format(i) for i in range(1, 36, 1) if not np.isin(i, [4, 15, 19, 30])]
        elif self.mode == 'testing':
            cases_O = []
            cases_P = ['P{:02d}'.format(i) for i in [16]]
            cases_R = ['R{:02d}'.format(i) for i in [31]]
            cases_S = ['S{:02d}'.format(i) for i in [13]]
            cases_T = ['T{:02d}'.format(i) for i in [36]]
        elif self.mode == 'validation':
            cases_O = []
            cases_P = ['P{:02d}'.format(i) for i in [17]]
            cases_R = ['R{:02d}'.format(i) for i in [32]]
            cases_S = ['S{:02d}'.format(i) for i in [15]]
            cases_T = ['T{:02d}'.format(i) for i in [37]]

        # one list
        mode_cases = []
        mode_cases.extend(cases_O)
        mode_cases.extend(cases_P)
        mode_cases.extend(cases_R)
        mode_cases.extend(cases_S)
        mode_cases.extend(cases_T)

        id_cases = np.where(np.isin(self.data['case'], mode_cases))[0]
        print('ODEDataset:: split cases for mode={} with remaining mode_cases={}, id_cases={}'.format(self.mode, mode_cases, id_cases))

        return id_cases

    def filter(self):
        # filter_file example: 'filter_mtg.npy'
        pickle_file = open(os.path.join(self.data_dir, self.filter_file), "rb")
        filter_dict = pickle.load(pickle_file)
        pickle_file.close()

        filter_cases = np.where(np.isin(self.data['case'], list(filter_dict.keys())))[0]
        overlap = np.where(np.isin(self.cases, filter_cases))
        filter_cases = self.cases[overlap[0]]
        print('ODEDataset:: filter loaded with keys={} filter={} resulted in filter_cases={}'.format(list(filter_dict.keys()), filter_dict, filter_cases))

        return filter_dict, filter_cases

    def create(self):
        X, y = None, None

        # dict to save data
        dataset = {'case': [], 'name': [], 'points': [], 'depth': [], 'plan_dir': [], 'lu': [], 'gu': []}

        # iterate through cases
        for i in self.cases:
            case = self.data['case'][i]
            plan = self.data['plan'][i]
            impl = self.data['impl'][i]
            delta_l = self.data['local_delta'][i]
            delta_g = self.data['delta'][i]

            print('     case={}'.format(case))

            # by default all electrodes unless filtered
            electrodes = np.arange(len(plan['name']))
            if self.filter_file is not None:
                electrodes = np.where(np.isin(plan['id'], self.filter_dict[case]))[0]

            # iterate through electrodes
            for j in electrodes:
                if len(plan['points'][j]) == 0:
                    continue

                print('         electrode={}'.format(plan['name'][j]))

                # compute direction from EP to TP of planned trajectory
                ep = plan['ep'][j]
                tp = plan['tp'][j]
                dir = tp - ep
                dir /= np.linalg.norm(dir)
                # print('   ep={} tp={} dir={}'.format(ep, tp, dir))

                # ensure points go from EP towards TP
                points_plan = np.flip(plan['points'][j], 0)
                points_impl = np.flip(impl['points'][j], 0)
                if not self.data_augment:
                    show_delta = np.where(np.isin(delta_l['name'], impl['name'][j]))[0][0]
                    ldu = np.flip(delta_l['du'][show_delta], 0)
                    lu = np.flip(delta_l['u'][show_delta], 0)
                    for k in range(len(ldu)):
                        lu[k] *= ldu[k]
                    gu = np.flip(delta_g['u'][show_delta], 0)
                    # print('    points: ', len(points_plan), len(points_impl), len(lu), len(gu))

                # create true_y useful when comparing to prediction
                num_points = len(points_plan)
                input_depth = torch.from_numpy(np.arange(num_points, dtype=np.float32))
                input_points = torch.from_numpy(np.zeros(shape=(num_points, 1, 3), dtype=np.float32))
                input_plan_dir = torch.from_numpy(np.zeros(shape=(1, 3), dtype=np.float32))
                inputs = np.zeros(shape=(num_points, 7), dtype=np.float32)
                label_lu = torch.from_numpy(np.zeros(shape=(num_points, 1, 3), dtype=np.float32))
                label_gu = torch.from_numpy(np.zeros(shape=(num_points, 1, 3), dtype=np.float32))
                labels = np.zeros(shape=(num_points, 3), dtype=np.float32)

                for d in range(3):
                    input_plan_dir[0][d] = dir[d]
                for p in range(num_points):
                    for d in range(3):
                        input_points[p][0][d] = points_plan[p][d]
                        label_lu[p][0][d] = lu[p][d]
                        label_gu[p][0][d] = gu[p][d]
                    inputs[p][0] = input_depth[p]
                    inputs[p][1] = input_points[0][0][0]
                    inputs[p][2] = input_points[0][0][1]
                    inputs[p][3] = input_points[0][0][2]
                    inputs[p][4] = input_plan_dir[0][0]
                    inputs[p][5] = input_plan_dir[0][1]
                    inputs[p][6] = input_plan_dir[0][2]
                    labels[p][0] = label_gu[p][0][0]
                    labels[p][1] = label_gu[p][0][1]
                    labels[p][2] = label_gu[p][0][2]

                # save into training set
                dataset['case'].append(case)
                dataset['name'].append(impl['name'][j])
                dataset['points'].append(points_impl)
                dataset['depth'].append(input_depth)
                dataset['plan_dir'].append(input_plan_dir)
                dataset['lu'].append(label_lu)
                dataset['gu'].append(label_gu)

                if X is None:
                    X = inputs
                    y = labels
                else:
                    X = np.concatenate((X, inputs), axis=0)
                    y = np.concatenate((y, labels), axis=0)

        print('GPyDataset:: {} set with {} cases / {} electrodes: '.format(self.mode, len(self.cases), len(dataset['case'])))

        return dataset, X, y
