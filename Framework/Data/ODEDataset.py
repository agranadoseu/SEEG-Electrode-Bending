"""
Dataset for ODENet model

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


class ODEDataset(data.Dataset):
    # parameters
    #dof = 3 #9
    batch_time = 8
    t_scale = 0.01

    def __init__(self, data_dir=None, data=None, space=None, filter_file=None, mode=None, cases=None, data_augment=False, dof=None, component=None, batch_time=0):
        self.data_dir = data_dir
        self.space = space
        self.filter_file = filter_file
        self.mode = mode
        self.data_augment = data_augment
        self.dof = dof
        self.component = component
        self.batch_time = batch_time  # set to zero for testing entire trajectory

        # load data (use mode only for compatibility)
        if mode != None:
            self.data = self.load()
            self.cases = np.arange(len(self.data['case']))

            # split dataset based on mode
            self.cases = self.split()
        else:
            self.data = data
            self.cases = cases

        # filter data
        if self.filter_file is not None:
            self.filter_dict, self.cases = self.filter()

        # create dataset
        self.dataset = self.create()

    def __len__(self):
        """ Defines the total number of samples """
        return len(self.dataset['case'])

    def __getitem__(self, index):
        """ Generates one sample of data """
        case = self.dataset['case'][index]
        name = self.dataset['name'][index]
        ep = self.dataset['ep'][index]
        dir = self.dataset['dir'][index]
        num_points = len(self.dataset['points_impl'][index])
        points_plan = self.dataset['points_plan'][index]

        if self.batch_time != 0:
            ''' Generate batch_time-size windows for training and evaluation '''
            # randomise section of trajectory for each electrode while keeping away enough time_batch points
            batch_t = np.arange(self.batch_time, dtype=np.float32) * self.t_scale
            batch_y0 = np.zeros(shape=(1, 1, self.dof), dtype=np.float32)
            batch_y = np.zeros(shape=(self.batch_time, 1, self.dof), dtype=np.float32)
            batch_plan = np.zeros(shape=(self.batch_time, 1, self.dof), dtype=np.float32)

            random_point = np.random.choice(np.arange(num_points - self.batch_time, dtype=np.int64), 1, replace=False)[0]
            batch_y0 = self.dataset['true_y'][index][random_point, :]
            point_time_series = self.dataset['true_y'][index][random_point:random_point + self.batch_time, :]
            for t in range(self.batch_time):
                batch_y[t] = point_time_series[t]
            batch_ep = np.zeros(shape=(1, 1, 3), dtype=np.float32)
            batch_dir = np.zeros(shape=(1, 1, 3), dtype=np.float32)

        else:
            ''' Generate one batch for the entire trajectory for testing '''
            batch_t = np.arange(num_points, dtype=np.float32) * self.t_scale
            batch_y0 = np.zeros(shape=(1, 1, self.dof), dtype=np.float32)
            batch_y = np.zeros(shape=(num_points, 1, self.dof), dtype=np.float32)
            batch_y0 = self.dataset['true_y'][index][0, :]
            point_time_series = self.dataset['true_y'][index]
            for t in range(num_points):
                batch_y[t] = point_time_series[t]
            batch_ep = np.zeros(shape=(1, 3), dtype=np.float32)
            batch_dir = np.zeros(shape=(1, 3), dtype=np.float32)
            for d in range(3):
                batch_ep[0, d] = ep[d]
                batch_dir[0, d] = dir[d]

        return batch_t, batch_y0, batch_y, points_plan, batch_ep, batch_dir, case, name

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

    def save(self, checkpoint_dir=None, fold=None, type=None):
        data_pkl_file = os.path.join(checkpoint_dir, 'dataset_' + type + '_f' + str(fold) + '.pkl')
        pickle_file = open(data_pkl_file, "wb")
        pickle.dump(self.dataset, pickle_file)
        pickle_file.close()

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
        # dict to save data
        dataset = {'case': [], 'name': [], 'ep': [], 'dir': [], 'points_impl': [], 'points_plan': [], 't': [], 'true_y0': [], 'true_y': []}

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
                ep = np.float32(ep)
                dir = np.float32(dir)
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
                t = torch.from_numpy(np.arange(num_points, dtype=np.float32)) * self.t_scale
                # init_true_y0 = torch.from_numpy(np.zeros(shape=(1, self.dof), dtype=np.float32))
                # init_true_y = torch.from_numpy(np.zeros(shape=(num_points, 1, self.dof), dtype=np.float32))
                true_y0 = torch.from_numpy(np.zeros(shape=(1, self.dof), dtype=np.float32))
                true_y = torch.from_numpy(np.zeros(shape=(num_points, 1, self.dof), dtype=np.float32))

                if self.dof == 1:
                    ''' 1dof '''
                    true_y0[0][0] = points_impl[0][self.component]
                    for p in range(num_points):
                        true_y[p][0][0] = points_impl[p][self.component]

                elif self.dof == 3:
                    ''' 3dof '''
                    for d in range(3):
                        true_y0[0][d] = points_impl[0][d]

                    for p in range(num_points):
                        for d in range(3):
                            true_y[p][0][d] = points_impl[p][d]

                elif self.dof == 8:
                    ''' yz '''
                    for d in range(2):
                        # if self.mode == 'init':
                        #     true_y0[0][d] = points_plan[0][d]
                        # else:
                        true_y0[0][d] = points_impl[0][d+1]
                    for d in range(2, 5):
                        true_y0[0][d] = ep[d - 3]
                    for d in range(5, 8):
                        true_y0[0][d] = dir[d - 6]

                    for p in range(num_points):
                        for d in range(2):
                            # if self.mode == 'init':
                            #     true_y[p][0][d] = points_plan[p][d]
                            # else:
                            true_y[p][0][d] = points_impl[p][d+1]
                        for d in range(2, 5):
                            true_y[p][0][d] = ep[d - 3]
                        for d in range(5, 8):
                            true_y[p][0][d] = dir[d - 6]

                elif self.dof == 9:
                    ''' xyz '''
                    for d in range(3):
                        # if self.mode == 'init':
                        #     true_y0[0][d] = points_plan[0][d]
                        # else:
                        true_y0[0][d] = points_impl[0][d]
                    for d in range(3, 6):
                        true_y0[0][d] = ep[d - 3]
                    for d in range(6, 9):
                        true_y0[0][d] = dir[d - 6]

                    for p in range(num_points):
                        for d in range(3):
                            # if self.mode == 'init':
                            #     true_y[p][0][d] = points_plan[p][d]
                            # else:
                            true_y[p][0][d] = points_impl[p][d]
                        for d in range(3, 6):
                            true_y[p][0][d] = ep[d - 3]
                        for d in range(6, 9):
                            true_y[p][0][d] = dir[d - 6]

                # save into training set
                dataset['case'].append(case)
                dataset['name'].append(impl['name'][j])
                dataset['ep'].append(ep)
                dataset['dir'].append(dir)
                dataset['points_impl'].append(points_impl)
                dataset['points_plan'].append(points_plan)
                dataset['t'].append(t)
                dataset['true_y0'].append(true_y0)
                dataset['true_y'].append(true_y)

        # print('ODEDataset:: {} set with {} cases / {} electrodes: '.format(self.mode, len(self.cases), len(dataset['case'])))
        print('ODEDataset:: with {} cases / {} electrodes: '.format(len(self.cases), len(dataset['case'])))

        return dataset
