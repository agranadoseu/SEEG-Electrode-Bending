"""
Dataset including windows

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


class WindowDataset(data.Dataset):
    # parameters
    batch_time = 8
    t_scale = 0.01
    labels = []

    def __init__(self, data_dir=None, data=None, space=None, filter_file=None, mode=None, cases=None, data_augment=False, window_type=None, window_size=0, batch_time=0):
        self.data_dir = data_dir
        self.space = space
        self.filter_file = filter_file
        self.mode = mode
        self.data_augment = data_augment
        self.window_type = window_type
        self.window_size = window_size
        self.batch_time = batch_time

        # load data
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

        # create dataset (mri, gif, cwd)
        self.dataset = self.create(window_type=self.window_type, window_size=self.window_size)

    def __len__(self):
        """ Defines the total number of samples """
        return len(self.dataset['case'])

    def __getitem__(self, index):
        """ Generates one sample per point """
        case = self.dataset['case'][index]
        name = self.dataset['name'][index]
        interpolation = self.dataset['interpolation'][index]
        impl = self.dataset['impl'][index]
        plan_next = self.dataset['plan_next'][index]
        plan = self.dataset['plan'][index]
        impl_next = self.dataset['impl_next'][index]
        vector = self.dataset['vector'][index]
        direction = self.dataset['direction'][index]
        window = self.dataset['window'][index]
        lu = self.dataset['lu'][index]
        gu = self.dataset['gu'][index]
        gu_next = self.dataset['gu_next'][index]

        # # convert to tensors
        # direction = torch.from_numpy(direction.copy())
        # window = torch.from_numpy(window.copy())
        # lu = torch.from_numpy(lu.copy())
        # gu = torch.from_numpy(gu.copy())
        # vector = torch.from_numpy(vector.copy())
        # impl = torch.from_numpy(impl.copy())
        # impl_next = torch.from_numpy(impl_next.copy())
        # plan_next = torch.from_numpy(plan_next.copy())

        # print('WindowSataset :: case={} name={} interpolation[{}] impl=[{}] plan_next[{}] impl_next[{}] vector[{}] direction[{}] window[{}] lu[{}] gu[{}]'.format(
        #         case, name, interpolation.shape, impl.shape, plan_next.shape, impl_next.shape, vector.shape, direction.shape, window.shape, lu.shape, gu.shape))

        """ Generates one sample per electrode """
        return direction, window, lu, gu, gu_next, vector, impl, impl_next, plan, plan_next, case, name, interpolation

    def load(self):
        if not self.data_augment:
            data_file = 'data_' + self.space + '_window_norm.npy'
        else:
            data_file = 'data_' + self.space + '_window_norm_da.npy'

        pickle_file = open(os.path.join(self.data_dir, data_file), "rb")
        data = pickle.load(pickle_file)
        pickle_file.close()

        print('CNNDataset:: loaded data from {} with cases={}'.format(data_file, data['case']))

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
        print('CNNDataset:: split cases for mode={} with remaining mode_cases={}, id_cases={}'.format(self.mode, mode_cases, id_cases))

        return id_cases

    def filter(self):
        # filter_file example: 'filter_mtg.npy'
        pickle_file = open(os.path.join(self.data_dir, self.filter_file), "rb")
        filter_dict = pickle.load(pickle_file)
        pickle_file.close()

        filter_cases = np.where(np.isin(self.data['case'], list(filter_dict.keys())))[0]
        overlap = np.where(np.isin(self.cases, filter_cases))
        filter_cases = self.cases[overlap[0]]
        print('CNNDataset:: filter loaded with keys={} filter={} resulted in filter_cases={}'.format(list(filter_dict.keys()), filter_dict, filter_cases))

        return filter_dict, filter_cases

    def create(self, window_type=None, window_size=None):
        # dict to save data
        # dataset = {'case': [], 'name': [], 'points': [], 't': [], 'true_y0': [], 'true_y': []}
        dataset = {'case': [], 'name': [], 'ep': [], 'tp': [], 'ep_gif': [], 'tp_gif': [], 'contacts': [], 'interpolation': [], 'impl': [], 'impl_next': [], 'plan': [], 'plan_next': [], 'vector': [], 'direction': [], 'window': [], 'lu': [], 'gu': [], 'gu_next': []}

        window_size_feature = ''
        if window_size == 9:
            window_size_feature = 'window9'
        elif window_size == 11:
            window_size_feature = 'window11'

        # iterate through cases
        num_electrodes = 0
        for i in self.cases:
            case = self.data['case'][i]
            ep_gif = self.data['ep'][i]
            tp_gif = self.data['tp'][i]
            plan = self.data['plan'][i]
            impl = self.data['impl'][i]
            delta_l = self.data['local_delta'][i]
            delta_g = self.data['delta'][i]
            window_l = self.data[window_size_feature][i]

            # print('     case={}'.format(case))
            # print('     ep_gif={}'.format(ep_gif))
            # print('     tp_gif={}'.format(tp_gif))
            # print('     plan[{}]={}'.format(plan.keys(), plan))

            # by default all electrodes unless filtered
            electrodes = np.arange(len(plan['name']))
            if self.filter_file is not None:
                electrodes = np.where(np.isin(plan['id'], self.filter_dict[case]))[0]

            # iterate through electrodes
            for j in electrodes:
                if len(plan['points'][j]) == 0:
                    continue

                num_electrodes += 1
                print('         electrode={}'.format(plan['name'][j]))

                # ensure points go from EP towards TP
                ep = plan['ep'][j]
                tp = plan['tp'][j]
                num_contacts = plan['num_contacts'][j]
                points_plan = np.flip(plan['points'][j], 0)
                points_impl = np.flip(impl['points'][j], 0)
                window = np.flip(window_l[window_type][j], 0)
                # print('window[{}]'.format(window.shape))
                if not self.data_augment:
                    show_delta = np.where(np.isin(delta_l['name'], impl['name'][j]))[0][0]
                    v_dir = np.flip(delta_l['v_dir'][show_delta], 0)
                    lu_dir = np.flip(delta_l['dir'][show_delta], 0)
                    lu_du = np.flip(delta_l['du'][show_delta], 0)
                    lu = np.flip(delta_l['u'][show_delta], 0)
                    for k in range(len(lu_du)):
                        lu[k] *= lu_du[k]
                    gu = np.flip(delta_g['u'][show_delta], 0)
                #print('    points_plan={} points_impl={} window={} lu_dir={} lu_du={} lu={} gu={}: '.format(points_plan.shape, points_impl.shape, window.shape, lu_dir.shape, lu_du.shape, lu.shape, gu.shape))
                #print('    points_plan={} points_impl={} window={} lu_dir={} lu_du={} lu={} gu={}: '.format(points_plan.dtype, points_impl.dtype, window.dtype, lu_dir.dtype, lu_du.dtype, lu.dtype, gu.dtype))

                # # POINTS
                # num_points = len(points_plan)
                # # for p in range(num_points):
                # for p in range(num_points-1):   # all but TP since we don't have local displacement information
                #
                #     wdata = window[p]
                #     if window_type == 'mri' or window_type == 'gif':
                #         # 1 channel window
                #         wdata = wdata.reshape((wdata.shape[0], wdata.shape[1], wdata.shape[2], 1))
                #         print('     wdata[{}]'.format(wdata.shape))
                #     elif window_type == 'cwd':
                #         # 4-channel window (1-hot vector)
                #         # channels = np.zeros((wdata.shape[0],wdata.shape[1],wdata.shape[2],4), dtype=np.float32)
                #         # channels[wdata == 0, :] = [1, 0, 0, 0]
                #         # channels[wdata == 1, :] = [0, 1, 0, 0]
                #         # channels[wdata == 2, :] = [0, 0, 1, 0]
                #         # channels[wdata == 3, :] = [0, 0, 0, 1]
                #         # wdata = channels
                #
                #         # 1-channel window (normalise)
                #         wdata /= 4
                #         wdata = wdata.reshape((wdata.shape[0], wdata.shape[1], wdata.shape[2], 1))
                #
                #     dataset['case'].append(case)
                #     dataset['name'].append(impl['name'][j])
                #     dataset['interpolation'].append(num_points - 1 - p)
                #     dataset['impl'].append(points_impl[p])          # centre of window
                #     dataset['impl_next'].append(points_impl[p+1])   # save next where it should get closer to
                #     dataset['plan_next'].append(points_plan[p+1])   # save next
                #     dataset['vector'].append(v_dir[p])
                #     dataset['direction'].append(lu_dir[p])
                #     dataset['window'].append(wdata)
                #     dataset['lu'].append(lu[p])
                #     dataset['gu'].append(gu[p])

                # ELECTRODES
                window = window.reshape((window.shape[0], 1, window.shape[1], window.shape[2], window.shape[3]))
                interpolation = np.flip(np.arange(points_impl.shape[0])[1:], 0)
                interpolation = np.reshape(interpolation, (interpolation.shape[0], 1))
                dataset['case'].append(case)
                dataset['name'].append(impl['name'][j])
                dataset['ep'].append(ep)
                dataset['tp'].append(tp)
                dataset['ep_gif'].append(ep_gif[j])
                dataset['tp_gif'].append(tp_gif[j])
                dataset['contacts'].append(num_contacts)
                dataset['interpolation'].append(interpolation)
                dataset['impl'].append(points_impl[0:-1])  # centre of window
                dataset['impl_next'].append(points_impl[1:])  # save next where it should get closer to
                dataset['plan'].append(points_plan[0:-1])  # centre of window
                dataset['plan_next'].append(points_plan[1:])  # save next
                dataset['vector'].append(v_dir[0:-1])
                dataset['direction'].append(lu_dir[0:-1])
                dataset['window'].append(window[0:-1])
                dataset['lu'].append(lu[0:-1])
                dataset['gu'].append(gu[0:-1])
                dataset['gu_next'].append(gu[1:])    # next gu to predict

        # print('WindowDataset:: {} set with {} cases / {} electrodes / {} points: '.format(self.mode, len(self.cases), num_electrodes, len(dataset['case'])))
        print('WindowDataset:: {} set with {} cases / {}=={} electrodes: '.format(self.mode, len(self.cases), num_electrodes, len(dataset['case'])))

        return dataset

    def database_backup(self):
        self.backup = self.dataset

    def database_restore(self):
        self.dataset = self.backup

    def get_names(self):
        return self.dataset['name']

    def database_by_electrode(self, name=None):
        ''' points go from EP to TP '''
        electrode = {'case': [], 'name': [], 'ep': [], 'tp': [], 'ep_gif': [], 'tp_gif': [], 'contacts': [], 'interpolation': [], 'impl': [], 'impl_next': [], 'plan': [], 'plan_next': [], 'vector': [], 'direction': [], 'window': [], 'lu': [], 'gu': [], 'gu_next': []}

        # find index by value
        idx = self.dataset['name'].index(name)

        # add values to dictionary
        electrode['case'].append(self.dataset['case'][idx])
        electrode['name'].append(self.dataset['name'][idx])
        electrode['ep'].append(self.dataset['ep'][idx])
        electrode['tp'].append(self.dataset['tp'][idx])
        electrode['ep_gif'].append(self.dataset['ep_gif'][idx])
        electrode['tp_gif'].append(self.dataset['tp_gif'][idx])
        electrode['contacts'].append(self.dataset['contacts'][idx])
        electrode['interpolation'].append(self.dataset['interpolation'][idx])
        electrode['impl'].append(self.dataset['impl'][idx])
        electrode['impl_next'].append(self.dataset['impl_next'][idx])
        electrode['plan'].append(self.dataset['plan'][idx])
        electrode['plan_next'].append(self.dataset['plan_next'][idx])
        electrode['vector'].append(self.dataset['vector'][idx])
        electrode['direction'].append(self.dataset['direction'][idx])
        electrode['window'].append(self.dataset['window'][idx])
        electrode['lu'].append(self.dataset['lu'][idx])
        electrode['gu'].append(self.dataset['gu'][idx])
        electrode['gu_next'].append(self.dataset['gu_next'][idx])

        # replace dataset with electrode
        self.dataset = electrode

        return self.dataset['ep_gif'][0], self.dataset['tp_gif'][0],\
               self.dataset['ep'][0], self.dataset['tp'][0],\
               self.dataset['contacts'][0], \
               np.vstack((self.dataset['plan'][0], self.dataset['plan_next'][0][-1, :])), \
               np.vstack((self.dataset['impl'][0], self.dataset['impl_next'][0][-1, :]))
