"""
Script to train, validate and test a regression model

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import torch

from Framework.Data import ODEDataset
from Framework.Data import GPyDataset
from Framework.Data import WindowDataset
from Framework.Data import ODECnnDataset
from Framework.Regression import ODERegression
from Framework.Regression import GPyRegression
from Framework.Regression import NNRegression
from Framework.Regression import ODEWindowRegression


data_dir = 'C:\\UCL\\PhysicsSimulation\\Python\\NSElectrodeBending\\Framework\\Data'
space = 'mni_aff'     # 'patient', 'mni_f3d'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
method = 'odewindow'      # cnn, nn, gpy, ode, odewindow, odecnn
data_augment = False


def main():
    datasets = {}

    if method == 'ode':
        # create dataset
        datasets['init'] = ODEDataset.ODEDataset(data_dir=data_dir, space=space, mode='init', filter_file='filter_mtg.npy', data_augment=data_augment, batch_time=8)
        datasets['training'] = ODEDataset.ODEDataset(data_dir=data_dir, space=space, mode='training', filter_file='filter_mtg.npy', data_augment=data_augment, batch_time=8)
        datasets['validation'] = ODEDataset.ODEDataset(data_dir=data_dir, space=space, mode='validation', filter_file='filter_mtg.npy', data_augment=data_augment, batch_time=8)
        datasets['testing'] = ODEDataset.ODEDataset(data_dir=data_dir, space=space, mode='testing', filter_file='filter_mtg.npy', data_augment=data_augment, batch_time=0)

        input('break')

        # create regression model
        odemodel = ODERegression.ODERegression()
        odemodel.init_train(datasets=datasets)

        # train
        odemodel.train()

        # test
        odemodel.test()

    elif method == 'gpy':
        # create dataset
        datasets['init'] = GPyDataset.GPyDataset(data_dir=data_dir, space=space, mode='init', filter_file='filter_mtg.npy')
        datasets['training'] = GPyDataset.GPyDataset(data_dir=data_dir, space=space, mode='training', filter_file='filter_mtg.npy')
        datasets['testing'] = GPyDataset.GPyDataset(data_dir=data_dir, space=space, mode='testing', filter_file='filter_mtg.npy')
        datasets['validation'] = GPyDataset.GPyDataset(data_dir=data_dir, space=space, mode='validation', filter_file='filter_mtg.npy')

        # create regression model
        gpymodel = GPyRegression.GPyRegression(datasets=datasets)

        # train
        gpymodel.train()

        # test
        gpymodel.test()

    elif method == 'cnn':
        # create dataset
        datasets['init'] = WindowDataset.WindowDataset(data_dir=data_dir, space=space, mode='init', filter_file='filter_mtg.npy', type='cwd')
        datasets['training'] = WindowDataset.WindowDataset(data_dir=data_dir, space=space, mode='training', filter_file='filter_mtg.npy', type='cwd')
        datasets['testing'] = WindowDataset.WindowDataset(data_dir=data_dir, space=space, mode='testing', filter_file='filter_mtg.npy', type='cwd')
        datasets['validation'] = WindowDataset.WindowDataset(data_dir=data_dir, space=space, mode='validation', filter_file='filter_mtg.npy', type='cwd')

        # create regression model
        # cnnmodel = NNRegression.NNRegression(datasets=datasets, channels=1)     # mri, cwd
        cnnmodel = NNRegression.NNRegression(datasets=datasets, channels=4)       # one-hot vector

        # train
        cnnmodel.train()

        # test
        cnnmodel.test()

    elif method == 'odewindow':
        # create dataset
        datasets['init'] = ODECnnDataset.ODECnnDataset(data_dir=data_dir, space=space, mode='init', filter_file='filter_mtg.npy', type='mri')
        datasets['training'] = ODECnnDataset.ODECnnDataset(data_dir=data_dir, space=space, mode='training', filter_file='filter_mtg.npy', type='mri')
        datasets['testing'] = ODECnnDataset.ODECnnDataset(data_dir=data_dir, space=space, mode='testing', filter_file='filter_mtg.npy', type='mri')
        datasets['validation'] = ODECnnDataset.ODECnnDataset(data_dir=data_dir, space=space, mode='validation', filter_file='filter_mtg.npy', type='mri')

        # create regression model
        cnnwindow = ODEWindowRegression.ODEWindowRegression(datasets=datasets)     # mri, cwd
        # cnnwindow = ODEWindowRegression.ODEWindowRegression(datasets=datasets, channels=4)       # one-hot vector

        # train
        cnnwindow.train()

        # test
        cnnwindow.test()


if __name__ == '__main__':
    main()