"""
Script to load a regression model and predict electrode trajectory in patient space

- load a model or load best models of each fold to create ensemble  [DONE]
- run inference using test set (MNI space)                          [DONE]
- compute distance metric from ground truth (MNI space)             [DONE]
- visualise in MNI space
    * plot plan and impl
    * plot predictions
- transform electrode from MNI space to patient space
- compute distance metric from ground truth (patient space)
- visualise in patient space

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

from Framework.Data import ODEDataset
from Framework.Regression import ODERegression

# parameters
data_dir = 'C:\\UCL\\PhysicsSimulation\\Python\\NSElectrodeBending\\Framework\\Data'
data_augment = False
space = 'mni_aff'     # 'patient', 'mni_f3d'
method = 'ode'      # cnn, nn, gpy, ode, odewindow, odecnn
timestamp = '20200914-1616'
file_models = ['f1-i300.pth', 'f2-i300.pth', 'f3-i300.pth', 'f4-i300.pth', 'f5-i300.pth']
test_indices = np.asarray([2, 13, 30, 41, 43, 45, 53, 74], dtype=np.int64)
test_cases = ['R03', 'R14', 'R31', 'T11', 'T13', 'T16', 'T25', 'P10']


def main():
    datasets = {}

    # load data
    data_file = 'data_' + space + '.npy' if not data_augment else 'data_' + space + '_da.npy'
    pickle_file = open(os.path.join(data_dir, data_file), "rb")
    data = pickle.load(pickle_file)
    pickle_file.close()
    print('ODEDataset:: {} cases in total loaded from {} with ids={}'.format(len(data['case']), data_file, data['case']))

    if method == 'ode':

        for i in range(len(test_cases)):
            case = test_cases[i]
            idx = test_indices[i]
            print("Predicting trajectories for case={} (id={}:{})".format(case, idx, idx.dtype))

            # create dataset
            #datasets['testing'] = ODEDataset.ODEDataset(data_dir=data_dir, data=data, space=space, cases=test_indices, filter_file='filter_mtg.npy', data_augment=data_augment, batch_time=0)
            datasets['testing'] = ODEDataset.ODEDataset(data_dir=data_dir, data=data, space=space, cases=np.asarray([idx]), filter_file='filter_mtg.npy', data_augment=data_augment, batch_time=0)
            print('dataset size = {}'.format(datasets['testing'].__len__()))
            if datasets['testing'].__len__() == 0:
                print('no electrode were found for inference')
                continue

            for file in file_models:
                # create regression model
                odemodel = ODERegression.ODERegression()
                odemodel.init_test(datasets=datasets)
                odemodel.load_state(timestamp=timestamp, filename=file)

                # test
                pred = odemodel.infer()
                print('Prediction:')
                for e in range(len(pred['name'])):
                    print('case={} name={}\npoints({})={}'.format(pred['case'][e], pred['name'][e], pred['points'][e].size(), pred['points'][e]))

                # print('model={} test_loss={}'.format(file, odemodel.test_loss))

if __name__ == '__main__':
    main()