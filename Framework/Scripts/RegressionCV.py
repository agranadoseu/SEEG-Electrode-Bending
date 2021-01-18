"""
Script to train, validate and test a regression model using cross validation

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import random
import pickle
import numpy as np
import torch
from optparse import OptionParser

from sklearn.model_selection import KFold

from Framework.Data import PlanGenerator
from Framework.Data import InputData
from Framework.Data import InputFeatures
from Framework.Data import ODEDataset
from Framework.Data import GPyDataset
from Framework.Data import NNDataset
from Framework.Data import WindowDataset
from Framework.Data import ODECnnDataset
from Framework.Regression import ODERegression
from Framework.Regression import GPyRegression
from Framework.Regression import NNRegression
from Framework.Regression import WindowRegression
from Framework.Regression import ODEWindowRegression


# parameters
# data_dir = 'C:\\UCL\\PhysicsSimulation\\Python\\NSElectrodeBending\\Framework\\Data'
# space = 'mni_aff'     # 'patient', 'mni_f3d'
# method = 'nn'      # cnn, nn, gpy, ode, odewindow, odecnn
# data_augment = False
# folds = 10
# labels = ['lu_x', 'lu_y', 'lu_z']
# labels = ['gu_x', 'gu_y', 'gu_z']
# labels = ['elec_dir_x', 'elec_dir_y', 'elec_dir_z']


class Options:
    data_dir = '..\\Data'
    space = 'mni_aff'
    method = 'cnn'
    data = 'tg'
    labels = 'gu'
    da = False
    folds = 10
    odedof = 3
    odecom = 0
    winsize = 9
    wintype = 'mri'
    mcdropout = True
    leave_out_test = 0.1
    niters = 200
    valfreq = 5
    lr = 0.001

options = Options()


def main():
    ''' Results '''
    datasets = {}
    cv_options = {}
    cv_results = {'fold': [],
                  'val_loss': [], 'best_val_iter': [],
                  'test_loss_standard': [],
                  'test_loss_mc_mean': [], 'test_loss_mc_std': [],
                  'file': [], 'train_idx': [], 'val_idx': [], 'options': []}

    ''' Random seed '''
    np.random.seed(0)

    ''' Parse arguments '''
    parser = OptionParser()
    parser.add_option("--datadir", dest="datadir", help="directory where all the data is stored", default='..\\Data', metavar="DATADIR")
    parser.add_option("--space", dest="space", help="space where the data is to be processed: ['patient', 'acpc', 'mni_aff', 'mni_f3d']", default='mni_aff', metavar="SPACE")
    parser.add_option("--method", dest="method", help="machine learning method to use for regression; ['nn', 'cnn', 'ode']", metavar="METHOD")
    parser.add_option("--data", dest="data", help="type of data to use as input: ['all', 'sfg', 'mfg', 'ifog', 'tg', 'apcg', 'po']", default='all', metavar="DATA")
    parser.add_option("--labels", dest="labels", help="labels to use for regression: ['lu', 'gu', 'vector']", metavar="LABELS")
    parser.add_option("--da", dest="da", help="data augment: ['yes', 'no']", default='no', metavar="DA")
    parser.add_option("--folds", dest="folds", help="number of folds for cross-validation", default=10, metavar="FOLDS")
    parser.add_option("--odedof", dest="odedof", help="ODE degrees of freedom: [1, 3]", default=3, metavar="ODEDOF")
    parser.add_option("--odecom", dest="odecom", help="ODE component to regress", default=0, metavar="ODECOM")
    parser.add_option("--winsize", dest="winsize", help="CNN window size", default=9, metavar="WINSIZE")
    parser.add_option("--wintype", dest="wintype", help="CNN window type: ['mri', 'gif', 'cwd']", default='mri', metavar="WINTYPE")
    parser.add_option("--mcdropout", dest="mcdropout", help="NN MC dropout: ['yes', 'no']", default='yes', metavar="MCDROPout")
    parser.add_option("--leaveouttest", dest="leaveouttest", help="Leave out test ratio", default=0.1, metavar="LEAVEOUTTEST")
    parser.add_option("--niters", dest="niters", help="number of iterations", default=100, metavar="NITERS")
    parser.add_option("--valfreq", dest="valfreq", help="iterations frequency of validation", default=5, metavar="VALFREQ")
    parser.add_option("--lr", dest="lr", help="learning rate", default=0.001, metavar="LR")
    # (options, args) = parser.parse_args()

    ''' Filter data based on region type '''
    plangen = PlanGenerator.PlanGenerator()
    if options.data == 'sfg':
        plangen.ep_superior_frontal_gyrus()
    elif options.data == 'mfg':
        plangen.ep_middle_frontal_gyrus()
    elif options.data == 'ifog':
        plangen.ep_inferior_frontal_orbital_gyrus()
    elif options.data == 'tg':
        plangen.ep_temporal_gyrus()
    elif options.data == 'apcg':
        plangen.ep_anterior_posterior_central_gyrus()
    elif options.data == 'po':
        plangen.ep_parietal_occipital()
    filter_file = plangen.filename
    filter_name = '' if filter_file is None else filter_file[0:filter_file.find('.')]

    ''' Labels '''
    labels = []
    if options.labels == 'lu':
        labels = ['lu_x', 'lu_y', 'lu_z']
    elif options.labels == 'gu':
        labels = ['gu_x', 'gu_y', 'gu_z']
    elif options.labels == 'vector':
        labels = ['elec_dir_x', 'elec_dir_y', 'elec_dir_z']

    ''' MC dropout '''
    options.da = True if options.da == 'yes' else False

    ''' MC dropout '''
    options.mcdropout = True if options.mcdropout == 'yes' else False

    ''' Input data '''
    data_file = 'data_' + options.space + '_window_norm.npy'
    features_data_file = 'features_' + options.space + '.pkl'
    input_data = InputData.InputData(directory=options.data_dir, file=data_file)
    input_features = InputFeatures.InputFeatures(directory=options.data_dir, file=features_data_file, labels=labels)
    print('ODEDataset:: {} cases in total loaded from {} with ids={}'.format(len(input_data.data['case']), data_file, input_data.data['case']))
    print('NNDataset:: {} cases in total loaded from {} with ids={}'.format(len(np.unique(input_features.df['case'])), features_data_file, np.unique(input_features.df['case'])))

    ''' Randomly assign testing set '''
    N = len(input_data.data['case'])
    cases = np.asarray(input_data.data['case'])
    indices = np.arange(N, dtype=np.int64)
    test_indices = np.random.choice(indices, int(N * options.leave_out_test), replace=False)
    test_indices = np.sort(test_indices)
    training_indices = np.setdiff1d(indices, test_indices)
    test_cases = cases[list(test_indices)]
    training_cases = cases[list(training_indices)]
    print('test_indices={}'.format(test_indices))
    print('test_cases = {}'.format(test_cases))
    # input('break')

    ''' K-fold '''
    kf = KFold(n_splits=options.folds)
    kf.get_n_splits(training_cases)
    cv_fold = 1
    timestamp = None
    model = None
    for train_idx, val_idx in kf.split(training_indices):
        # print('CV: fold={} train={} val={}'.format(cv_fold, training_indices[train_idx], training_indices[val_idx]))
        print('CV: fold={} train={:.2f}% val={:.2f}% test={:.2f}%'.format(cv_fold, len(train_idx)/N*100.0, len(val_idx)/N*100.0, len(test_indices)/N*100.0))
        # input('break')

        if options.method == 'ode':
            # create dataset
            datasets['training'] = ODEDataset.ODEDataset(data_dir=options.data_dir, data=input_data.data, space=options.space, cases=training_indices[train_idx], filter_file=filter_file, data_augment=options.da, dof=options.odedof, component=options.odecom, batch_time=0)
            datasets['validation'] = ODEDataset.ODEDataset(data_dir=options.data_dir, data=input_data.data, space=options.space, cases=training_indices[val_idx], filter_file=filter_file, data_augment=options.da, dof=options.odedof, component=options.odecom, batch_time=0)
            datasets['testing'] = ODEDataset.ODEDataset(data_dir=options.data_dir, data=input_data.data, space=options.space, cases=test_indices, filter_file=filter_file, data_augment=options.da, dof=options.odedof, component=options.odecom, batch_time=0)

            # create regression model
            model = ODERegression.ODERegression(dof=options.odedof)
            model.init_train(datasets=datasets, fold=cv_fold, niters=options.niters, valfreq=options.valfreq, lr=options.lr)

        if options.method == 'cnn':
            # create dataset
            datasets['training'] = WindowDataset.WindowDataset(data_dir=options.data_dir, data=input_data.data, space=options.space, cases=training_indices[train_idx], filter_file=filter_file, window_type=options.wintype, window_size=options.winsize, batch_time=0)
            datasets['validation'] = WindowDataset.WindowDataset(data_dir=options.data_dir, data=input_data.data, space=options.space, cases=training_indices[val_idx], filter_file=filter_file, window_type=options.wintype, window_size=options.winsize, batch_time=0)
            datasets['testing'] = WindowDataset.WindowDataset(data_dir=options.data_dir, data=input_data.data, space=options.space, cases=test_indices, filter_file=filter_file, window_type=options.wintype, window_size=options.winsize, batch_time=0)

            # labels
            datasets['training'].labels = labels

            # create regression model
            # model = WindowRegression.WindowRegression(datasets=datasets, channels=4)  # one-hot vector
            model = WindowRegression.WindowRegression(channels=1, window_size=options.winsize, labels=options.labels)
            model.init_train(datasets=datasets, fold=cv_fold, mcdropout=options.mcdropout, niters=options.niters, valfreq=options.valfreq, lr=options.lr)

        if options.method == 'nn':
            # create dataset from features
            datasets['training'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=training_cases[train_idx], filter_file=filter_file, data_augment=options.da, batch_time=0)
            datasets['validation'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=training_cases[val_idx], filter_file=filter_file, data_augment=options.da, batch_time=0)
            datasets['testing'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=test_cases, filter_file=filter_file, data_augment=options.da, batch_time=0)

            # normalisation of data
            scaler_file = 'scaler_'+options.space+'_'+filter_name+'_f'+str(cv_fold)+'.joblib' if filter_file is not None else 'scaler_'+options.space+'_f'+str(cv_fold)+'.joblib'
            datasets['training'].create_scaler(file=scaler_file)
            datasets['validation'].scaler = datasets['training'].scaler
            datasets['testing'].scaler = datasets['training'].scaler
            datasets['training'].normalise()
            datasets['validation'].normalise()
            datasets['testing'].normalise()

            # TODO features selection

            # create regression model
            model = NNRegression.NNRegression(num_features=datasets['training'].feature_sz, num_outputs=datasets['training'].output_sz)
            model.init_train(datasets=datasets, fold=cv_fold, mcdropout=options.mcdropout, niters=options.niters, valfreq=options.valfreq, lr=options.lr)

        if cv_fold > 1:
            model.timestamp = timestamp

        # train
        try:
            model.train()
        except AssertionError as error:
            print('assertion error found ...\n', error)

        # test
        model.test()

        # save
        cv_results['fold'].append(cv_fold)
        cv_results['val_loss'].append(model.best_val_loss)
        cv_results['best_val_iter'].append(model.best_val_itr)
        cv_results['test_loss_standard'].append(model.test_loss)
        if options.method == 'nn' or options.method == 'cnn':
            cv_results['test_loss_mc_mean'].append(model.test_loss_mc_mean)
            cv_results['test_loss_mc_std'].append(model.test_loss_mc_std)
        else:
            cv_results['test_loss_mc_mean'].append(0.0)
            cv_results['test_loss_mc_std'].append(0.0)
        cv_results['file'].append('f'+str(cv_fold)+'-i'+str(model.best_val_itr)+'.pth')
        cv_results['train_idx'].append(train_idx)
        cv_results['val_idx'].append(val_idx)

        if cv_fold == 1:
            timestamp = model.timestamp
        cv_fold += 1

    ''' Results '''
    print("\n\n\nResults")
    print("=======================================")
    print('fold     iter     val        test    mc_mean     mc_std')
    for i in range(len(cv_results['fold'])):
        print('{:02d} | {:04d} | {:.6f} | {:.6f} | {:.6f} | {:.6f}'.format(cv_results['fold'][i], cv_results['best_val_iter'][i], cv_results['val_loss'][i], cv_results['test_loss_standard'][i], cv_results['test_loss_mc_mean'][i], cv_results['test_loss_mc_std'][i]))
    print('     Mean     {:.6f} | {:.6f} | {:.6f} | {:.6f}'.format(np.mean(cv_results['val_loss']), np.mean(cv_results['test_loss_standard']), np.mean(cv_results['test_loss_mc_mean']), np.mean(cv_results['test_loss_mc_std'])))
    print('     Std      {:.6f} | {:.6f} | {:.6f} | {:.6f}'.format(np.std(cv_results['val_loss']), np.std(cv_results['test_loss_standard']), np.std(cv_results['test_loss_mc_mean']), np.std(cv_results['test_loss_mc_std'])))

    cv_options['space'] = options.space
    cv_options['method'] = options.method
    cv_options['data'] = options.data
    cv_options['labels'] = options.labels
    cv_options['folds'] = options.folds
    cv_options['odedof'] = options.odedof
    cv_options['odecom'] = options.odecom
    cv_options['winsize'] = options.winsize
    cv_options['wintype'] = options.wintype
    cv_options['mcdropout'] = options.mcdropout
    cv_options['lot'] = options.leave_out_test
    cv_options['niters'] = options.niters
    cv_options['valfreq'] = options.valfreq
    cv_options['lr'] = options.lr
    cv_options['training_indices'] = training_indices
    cv_options['training_cases'] = training_cases
    cv_options['test_indices'] = test_indices
    cv_options['test_cases'] = test_cases
    cv_results['options'] = cv_options

    data_pkl_file = os.path.join(model.checkpoint_dir, 'cv_results.pkl')
    pickle_file = open(data_pkl_file, "wb")
    pickle.dump(cv_results, pickle_file)
    pickle_file.close()

if __name__ == '__main__':
    main()