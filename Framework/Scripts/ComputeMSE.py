"""
Once trajectories are inferred this script computes MSE

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
import pandas as pd
import torch
from optparse import OptionParser

from Framework.Data import PlanGenerator
from Framework.Data import InputData
from Framework.Data import InputFeatures
from Framework.Data import NNDataset
from Framework.Data import WindowDataset

from Framework.Regression import NNRegression
from Framework.Regression import WindowRegression

from Framework.Inference import DataGeneration

from Framework.Tools import XMLParser


class Options:
    data_dir = '..\\Data'
    models_dir = '.\\models'
    space = 'mni_aff'
    method = 'cnn'
    labels = 'lu'
    winsize = 9
    wintype = 'cwd'
    mcdropout = True

options = Options()


def main():
    datasets = {}

    directory = 'E:\\UCL\\Backup\\medic-biomechanics\\Datasets\\BendingJournal\\datasets'
    xmlparser = XMLParser.XMLParser()

    ''' Parse arguments '''
    parser = OptionParser()
    parser.add_option("--datadir", dest="datadir", help="directory where all the data is stored", default='..\\Data', metavar="DATADIR")
    parser.add_option("--modelsdir", dest="modelsdir", help="directory where all the trained models are stored", default='.\\models', metavar="MODELSDIR")
    parser.add_option("--space", dest="space", help="space where the data is to be processed: ['patient', 'acpc', 'mni_aff', 'mni_f3d']", default='mni_aff', metavar="SPACE")
    parser.add_option("--method", dest="method", help="machine learning method to use for regression; ['nn', 'cnn', 'ode']", metavar="METHOD")
    parser.add_option("--labels", dest="labels", help="labels to use for regression: ['lu', 'gu', 'vector']", metavar="LABELS")
    parser.add_option("--winsize", dest="winsize", help="CNN window size", default=9, metavar="WINSIZE")
    parser.add_option("--wintype", dest="wintype", help="CNN window type: ['mri', 'gif', 'cwd']", default='mri', metavar="WINTYPE")
    parser.add_option("--mcdropout", dest="mcdropout", help="NN MC dropout: ['yes', 'no']", default='yes', metavar="MCDROPout")
    # (options, args) = parser.parse_args()

    ''' test cases chosen at random (seed 0) by DGX server '''
    test_indices = [2, 13, 30, 41, 43, 45, 53, 74]
    test_cases = ['R03', 'R14', 'R31', 'T11', 'T13', 'T16', 'T25', 'P10']

    ''' MSE loss per fold '''
    loss = {'case': [], 'name': [], 'points': [],
            'f1': [], 'f2': [], 'f3': [], 'f4': [], 'f5': [],
            'f6': [], 'f7': [], 'f8': [], 'f9': [], 'f10': []}
    results = {'sfg': loss, 'mfg': loss, 'ifog': loss, 'tg': loss, 'apcg': loss, 'po': loss}
    mse_col_df = pd.DataFrame(columns=['method', 'label', 'region', 'fold', 'case', 'elec', 'points', 'mse_gt', 'mse_plan', 'mse_impl'])
    mse_row_df = pd.DataFrame(columns=['method', 'label', 'region', 'fold', 'case', 'elec', 'points', 'type', 'mse'])

    ''' model dictionary '''
    folds = {}
    if options.method == 'nn':
        if options.labels == 'lu':
            folds = {'sfg': [55, 80, 20, 25, 70, 5, 35, 45, 75, 10],
                     'mfg': [100, 65, 10, 90, 95, 95, 95, 100, 100, 55],
                     'ifog': [95, 100, 75, 70, 85, 35, 60, 95, 95, 90],
                     'tg': [95, 70, 90, 15, 60, 95, 50, 30, 70, 65],
                     'apcg': [100, 100, 100, 90, 50, 100, 95, 95, 50, 75],
                     'po': [75, 100, 25, 95, 95, 100, 15, 90, 100, 60]}
        elif options.labels == 'vector':
            folds = {'sfg': [200, 15, 35, 140, 190, 150, 105, 75, 95, 25],
                     'mfg': [95, 150, 40, 195, 95, 85, 155, 200, 200, 200],
                     'ifog': [115, 40, 85, 5, 40, 170, 50, 100, 35, 140],
                     'tg': [80, 10, 100, 50, 110, 10, 75, 60, 135, 40],
                     'apcg': [105, 105, 105, 140, 135, 115, 115, 20, 140, 125],
                     'po': [140, 145, 105, 90, 85, 80, 125, 25, 120, 80]}
    elif options.method == 'cnn':
        if options.wintype == 'mri':
            if options.labels == 'lu':
                folds = {'sfg': [20, 45, 125, 10, 185, 200, 15, 10, 20, 20],
                         'mfg': [60, 85, 45, 195, 15, 195, 20, 200, 195, 15],
                         'ifog': [190, 190, 160, 195, 200, 130, 175, 200, 200, 195],
                         'tg': [30, 185, 70, 165, 150, 160, 200, 190, 200, 180],
                         'apcg': [175, 115, 170, 130, 50, 195, 140, 185, 175, 180],
                         'po': [115, 125, 190, 90, 100, 185, 45, 155, 195, 1]}
            elif options.labels == 'vector':
                folds = {'sfg': [160, 150, 185, 170, 175, 155, 195, 170, 190, 195],
                         'mfg': [200, 120, 195, 200, 125, 160, 200, 200, 180, 170],
                         'ifog': [185, 140, 200, 130, 155, 190, 35, 95, 180, 190],
                         'tg': [195, 195, 140, 195, 200, 200, 90, 150, 200, 180],
                         'apcg': [175, 180, 25, 180, 185, 185, 110, 175, 195, 190],
                         'po': [180, 195, 145, 140, 190, 165, 190, 180, 175, 190]}
        elif options.wintype == 'gif':
            if options.labels == 'lu':
                folds = {'sfg': [170, 165, 195, 200, 200, 200, 190, 200, 170, 200],
                         'mfg': [60, 185, 200, 180, 200, 200, 190, 200, 200, 195],
                         'ifog': [140, 175, 115, 200, 165, 190, 200, 110, 195, 200],
                         'tg': [105, 145, 195, 130, 170, 200, 200, 180, 135],
                         'apcg': [135, 110, 25, 190, 200, 150, 175, 165, 175, 155],
                         'po': [40, 190, 125, 165, 200, 95, 180, 190, 165, 65]}
            elif options.labels == 'vector':
                folds = {'sfg': [190, 10, 175, 200, 200, 200, 190, 90, 160, 40],
                         'mfg': [200, 200, 200, 110, 195, 200, 45, 185, 190, 200],
                         'ifog': [125, 180, 55, 35, 200, 200, 20, 25, 40, 5],
                         'tg': [165, 190, 135, 55, 185, 190, 140, 200, 175, 200],
                         'apcg': [145, 180, 35, 155, 40, 190, 190, 60, 45, 20],
                         'po': [140, 200, 5, 195, 100, 195, 170, 15, 165, 180]}
        elif options.wintype == 'cwd':
            if options.labels == 'lu':
                folds = {'sfg': [130, 80, 190, 155, 195, 195, 200, 195, 20, 200],
                         'mfg': [175, 40, 195, 30, 185, 190, 200, 200, 195, 200],
                         'ifog': [185, 155, 200, 170, 200, 200, 155, 25, 200, 195],
                         'tg': [95, 110, 125, 70, 125, 180, 10, 190, 195, 200],
                         'apcg': [155, 65, 125, 135, 190, 195, 195, 155, 150, 195],
                         'po': [15, 160, 135, 75, 30, 95, 165, 110, 170, 70]}
            elif options.labels == 'vector':
                folds = {'sfg': [190, 135, 135, 190, 200, 120, 190, 200, 120, 165],
                         'mfg': [145, 185, 170, 140, 185, 200, 195, 190, 170, 175],
                         'ifog': [185, 10, 150, 170, 200, 185, 155, 135, 35, 160],
                         'tg': [195, 190, 145, 190, 115, 200, 65, 190, 5, 190],
                         'apcg': [115, 145, 160, 200, 140, 165, 190, 130, 140, 200],
                         'po': [200, 185, 170, 195, 185, 190, 190, 5, 135, 145]}

    ''' Labels '''
    labels = []
    colour = np.array([1., 1., 1.])
    if options.labels == 'lu':
        labels = ['lu_x', 'lu_y', 'lu_z']
        colour = np.array([.5, .5, 1.])
    elif options.labels == 'gu':
        labels = ['gu_x', 'gu_y', 'gu_z']
    elif options.labels == 'vector':
        labels = ['elec_dir_x', 'elec_dir_y', 'elec_dir_z']
        colour = np.array([1., .5, 1.])

    ''' Input data '''
    data_file = 'data_' + options.space + '_window_norm.npy'
    features_data_file = 'features_' + options.space + '.pkl'
    input_data = InputData.InputData(directory=options.data_dir, file=data_file)
    input_features = InputFeatures.InputFeatures(directory=options.data_dir, file=features_data_file, labels=labels)
    print('ODEDataset:: {} cases in total loaded from {} with ids={}'.format(len(input_data.data['case']), data_file,
                                                                             input_data.data['case']))
    print('NNDataset:: {} cases in total loaded from {} with ids={}'.format(len(np.unique(input_features.df['case'])),
                                                                            features_data_file,
                                                                            np.unique(input_features.df['case'])))

    ''' plan generator to find type of data region to use '''
    plangen = PlanGenerator.PlanGenerator()

    ''' text for filenames '''
    method_name = options.method if options.method == 'nn' else options.wintype
    uncertainty = 'mc' if options.mcdropout else 'std'

    ''' iterate through test cases '''
    for c in range(len(test_cases)):
        case = test_cases[c]
        idx = test_indices[c]
        case_dir = os.path.join(directory, case)
        mni_dir = os.path.join(case_dir, 'mniE')
        pred_dir = os.path.join(mni_dir, 'pred')
        print("Predicting trajectories for case={} (id={})".format(case, idx))

        # ground truth and regression model
        if options.method == 'cnn':
            # dataset
            datasets['testing'] = WindowDataset.WindowDataset(data_dir=options.data_dir, data=input_data.data,
                                                              space=options.space, cases=np.asarray([idx]),
                                                              filter_file=None, window_type=options.wintype,
                                                              window_size=options.winsize, batch_time=0)

            # labels
            datasets['testing'].labels = labels

            # regression model
            model = WindowRegression.WindowRegression(channels=1, window_size=options.winsize, labels=options.labels)

        elif options.method == 'nn':
            # dataset
            datasets['testing'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features,
                                                      cases=[case], filter_file=None,
                                                      data_augment=False, batch_time=0)

            # regression model
            model = NNRegression.NNRegression(num_features=datasets['testing'].feature_sz, num_outputs=datasets['testing'].output_sz)

        # initialise model for testing
        model.init_loss_function()

        # iterate through electrodes
        datasets['testing'].database_backup()
        electrodes = datasets['testing'].get_names()
        for e in range(len(electrodes)):
            name = electrodes[e]

            # ground truth: dataset containing only one electrode
            ep_gif, tp_gif, ep, tp, contacts, plan, impl = datasets['testing'].database_by_electrode(name=name)

            # distance between interpolated points
            p_dist = np.linalg.norm(plan[1:,:] - plan[0:-1,:], axis=1)
            p_dist_mean = np.mean(p_dist)
            p_dist_std = np.mean(p_dist)
            print('     p distances: {}({}) = {}'.format(p_dist_mean, p_dist_std, p_dist))
            i_dist = np.linalg.norm(impl[1:, :] - impl[0:-1, :], axis=1)
            i_dist_mean = np.mean(i_dist)
            i_dist_std = np.mean(i_dist)
            print('     i distances: {}({}) = {}'.format(i_dist_mean, i_dist_std, i_dist))

            # select data group
            type = plangen.get_type(ep=ep_gif, tp=tp_gif)

            results[type]['case'].append(case)
            results[type]['name'].append(name)
            results[type]['points'].append(len(plan))

            # compute loss
            mse_gt = model.compute_loss(truth=plan, pred=impl)
            mse_row_df = mse_row_df.append({'method':method_name, 'label':options.labels, 'region':type, 'fold':0, 'case':case, 'elec':name, 'points':len(plan), 'type':'gt', 'mse':mse_gt}, ignore_index=True)

            # iterate through folds
            for f in range(len(folds[type])):
                print('     electrode={} type={} fold={} i={}'.format(name, type, f + 1, folds[type][f]))

                # for electrode filename
                pred_id = int(''.join([i for i in name if i.isdigit()]))
                pred_name = 'I' + str(pred_id) + 'i-' + method_name + '-' + uncertainty + '-' + options.labels + '-' + type + '-f' + str(f + 1)
                xml_file = os.path.join(pred_dir, pred_name + '.xmlE')
                if os.path.exists(xml_file):
                    xml_ep, xml_tp, xml_points = xmlparser.load_electrode(filename=xml_file)
                    pred_points = np.flip(xml_points, 0).copy()

                    d_dist = np.linalg.norm(pred_points[1:, :] - pred_points[0:-1, :], axis=1)
                    d_dist_mean = np.mean(d_dist)
                    d_dist_std = np.mean(d_dist)
                    print('     d distances: {}({}) = {}'.format(d_dist_mean, d_dist_std, d_dist))

                    assert len(plan) == len(pred_points)
                else:
                    print('     ERROR: xml_file={} not found'.format(xml_file))
                    input('break')

                # compute loss
                # mse_gt = model.compute_loss(truth=plan, pred=impl)
                mse_plan = model.compute_loss(truth=plan, pred=pred_points)
                mse_impl = model.compute_loss(truth=impl, pred=pred_points)
                print('     MSE: plan-impl={} plan-pred={} impl-pred={}'.format(mse_gt, mse_plan, mse_impl))
                results[type]['f' + str(f + 1)].append(mse_gt)

                # dataframe
                mse_col_df = mse_col_df.append({'method':method_name, 'label':options.labels, 'region':type, 'fold':f+1, 'case':case, 'elec':name, 'points': len(plan), 'mse_gt': mse_gt, 'mse_plan': mse_plan, 'mse_impl':mse_impl}, ignore_index=True)
                # mse_row_df = mse_row_df.append({'method':method_name, 'label':options.labels, 'region':type, 'fold':f+1, 'case':case, 'elec':name, 'type':'gt', 'mse':mse_gt}, ignore_index=True)
                mse_row_df = mse_row_df.append({'method':method_name, 'label':options.labels, 'region':type, 'fold':f+1, 'case':case, 'elec':name, 'points': len(plan), 'type':'plan', 'mse':mse_plan}, ignore_index=True)
                mse_row_df = mse_row_df.append({'method':method_name, 'label':options.labels, 'region':type, 'fold':f+1, 'case':case, 'elec':name, 'points': len(plan), 'type':'impl', 'mse':mse_impl}, ignore_index=True)

            # restore dataset
            datasets['testing'].database_restore()

    # dtypes
    mse_col_df = mse_col_df.astype({'fold': 'int32'})
    mse_col_df = mse_col_df.astype({'points': 'int32'})
    mse_col_df = mse_col_df.astype({'mse_gt': 'float32'})
    mse_col_df = mse_col_df.astype({'mse_plan': 'float32'})
    mse_col_df = mse_col_df.astype({'mse_impl': 'float32'})
    mse_row_df = mse_row_df.astype({'fold': 'int32'})
    mse_row_df = mse_row_df.astype({'points': 'int32'})
    mse_row_df = mse_row_df.astype({'mse': 'float32'})

    # save results
    results_filename = method_name + '-' + uncertainty + '-' + options.labels + '.pkl'
    data_pkl_file = os.path.join(options.data_dir, results_filename)
    pickle_file = open(data_pkl_file, "wb")
    pickle.dump(results, pickle_file)
    pickle_file.close()

    results_filename = method_name + '-' + uncertainty + '-' + options.labels + '-df-col.pkl'
    data_pkl_file = os.path.join(options.data_dir, results_filename)
    pickle_file = open(data_pkl_file, "wb")
    pickle.dump(mse_col_df, pickle_file)
    pickle_file.close()

    results_filename = method_name + '-' + uncertainty + '-' + options.labels + '-df-row.pkl'
    data_pkl_file = os.path.join(options.data_dir, results_filename)
    pickle_file = open(data_pkl_file, "wb")
    pickle.dump(mse_row_df, pickle_file)
    pickle_file.close()


if __name__ == '__main__':
    main()