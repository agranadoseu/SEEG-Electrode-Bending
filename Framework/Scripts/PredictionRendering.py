"""
Once trajectories are inferred this script plot predictions against planned and implanted trajectories

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

from Framework.Patient import Surfaces
from Framework.Preprocessing import PatientImageLoader
from Framework.Visualisation import VTKRenderer

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

    pickle_file = open(os.path.join(options.data_dir, 'cwd-mc-lu-df-col.pkl'), "rb")
    mse_col_df = pickle.load(pickle_file)
    pickle_file.close()
    # split dataframes
    over_df = mse_col_df[(mse_col_df.mse_plan > mse_col_df.mse_gt) &
                         (mse_col_df.mse_impl < mse_col_df.mse_plan)]
    under_df = mse_col_df[(mse_col_df.mse_plan <= mse_col_df.mse_gt) &
                          (mse_col_df.mse_impl <= mse_col_df.mse_gt)]
    wrong_df = mse_col_df[(mse_col_df.mse_impl > mse_col_df.mse_gt) &
                          (mse_col_df.mse_impl > mse_col_df.mse_plan)]

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
    # test_indices = [2, 13, 30, 41, 43, 45, 53, 74]
    # test_cases = ['R03', 'R14', 'R31', 'T11', 'T13', 'T16', 'T25', 'P10']

    # normal
    # test_indices, test_cases, test_elec = [43], ['T13'], ['E2i']  # sfg     # test_indices, test_cases, test_elec = [74], ['P10'], ['E1i']  # sfg (best)    # test_indices, test_cases, test_elec = [2], ['R03'], ['E10i']    # sfg
    # test_indices, test_cases, test_elec = [13], ['R14'], ['E12i']  # mfg
    # test_indices, test_cases, test_elec = [41], ['T11'], ['E4i']  # ifog
    # test_indices, test_cases, test_elec = [13], ['R14'], ['E5i']  # tg
    # test_indices, test_cases, test_elec = [2], ['R03'], ['E12i']  # cg
    # test_indices, test_cases, test_elec = [30], ['R31'], ['E8i']  # po

    # bad
    # test_indices, test_cases, test_elec = [74], ['P10'], ['E6i']    # sfg
    # test_indices, test_cases, test_elec = [45], ['T16'], ['E8i']  # mfg   # test_indices, test_cases, test_elec = [74], ['P10'], ['E8i']  # mfg (super worst)
    # test_indices, test_cases, test_elec = [2], ['R03'], ['E4i']  # ifog
    # test_indices, test_cases, test_elec = [53], ['T25'], ['E8i']  # tg
    # test_indices, test_cases, test_elec = [45], ['T16'], ['E6i']  # cg
    test_indices, test_cases, test_elec = [13], ['R14'], ['E13i']  # po     # test_indices, test_cases, test_elec = [13], ['R14'], ['E9i']  # po

    colours = {'sfg': [0.298, 0.447, 0.690],
               'mfg': [0.506, 0.447, 0.702],
               'ifog': [0.769, 0.306, 0.322],
               'tg': [0.33, 0.659, 0.408],
               'apcg': [0.866, 0.518, 0.322],
               'po': [0.600, 0.490, 0.376]}
    c_electrode = [1.0, 0.776, 0.0]
    c_wrong = [0.988, 0.0, 0.066]
    c_over = [0.0, 0.792, 0.992]

    ''' Surfaces '''
    gif_ep, gif_tp = [], []
    images = PatientImageLoader.PatientImageLoader(datasets=directory, case=test_cases[0], space='mni_aff')
    images.open_patient_images()
    surfaces = Surfaces.Surfaces()

    ''' Visualisation'''
    renderer = VTKRenderer.VTKRenderer()
    renderer.create_vtk_renderer()

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
        print("Plotting trajectories for case={} (id={})".format(case, idx))

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
            # model = WindowRegression.WindowRegression(channels=1, window_size=options.winsize, labels=options.labels)

        elif options.method == 'nn':
            # dataset
            datasets['testing'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features,
                                                      cases=[case], filter_file=None,
                                                      data_augment=False, batch_time=0)

            # regression model
            # model = NNRegression.NNRegression(num_features=datasets['testing'].feature_sz, num_outputs=datasets['testing'].output_sz)

        # initialise model for testing
        # model.init_loss_function()

        # iterate through electrodes
        datasets['testing'].database_backup()
        electrodes = datasets['testing'].get_names()
        for e in range(len(electrodes)):
            name = electrodes[e]

            if name != test_elec[0]:
                continue

            # ground truth: dataset containing only one electrode
            print('name={}'.format(name))
            ep_gif, tp_gif, ep, tp, contacts, plan, impl = datasets['testing'].database_by_electrode(name=name)

            # select data group
            type = plangen.get_type(ep=ep_gif, tp=tp_gif)

            # render electrode
            gif_ep += [ep_gif]
            gif_tp += [tp_gif]
            print('ep_gif={} tp_gif={}'.format(ep_gif, tp_gif))
            direction = plan[0] - ep
            direction /= np.linalg.norm(direction)
            plan_flip = np.flip(plan, 0)
            impl_flip = np.flip(impl, 0)
            renderer.create_ep_actors(idx=0, num_elec=1, ep=ep, direction=direction, colour=[1.0, 1.0, 1.0], radius=0.4, opacity=0.8)
            renderer.create_trajectory_actor(idx=0, num_elec=1, x_points=plan_flip, ep=ep, colour=[1.0, 1.0, 1.0], radius=0.4, opacity=1.0)
            renderer.create_trajectory_actor(idx=0, num_elec=1, x_points=impl_flip, ep=ep, colour=c_electrode, radius=0.4, opacity=1.0)

            # iterate through folds
            for f in range(10):
                print('     electrode={} type={} fold={}'.format(name, type, f+1))

                # for electrode filename
                pred_id = int(''.join([i for i in name if i.isdigit()]))
                pred_name = 'I' + str(pred_id) + 'i-' + method_name + '-' + uncertainty + '-' + options.labels + '-' + type + '-f' + str(f+1)
                xml_file = os.path.join(pred_dir, pred_name + '.xmlE')
                if os.path.exists(xml_file):
                    xml_ep, xml_tp, xml_points = xmlparser.load_electrode(filename=xml_file)
                    pred_points = np.flip(xml_points, 0).copy()

                    d_tp = np.linalg.norm(pred_points[-1] - plan[-1])
                    print(' d_tp={}'.format(d_tp))

                    colour_pred = colours[type]
                    pred_df = mse_col_df[(mse_col_df.case==case) & (mse_col_df.elec==name) & (mse_col_df.fold==f+1)]
                    if pred_df.mse_impl.values[0] <= 1.0:
                        colour_pred = colours[type]
                    elif len(over_df[(over_df.case==case) & (over_df.elec==name) & (over_df.fold==f+1)]):
                        colour_pred = c_over
                    elif len(wrong_df[(wrong_df.case==case) & (wrong_df.elec==name) & (wrong_df.fold==f+1)]):
                        colour_pred = c_wrong

                    renderer.create_trajectory_actor(idx=0, num_elec=1, x_points=xml_points, ep=ep, colour=colour_pred, radius=0.4, opacity=0.7)

                else:
                    print('     ERROR: xml_file={} not found'.format(xml_file))
                    input('break')

                # visualise


            # restore dataset
            datasets['testing'].database_restore()

        print('gif_ep={} gif_tp={}'.format(gif_ep, gif_tp))
        for s in list(set(gif_ep + gif_tp)):
            print('s', int(s))
            surfaces.load(directory=images.space_dir, image=images.image['GIF'], gif=int(s))
            renderer.render_surface(polydata=surfaces.polydata[int(s)], colour=[1., 1., 1.], opacity=0.2)

    # render
    renderer.execute()

if __name__ == '__main__':
    main()