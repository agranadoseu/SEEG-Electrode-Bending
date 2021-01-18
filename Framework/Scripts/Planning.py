"""
Script to assist surgical planning of electrode implantation
The idea is to bring previous cases near planning to see how they deviated
This will include predictions from regression and uncertainty confidence

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import numpy as np
import pickle

from Framework.Patient import Surfaces
from Framework.Data import CaseGenerator
from Framework.Preprocessing import PatientDataGenerator
from Framework.Preprocessing import PatientDataLoader
from Framework.Preprocessing import TrajectoryFilter
from Framework.Preprocessing import PatientImageLoader
from Framework.Visualisation import VTKRenderer

from Framework.Data import ODEDataset
from Framework.Regression import ODERegression


def main():
    datasets = 'E:\\UCL\\Backup\\medic-biomechanics\\Datasets\\BendingJournal\\datasets'
    space = 'mni_aff'  # 'patient', 'acpc', 'mni_aff', 'mni_f3d'
    data_augment = False
    method = 'ode'  # cnn, nn, gpy, ode, odewindow, odecnn
    timestamp = '20200914-1616'
    file_models = ['f1-i300.pth', 'f2-i300.pth', 'f3-i300.pth', 'f4-i300.pth', 'f5-i300.pth']
    timestamp = '20200915-2041'
    file_models = ['f1-i135.pth']
    timestamp = '20200917-1052'
    file_models = ['f1-i1.pth']
    timestamp = '20200917-1709'
    file_models = ['f1-i100.pth']

    ''' Pre-processed patient data '''
    pickle_file = open('../Data/data_' + space + '.npy', "rb")
    predata = pickle.load(pickle_file)
    pickle_file.close()
    # cases_generator = CaseGenerator.CaseGenerator()
    # cases = cases_generator.get_all()
    case = ['R10']

    ''' Surfaces '''
    gif_ep, gif_tp = [], []
    images = PatientImageLoader.PatientImageLoader(datasets=datasets, case=case[0], space=space)
    images.open_patient_images()
    surfaces = Surfaces.Surfaces()

    ''' Visualisation'''
    renderer = VTKRenderer.VTKRenderer()
    renderer.create_vtk_renderer()

    # iterate through cases
    show = np.where(np.isin(predata['case'], case))
    print('show', show, show[0])
    for i in show[0]:
        print('Filtered cases: ', predata['case'][i])

        ''' Pre-computed data '''
        plan = predata['plan'][i]
        impl = predata['impl'][i]
        local_delta = predata['local_delta'][i]
        delta = predata['delta'][i]
        gif_ep += predata['ep'][i]
        gif_tp += predata['tp'][i]
        print('pre-computed data names={}'.format(impl['name']))

        ''' Visualisation '''
        if space is not 'mni_f3d':
            renderer.render_trajectories(data=plan, colour=[1.0, 1.0, 1.0], radius=0.4, opacity=0.4)
            renderer.render_trajectories(data=impl, colour=[], radius=0.4, opacity=0.4)
        renderer.render_vectorfield(data=delta, source='from', vector='u', colour=[])
        renderer.render_vectorfield(data=local_delta, source='from', vector='u', colour=[1.0, 0.0, 0.0])
        renderer.render_vectorfield(data=local_delta, source='from_dir', vector='dir', colour=[0.0, 1.0, 0.0])

        ''' Integrate prediction '''
        # create dataset
        datasets = {}
        datasets['testing'] = ODEDataset.ODEDataset(data_dir='../Data', data=predata, space=space, cases=np.asarray([i]), filter_file='filter_mtg.npy', data_augment=data_augment, batch_time=0)
        print('dataset size = {}'.format(datasets['testing'].__len__()))
        if datasets['testing'].__len__() != 0:
            # create structure for visualisation
            pred = {'name': [], 'id': [], 'num_contacts': [], 'ep': [], 'tp': [], 'stylet': [], 'contacts': [], 'points': []}

            # create regression model
            odemodel = ODERegression.ODERegression()
            odemodel.init_test(datasets=datasets)
            odemodel.load_state(timestamp=timestamp, filename=file_models[0])

            # test
            pred_y = odemodel.infer()
            print('Prediction:')
            for e in range(len(pred_y['name'])):
                e_case = pred_y['case'][e][0]
                e_name = pred_y['name'][e][0]
                e_id = int(''.join([i for i in e_name if i.isdigit()]))
                e_name_idx = impl['name'].index(e_name)
                e_points = pred_y['points'][e].cpu().numpy()
                print('case={} name={}\npoints({})={}'.format(e_case, e_name, e_points.shape, e_points))
                pred['name'].append(e_name)
                pred['id'].append(e_id)
                pred['ep'].append(impl['ep'][e_name_idx])
                pred['points'].append(np.flip(e_points, 0))

            # Visualisation of prediction
            renderer.render_prediction(data=pred, colour=[0.4, 0.2, 0.6], radius=0.4, opacity=0.4)

        # surfaces
        print('gif_ep', gif_ep)
        print('gif_tp', gif_tp)
        for s in list(set(gif_ep + gif_tp)):
            # print('s', int(s))
            surfaces.load(directory=images.space_dir, image=images.image['GIF'], gif=int(s))
            renderer.render_surface(polydata=surfaces.polydata[int(s)], colour=[1., 1., 1.], opacity=0.2)

    # render
    renderer.execute()


if __name__ == '__main__':
    main()