"""
Script to test functionality

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
import pandas as pd
import pickle

from Framework.Patient import Surfaces
from Framework.Data import CaseGenerator
from Framework.Preprocessing import PatientDataGenerator
from Framework.Preprocessing import PatientDataLoader
from Framework.Preprocessing import TrajectoryFilter
from Framework.Preprocessing import SurfaceCollisionFilter
from Framework.Preprocessing import AssemblyFilter
from Framework.Preprocessing import TemplateImageLoader

from Framework.Visualisation import VTKRenderer

datasets = 'E:\\UCL\\Backup\\medic-biomechanics\\Datasets\\BendingJournal\\datasets'
cases_generator = CaseGenerator.CaseGenerator()
cases = cases_generator.get_all()
space = 'mni_aff'  # 'patient', 'acpc', 'mni_aff', 'mni_f3d'
# cases = ['R03'] # delete
data = {'case': [], 'plan': [], 'impl': [], 'ghost': [], 'ep': [], 'tp': [],
        'local_delta': [], 'delta': [],
        'window3': [], 'window5': [], 'window9': [], 'window11': []}

# features
features = {'case': [], 'plan': [], 'impl': [],
            'local_delta': [], 'delta': [],
            'bending': [], 'structure': [], 'collision': []}
features_df = []

''' Surfaces '''
gif_ep, gif_tp = [], []
templates = TemplateImageLoader.TemplateImageLoader(datasets=datasets)
templates.open_mni_images()
surfaces = Surfaces.Surfaces()

''' Visualisation'''
# renderer = VTKRenderer.VTKRenderer()
# renderer.create_vtk_renderer()

# iterate through cases
for case in cases:
    data['case'].append(case)

    ''' Data generator '''
    generator = PatientDataGenerator.PatientDataGenerator(directory=datasets, case=case)
    generator.execute()
    # input('break')

    ''' Data loader '''
    patient = PatientDataLoader.PatientDataLoader(directory=datasets, case=case)
    patient.execute()
    data['plan'].append(patient.electrodes.plan[space])
    data['impl'].append(patient.electrodes.impl[space])
    data['ghost'].append(patient.electrodes.ghost[space])

    # ''' Surfaces '''
    # data['ep'].append(list(np.unique(patient.features.EP_region)))
    # data['tp'].append(list(np.unique(patient.features.TP_region)))
    # gif_ep += list(np.unique(patient.features.EP_region))
    # gif_tp += list(np.unique(patient.features.TP_region))

    ''' Pre-processing '''
    # compute displacements
    trajectories = TrajectoryFilter.TrajectoryFilter(images=patient.medicalImages, electrodes=patient.electrodes)
    trajectories.execute(space=space)
    data['local_delta'].append(trajectories.local_delta)
    data['delta'].append(trajectories.delta)
    data['window3'].append(trajectories.window3)
    data['window5'].append(trajectories.window5)
    data['window9'].append(trajectories.window9)
    data['window11'].append(trajectories.window11)

    # features
    features_file = os.path.join(patient.mni_aff, 'features.pkl')
    if not os.path.exists(features_file):
        features['case'].append(case)
        features['plan'].append(patient.electrodes.plan[space])
        features['impl'].append(patient.electrodes.impl[space])
        features['local_delta'].append(trajectories.local_delta)
        features['delta'].append(trajectories.delta)
        features['bending'].append(trajectories.bending)
        features['structure'].append(trajectories.structure)

        # collision
        surface_collision = SurfaceCollisionFilter.SurfaceCollisionFilter(images=patient.medicalImages, electrodes=patient.electrodes, space=space)
        surface_collision.execute()
        features['collision'].append(surface_collision.collision)

        # assembly of features
        assembly = AssemblyFilter.AssemblyFilter(images=patient.medicalImages, electrodes=patient.electrodes, space=space, features=features)
        case_features_df = assembly.execute(case=case, plan=patient.electrodes.plan[space], impl=patient.electrodes.impl[space])
        assembly.save(case_features_df)
    else:
        features_pickle_file = open(features_file, "rb")
        case_features_df = pickle.load(features_pickle_file)
        features_pickle_file.close()
    features_df.append(case_features_df)

    ''' Surfaces '''
    case_ep, case_tp = [], []
    for elec_name in patient.electrodes.impl[space]['name']:
        case_ep += list(np.unique(case_features_df[case_features_df.electrode == elec_name].EP_region.values))
        case_tp += list(np.unique(case_features_df[case_features_df.electrode == elec_name].TP_region.values))
    data['ep'].append(case_ep)
    data['tp'].append(case_tp)
    gif_ep += case_ep
    gif_tp += case_tp

    ''' Visualisation '''
    # if space is not 'mni_f3d':
    #     renderer.render_trajectories(data=patient.electrodes.plan[space], colour=[1.0, 1.0, 1.0], radius=0.4, opacity=0.4)
    #     renderer.render_trajectories(data=patient.electrodes.impl[space], colour=[], radius=0.4, opacity=0.4)
    # renderer.render_vectorfield(data=trajectories.delta, source='from', vector='u', colour=[])
    # renderer.render_vectorfield(data=trajectories.local_delta, source='from', vector='u', colour=[1.0, 0.0, 0.0])
    # renderer.render_vectorfield(data=trajectories.local_delta, source='from_dir', vector='dir', colour=[1.0, 1.0, 1.0])

# surfaces
# for s in list(set(gif_ep + gif_tp)):
#     # print('s', int(s))
#     surfaces.load(directory=templates.templates_dir, image=templates.mni['GIF'], gif=int(s))
#     renderer.render_surface(polydata=surfaces.polydata[int(s)], colour=[1., 1., 1.], opacity=0.2)

# save
# pickle_file = open('../Data/data_'+space+'.npy', "wb")
pickle_file = open('../Data/data_'+space+'_window_norm.npy', "wb")
pickle.dump(data, pickle_file)
pickle_file.close()

# save features
all_df = pd.concat(features_df)
all_df.to_csv('../Data/features_'+space+'.csv', header=True, index=False)
all_df.to_pickle('../Data/features_'+space+'.pkl')

# render
# renderer.execute()