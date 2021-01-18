"""
Script to create synthetic data and be able to evaluate methods
This will be based on the MNI template and similar to FullPipeline to generate a npy data file

Features:
- 5 main trajectories each side along the temporal lobe
    * no bending
    * bending upward/downward
    * bending left/right
- introduce noise at EP and bolt direction


Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import numpy as np
import pickle

from Framework.Patient import Surfaces
from Framework.Data import CaseGenerator
from Framework.Preprocessing import PatientDataGenerator
from Framework.Preprocessing import PatientDataLoader
from Framework.Preprocessing import TrajectoryFilter
from Framework.Preprocessing import TemplateImageLoader
from Framework.Visualisation import VTKRenderer


datasets = 'E:\\UCL\\Backup\\medic-biomechanics\\Datasets\\BendingJournal\\datasets'
num_cases = 10
cases = ['S{:02d}'.format(i) for i in range(1,num_cases+1,1)]
space = 'mni_aff'
data = {'case': [], 'plan': [], 'impl': [], 'ep': [], 'tp': [],
        'local_delta': [], 'delta': [],
        'window3': [], 'window5': [], 'window9': [], 'window11': []}

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

    ''' Data loader '''

    ''' Surfaces '''
    data['ep'].append(list(np.unique(patient.features.EP_region)))
    data['tp'].append(list(np.unique(patient.features.TP_region)))
    gif_ep += list(np.unique(patient.features.EP_region))
    gif_tp += list(np.unique(patient.features.TP_region))

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
pickle_file = open('../Data/synthetic_data_' + space + '_window_norm.npy', "wb")
pickle.dump(data, pickle_file)
pickle_file.close()

# render
# renderer.execute()