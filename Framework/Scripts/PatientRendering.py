"""
Rendering only one case in patient/acpc space

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

datasets = 'E:\\UCL\\Backup\\medic-biomechanics\\Datasets\\BendingJournal\\datasets'
space = 'acpc'  # 'patient', 'acpc', 'mni_aff', 'mni_f3d'

''' Pre-processed patient data '''
pickle_file = open('../Data/data_' + space + '.npy', "rb")
predata = pickle.load(pickle_file)
pickle_file.close()
# cases_generator = CaseGenerator.CaseGenerator()
# cases = cases_generator.get_all()
case = ['R10raw']

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

    ''' Visualisation '''
    if space is not 'mni_f3d':
        renderer.render_trajectories(data=plan, colour=[1.0, 1.0, 1.0], radius=0.4, opacity=0.4)
        renderer.render_trajectories(data=impl, colour=[], radius=0.4, opacity=0.4)
    renderer.render_vectorfield(data=delta, source='from', vector='u', colour=[])
    renderer.render_vectorfield(data=local_delta, source='from', vector='u', colour=[1.0, 0.0, 0.0])
    renderer.render_vectorfield(data=local_delta, source='from_dir', vector='dir', colour=[0.0, 1.0, 0.0])

    # renderer.render_prediction(data=impl, colour=[0.4, 0.2, 0.6], radius=0.4, opacity=0.4)

    # surfaces
    print('gif_ep', gif_ep)
    print('gif_tp', gif_tp)
    for s in list(set(gif_ep + gif_tp)):
        # print('s', int(s))
        surfaces.load(directory=images.space_dir, image=images.image['GIF'], gif=int(s))
        renderer.render_surface(polydata=surfaces.polydata[int(s)], colour=[1., 1., 1.], opacity=0.2)

# render
renderer.execute()