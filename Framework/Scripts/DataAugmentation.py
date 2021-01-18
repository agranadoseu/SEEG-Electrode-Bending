"""
Script to perform data augmentation of the output of FullPipeline

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import copy
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
space = 'mni_aff'  # 'patient', 'mni_f3d'

''' Pre-processed patient data '''
pickle_file = open('../Data/data_'+space+'.npy', "rb")
predata = pickle.load(pickle_file)
pickle_file.close()
# cases_generator = CaseGenerator.CaseGenerator()
# cases = cases_generator.get_all()

''' Surfaces '''
gif_ep, gif_tp = [], []
templates = TemplateImageLoader.TemplateLoader(datasets=datasets)
templates.open_mni_images()
surfaces = Surfaces.Surfaces()

''' Visualisation'''
renderer = VTKRenderer.VTKRenderer()
renderer.create_vtk_renderer()

# iterate through cases
N = 0
N_da = 0
for i in range(len(predata['case'])):
    ''' Pre-computed data '''
    plan = predata['plan'][i]
    impl = predata['impl'][i]
    local_delta = predata['local_delta'][i]
    delta = predata['delta'][i]
    gif_ep += predata['ep'][i]
    gif_tp += predata['tp'][i]
    # print('local_delta', local_delta)
    # print('delta', delta)

    num_elec = len(plan['name'])
    N += num_elec
    print('Original case: {} with {} electrodes'.format(predata['case'][i], len(plan['name'])))
    for c in range(num_elec):
        # print('   plan={} impl={}'.format(plan['name'][c], impl['name'][c]))
        print('   plan={} impl={} contacts={} points={}'.format(plan['name'][c], impl['name'][c], len(plan['contacts'][c]), len(plan['points'][c])))
        # print('   local_delta[u]={} delta[u]={}'.format(len(local_delta['u'][c]), len(delta['u'][c])))

    ''' Visualisation '''
    if space is not 'mni_f3d':
        renderer.render_trajectories(data=plan, colour=[1.0, 1.0, 1.0], radius=0.4, opacity=0.4)
        renderer.render_trajectories(data=impl, colour=[], radius=0.4, opacity=0.4)
    renderer.render_vectorfield(data=delta, source='from', vector='u', colour=[])
    renderer.render_vectorfield(data=local_delta, source='from', vector='u', colour=[1.0, 0.0, 0.0])
    renderer.render_vectorfield(data=local_delta, source='from_dir', vector='dir', colour=[0.0, 1.0, 0.0])

    ''' Data augmentation '''
    plan_da = copy.deepcopy(plan)
    impl_da = copy.deepcopy(impl)
    # local_delta_da = copy.deepcopy(local_delta)
    # delta_da = copy.deepcopy(delta)
    for c in range(num_elec):
        plan_da['ep'][c][0] *= -1.0
        impl_da['ep'][c][0] *= -1.0
        plan_da['tp'][c][0] *= -1.0
        impl_da['tp'][c][0] *= -1.0
        for d in range(len(plan_da['contacts'][c])):
            plan_da['contacts'][c][d][0] *= -1.0
            impl_da['contacts'][c][d][0] *= -1.0
        for d in range(len(plan_da['points'][c])):
            plan_da['points'][c][d][0] *= -1.0
            impl_da['points'][c][d][0] *= -1.0
            # local_delta_da['u'][c][d][0] *= -1.0
            # local_delta_da['from'][c][d][0] *= -1.0
            # local_delta_da['to'][c][d][0] *= -1.0
            # local_delta_da['dir'][c][d][0] *= -1.0
            # local_delta_da['from_dir'][c][d][0] *= -1.0
            # delta_da['u'][c][d][0] *= -1.0
            # delta_da['from'][c][d][0] *= -1.0
            # delta_da['to'][c][d][0] *= -1.0
            # delta_da['dir'][c][d][0] *= -1.0
    for c in range(num_elec):
        # update dictionary
        plan['name'].append('D'+str(plan['id'][c])+'i')
        plan['id'].append(plan['id'][c])
        plan['num_contacts'].append(plan['contacts'][c])
        plan['ep'].append(plan_da['ep'][c])
        plan['tp'].append(plan_da['tp'][c])
        plan['stylet'].append(plan['stylet'][c])
        plan['contacts'].append(plan_da['contacts'][c])
        plan['points'].append(plan_da['points'][c])
        impl['name'].append('A' + str(impl['id'][c]) + 'i')
        impl['id'].append(impl['id'][c])
        impl['num_contacts'].append(impl['contacts'][c])
        impl['ep'].append(impl_da['ep'][c])
        impl['tp'].append(impl_da['tp'][c])
        impl['stylet'].append(impl['stylet'][c])
        impl['contacts'].append(impl_da['contacts'][c])
        impl['points'].append(impl_da['points'][c])

        # if len(plan['points'][c]) is not 0:
        #     local_delta['name'].append('A'+str(local_delta['id'][c])+'i')
        #     local_delta['id'].append(local_delta['id'][c])
        #     local_delta['du'].append(local_delta['du'][c])
        #     local_delta['u'].append(local_delta_da['u'][c])
        #     local_delta['from'].append(local_delta_da['from'][c])
        #     local_delta['to'].append(local_delta_da['to'][c])
        #     local_delta['dir'].append(local_delta_da['dir'][c])
        #     local_delta['from_dir'].append(local_delta_da['from_dir'][c])
        #     delta['name'].append('A' + str(delta['id'][c]) + 'i')
        #     delta['id'].append(delta['id'][c])
        #     delta['du'].append(delta['du'][c])
        #     delta['u'].append(delta_da['u'][c])
        #     delta['from'].append(delta_da['from'][c])
        #     delta['to'].append(delta_da['to'][c])
        #     delta['dir'].append(delta_da['dir'][c])

    # update data
    predata['plan'][i] = plan
    predata['impl'][i] = impl
    predata['local_delta'][i] = local_delta
    predata['delta'][i] = delta

    N_da += len(plan['name'])
    print('Augmenting case: {} with {} electrodes'.format(predata['case'][i], len(plan['name'])))
    for c in range(len(plan['name'])):
        print('   plan={} impl={}'.format(plan['name'][c], impl['name'][c]))

    ''' Visualisation '''
    if space is not 'mni_f3d':
        renderer.render_trajectories(data=plan_da, colour=[0.82, 0.73, 0.91], radius=0.4, opacity=0.4)
        renderer.render_trajectories(data=impl_da, colour=[0.4, 0.2, 0.6], radius=0.4, opacity=0.4)

print('Total number of electrodes: ', N)
print('Total number of electrodes after data augmentation: ', N_da)

# surfaces
for s in list(set(gif_ep + gif_tp)):
    # print('s', int(s))
    surfaces.load(directory=templates.templates_dir, image=templates.mni['GIF'], gif=int(s))
    renderer.render_surface(polydata=surfaces.polydata[int(s)], colour=[1., 1., 1.], opacity=0.2)

# save
pickle_file = open('../Data/data_'+space+'_da.npy', "wb")
pickle.dump(predata, pickle_file)
pickle_file.close()

# render
renderer.execute()

