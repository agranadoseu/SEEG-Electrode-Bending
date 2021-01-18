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
import copy
import numpy as np
import pickle
import torch
from torch.utils import data

from Framework.Patient import Surfaces
from Framework.Data import InputData
from Framework.Data import CaseGenerator
from Framework.Patient import MedicalImages
from Framework.Data import PlanGenerator
from Framework.Data import InputFeatures
from Framework.Data import CategoricalData
from Framework.Data import FilterDataset
from Framework.Preprocessing import PatientDataGenerator
from Framework.Preprocessing import PatientDataLoader
from Framework.Preprocessing import TrajectoryFilter
from Framework.Preprocessing import TemplateImageLoader
from Framework.Visualisation import VTKRenderer


data_dir = 'C:\\UCL\\PhysicsSimulation\\Python\\NSElectrodeBending\\Framework\\Data'
datasets = 'E:\\UCL\\Backup\\medic-biomechanics\\Datasets\\BendingJournal\\datasets'
space = 'mni_aff'     # 'patient', 'mni_f3d'
data_file = 'data_' + space + '_window_norm.npy'


def main():

    # select regions of interest
    plangen = PlanGenerator.PlanGenerator()
    # plangen.ep_superior_frontal_gyrus()
    # plangen.ep_middle_frontal_gyrus()
    # plangen.ep_inferior_frontal_orbital_gyrus()

    # plangen.ep_temporal_gyrus()
    # plangen.ep_inferior_medial_temporal_gyrus()
    # plangen.ep_superior_temporal_gyrus()

    # plangen.ep_anterior_posterior_central_gyrus()

    plangen.ep_parietal_occipital()
    # plangen.ep_parietal_lobule()
    # plangen.ep_occipital()

    print('filter: filename={} ep={} tp={}'.format(plangen.filename, plangen.ep, plangen.tp))

    # open data
    input_data = InputData.InputData(directory=data_dir, file=data_file)
    selection = input_data.search_by_region(ep=plangen.ep, tp=plangen.tp)
    print('selection', selection)

    # save selection
    pickle_file = open(os.path.join(data_dir, plangen.filename), "wb")
    pickle.dump(selection, pickle_file)
    pickle_file.close()

    ''' Visualisation '''
    renderer = VTKRenderer.VTKRenderer()
    renderer.create_vtk_renderer()

    ''' Surfaces '''
    templates = TemplateImageLoader.TemplateImageLoader(datasets=datasets)
    templates.open_mni_images()
    surfaces = Surfaces.Surfaces()
    for s in plangen.ep + plangen.tp:
        surfaces.load(directory=templates.templates_dir, image=templates.mni['GIF'], gif=s)

    # iterate through selection
    print('Selection:')
    N = 0
    for case in selection.keys():
        # index of case
        c = input_data.data['case'].index(case)
        N += len(selection[case])
        print('case={} electrodes={}'.format(case, selection[case]))

        # Pre-computed data
        plan = input_data.data['plan'][c]
        impl = input_data.data['impl'][c]
        local_delta = input_data.data['local_delta'][c]
        delta = input_data.data['delta'][c]

        # Visualisation
        if space is not 'mni_f3d':
            renderer.render_trajectories(data=plan, filter=selection[case], colour=[1.0, 1.0, 1.0], radius=0.4, opacity=0.4)
            renderer.render_trajectories(data=impl, filter=selection[case], colour=[], radius=0.4, opacity=0.4)
        renderer.render_vectorfield(data=delta, source='from', vector='u', filter=selection[case], colour=[])
        renderer.render_vectorfield(data=local_delta, source='from', vector='u', filter=selection[case], colour=[1.0, 0.0, 0.0])
        renderer.render_vectorfield(data=local_delta, source='from_dir', vector='dir', filter=selection[case], colour=[0.0, 1.0, 0.0])

    for s in list(np.unique(plangen.ep + list(np.asarray(plangen.tp)[np.isin(plangen.tp, plangen.all_eps)]))):
        if s not in list(MedicalImages.wmGif):
            renderer.render_surface(polydata=surfaces.polydata[s], colour=[1.,1.,1.], opacity=0.2)
    for s in list(np.asarray(plangen.tp)[np.isin(plangen.tp, plangen.all_eps, invert=True)]):
        if s not in list(MedicalImages.wmGif):
            renderer.render_surface(polydata=surfaces.polydata[s], colour=[0.,0.95,0.95], opacity=0.2)

    print('filter={} num_cases={} num_electrodes={}'.format(plangen.filename, len(selection.keys()), N))

    # render
    renderer.execute()


if __name__ == '__main__':
    main()