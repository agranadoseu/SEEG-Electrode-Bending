"""
This class generates plots of the data for publication

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from Framework.Data import PlanGenerator
from Framework.Data import InputData
from Framework.Data import InputFeatures
from Framework.Data import NNDataset


class Options:
    data_dir = '..\\Data'
    models_dir = '.\\models'
    space = 'mni_aff'
    labels = 'vector'

options = Options()


def main():
    datasets = {}

    ''' Labels '''
    labels = []
    if options.labels == 'lu':
        labels = ['lu_x', 'lu_y', 'lu_z']
    elif options.labels == 'gu':
        labels = ['gu_x', 'gu_y', 'gu_z']
    elif options.labels == 'vector':
        labels = ['elec_dir_x', 'elec_dir_y', 'elec_dir_z']

    ''' Input data '''
    data_file = 'data_' + options.space + '_window_norm.npy'
    features_data_file = 'features_' + options.space + '.pkl'
    input_data = InputData.InputData(directory=options.data_dir, file=data_file)
    input_features = InputFeatures.InputFeatures(directory=options.data_dir, file=features_data_file, labels=labels)
    print('ODEDataset:: {} cases in total loaded from {} with ids={}'.format(len(input_data.data['case']), data_file, input_data.data['case']))
    print('NNDataset:: {} cases in total loaded from {} with ids={}'.format(len(np.unique(input_features.df['case'])), features_data_file, np.unique(input_features.df['case'])))

    ''' Cases '''
    N = len(input_data.data['case'])
    cases = np.asarray(input_data.data['case'])
    indices = np.arange(N, dtype=np.int64)

    ''' Dataset '''
    plangen = PlanGenerator.PlanGenerator()
    plangen.ep_superior_frontal_gyrus()
    datasets['sfg'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=cases, filter_file=plangen.filename, data_augment=False, batch_time=0)
    plangen.ep_middle_frontal_gyrus()
    datasets['mfg'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=cases, filter_file=plangen.filename, data_augment=False, batch_time=0)
    plangen.ep_inferior_frontal_orbital_gyrus()
    datasets['ifog'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=cases, filter_file=plangen.filename, data_augment=False, batch_time=0)
    plangen.ep_temporal_gyrus()
    datasets['tg'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=cases, filter_file=plangen.filename, data_augment=False, batch_time=0)
    plangen.ep_anterior_posterior_central_gyrus()
    datasets['apcg'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=cases, filter_file=plangen.filename, data_augment=False, batch_time=0)
    plangen.ep_parietal_occipital()
    datasets['po'] = NNDataset.NNDataset(data_dir=options.data_dir, input_features=input_features, cases=cases, filter_file=plangen.filename, data_augment=False, batch_time=0)

    ''' Plot '''
    df_sfg = datasets['sfg'].generate_gu_dataframe(region_name='sfg', plot=False)
    df_mfg = datasets['mfg'].generate_gu_dataframe(region_name='mfg', plot=False)
    df_ifog = datasets['ifog'].generate_gu_dataframe(region_name='ifog', plot=False)
    df_tg = datasets['tg'].generate_gu_dataframe(region_name='tg', plot=False)
    df_apcg = datasets['apcg'].generate_gu_dataframe(region_name='apcg', plot=False)
    df_po = datasets['po'].generate_gu_dataframe(region_name='po', plot=False)
    df = df_sfg.append(df_mfg)
    df = df.append(df_ifog)
    df = df.append(df_tg)
    df = df.append(df_apcg)
    df = df.append(df_po)
    df = df.reset_index(drop=True)

    # all
    # sns.set_theme(style="darkgrid")
    # sns.lineplot(x="i", y="gu",
    #              hue="region", style="component",
    #              data=df)
    # plt.show()



if __name__ == '__main__':
    main()