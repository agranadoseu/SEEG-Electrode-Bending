"""
This class handles electrode information from a specific patient.
The idea is to have a single class that has access to all electrode related information

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

import xml.etree.ElementTree as ET

from Framework.Tools import FileSystem
from Framework.Tools import XMLParser


# @TODO organise electrode files: same folder, same suffix, same extension

class Electrodes:

    def __init__(self, main='', acpc='', mni_aff='', mni_f3d=''):
        # electrodes
        self.N = 0
        self.plan = {'patient': None, 'acpc': None, 'mni_aff': None, 'mni_f3d': None}
        self.impl = {'patient': None, 'acpc': None, 'mni_aff': None, 'mni_f3d': None}
        self.ghost = {'patient': None, 'acpc': None, 'mni_aff': None, 'mni_f3d': None}

        # sub folders
        self.dir = main
        self.acpc = acpc
        self.mni_aff = mni_aff
        self.mni_f3d = mni_f3d

        self.filesystem = FileSystem.FileSystem()
        self.xmlparser = XMLParser.XMLParser()

    def get_directory(self, type=None, space=None, suffix=None):
        # set directory depending on space
        directory = None
        if space == 'patient':
            directory = self.dir
            if suffix is not '' or type == 'plan':
                directory = os.path.join(self.dir, 'bending')
        elif space == 'acpc':
            directory = os.path.join(self.dir, self.acpc)
        elif space == 'mni_aff':
            directory = os.path.join(self.dir, self.mni_aff)
        elif space == 'mni_f3d':
            directory = os.path.join(self.dir, self.mni_f3d)

        return directory

    def load(self, type=None, suffix=None, space=None):
        """
        Opens interpolated electrodes
        :param type: type of electrode ['plan', 'impl']
        :param suffix:
        :param space: space of image ['patient', 'acpc', 'mni_aff', 'mni_f3d']
        :return:
        """
        # TODO load/create interpolated electrode trajectories

        # dictionary to save data
        data = {'name': [], 'id': [], 'num_contacts': [], 'ep': [], 'tp': [], 'stylet': [], 'contacts': [], 'points': []}

        # prefix depends on type
        prefix = None
        if type == 'plan':
            prefix = 'R'
        elif type == 'impl':
            prefix = 'E'

        # set directory depending on space
        directory = self.get_directory(type=type, space=space, suffix=suffix)

        # Load stylet file
        stylet_df = self.read_stylet(filename=os.path.join(self.dir,'stylet.txt'))
        print('stylet_df', stylet_df)

        # find electrode files under directory
        for file in os.listdir(directory):
            name = file.split('.')[0]
            condition = name[-1].isnumeric() if suffix == '' else name.endswith(suffix)
            if name.startswith(prefix) and condition and (file.endswith(".xmlE") or file.endswith(".xmlSE")):
                filename = os.path.join(directory, file)

                # parse interpolation
                tree = ET.parse(filename)
                root = tree.getroot()

                # get entry and target point
                ep = self.xmlparser.xml_get_point(parent=root, node='specifiedEntryPoint')
                tp = self.xmlparser.xml_get_point(parent=root, node='specifiedTargetPoint')

                # get point coordinates
                points = self.xmlparser.xml_get_pointarray(parent=root, node='contactPointArray')

                # parse original electrode and get contacts
                original_filename = None
                if space == 'patient':
                    if type == 'impl':
                        original_filename = os.path.join(self.dir, name.replace(suffix,'')+'.xmlE')
                    else:
                        original_filename = os.path.join(directory, name.replace(suffix, '') + '.xmlE')
                elif space == 'acpc' or space == 'mni_aff' or space == 'mni_f3d':
                    original_filename = os.path.join(directory, name.replace(suffix, '') + '.xmlE')
                print('opening: ', original_filename)
                tree = ET.parse(original_filename)
                root = tree.getroot()
                contacts = self.xmlparser.xml_get_pointarray(parent=root, node='contactPointArray')

                # stylet information
                stylet_val = -1
                if stylet_df is not None:
                    elec_id = int(''.join([i for i in name if i.isdigit()]))
                    for index, row in stylet_df.iterrows():
                        stylet_id = int(''.join([i for i in row['name'] if i.isdigit()]))
                        if elec_id == stylet_id:
                            stylet_val = row['stylet']
                            break

                # update dictionary
                data['name'].append(name)
                data['id'].append(int(''.join([i for i in name if i.isdigit()])))
                data['num_contacts'].append(len(contacts))
                data['ep'].append(ep)
                data['tp'].append(tp)
                data['stylet'].append(stylet_val)
                data['contacts'].append(contacts)
                data['points'].append(points)

        # save reference
        if type == 'plan':
            self.plan[space] = data
        elif type == 'impl':
            self.impl[space] = data

        return data

    def load_ghost(self, space=None):
        # dictionary to save data
        data = {'name': [], 'id': [], 'ep': [], 'tp': [], 'points': []}

        # set directory depending on space
        directory = self.get_directory(type='impl', space=space, suffix='i')

        # find electrode files under directory
        for file in os.listdir(directory):
            name = file.split('.')[0]
            if name.startswith('G') and name.endswith('i') and (file.endswith(".xmlE") or file.endswith(".xmlSE")):
                filename = os.path.join(directory, file)

                # parse interpolation
                tree = ET.parse(filename)
                root = tree.getroot()

                # get entry and target point
                ep = self.xmlparser.xml_get_point(parent=root, node='specifiedEntryPoint')
                tp = self.xmlparser.xml_get_point(parent=root, node='specifiedTargetPoint')

                # get point coordinates
                points = self.xmlparser.xml_get_pointarray(parent=root, node='contactPointArray')

                # update dictionary
                data['name'].append(name)
                data['id'].append(int(''.join([i for i in name if i.isdigit()])))
                data['ep'].append(ep)
                data['tp'].append(tp)
                data['points'].append(points)

        # save reference
        self.ghost[space] = data

        return data

    # def load_features(self):
    #     # set directory depending on space
    #     directory = self.get_directory(type='impl', space='patient', suffix='')
    #     features_df = pd.read_pickle(os.path.join(directory, 'features.pkl'))
    #
    #     # iterate through electrodes
    #     for i in range(len(self.impl['patient']['name'])):
    #         elec_name = self.impl['patient']['name'][i]
    #         electrode_df = features_df[features_df.electrode == elec_name]
    #         gif_ep = electrode_df.EP_region.values[0]
    #         gif_tp = electrode_df.TP_region.values[0]

    def read_stylet(self, filename=None):
        """
        File contains how short the stylet was for a list of electrodes by name
        :param filename:
        :return: dataframe
        """
        data_df = None
        stylet_file = os.path.join(self.dir, filename)
        stylet_exists = os.path.exists(stylet_file)

        # open txt
        if stylet_exists:
            data_df = pd.read_csv(stylet_file, sep=",", header=0)

        return data_df

    def save_txt_as_xml(self, type=None, space=None, colour=None):
        # set directory depending on space
        directory = self.get_directory(type=type, space=space, suffix='i')

        prefix = None
        if type == 'plan':
            prefix = 'R'
        elif type == 'impl':
            prefix = 'E'

        # find electrode files under directory
        for file in os.listdir(directory):
            name = file.split('.')[0]
            if name.startswith(prefix) and file.endswith(".txt"):
                txt_file = os.path.join(directory, name + '.txt')
                xml_file = os.path.join(directory, name + '.xmlE')
                if not os.path.exists(xml_file):
                    # load points
                    # points_df = self.filesystem.open_csv(file=txt_file, cols=['x','y','z','null'], delim=" ")
                    points_df = self.filesystem.open_csv(file=txt_file, cols=['x', 'y', 'z'], delim=" ")
                    ep = np.asarray([-points_df.iloc[0].x, -points_df.iloc[0].y, points_df.iloc[0].z], dtype=np.float32)
                    points = np.zeros((points_df.shape[0] - 1, 3), dtype=np.float32)
                    for p in range(1, len(points) + 1, 1):
                        points[p - 1][0] = -points_df.iloc[p].x
                        points[p - 1][1] = -points_df.iloc[p].y
                        points[p - 1][2] = points_df.iloc[p].z

                    self.xmlparser.save_points_as_xml(name=name, ep=ep, points=points, colour=colour, xml_file=xml_file)

    def save_as_txt(self, type=None, space=None):
        data = None
        if type=='plan':
            data = self.plan[space]
        elif type=='impl':
            data = self.impl[space]

        directory = self.get_directory(type=type, space=space, suffix='i')

        for i in range(len(data['name'])):
            # file to write
            str_file = os.path.join(directory, data['name'][i][0:-1] + ".txt")
            str_file_i = os.path.join(directory, data['name'][i] + ".txt")

            # original
            if not os.path.exists(str_file):
                wfile = open(str_file, "w+")
                # IMPORTANT: negate x, y for EpiNav
                wfile.write('{:.4f} {:.4f} {:.4f}'.format(-data['ep'][i][0], -data['ep'][i][1], data['ep'][i][2]))
                for c in range(len(data['contacts'][i])):
                    wfile.write('\n{:.4f} {:.4f} {:.4f}'.format(-data['contacts'][i][c][0], -data['contacts'][i][c][1], data['contacts'][i][c][2]))
                wfile.close()

            # interpolation
            if not os.path.exists(str_file_i):
                wfile_i = open(str_file_i, "w+")
                # IMPORTANT: negate x, y for EpiNav
                wfile_i.write('{:.4f} {:.4f} {:.4f}'.format(-data['ep'][i][0], -data['ep'][i][1], data['ep'][i][2]))
                for c in range(len(data['points'][i])):
                    wfile_i.write('\n{:.4f} {:.4f} {:.4f}'.format(-data['points'][i][c][0], -data['points'][i][c][1], data['points'][i][c][2]))
                wfile_i.close()
