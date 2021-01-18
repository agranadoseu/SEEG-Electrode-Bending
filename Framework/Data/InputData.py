"""
Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import pandas as pd
import numpy as np
import pickle


class InputData:

    def __init__(self, directory=None, file=None):
        self.directory = directory
        self.file = file

        self.data = self.load_data()
        self.fix_data()

    def load_data(self):
        # load cases from file
        pickle_file = open(os.path.join(self.directory, self.file), "rb")
        data = pickle.load(pickle_file)
        pickle_file.close()

        return data

    def fix_data(self):
        # TP_region
        self.replace_value(case='O05', elec='E0i', column='TP_region', value=172)
        self.replace_value(case='O05', elec='E2i', column='TP_region', value=172)
        self.replace_value(case='O10', elec='E10i', column='TP_region', value=139)
        self.replace_value(case='P01', elec='E5i', column='TP_region', value=102)
        self.replace_value(case='P01', elec='E9i', column='TP_region', value=194)
        self.replace_value(case='P05', elec='E7i', column='TP_region', value=106)
        self.replace_value(case='P06', elec='E10i', column='TP_region', value=153)
        self.replace_value(case='P06', elec='E12i', column='TP_region', value=137)
        self.replace_value(case='P08', elec='E1i', column='TP_region', value=138)
        self.replace_value(case='P08', elec='E4i', column='TP_region', value=194)
        self.replace_value(case='P10', elec='E0i', column='TP_region', value=118)
        self.replace_value(case='P10', elec='E8i', column='TP_region', value=154)
        self.replace_value(case='P11', elec='E2i', column='TP_region', value=118)
        self.replace_value(case='P11', elec='E8i', column='TP_region', value=148)
        self.replace_value(case='P12', elec='E1i', column='TP_region', value=137)
        self.replace_value(case='P17', elec='E0i', column='TP_region', value=169)
        self.replace_value(case='P17', elec='E1i', column='TP_region', value=101)
        self.replace_value(case='P17', elec='E2i', column='TP_region', value=139)
        self.replace_value(case='P17', elec='E3i', column='TP_region', value=119)
        self.replace_value(case='P17', elec='E4i', column='TP_region', value=193)
        self.replace_value(case='P17', elec='E6i', column='TP_region', value=193)
        self.replace_value(case='P17', elec='E7i', column='TP_region', value=151)
        self.replace_value(case='P17', elec='E9i', column='TP_region', value=179)

        self.replace_value(case='R02', elec='E3i', column='TP_region', value=154)
        self.replace_value(case='R03', elec='E6i', column='TP_region', value=101)
        self.replace_value(case='R03', elec='E10i', column='TP_region', value=193)
        self.replace_value(case='R06', elec='E10i', column='TP_region', value=48)
        self.replace_value(case='R07', elec='E10i', column='TP_region', value=194)
        self.replace_value(case='R07', elec='E8i', column='TP_region', value=102)
        self.replace_value(case='R08', elec='E3i', column='TP_region', value=141)
        self.replace_value(case='R08', elec='E6i', column='TP_region', value=153)
        self.replace_value(case='R08', elec='E8i', column='TP_region', value=139)
        self.replace_value(case='R12', elec='E6i', column='TP_region', value=102)
        self.replace_value(case='R13', elec='E3i', column='TP_region', value=172)
        self.replace_value(case='R13', elec='E13i', column='TP_region', value=152)
        self.replace_value(case='R14', elec='E7i', column='TP_region', value=110)
        self.replace_value(case='R14', elec='E14i', column='TP_region', value=170)
        self.replace_value(case='R15', elec='E6i', column='TP_region', value=208)
        self.replace_value(case='R17', elec='E5i', column='TP_region', value=206)
        self.replace_value(case='R20', elec='E6i', column='TP_region', value=154)
        self.replace_value(case='R20', elec='E10i', column='TP_region', value=154)
        self.replace_value(case='R22', elec='E4i', column='TP_region', value=182)
        self.replace_value(case='R24', elec='E11i', column='TP_region', value=194)
        self.replace_value(case='R24', elec='E12i', column='TP_region', value=152)
        self.replace_value(case='R31', elec='E3i', column='TP_region', value=135)
        self.replace_value(case='R31', elec='E7i', column='TP_region', value=169)
        self.replace_value(case='T01', elec='E0i', column='TP_region', value=147)
        self.replace_value(case='T05', elec='E4i', column='TP_region', value=139)
        self.replace_value(case='T09', elec='E6i', column='TP_region', value=181)
        self.replace_value(case='T17', elec='E6i', column='TP_region', value=181)
        self.replace_value(case='T20', elec='E4i', column='TP_region', value=204)
        self.replace_value(case='T22', elec='E1i', column='TP_region', value=104)
        self.replace_value(case='T22', elec='E2i', column='TP_region', value=102)
        self.replace_value(case='T22', elec='E3i', column='TP_region', value=154)
        self.replace_value(case='T23', elec='E10i', column='TP_region', value=152)
        self.replace_value(case='T24', elec='E12i', column='TP_region', value=153)
        self.replace_value(case='T27', elec='E2i', column='TP_region', value=153)
        self.replace_value(case='T33', elec='E0i', column='TP_region', value=153)
        self.replace_value(case='T33', elec='E2i', column='TP_region', value=101)
        self.replace_value(case='T33', elec='E4i', column='TP_region', value=117)
        self.replace_value(case='T33', elec='E6i', column='TP_region', value=48)
        self.replace_value(case='T34', elec='E10i', column='TP_region', value=194)
        self.replace_value(case='T34', elec='E2i', column='TP_region', value=102)
        self.replace_value(case='T34', elec='E4i', column='TP_region', value=180)
        self.replace_value(case='T36', elec='E5i', column='TP_region', value=167)

        # EP_region
        self.replace_value(case='O05', elec='E10i', column='EP_region', value=178)
        self.replace_value(case='P07', elec='E4i', column='EP_region', value=157)
        self.replace_value(case='R02', elec='E9i', column='EP_region', value=178)
        self.replace_value(case='R14', elec='E7i', column='EP_region', value=158)
        self.replace_value(case='R29', elec='E2i', column='EP_region', value=163)
        self.replace_value(case='T02', elec='E9i', column='EP_region', value=183)
        self.replace_value(case='T05', elec='E2i', column='EP_region', value=201)
        self.replace_value(case='T09', elec='E3i', column='EP_region', value=155)
        self.replace_value(case='T28', elec='E7i', column='EP_region', value=198)
        self.replace_value(case='T31', elec='E5i', column='EP_region', value=196)

        return

    def replace_value(self, case=None, elec=None, column=None, value=None):
        # index of case
        # data.keys(): ['case', 'plan', 'impl', 'ghost', 'ep', 'tp', 'local_delta', 'delta', 'window3', 'window5', 'window9', 'window11']
        c = self.data['case'].index(case)

        # index of electrode
        # data['impl'][c].keys(): ['name', 'id', 'num_contacts', 'ep', 'tp', 'stylet', 'contacts', 'points']
        e = self.data['impl'][c]['name'].index(elec)

        # update data
        if column == 'EP_region':
            self.data['ep'][c][e] = value
        elif column == 'TP_region':
            self.data['tp'][c][e] = value

    def search_by_region(self, ep=None, tp=None):
        ''' return a dictionary of cases (keys) and electrode indices (values) '''
        selection = {}

        # iterate through cases
        for c in range(len(self.data['case'])):
            case = self.data['case'][c]

            # iterate through electrodes
            for e in range(len(self.data['impl'][c]['id'])):
                elec_id = self.data['impl'][c]['id'][e]
                elec_ep = self.data['ep'][c][e]
                elec_tp = self.data['tp'][c][e]

                # select
                if ep is None and tp is None:
                    if case not in list(selection.keys()):
                        selection[case] = [elec_id]
                    else:
                        selection[case].append(elec_id)
                elif ep is None:
                    if np.isin(elec_tp, tp):
                        if case not in list(selection.keys()):
                            selection[case] = [elec_id]
                        else:
                            selection[case].append(elec_id)
                elif tp is None:
                    if np.isin(elec_ep, ep):
                        if case not in list(selection.keys()):
                            selection[case] = [elec_id]
                        else:
                            selection[case].append(elec_id)
                else:
                    if np.isin(elec_ep, ep) and np.isin(elec_tp, tp):
                        if case not in list(selection.keys()):
                            selection[case] = [elec_id]
                        else:
                            selection[case].append(elec_id)

        return selection
