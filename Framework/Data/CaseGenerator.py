"""
Generates a list of patient cases

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import numpy as np


class CaseGenerator:

    def __init__(self):
        self.ids = []

    def get_all(self):
        self.ids = []

        r = self.get_r()
        t = self.get_t()
        p = self.get_p()
        s = self.get_s()
        o = self.get_o()

        return self.ids

    def get_r(self):
        cases = []
        skip = [0]
        for i in range(1, 33, 1):
            if not np.isin(i, skip):
                if i < 10:
                    cases.append('R0' + str(i))
                else:
                    cases.append('R' + str(i))

        self.ids.extend(cases)
        return cases

    def get_t(self):
        cases = []
        skip = [4, 15, 19, 30]
        for i in range(1, 38, 1):
            if not np.isin(i, skip):
                if i < 10:
                    cases.append('T0' + str(i))
                else:
                    cases.append('T' + str(i))

        self.ids.extend(cases)
        return cases

    def get_p(self):
        cases = []
        skip = [14]
        for i in range(1, 18, 1):
            if not np.isin(i, skip):
                if i < 10:
                    cases.append('P0' + str(i))
                else:
                    cases.append('P' + str(i))

        self.ids.extend(cases)
        return cases

    def get_s(self):
        cases = []
        skip = [0]
        for i in [7, 13, 15]:
            if not np.isin(i, skip):
                if i < 10:
                    cases.append('S0' + str(i))
                else:
                    cases.append('S' + str(i))

        self.ids.extend(cases)
        return cases

    def get_o(self):
        cases = []
        skip = [0]
        for i in [5, 10]:
            if not np.isin(i, skip):
                if i < 10:
                    cases.append('O0' + str(i))
                else:
                    cases.append('O' + str(i))

        self.ids.extend(cases)
        return cases
