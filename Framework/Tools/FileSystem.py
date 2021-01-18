"""
This class handles all file related functions

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


class FileSystem:

    def __init__(self):
        return

    def create_dir(self, name=None):
        try:
            os.mkdir(name)
        except OSError:
            print('Creation of the directory {} failed'.format(name))
        else:
            print('Successfully created the directory {}'.format(name))

    def open_csv(self, file=None, cols=None, delim=None):
        data_df = pd.read_csv(file, sep=delim, header=None)
        data_df.columns = cols
        return data_df

    def get_points_from_txt(self, file=None, delim=None):
        points = []
        if os.path.exists(file):
            wfile = open(file, "r")
            for x in wfile:
                point = x.split(delim)
                points.append([float(point[0]), float(point[1]), float(point[2])])
            wfile.close()
        return points

    def save_points_as_txt(self, points=None, file=None):
        if not os.path.exists(file):
            wfile = open(file, "w+")
            # IMPORTANT: negate x, y for EpiNav
            for p in range(len(points)-1):
                # wfile.write('{:.4f} {:.4f} {:.4f}\n'.format(-points[p][0], -points[p][1], points[p][2]))
                wfile.write('{:.4f} {:.4f} {:.4f}\n'.format(points[p][0], points[p][1], points[p][2]))
            # wfile.write('{:.4f} {:.4f} {:.4f}'.format(-points[len(points)-1][0], -points[len(points)-1][1], points[len(points)-1][2]))
            wfile.write('{:.4f} {:.4f} {:.4f}'.format(points[len(points) - 1][0], points[len(points) - 1][1], points[len(points) - 1][2]))
            wfile.close()

    def save_registration_matrix(self, matrix=None, file=None):
        np.savetxt(file, matrix, delimiter=' ', fmt='%1.4f')

    def open_registration_matrix(self, file=None):
        M = np.loadtxt(file, delimiter=' ')
        return M
