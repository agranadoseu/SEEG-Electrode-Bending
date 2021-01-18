"""
This class handles all skull stripping related functions

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import subprocess


class Robex:

    def __init__(self):
        return

    def skull_stripping(self, in_img='', out_img=''):
        # runROBEX.bat ./R01/T1-mni.nii.gz ./R01/T1-mni-stripped.nii.gz

        command = 'C:\\UCL\\PhysicsSimulation\\ROBEX\\runROBEX.bat'
        params10, params11 = in_img, out_img

        print('Executing:\n{} {} {}'.format(command, params10, params11))
        subprocess.check_call(['cmd.exe', '/c', command, params10, params11], shell=True)
