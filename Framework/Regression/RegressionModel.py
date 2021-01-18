"""
Abstract class for all regression models

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""


class RegressionModel:
    def __init__(self):
        return

    def save_state(self, itr=None):
        return

    def load_state(self, timestamp=None, filename=None):
        return

    def create(self):
        return

    def train(self):
        return

    def test(self):
        return

    def infer(self):
        return

