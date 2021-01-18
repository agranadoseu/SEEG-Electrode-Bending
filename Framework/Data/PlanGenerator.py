"""
Generates a list of entry and target points based on GIF parcellation

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import numpy as np


class PlanGenerator:
    ''' EP regions '''
    angular_gyrus = [107, 108]
    frontal_operculum = [119, 120]
    frontal_pole = [121, 122]
    inferior_occipital_gyrus = [129, 130]
    inferior_temporal_gyrus = [133, 134]
    lateral_orbital_gyrus = [137, 138]
    middle_frontal_gyrus = [143,144]
    middle_occipital_gyrus = [145, 146]
    medial_temporal_gyrus = [155, 156]
    occipital_pole = [157, 158]
    opercular_inferior_frontal_gyrus = [163, 164]
    orbital_inferior_frontal_gyrus = [165, 166]
    posterior_central_gyrus = [177, 178]
    precentral_gyrus = [183, 184]
    superior_frontal_gyrus = [191, 192]
    supplementary_motor_cortex = [193, 194]
    supramarginal_gyrus = [195, 196]
    superior_occipital_gyrus = [197,198]
    superior_parietal_lobule = [199, 200]
    superior_temporal_gyrus = [201, 202]
    temporal_lobe = [203, 204]
    triangular_inferior_frontal_gyrus = [205, 206]

    ''' TP regions '''
    amygdala = [32, 33]
    cerebellum = [39, 40]
    cerebral_white_matter = [45, 46]
    hippocampus = [48, 49]
    inferior_lateral_ventricle = [50, 51]
    lateral_ventricle = [52, 53]
    putamen = [58, 59]
    ventral_dc = [62, 63]
    periventricular_white_matter = [66,67]
    temporal_white_matter = [81,89]
    insula_white_matter = [82, 90]
    cingulate_white_matter = [83, 91]
    frontal_white_matter = [84, 92]
    occipital_white_matter = [85, 93]
    parietal_white_matter = [86,94]
    corpus_callosum = [87]
    claustrum = [96, 97]
    anterior_cingulate_gyrus = [101, 102]
    anterior_insula = [103, 104]
    anterior_orbital_gyrus = [105, 106]
    calcarine_cortex = [109,110]
    central_operculum = [113, 114]
    cuneus = [115, 116]
    entorhinal = [117, 118]
    fusiform_gyrus = [123, 124]
    gyrus_rectus = [125, 126]
    lingual_gyrus = [135, 136]
    middle_cingulate_gyrus = [139, 140]
    medial_frontal_cortex = [141, 142]
    medial_orbital_gyrus = [147, 148]
    precentral_gyrus_medial_segment = [151, 152]
    sfg_medial_segment = [153, 154]
    posterior_cingulate_gyrus = [167, 168]
    precuneus = [169, 170]
    parahippocampal_gyrus = [171, 172]
    posterior_insula = [173, 174]
    parietal_operculum = [175, 176]
    posterior_orbital_gyrus = [179, 180]
    planun_polare = [181, 182]
    traverse_temporal_gyrus = [207, 208]

    def __init__(self):
        self.ep = []
        self.tp = []
        self.filename = None

        self.all_eps = self.angular_gyrus + \
                       self.frontal_operculum + \
                       self.frontal_pole + \
                       self.inferior_occipital_gyrus + \
                       self.inferior_temporal_gyrus + \
                       self.lateral_orbital_gyrus + \
                       self.middle_frontal_gyrus + \
                       self.middle_occipital_gyrus + \
                       self.medial_temporal_gyrus + \
                       self.occipital_pole + \
                       self.opercular_inferior_frontal_gyrus + \
                       self.orbital_inferior_frontal_gyrus + \
                       self.posterior_central_gyrus + \
                       self.precentral_gyrus + \
                       self.superior_frontal_gyrus + \
                       self.supplementary_motor_cortex + \
                       self.supramarginal_gyrus + \
                       self.superior_occipital_gyrus + \
                       self.superior_parietal_lobule + \
                       self.superior_temporal_gyrus + \
                       self.temporal_lobe + \
                       self.triangular_inferior_frontal_gyrus

    def get_type(self, ep=None, tp=None):
        ''' returns a vector (6x1) indicating which group type the ep/tp pair belongs to '''
        # type = np.zeros(6, dtype=np.bool)

        self.ep_superior_frontal_gyrus()
        if np.isin(ep, self.ep) and np.isin(tp, self.tp): return 'sfg'
        self.ep_middle_frontal_gyrus()
        if np.isin(ep, self.ep) and np.isin(tp, self.tp): return 'mfg'
        self.ep_inferior_frontal_orbital_gyrus()
        if np.isin(ep, self.ep) and np.isin(tp, self.tp): return 'ifog'
        self.ep_temporal_gyrus()
        if np.isin(ep, self.ep) and np.isin(tp, self.tp): return 'tg'
        self.ep_anterior_posterior_central_gyrus()
        if np.isin(ep, self.ep) and np.isin(tp, self.tp): return 'apcg'
        self.ep_parietal_occipital()
        if np.isin(ep, self.ep) and np.isin(tp, self.tp): return 'po'

    def ep_frontal_pole(self):
        ''' DISCARD '''
        gif_ep = self.frontal_pole
        gif_tp = self.frontal_white_matter + self.medial_orbital_gyrus

        self.ep = gif_ep
        self.tp = gif_tp

    def ep_superior_frontal_gyrus(self):
        ''' IPCAI2021 '''
        gif_ep = self.superior_frontal_gyrus + self.supplementary_motor_cortex + \
                 self.frontal_pole
        gif_tp = self.cingulate_white_matter + \
                 self.frontal_white_matter + \
                 self.claustrum + \
                 self.anterior_cingulate_gyrus + \
                 self.anterior_insula + \
                 self.anterior_orbital_gyrus + \
                 self.gyrus_rectus + \
                 self.middle_cingulate_gyrus + \
                 self.medial_frontal_cortex + \
                 self.medial_orbital_gyrus + \
                 self.sfg_medial_segment + \
                 self.posterior_orbital_gyrus
        gif_tp += self.superior_frontal_gyrus + self.supplementary_motor_cortex

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_sfg.npy'

    def ep_middle_frontal_gyrus(self):
        ''' IPCAI2021 '''
        gif_ep = self.middle_frontal_gyrus
        gif_tp = self.cerebral_white_matter + \
                 self.lateral_ventricle + \
                 self.insula_white_matter + \
                 self.cingulate_white_matter + \
                 self.frontal_white_matter + \
                 self.corpus_callosum + \
                 self.claustrum + \
                 self.anterior_cingulate_gyrus + \
                 self.anterior_insula + \
                 self.anterior_orbital_gyrus + \
                 self.central_operculum + \
                 self.gyrus_rectus + \
                 self.middle_cingulate_gyrus + \
                 self.medial_frontal_cortex + \
                 self.medial_orbital_gyrus + \
                 self.sfg_medial_segment + \
                 self.posterior_orbital_gyrus
        gif_tp += self.middle_frontal_gyrus + \
                  self.frontal_operculum + \
                  self.lateral_orbital_gyrus + \
                  self.superior_frontal_gyrus + \
                  self.supplementary_motor_cortex

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_mfg.npy'

    def ep_inferior_frontal_orbital_gyrus(self):
        ''' IPCAI2021 '''
        gif_ep = self.frontal_operculum + \
                 self.lateral_orbital_gyrus + \
                 self.opercular_inferior_frontal_gyrus + \
                 self.orbital_inferior_frontal_gyrus + \
                 self.triangular_inferior_frontal_gyrus
        gif_tp = self.cerebral_white_matter + \
                 self.putamen + \
                 self.frontal_white_matter + \
                 self.corpus_callosum + \
                 self.claustrum + \
                 self.anterior_cingulate_gyrus + \
                 self.anterior_insula + \
                 self.central_operculum + \
                 self.gyrus_rectus + \
                 self.medial_frontal_cortex + \
                 self.medial_orbital_gyrus
        gif_tp += self.frontal_operculum + \
                  self.lateral_orbital_gyrus + \
                  self.opercular_inferior_frontal_gyrus + \
                  self.triangular_inferior_frontal_gyrus

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_ifog.npy'

    def ep_temporal_gyrus(self):
        ''' IPCAI2021 '''
        # ep_inferior_medial_temporal_gyrus()  + ep_superior_temporal_gyrus()
        gif_ep = self.inferior_temporal_gyrus + \
                 self.medial_temporal_gyrus + \
                 self.temporal_lobe + \
                 self.superior_temporal_gyrus
        gif_tp = self.amygdala + \
                 self.cerebellum + \
                 self.cerebral_white_matter + \
                 self.hippocampus + \
                 self.inferior_lateral_ventricle + \
                 self.ventral_dc + \
                 self.temporal_white_matter + \
                 self.insula_white_matter + \
                 self.frontal_white_matter + \
                 self.occipital_white_matter + \
                 self.entorhinal + \
                 self.fusiform_gyrus + \
                 self.lingual_gyrus + \
                 self.parahippocampal_gyrus + \
                 self.lateral_ventricle + \
                 self.periventricular_white_matter + \
                 self.claustrum + \
                 self.posterior_cingulate_gyrus + \
                 self.planun_polare + \
                 self.traverse_temporal_gyrus
        gif_tp += self.inferior_temporal_gyrus + \
                  self.temporal_lobe + \
                  self.medial_temporal_gyrus

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_tg.npy'

    def ep_inferior_medial_temporal_gyrus(self):
        ''' IPCAI2021 '''
        gif_ep = self.inferior_temporal_gyrus + \
                 self.medial_temporal_gyrus + \
                 self.temporal_lobe
        gif_tp = self.amygdala + \
                 self.cerebral_white_matter + \
                 self.hippocampus + \
                 self.inferior_lateral_ventricle + \
                 self.temporal_white_matter + \
                 self.insula_white_matter + \
                 self.frontal_white_matter + \
                 self.occipital_white_matter + \
                 self.entorhinal + \
                 self.fusiform_gyrus + \
                 self.lingual_gyrus + \
                 self.parahippocampal_gyrus
        gif_tp += self.inferior_temporal_gyrus + \
                  self.temporal_lobe

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_imtg.npy'

    def ep_superior_temporal_gyrus(self):
        ''' IPCAI2021 '''
        gif_ep = self.superior_temporal_gyrus
        gif_tp = self.lateral_ventricle + \
                 self.periventricular_white_matter + \
                 self.temporal_white_matter + \
                 self.insula_white_matter + \
                 self.claustrum + \
                 self.posterior_cingulate_gyrus + \
                 self.planun_polare + \
                 self.traverse_temporal_gyrus
        gif_tp += self.medial_temporal_gyrus

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_stg.npy'

    def ep_anterior_posterior_central_gyrus(self):
        ''' IPCAI2021 '''
        gif_ep = self.posterior_central_gyrus + \
                 self.precentral_gyrus
        gif_tp = self.cerebral_white_matter + \
                 self.insula_white_matter + \
                 self.cingulate_white_matter + \
                 self.frontal_white_matter + \
                 self.parietal_white_matter + \
                 self.corpus_callosum + \
                 self.claustrum + \
                 self.anterior_insula + \
                 self.central_operculum + \
                 self.middle_cingulate_gyrus + \
                 self.precentral_gyrus_medial_segment + \
                 self.posterior_cingulate_gyrus + \
                 self.posterior_insula
        gif_tp += self.posterior_central_gyrus + \
                  self.precentral_gyrus + \
                  self.frontal_operculum + \
                  self.supplementary_motor_cortex

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_apcg.npy'

    def ep_parietal_occipital(self):
        ''' IPCAI2021 '''
        # ep_parietal_lobule() + ep_occipital()
        gif_ep = self.angular_gyrus + \
                 self.supramarginal_gyrus + \
                 self.superior_parietal_lobule + \
                 self.inferior_occipital_gyrus + \
                 self.middle_occipital_gyrus + \
                 self.occipital_pole + \
                 self.superior_occipital_gyrus
        gif_tp = self.cerebral_white_matter + \
                 self.cingulate_white_matter + \
                 self.parietal_white_matter + \
                 self.corpus_callosum + \
                 self.middle_cingulate_gyrus + \
                 self.posterior_cingulate_gyrus + \
                 self.precuneus + \
                 self.posterior_insula + \
                 self.parietal_operculum + \
                 self.amygdala + \
                 self.lateral_ventricle + \
                 self.temporal_white_matter + \
                 self.occipital_white_matter + \
                 self.calcarine_cortex + \
                 self.cuneus + \
                 self.fusiform_gyrus + \
                 self.lingual_gyrus
        gif_tp += self.angular_gyrus + \
                  self.superior_parietal_lobule + \
                  self.inferior_occipital_gyrus + \
                  self.superior_temporal_gyrus

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_parocc.npy'

    def ep_parietal_lobule(self):
        ''' IPCAI2021 '''
        gif_ep = self.angular_gyrus + \
                 self.supramarginal_gyrus + \
                 self.superior_parietal_lobule
        gif_tp = self.cerebral_white_matter + \
                 self.cingulate_white_matter + \
                 self.parietal_white_matter + \
                 self.corpus_callosum + \
                 self.middle_cingulate_gyrus + \
                 self.posterior_cingulate_gyrus + \
                 self.precuneus + \
                 self.posterior_insula + \
                 self.parietal_operculum
        gif_tp += self.angular_gyrus + \
                  self.superior_parietal_lobule

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_par.npy'

    def ep_occipital(self):
        ''' IPCAI2021 '''
        gif_ep = self.inferior_occipital_gyrus + \
                 self.middle_occipital_gyrus + \
                 self.occipital_pole + \
                 self.superior_occipital_gyrus
        gif_tp = self.amygdala + \
                 self.lateral_ventricle + \
                 self.temporal_white_matter + \
                 self.cingulate_white_matter + \
                 self.occipital_white_matter + \
                 self.calcarine_cortex + \
                 self.cuneus + \
                 self.fusiform_gyrus + \
                 self.lingual_gyrus + \
                 self.precuneus
        gif_tp += self.inferior_occipital_gyrus + \
                  self.superior_temporal_gyrus

        self.ep = gif_ep
        self.tp = gif_tp
        self.filename = 'filter_occ.npy'

    def tp_amygdala(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [32, 33]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_cerebellum(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [39,40]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_lateral_whitematter(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [45,81,82,85,46,89,90,93]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [46,89,90,93]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [45,81,82,85]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_mfg(self, hemisphere=None):
        ''' straight '''
        gif_ep, gif_tp = [143, 144], [45, 46]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_hippocampus(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [48, 49]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_lateral_ventricle(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [50, 51]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_entorhinal(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [117,118]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_fusiform_gyrus(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [123,124]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_inferior_temporal_gyrus(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [133,134]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_lingual_gyrus(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [135,136]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_parahippocampal_gyrus(self, hemisphere=None):
        gif_ep, gif_tp = [155, 156], [171,172]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp

    def tp_smc(self, hemisphere=None):
        gif_ep, gif_tp = [191, 192], [193, 194]
        if hemisphere == 'L':
            gif_ep = [gif_ep[1]]
            gif_tp = [gif_tp[1]]
        elif hemisphere == 'R':
            gif_ep = [gif_ep[0]]
            gif_tp = [gif_tp[0]]
        self.ep += gif_ep
        self.tp += gif_tp
