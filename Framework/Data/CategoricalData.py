"""
Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import pandas as pd


class CategoricalData:

    def __init__(self):
        self.stylet_cat = ['stylet']
        self.stylet_cols = ['stylet_-1', 'stylet_0', 'stylet_1']

        self.cwd_cat = ['cwd']
        self.cwd_cols = ['cwd_0', 'cwd_1', 'cwd_2', 'cwd_3']

        self.ep_cat = ['EP_region']
        self.ep_cols = []

        self.tp_cat = ['TP_region']
        self.tp_cols = []

        return

    def stylet(self, df):
        df_all = df

        # Stylet
        df['stylet_-1'] = 0
        df['stylet_0'] = 0
        df['stylet_1'] = 0
        df_cat = df[self.stylet_cat]
        df_cat = pd.get_dummies(df_cat, columns=self.stylet_cat)
        if 'stylet_-1.0' in df_cat.columns:
            df_all['stylet_-1'] = df_cat['stylet_-1.0']
        if 'stylet_0.0' in df_cat.columns:
            df_all.stylet_0 = df_cat['stylet_0.0']
        if 'stylet_1.0' in df_cat.columns:
            df_all.stylet_1 = df_cat['stylet_1.0']

        return df_all

    def cwd(self, df):
        df_all = df

        # Cortex White Deep
        df['cwd_0'] = 0
        df['cwd_1'] = 0
        df['cwd_2'] = 0
        df['cwd_3'] = 0
        df_cat = df[self.cwd_cat]
        df_cat = pd.get_dummies(df_cat, columns=self.cwd_cat)
        if 'cwd_0' in df_cat.columns:
            df_all['cwd_0'] = df_cat['cwd_0']
        if 'cwd_1' in df_cat.columns:
            df_all['cwd_1'] = df_cat['cwd_1']
        if 'cwd_2' in df_cat.columns:
            df_all['cwd_2'] = df_cat['cwd_2']
        if 'cwd_3' in df_cat.columns:
            df_all['cwd_3'] = df_cat['cwd_3']

        return df_all

    def ep_by_lobes(self, df):
        df_all = df

        # ENTRY POINT
        ep_outside_region = [0, 1, 2, 3, 4]
        ep_frontal_region = [119, 120, 121, 122, 191, 192, 143, 144, 205, 206, 137, 138, 165, 166, 163, 164]
        ep_central_region = [183, 184, 177, 178]
        ep_temporal_region = [203, 204, 201, 202, 155, 156, 133, 134]
        ep_parietal_region = [195, 196, 107, 108, 94, 199, 200]
        ep_occipital_region = [157, 158, 145, 146, 129, 130, 197, 198, 93]
        print("Unique EP_region (sample) = ", df['EP_region'].unique())
        df['EP_lobe'] = df['EP_region']
        df.loc[df['EP_region'].isin(ep_outside_region), 'EP_lobe'] = 0
        df.loc[df['EP_region'].isin(ep_frontal_region), 'EP_lobe'] = 1
        df.loc[df['EP_region'].isin(ep_central_region), 'EP_lobe'] = 2
        df.loc[df['EP_region'].isin(ep_temporal_region), 'EP_lobe'] = 3
        df.loc[df['EP_region'].isin(ep_parietal_region), 'EP_lobe'] = 4
        df.loc[df['EP_region'].isin(ep_occipital_region), 'EP_lobe'] = 5
        print("EP_lobe", df['EP_lobe'].shape, df['EP_lobe'].unique())
        self.ep_cols = ['EP_lobe_0', 'EP_lobe_1', 'EP_lobe_2', 'EP_lobe_3', 'EP_lobe_4', 'EP_lobe_5']
        df['EP_lobe_0'] = 0
        df['EP_lobe_1'] = 0
        df['EP_lobe_2'] = 0
        df['EP_lobe_3'] = 0
        df['EP_lobe_4'] = 0
        df['EP_lobe_5'] = 0
        df_cat = df[['EP_lobe']]
        df_cat = pd.get_dummies(df_cat, columns=['EP_lobe'])
        if 'EP_lobe_0.0' in df_cat.columns:
            df_all['EP_lobe_0'] = df_cat['EP_lobe_0.0']
        if 'EP_lobe_1.0' in df_cat.columns:
            df_all['EP_lobe_1'] = df_cat['EP_lobe_1.0']
        if 'EP_lobe_2.0' in df_cat.columns:
            df_all['EP_lobe_2'] = df_cat['EP_lobe_2.0']
        if 'EP_lobe_3.0' in df_cat.columns:
            df_all['EP_lobe_3'] = df_cat['EP_lobe_3.0']
        if 'EP_lobe_4.0' in df_cat.columns:
            df_all['EP_lobe_4'] = df_cat['EP_lobe_4.0']
        if 'EP_lobe_5.0' in df_cat.columns:
            df_all['EP_lobe_5'] = df_cat['EP_lobe_5.0']

        return df_all

    def tp_by_lobes(self, df):
        df_all = df

        # TARGET POINT
        # for region in TP:
        #     if not np.isin(region, lobes):
        #         print('region {} not in lobes'.format(region))
        tp_outside = [0, 1, 2, 3, 4]
        tp_frontal_region = [84, 92, 179, 180, 163, 164, 125, 126, 147, 148, 137, 138, 141, 142, 143, 144, 153, 154, 191, 192,
                             193, 194, 105, 106, 119, 120, 205, 206]
        tp_central_region = [113, 114, 177, 178, 149, 150, 151, 152, 183, 184]
        tp_temporal_region = [87, 88, 81, 89, 171, 172, 201, 202, 203, 204, 123, 124, 117, 118, 39, 40, 48, 49, 32, 33,
                              52, 53, 62, 63, 60, 61, 58, 59, 66, 67, 96, 97, 50, 51]
        tp_parietal_region = [86, 94, 175, 176, 169, 170, 107, 108, 109, 110]
        tp_occipital_region = [85, 93, 115, 116, 135, 136]
        tp_insula_region = [82, 90, 173, 174, 103, 104, 45, 46]  # 45 46 is white matter
        tp_cingulum_region = [83, 91, 167, 168, 139, 140, 101, 102]
        print("Unique TP_region (sample) = ", df['TP_region'].unique())
        df['TP_lobe'] = df['TP_region']
        df.loc[df['TP_region'].isin(tp_outside), 'TP_lobe'] = 0
        df.loc[df['TP_region'].isin(tp_frontal_region), 'TP_lobe'] = 1
        df.loc[df['TP_region'].isin(tp_central_region), 'TP_lobe'] = 2
        df.loc[df['TP_region'].isin(tp_temporal_region), 'TP_lobe'] = 3
        df.loc[df['TP_region'].isin(tp_parietal_region), 'TP_lobe'] = 4
        df.loc[df['TP_region'].isin(tp_occipital_region), 'TP_lobe'] = 5
        df.loc[df['TP_region'].isin(tp_insula_region), 'TP_lobe'] = 6
        df.loc[df['TP_region'].isin(tp_cingulum_region), 'TP_lobe'] = 7
        print("TP_lobe", df['TP_lobe'].shape, df['TP_lobe'].unique())
        self.tp_cols = ['TP_lobe_0', 'TP_lobe_1', 'TP_lobe_2', 'TP_lobe_3', 'TP_lobe_4', 'TP_lobe_5', 'TP_lobe_6', 'TP_lobe_7']
        df['TP_lobe_0'] = 0
        df['TP_lobe_1'] = 0
        df['TP_lobe_2'] = 0
        df['TP_lobe_3'] = 0
        df['TP_lobe_4'] = 0
        df['TP_lobe_5'] = 0
        df['TP_lobe_6'] = 0
        df['TP_lobe_7'] = 0
        df_cat = df[['TP_lobe']]
        df_cat = pd.get_dummies(df_cat, columns=['TP_lobe'])
        if 'TP_lobe_0.0' in df_cat.columns:
            df_all['TP_lobe_0'] = df_cat['TP_lobe_0.0']
        if 'TP_lobe_1.0' in df_cat.columns:
            df_all['TP_lobe_1'] = df_cat['TP_lobe_1.0']
        if 'TP_lobe_2.0' in df_cat.columns:
            df_all['TP_lobe_2'] = df_cat['TP_lobe_2.0']
        if 'TP_lobe_3.0' in df_cat.columns:
            df_all['TP_lobe_3'] = df_cat['TP_lobe_3.0']
        if 'TP_lobe_4.0' in df_cat.columns:
            df_all['TP_lobe_4'] = df_cat['TP_lobe_4.0']
        if 'TP_lobe_5.0' in df_cat.columns:
            df_all['TP_lobe_5'] = df_cat['TP_lobe_5.0']
        if 'TP_lobe_6.0' in df_cat.columns:
            df_all['TP_lobe_6'] = df_cat['TP_lobe_6.0']
        if 'TP_lobe_7.0' in df_cat.columns:
            df_all['TP_lobe_7'] = df_cat['TP_lobe_7.0']

        return df_all

    def by_LRlobes(self, df):
        """ Categories based on Left/Right Lobes """
        df_all = df
        cat_cols = []

        # ENTRY POINT
        ep_outside_region = [0, 1, 2, 3]
        ep_frontal_region_L = [120, 122, 192, 144, 206, 138, 166, 164]
        ep_frontal_region_R = [119, 121, 191, 143, 205, 137, 165, 163]
        ep_central_region_L = [184, 178]
        ep_central_region_R = [183, 177]
        ep_temporal_region_L = [204, 202, 156, 134]
        ep_temporal_region_R = [203, 201, 155, 133]
        ep_parietal_region_L = [196, 108, 200, 94]
        ep_parietal_region_R = [195, 107, 199]
        ep_occipital_region_L = [158, 146, 130, 198, 93]
        ep_occipital_region_R = [157, 145, 129, 197]
        print("Unique EP_region = ", sorted(df['EP_region'].unique()))
        df['EP_continent'] = df['EP_region']
        df.loc[df['EP_region'].isin(ep_outside_region), 'EP_continent'] = 0
        df.loc[df['EP_region'].isin(ep_frontal_region_L), 'EP_continent'] = 1
        df.loc[df['EP_region'].isin(ep_frontal_region_R), 'EP_continent'] = 2
        df.loc[df['EP_region'].isin(ep_central_region_L), 'EP_continent'] = 3
        df.loc[df['EP_region'].isin(ep_central_region_R), 'EP_continent'] = 4
        df.loc[df['EP_region'].isin(ep_temporal_region_L), 'EP_continent'] = 5
        df.loc[df['EP_region'].isin(ep_temporal_region_R), 'EP_continent'] = 6
        df.loc[df['EP_region'].isin(ep_parietal_region_L), 'EP_continent'] = 7
        df.loc[df['EP_region'].isin(ep_parietal_region_R), 'EP_continent'] = 8
        df.loc[df['EP_region'].isin(ep_occipital_region_L), 'EP_continent'] = 9
        df.loc[df['EP_region'].isin(ep_occipital_region_R), 'EP_continent'] = 10
        print("EP_continent", df['EP_continent'].shape, sorted(df['EP_continent'].unique()))
        cat_cols += ['EP_continent_0', 'EP_continent_1', 'EP_continent_2', 'EP_continent_3', 'EP_continent_4',
                     'EP_continent_5', 'EP_continent_6', 'EP_continent_7', 'EP_continent_8', 'EP_continent_9',
                     'EP_continent_10']
        df['EP_continent_0'] = 0
        df['EP_continent_1'] = 0
        df['EP_continent_2'] = 0
        df['EP_continent_3'] = 0
        df['EP_continent_4'] = 0
        df['EP_continent_5'] = 0
        df['EP_continent_6'] = 0
        df['EP_continent_7'] = 0
        df['EP_continent_8'] = 0
        df['EP_continent_9'] = 0
        df['EP_continent_10'] = 0
        cols_cat = ['EP_continent']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'EP_continent_0' in df_cat.columns:
            df_all['EP_continent_0'] = df_cat['EP_continent_0']
        if 'EP_continent_1' in df_cat.columns:
            df_all['EP_continent_1'] = df_cat['EP_continent_1']
        if 'EP_continent_2' in df_cat.columns:
            df_all['EP_continent_2'] = df_cat['EP_continent_2']
        if 'EP_continent_3' in df_cat.columns:
            df_all['EP_continent_3'] = df_cat['EP_continent_3']
        if 'EP_continent_4' in df_cat.columns:
            df_all['EP_continent_4'] = df_cat['EP_continent_4']
        if 'EP_continent_5' in df_cat.columns:
            df_all['EP_continent_5'] = df_cat['EP_continent_5']
        if 'EP_continent_6' in df_cat.columns:
            df_all['EP_continent_6'] = df_cat['EP_continent_6']
        if 'EP_continent_7' in df_cat.columns:
            df_all['EP_continent_7'] = df_cat['EP_continent_7']
        if 'EP_continent_8' in df_cat.columns:
            df_all['EP_continent_8'] = df_cat['EP_continent_8']
        if 'EP_continent_9' in df_cat.columns:
            df_all['EP_continent_9'] = df_cat['EP_continent_9']
        if 'EP_continent_10' in df_cat.columns:
            df_all['EP_continent_10'] = df_cat['EP_continent_10']

        # TARGET POINT
        tp_frontal_region_L = [92, 180, 164, 126, 148, 138, 142, 154, 192, 194, 106, 120, 206]
        tp_frontal_region_R = [84, 179, 163, 125, 147, 137, 141, 153, 191, 193, 105, 119, 205]
        tp_central_region_L = [114, 178, 150, 152, 184]
        tp_central_region_R = [113, 177, 149, 151, 183]
        tp_temporal_region_L = [88, 89, 172, 202, 204, 124, 118, 40, 49, 33, 53, 63, 61, 59, 67, 97, 51]
        tp_temporal_region_R = [87, 81, 171, 201, 203, 123, 117, 39, 48, 32, 52, 62, 60, 58, 66, 96, 50]
        tp_parietal_region_L = [94, 176, 170, 108, 110]
        tp_parietal_region_R = [86, 175, 169, 107, 109]
        tp_occipital_region_L = [93, 116, 136]
        tp_occipital_region_R = [85, 115, 135]
        tp_insula_region_L = [90, 174, 104, 46]
        tp_insula_region_R = [82, 173, 103, 45]
        tp_cingulum_region_L = [4, 91, 168, 140, 102]
        tp_cingulum_region_R = [2, 83, 167, 139, 101]
        print("Unique TP_region = ", sorted(df['TP_region'].unique()))
        df['TP_continent'] = df['TP_region']
        df.loc[df['TP_region'].isin(tp_frontal_region_L), 'TP_continent'] = 0
        df.loc[df['TP_region'].isin(tp_frontal_region_R), 'TP_continent'] = 1
        df.loc[df['TP_region'].isin(tp_central_region_L), 'TP_continent'] = 2
        df.loc[df['TP_region'].isin(tp_central_region_R), 'TP_continent'] = 3
        df.loc[df['TP_region'].isin(tp_temporal_region_L), 'TP_continent'] = 4
        df.loc[df['TP_region'].isin(tp_temporal_region_R), 'TP_continent'] = 5
        df.loc[df['TP_region'].isin(tp_parietal_region_L), 'TP_continent'] = 6
        df.loc[df['TP_region'].isin(tp_parietal_region_R), 'TP_continent'] = 7
        df.loc[df['TP_region'].isin(tp_occipital_region_L), 'TP_continent'] = 8
        df.loc[df['TP_region'].isin(tp_occipital_region_R), 'TP_continent'] = 9
        df.loc[df['TP_region'].isin(tp_insula_region_L), 'TP_continent'] = 10
        df.loc[df['TP_region'].isin(tp_insula_region_R), 'TP_continent'] = 11
        df.loc[df['TP_region'].isin(tp_cingulum_region_L), 'TP_continent'] = 12
        df.loc[df['TP_region'].isin(tp_cingulum_region_R), 'TP_continent'] = 13
        print("TP_continent", df['TP_continent'].shape, sorted(df['TP_continent'].unique()))
        cat_cols += ['TP_continent_0', 'TP_continent_1', 'TP_continent_2', 'TP_continent_3', 'TP_continent_4',
                     'TP_continent_5', 'TP_continent_6', 'TP_continent_7', 'TP_continent_8', 'TP_continent_9',
                     'TP_continent_10', 'TP_continent_11', 'TP_continent_12', 'TP_continent_13']
        df['TP_continent_0'] = 0
        df['TP_continent_1'] = 0
        df['TP_continent_2'] = 0
        df['TP_continent_3'] = 0
        df['TP_continent_4'] = 0
        df['TP_continent_5'] = 0
        df['TP_continent_6'] = 0
        df['TP_continent_7'] = 0
        df['TP_continent_8'] = 0
        df['TP_continent_9'] = 0
        df['TP_continent_10'] = 0
        df['TP_continent_11'] = 0
        df['TP_continent_12'] = 0
        df['TP_continent_13'] = 0
        cols_cat = ['TP_continent']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'TP_continent_0' in df_cat.columns:
            df_all['TP_continent_0'] = df_cat['TP_continent_0']
        if 'TP_continent_1' in df_cat.columns:
            df_all['TP_continent_1'] = df_cat['TP_continent_1']
        if 'TP_continent_2' in df_cat.columns:
            df_all['TP_continent_2'] = df_cat['TP_continent_2']
        if 'TP_continent_3' in df_cat.columns:
            df_all['TP_continent_3'] = df_cat['TP_continent_3']
        if 'TP_continent_4' in df_cat.columns:
            df_all['TP_continent_4'] = df_cat['TP_continent_4']
        if 'TP_continent_5' in df_cat.columns:
            df_all['TP_continent_5'] = df_cat['TP_continent_5']
        if 'TP_continent_6' in df_cat.columns:
            df_all['TP_continent_6'] = df_cat['TP_continent_6']
        if 'TP_continent_7' in df_cat.columns:
            df_all['TP_continent_7'] = df_cat['TP_continent_7']
        if 'TP_continent_8' in df_cat.columns:
            df_all['TP_continent_8'] = df_cat['TP_continent_8']
        if 'TP_continent_9' in df_cat.columns:
            df_all['TP_continent_9'] = df_cat['TP_continent_9']
        if 'TP_continent_10' in df_cat.columns:
            df_all['TP_continent_10'] = df_cat['TP_continent_10']
        if 'TP_continent_11' in df_cat.columns:
            df_all['TP_continent_11'] = df_cat['TP_continent_11']
        if 'TP_continent_12' in df_cat.columns:
            df_all['TP_continent_12'] = df_cat['TP_continent_12']
        if 'TP_continent_13' in df_cat.columns:
            df_all['TP_continent_13'] = df_cat['TP_continent_13']

        # Cortex White Deep
        cat_cols += ['cwd_0', 'cwd_1', 'cwd_2', 'cwd_3']
        df['cwd_0'] = 0
        df['cwd_1'] = 0
        df['cwd_2'] = 0
        df['cwd_3'] = 0
        cols_cat = ['cwd']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'cwd_0' in df_cat.columns:
            df_all['cwd_0'] = df_cat['cwd_0']
        if 'cwd_1' in df_cat.columns:
            df_all['cwd_1'] = df_cat['cwd_1']
        if 'cwd_2' in df_cat.columns:
            df_all['cwd_2'] = df_cat['cwd_2']
        if 'cwd_3' in df_cat.columns:
            df_all['cwd_3'] = df_cat['cwd_3']

        # Stylet
        cat_cols += ['stylet_n', 'stylet_0', 'stylet_1']
        df['stylet_n'] = 0
        df['stylet_0'] = 0
        df['stylet_1'] = 0
        cols_cat = ['stylet']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'stylet_n' in df_cat.columns:
            df_all['stylet_n'] = df_cat['stylet_n']
        if 'stylet_0' in df_cat.columns:
            df_all['stylet_0'] = df_cat['stylet_0']
        if 'stylet_1' in df_cat.columns:
            df_all['stylet_1'] = df_cat['stylet_1']

        return df_all, cat_cols


    def by_gif_ep(self, df):
        """ Categories based on actual regions of EP """
        df_all = df
        cat_cols = []

        # ENTRY POINT
        ep_outside_region = [0, 1, 2, 3]
        ep_frontal_121 = [121]
        ep_frontal_122 = [122]
        ep_frontal_137 = [137]
        ep_frontal_138 = [138]
        ep_frontal_143 = [143]
        ep_frontal_144 = [144]
        ep_frontal_163 = [163]
        ep_frontal_164 = [164]
        ep_frontal_165 = [165]
        ep_frontal_166 = [166]
        ep_frontal_191 = [191]
        ep_frontal_192 = [192]
        ep_frontal_205 = [205]
        ep_frontal_206 = [206]
        ep_central_177 = [177]
        ep_central_178 = [178]
        ep_central_183 = [183]
        ep_central_184 = [184]
        ep_temporal_133 = [133]
        ep_temporal_134 = [134]
        ep_temporal_155 = [155]
        ep_temporal_156 = [156]
        ep_temporal_201 = [201]
        ep_temporal_202 = [202]
        ep_temporal_203 = [203]
        ep_temporal_204 = [204]
        ep_parietal_94 = [94]
        ep_parietal_107 = [107]
        ep_parietal_108 = [108]
        ep_parietal_195 = [195]
        ep_parietal_196 = [196]
        ep_parietal_199 = [199]
        ep_parietal_200 = [200]
        ep_occipital_93 = [93]
        ep_occipital_129 = [129]
        ep_occipital_130 = [130]
        ep_occipital_145 = [145]
        ep_occipital_146 = [146]
        ep_occipital_157 = [157]
        ep_occipital_158 = [158]
        ep_occipital_197 = [197]
        ep_occipital_198 = [198]
        print("Unique EP_region = ", sorted(df['EP_region'].unique()))
        df['EP_continent'] = df['EP_region']
        df.loc[df['EP_region'].isin(ep_outside_region), 'EP_continent'] = 0
        df.loc[df['EP_region'].isin(ep_frontal_121), 'EP_continent'] = 1
        df.loc[df['EP_region'].isin(ep_frontal_122), 'EP_continent'] = 2
        df.loc[df['EP_region'].isin(ep_frontal_137), 'EP_continent'] = 3
        df.loc[df['EP_region'].isin(ep_frontal_138), 'EP_continent'] = 4
        df.loc[df['EP_region'].isin(ep_frontal_143), 'EP_continent'] = 5
        df.loc[df['EP_region'].isin(ep_frontal_144), 'EP_continent'] = 6
        df.loc[df['EP_region'].isin(ep_frontal_163), 'EP_continent'] = 7
        df.loc[df['EP_region'].isin(ep_frontal_164), 'EP_continent'] = 8
        df.loc[df['EP_region'].isin(ep_frontal_165), 'EP_continent'] = 9
        df.loc[df['EP_region'].isin(ep_frontal_166), 'EP_continent'] = 10
        df.loc[df['EP_region'].isin(ep_frontal_191), 'EP_continent'] = 11
        df.loc[df['EP_region'].isin(ep_frontal_192), 'EP_continent'] = 12
        df.loc[df['EP_region'].isin(ep_frontal_205), 'EP_continent'] = 13
        df.loc[df['EP_region'].isin(ep_frontal_206), 'EP_continent'] = 14
        df.loc[df['EP_region'].isin(ep_central_177), 'EP_continent'] = 15
        df.loc[df['EP_region'].isin(ep_central_178), 'EP_continent'] = 16
        df.loc[df['EP_region'].isin(ep_central_183), 'EP_continent'] = 17
        df.loc[df['EP_region'].isin(ep_central_184), 'EP_continent'] = 18
        df.loc[df['EP_region'].isin(ep_temporal_133), 'EP_continent'] = 19
        df.loc[df['EP_region'].isin(ep_temporal_134), 'EP_continent'] = 20
        df.loc[df['EP_region'].isin(ep_temporal_155), 'EP_continent'] = 21
        df.loc[df['EP_region'].isin(ep_temporal_156), 'EP_continent'] = 22
        df.loc[df['EP_region'].isin(ep_temporal_201), 'EP_continent'] = 23
        df.loc[df['EP_region'].isin(ep_temporal_202), 'EP_continent'] = 24
        df.loc[df['EP_region'].isin(ep_temporal_203), 'EP_continent'] = 25
        df.loc[df['EP_region'].isin(ep_temporal_204), 'EP_continent'] = 26
        df.loc[df['EP_region'].isin(ep_parietal_94), 'EP_continent'] = 27
        df.loc[df['EP_region'].isin(ep_parietal_107), 'EP_continent'] = 28
        df.loc[df['EP_region'].isin(ep_parietal_108), 'EP_continent'] = 29
        df.loc[df['EP_region'].isin(ep_parietal_195), 'EP_continent'] = 30
        df.loc[df['EP_region'].isin(ep_parietal_196), 'EP_continent'] = 31
        df.loc[df['EP_region'].isin(ep_parietal_199), 'EP_continent'] = 32
        df.loc[df['EP_region'].isin(ep_parietal_200), 'EP_continent'] = 33
        df.loc[df['EP_region'].isin(ep_occipital_93), 'EP_continent'] = 34
        df.loc[df['EP_region'].isin(ep_occipital_129), 'EP_continent'] = 35
        df.loc[df['EP_region'].isin(ep_occipital_130), 'EP_continent'] = 36
        df.loc[df['EP_region'].isin(ep_occipital_145), 'EP_continent'] = 37
        df.loc[df['EP_region'].isin(ep_occipital_146), 'EP_continent'] = 38
        df.loc[df['EP_region'].isin(ep_occipital_157), 'EP_continent'] = 39
        df.loc[df['EP_region'].isin(ep_occipital_158), 'EP_continent'] = 40
        df.loc[df['EP_region'].isin(ep_occipital_197), 'EP_continent'] = 41
        df.loc[df['EP_region'].isin(ep_occipital_198), 'EP_continent'] = 42
        print("EP_continent", df['EP_continent'].shape, sorted(df['EP_continent'].unique()))
        cat_cols += ['EP_continent_0', 'EP_continent_1', 'EP_continent_2', 'EP_continent_3', 'EP_continent_4',
                     'EP_continent_5', 'EP_continent_6', 'EP_continent_7', 'EP_continent_8', 'EP_continent_9',
                     'EP_continent_10', 'EP_continent_11', 'EP_continent_12', 'EP_continent_13', 'EP_continent_14',
                     'EP_continent_15', 'EP_continent_16', 'EP_continent_17', 'EP_continent_18', 'EP_continent_19',
                     'EP_continent_20', 'EP_continent_21', 'EP_continent_22', 'EP_continent_23', 'EP_continent_24',
                     'EP_continent_25', 'EP_continent_26', 'EP_continent_27', 'EP_continent_28', 'EP_continent_29',
                     'EP_continent_30', 'EP_continent_31', 'EP_continent_32', 'EP_continent_33', 'EP_continent_34',
                     'EP_continent_35', 'EP_continent_36', 'EP_continent_37', 'EP_continent_38', 'EP_continent_39',
                     'EP_continent_40', 'EP_continent_41', 'EP_continent_42']
        df['EP_continent_0'] = 0
        df['EP_continent_1'] = 0
        df['EP_continent_2'] = 0
        df['EP_continent_3'] = 0
        df['EP_continent_4'] = 0
        df['EP_continent_5'] = 0
        df['EP_continent_6'] = 0
        df['EP_continent_7'] = 0
        df['EP_continent_8'] = 0
        df['EP_continent_9'] = 0
        df['EP_continent_10'] = 0
        df['EP_continent_11'] = 0
        df['EP_continent_12'] = 0
        df['EP_continent_13'] = 0
        df['EP_continent_14'] = 0
        df['EP_continent_15'] = 0
        df['EP_continent_16'] = 0
        df['EP_continent_17'] = 0
        df['EP_continent_18'] = 0
        df['EP_continent_19'] = 0
        df['EP_continent_20'] = 0
        df['EP_continent_21'] = 0
        df['EP_continent_22'] = 0
        df['EP_continent_23'] = 0
        df['EP_continent_24'] = 0
        df['EP_continent_25'] = 0
        df['EP_continent_26'] = 0
        df['EP_continent_27'] = 0
        df['EP_continent_28'] = 0
        df['EP_continent_29'] = 0
        df['EP_continent_30'] = 0
        df['EP_continent_31'] = 0
        df['EP_continent_32'] = 0
        df['EP_continent_33'] = 0
        df['EP_continent_34'] = 0
        df['EP_continent_35'] = 0
        df['EP_continent_36'] = 0
        df['EP_continent_37'] = 0
        df['EP_continent_38'] = 0
        df['EP_continent_39'] = 0
        df['EP_continent_40'] = 0
        df['EP_continent_41'] = 0
        df['EP_continent_42'] = 0
        cols_cat = ['EP_continent']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'EP_continent_0' in df_cat.columns:
            df_all['EP_continent_0'] = df_cat['EP_continent_0']
        if 'EP_continent_1' in df_cat.columns:
            df_all['EP_continent_1'] = df_cat['EP_continent_1']
        if 'EP_continent_2' in df_cat.columns:
            df_all['EP_continent_2'] = df_cat['EP_continent_2']
        if 'EP_continent_3' in df_cat.columns:
            df_all['EP_continent_3'] = df_cat['EP_continent_3']
        if 'EP_continent_4' in df_cat.columns:
            df_all['EP_continent_4'] = df_cat['EP_continent_4']
        if 'EP_continent_5' in df_cat.columns:
            df_all['EP_continent_5'] = df_cat['EP_continent_5']
        if 'EP_continent_6' in df_cat.columns:
            df_all['EP_continent_6'] = df_cat['EP_continent_6']
        if 'EP_continent_7' in df_cat.columns:
            df_all['EP_continent_7'] = df_cat['EP_continent_7']
        if 'EP_continent_8' in df_cat.columns:
            df_all['EP_continent_8'] = df_cat['EP_continent_8']
        if 'EP_continent_9' in df_cat.columns:
            df_all['EP_continent_9'] = df_cat['EP_continent_9']
        if 'EP_continent_10' in df_cat.columns:
            df_all['EP_continent_10'] = df_cat['EP_continent_10']
        if 'EP_continent_11' in df_cat.columns:
            df_all['EP_continent_11'] = df_cat['EP_continent_11']
        if 'EP_continent_12' in df_cat.columns:
            df_all['EP_continent_12'] = df_cat['EP_continent_12']
        if 'EP_continent_13' in df_cat.columns:
            df_all['EP_continent_13'] = df_cat['EP_continent_13']
        if 'EP_continent_14' in df_cat.columns:
            df_all['EP_continent_14'] = df_cat['EP_continent_14']
        if 'EP_continent_15' in df_cat.columns:
            df_all['EP_continent_15'] = df_cat['EP_continent_15']
        if 'EP_continent_16' in df_cat.columns:
            df_all['EP_continent_16'] = df_cat['EP_continent_16']
        if 'EP_continent_17' in df_cat.columns:
            df_all['EP_continent_17'] = df_cat['EP_continent_17']
        if 'EP_continent_18' in df_cat.columns:
            df_all['EP_continent_18'] = df_cat['EP_continent_18']
        if 'EP_continent_19' in df_cat.columns:
            df_all['EP_continent_19'] = df_cat['EP_continent_19']
        if 'EP_continent_20' in df_cat.columns:
            df_all['EP_continent_20'] = df_cat['EP_continent_20']
        if 'EP_continent_21' in df_cat.columns:
            df_all['EP_continent_21'] = df_cat['EP_continent_21']
        if 'EP_continent_22' in df_cat.columns:
            df_all['EP_continent_22'] = df_cat['EP_continent_22']
        if 'EP_continent_23' in df_cat.columns:
            df_all['EP_continent_23'] = df_cat['EP_continent_23']
        if 'EP_continent_24' in df_cat.columns:
            df_all['EP_continent_24'] = df_cat['EP_continent_24']
        if 'EP_continent_25' in df_cat.columns:
            df_all['EP_continent_25'] = df_cat['EP_continent_25']
        if 'EP_continent_26' in df_cat.columns:
            df_all['EP_continent_26'] = df_cat['EP_continent_26']
        if 'EP_continent_27' in df_cat.columns:
            df_all['EP_continent_27'] = df_cat['EP_continent_27']
        if 'EP_continent_28' in df_cat.columns:
            df_all['EP_continent_28'] = df_cat['EP_continent_28']
        if 'EP_continent_29' in df_cat.columns:
            df_all['EP_continent_29'] = df_cat['EP_continent_29']
        if 'EP_continent_30' in df_cat.columns:
            df_all['EP_continent_30'] = df_cat['EP_continent_30']
        if 'EP_continent_31' in df_cat.columns:
            df_all['EP_continent_31'] = df_cat['EP_continent_31']
        if 'EP_continent_32' in df_cat.columns:
            df_all['EP_continent_32'] = df_cat['EP_continent_32']
        if 'EP_continent_33' in df_cat.columns:
            df_all['EP_continent_33'] = df_cat['EP_continent_33']
        if 'EP_continent_34' in df_cat.columns:
            df_all['EP_continent_34'] = df_cat['EP_continent_34']
        if 'EP_continent_35' in df_cat.columns:
            df_all['EP_continent_35'] = df_cat['EP_continent_35']
        if 'EP_continent_36' in df_cat.columns:
            df_all['EP_continent_36'] = df_cat['EP_continent_36']
        if 'EP_continent_37' in df_cat.columns:
            df_all['EP_continent_37'] = df_cat['EP_continent_37']
        if 'EP_continent_38' in df_cat.columns:
            df_all['EP_continent_38'] = df_cat['EP_continent_38']
        if 'EP_continent_39' in df_cat.columns:
            df_all['EP_continent_39'] = df_cat['EP_continent_39']
        if 'EP_continent_40' in df_cat.columns:
            df_all['EP_continent_40'] = df_cat['EP_continent_40']
        if 'EP_continent_41' in df_cat.columns:
            df_all['EP_continent_41'] = df_cat['EP_continent_41']
        if 'EP_continent_42' in df_cat.columns:
            df_all['EP_continent_42'] = df_cat['EP_continent_42']

        # TARGET POINT
        tp_frontal_region_L = [92, 180, 164, 126, 148, 138, 142, 154, 192, 194, 106, 120, 206]
        tp_frontal_region_R = [84, 179, 163, 125, 147, 137, 141, 153, 191, 193, 105, 119, 205]
        tp_central_region_L = [114, 178, 150, 152, 184]
        tp_central_region_R = [113, 177, 149, 151, 183]
        tp_temporal_region_L = [88, 89, 172, 202, 204, 124, 118, 40, 49, 33, 53, 63, 61, 59, 67, 97, 51]
        tp_temporal_region_R = [87, 81, 171, 201, 203, 123, 117, 39, 48, 32, 52, 62, 60, 58, 66, 96, 50]
        tp_parietal_region_L = [94, 176, 170, 108, 110]
        tp_parietal_region_R = [86, 175, 169, 107, 109]
        tp_occipital_region_L = [93, 116, 136]
        tp_occipital_region_R = [85, 115, 135]
        tp_insula_region_L = [90, 174, 104, 46]
        tp_insula_region_R = [82, 173, 103, 45]
        tp_cingulum_region_L = [4, 91, 168, 140, 102]
        tp_cingulum_region_R = [2, 83, 167, 139, 101]
        print("Unique TP_region = ", sorted(df['TP_region'].unique()))
        df['TP_continent'] = df['TP_region']
        df.loc[df['TP_region'].isin(tp_frontal_region_L), 'TP_continent'] = 0
        df.loc[df['TP_region'].isin(tp_frontal_region_R), 'TP_continent'] = 1
        df.loc[df['TP_region'].isin(tp_central_region_L), 'TP_continent'] = 2
        df.loc[df['TP_region'].isin(tp_central_region_R), 'TP_continent'] = 3
        df.loc[df['TP_region'].isin(tp_temporal_region_L), 'TP_continent'] = 4
        df.loc[df['TP_region'].isin(tp_temporal_region_R), 'TP_continent'] = 5
        df.loc[df['TP_region'].isin(tp_parietal_region_L), 'TP_continent'] = 6
        df.loc[df['TP_region'].isin(tp_parietal_region_R), 'TP_continent'] = 7
        df.loc[df['TP_region'].isin(tp_occipital_region_L), 'TP_continent'] = 8
        df.loc[df['TP_region'].isin(tp_occipital_region_R), 'TP_continent'] = 9
        df.loc[df['TP_region'].isin(tp_insula_region_L), 'TP_continent'] = 10
        df.loc[df['TP_region'].isin(tp_insula_region_R), 'TP_continent'] = 11
        df.loc[df['TP_region'].isin(tp_cingulum_region_L), 'TP_continent'] = 12
        df.loc[df['TP_region'].isin(tp_cingulum_region_R), 'TP_continent'] = 13
        print("TP_continent", df['TP_continent'].shape, sorted(df['TP_continent'].unique()))
        cat_cols += ['TP_continent_0', 'TP_continent_1', 'TP_continent_2', 'TP_continent_3', 'TP_continent_4',
                     'TP_continent_5', 'TP_continent_6', 'TP_continent_7', 'TP_continent_8', 'TP_continent_9',
                     'TP_continent_10', 'TP_continent_11', 'TP_continent_12', 'TP_continent_13']
        df['TP_continent_0'] = 0
        df['TP_continent_1'] = 0
        df['TP_continent_2'] = 0
        df['TP_continent_3'] = 0
        df['TP_continent_4'] = 0
        df['TP_continent_5'] = 0
        df['TP_continent_6'] = 0
        df['TP_continent_7'] = 0
        df['TP_continent_8'] = 0
        df['TP_continent_9'] = 0
        df['TP_continent_10'] = 0
        df['TP_continent_11'] = 0
        df['TP_continent_12'] = 0
        df['TP_continent_13'] = 0
        cols_cat = ['TP_continent']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'TP_continent_0' in df_cat.columns:
            df_all['TP_continent_0'] = df_cat['TP_continent_0']
        if 'TP_continent_1' in df_cat.columns:
            df_all['TP_continent_1'] = df_cat['TP_continent_1']
        if 'TP_continent_2' in df_cat.columns:
            df_all['TP_continent_2'] = df_cat['TP_continent_2']
        if 'TP_continent_3' in df_cat.columns:
            df_all['TP_continent_3'] = df_cat['TP_continent_3']
        if 'TP_continent_4' in df_cat.columns:
            df_all['TP_continent_4'] = df_cat['TP_continent_4']
        if 'TP_continent_5' in df_cat.columns:
            df_all['TP_continent_5'] = df_cat['TP_continent_5']
        if 'TP_continent_6' in df_cat.columns:
            df_all['TP_continent_6'] = df_cat['TP_continent_6']
        if 'TP_continent_7' in df_cat.columns:
            df_all['TP_continent_7'] = df_cat['TP_continent_7']
        if 'TP_continent_8' in df_cat.columns:
            df_all['TP_continent_8'] = df_cat['TP_continent_8']
        if 'TP_continent_9' in df_cat.columns:
            df_all['TP_continent_9'] = df_cat['TP_continent_9']
        if 'TP_continent_10' in df_cat.columns:
            df_all['TP_continent_10'] = df_cat['TP_continent_10']
        if 'TP_continent_11' in df_cat.columns:
            df_all['TP_continent_11'] = df_cat['TP_continent_11']
        if 'TP_continent_12' in df_cat.columns:
            df_all['TP_continent_12'] = df_cat['TP_continent_12']
        if 'TP_continent_13' in df_cat.columns:
            df_all['TP_continent_13'] = df_cat['TP_continent_13']

        # Cortex White Deep
        cat_cols += ['cwd_0', 'cwd_1', 'cwd_2', 'cwd_3']
        df['cwd_0'] = 0
        df['cwd_1'] = 0
        df['cwd_2'] = 0
        df['cwd_3'] = 0
        cols_cat = ['cwd']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'cwd_0' in df_cat.columns:
            df_all['cwd_0'] = df_cat['cwd_0']
        if 'cwd_1' in df_cat.columns:
            df_all['cwd_1'] = df_cat['cwd_1']
        if 'cwd_2' in df_cat.columns:
            df_all['cwd_2'] = df_cat['cwd_2']
        if 'cwd_3' in df_cat.columns:
            df_all['cwd_3'] = df_cat['cwd_3']

        # Stylet
        cat_cols += ['stylet_n', 'stylet_0', 'stylet_1']
        df['stylet_n'] = 0
        df['stylet_0'] = 0
        df['stylet_1'] = 0
        cols_cat = ['stylet']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'stylet_n' in df_cat.columns:
            df_all['stylet_n'] = df_cat['stylet_n']
        if 'stylet_0' in df_cat.columns:
            df_all['stylet_0'] = df_cat['stylet_0']
        if 'stylet_1' in df_cat.columns:
            df_all['stylet_1'] = df_cat['stylet_1']

        return df_all, cat_cols


    def by_gif(self, df):
        """ Categories based on actual regions of EP & TP """
        df_all = df
        cat_cols = []

        # ENTRY POINT
        ep_outside_region = [0, 1, 2, 3]
        ep_frontal_121 = [121]
        ep_frontal_122 = [122]
        ep_frontal_137 = [137]
        ep_frontal_138 = [138]
        ep_frontal_143 = [143]
        ep_frontal_144 = [144]
        ep_frontal_163 = [163]
        ep_frontal_164 = [164]
        ep_frontal_165 = [165]
        ep_frontal_166 = [166]
        ep_frontal_191 = [191]
        ep_frontal_192 = [192]
        ep_frontal_205 = [205]
        ep_frontal_206 = [206]
        ep_central_177 = [177]
        ep_central_178 = [178]
        ep_central_183 = [183]
        ep_central_184 = [184]
        ep_temporal_133 = [133]
        ep_temporal_134 = [134]
        ep_temporal_155 = [155]
        ep_temporal_156 = [156]
        ep_temporal_201 = [201]
        ep_temporal_202 = [202]
        ep_temporal_203 = [203]
        ep_temporal_204 = [204]
        ep_parietal_94 = [94]
        ep_parietal_107 = [107]
        ep_parietal_108 = [108]
        ep_parietal_195 = [195]
        ep_parietal_196 = [196]
        ep_parietal_199 = [199]
        ep_parietal_200 = [200]
        ep_occipital_93 = [93]
        ep_occipital_129 = [129]
        ep_occipital_130 = [130]
        ep_occipital_145 = [145]
        ep_occipital_146 = [146]
        ep_occipital_157 = [157]
        ep_occipital_158 = [158]
        ep_occipital_197 = [197]
        ep_occipital_198 = [198]
        print("Unique EP_region = ", sorted(df['EP_region'].unique()))
        df['EP_continent'] = df['EP_region']
        df.loc[df['EP_region'].isin(ep_outside_region), 'EP_continent'] = 0
        df.loc[df['EP_region'].isin(ep_frontal_121), 'EP_continent'] = 1
        df.loc[df['EP_region'].isin(ep_frontal_122), 'EP_continent'] = 2
        df.loc[df['EP_region'].isin(ep_frontal_137), 'EP_continent'] = 3
        df.loc[df['EP_region'].isin(ep_frontal_138), 'EP_continent'] = 4
        df.loc[df['EP_region'].isin(ep_frontal_143), 'EP_continent'] = 5
        df.loc[df['EP_region'].isin(ep_frontal_144), 'EP_continent'] = 6
        df.loc[df['EP_region'].isin(ep_frontal_163), 'EP_continent'] = 7
        df.loc[df['EP_region'].isin(ep_frontal_164), 'EP_continent'] = 8
        df.loc[df['EP_region'].isin(ep_frontal_165), 'EP_continent'] = 9
        df.loc[df['EP_region'].isin(ep_frontal_166), 'EP_continent'] = 10
        df.loc[df['EP_region'].isin(ep_frontal_191), 'EP_continent'] = 11
        df.loc[df['EP_region'].isin(ep_frontal_192), 'EP_continent'] = 12
        df.loc[df['EP_region'].isin(ep_frontal_205), 'EP_continent'] = 13
        df.loc[df['EP_region'].isin(ep_frontal_206), 'EP_continent'] = 14
        df.loc[df['EP_region'].isin(ep_central_177), 'EP_continent'] = 15
        df.loc[df['EP_region'].isin(ep_central_178), 'EP_continent'] = 16
        df.loc[df['EP_region'].isin(ep_central_183), 'EP_continent'] = 17
        df.loc[df['EP_region'].isin(ep_central_184), 'EP_continent'] = 18
        df.loc[df['EP_region'].isin(ep_temporal_133), 'EP_continent'] = 19
        df.loc[df['EP_region'].isin(ep_temporal_134), 'EP_continent'] = 20
        df.loc[df['EP_region'].isin(ep_temporal_155), 'EP_continent'] = 21
        df.loc[df['EP_region'].isin(ep_temporal_156), 'EP_continent'] = 22
        df.loc[df['EP_region'].isin(ep_temporal_201), 'EP_continent'] = 23
        df.loc[df['EP_region'].isin(ep_temporal_202), 'EP_continent'] = 24
        df.loc[df['EP_region'].isin(ep_temporal_203), 'EP_continent'] = 25
        df.loc[df['EP_region'].isin(ep_temporal_204), 'EP_continent'] = 26
        df.loc[df['EP_region'].isin(ep_parietal_94), 'EP_continent'] = 27
        df.loc[df['EP_region'].isin(ep_parietal_107), 'EP_continent'] = 28
        df.loc[df['EP_region'].isin(ep_parietal_108), 'EP_continent'] = 29
        df.loc[df['EP_region'].isin(ep_parietal_195), 'EP_continent'] = 30
        df.loc[df['EP_region'].isin(ep_parietal_196), 'EP_continent'] = 31
        df.loc[df['EP_region'].isin(ep_parietal_199), 'EP_continent'] = 32
        df.loc[df['EP_region'].isin(ep_parietal_200), 'EP_continent'] = 33
        df.loc[df['EP_region'].isin(ep_occipital_93), 'EP_continent'] = 34
        df.loc[df['EP_region'].isin(ep_occipital_129), 'EP_continent'] = 35
        df.loc[df['EP_region'].isin(ep_occipital_130), 'EP_continent'] = 36
        df.loc[df['EP_region'].isin(ep_occipital_145), 'EP_continent'] = 37
        df.loc[df['EP_region'].isin(ep_occipital_146), 'EP_continent'] = 38
        df.loc[df['EP_region'].isin(ep_occipital_157), 'EP_continent'] = 39
        df.loc[df['EP_region'].isin(ep_occipital_158), 'EP_continent'] = 40
        df.loc[df['EP_region'].isin(ep_occipital_197), 'EP_continent'] = 41
        df.loc[df['EP_region'].isin(ep_occipital_198), 'EP_continent'] = 42
        print("EP_continent", df['EP_continent'].shape, sorted(df['EP_continent'].unique()))
        cat_cols += ['EP_continent_0', 'EP_continent_1', 'EP_continent_2', 'EP_continent_3', 'EP_continent_4',
                     'EP_continent_5', 'EP_continent_6', 'EP_continent_7', 'EP_continent_8', 'EP_continent_9',
                     'EP_continent_10', 'EP_continent_11', 'EP_continent_12', 'EP_continent_13', 'EP_continent_14',
                     'EP_continent_15', 'EP_continent_16', 'EP_continent_17', 'EP_continent_18', 'EP_continent_19',
                     'EP_continent_20', 'EP_continent_21', 'EP_continent_22', 'EP_continent_23', 'EP_continent_24',
                     'EP_continent_25', 'EP_continent_26', 'EP_continent_27', 'EP_continent_28', 'EP_continent_29',
                     'EP_continent_30', 'EP_continent_31', 'EP_continent_32', 'EP_continent_33', 'EP_continent_34',
                     'EP_continent_35', 'EP_continent_36', 'EP_continent_37', 'EP_continent_38', 'EP_continent_39',
                     'EP_continent_40', 'EP_continent_41', 'EP_continent_42']
        df['EP_continent_0'] = 0
        df['EP_continent_1'] = 0
        df['EP_continent_2'] = 0
        df['EP_continent_3'] = 0
        df['EP_continent_4'] = 0
        df['EP_continent_5'] = 0
        df['EP_continent_6'] = 0
        df['EP_continent_7'] = 0
        df['EP_continent_8'] = 0
        df['EP_continent_9'] = 0
        df['EP_continent_10'] = 0
        df['EP_continent_11'] = 0
        df['EP_continent_12'] = 0
        df['EP_continent_13'] = 0
        df['EP_continent_14'] = 0
        df['EP_continent_15'] = 0
        df['EP_continent_16'] = 0
        df['EP_continent_17'] = 0
        df['EP_continent_18'] = 0
        df['EP_continent_19'] = 0
        df['EP_continent_20'] = 0
        df['EP_continent_21'] = 0
        df['EP_continent_22'] = 0
        df['EP_continent_23'] = 0
        df['EP_continent_24'] = 0
        df['EP_continent_25'] = 0
        df['EP_continent_26'] = 0
        df['EP_continent_27'] = 0
        df['EP_continent_28'] = 0
        df['EP_continent_29'] = 0
        df['EP_continent_30'] = 0
        df['EP_continent_31'] = 0
        df['EP_continent_32'] = 0
        df['EP_continent_33'] = 0
        df['EP_continent_34'] = 0
        df['EP_continent_35'] = 0
        df['EP_continent_36'] = 0
        df['EP_continent_37'] = 0
        df['EP_continent_38'] = 0
        df['EP_continent_39'] = 0
        df['EP_continent_40'] = 0
        df['EP_continent_41'] = 0
        df['EP_continent_42'] = 0
        cols_cat = ['EP_continent']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'EP_continent_0' in df_cat.columns:
            df_all['EP_continent_0'] = df_cat['EP_continent_0']
        if 'EP_continent_1' in df_cat.columns:
            df_all['EP_continent_1'] = df_cat['EP_continent_1']
        if 'EP_continent_2' in df_cat.columns:
            df_all['EP_continent_2'] = df_cat['EP_continent_2']
        if 'EP_continent_3' in df_cat.columns:
            df_all['EP_continent_3'] = df_cat['EP_continent_3']
        if 'EP_continent_4' in df_cat.columns:
            df_all['EP_continent_4'] = df_cat['EP_continent_4']
        if 'EP_continent_5' in df_cat.columns:
            df_all['EP_continent_5'] = df_cat['EP_continent_5']
        if 'EP_continent_6' in df_cat.columns:
            df_all['EP_continent_6'] = df_cat['EP_continent_6']
        if 'EP_continent_7' in df_cat.columns:
            df_all['EP_continent_7'] = df_cat['EP_continent_7']
        if 'EP_continent_8' in df_cat.columns:
            df_all['EP_continent_8'] = df_cat['EP_continent_8']
        if 'EP_continent_9' in df_cat.columns:
            df_all['EP_continent_9'] = df_cat['EP_continent_9']
        if 'EP_continent_10' in df_cat.columns:
            df_all['EP_continent_10'] = df_cat['EP_continent_10']
        if 'EP_continent_11' in df_cat.columns:
            df_all['EP_continent_11'] = df_cat['EP_continent_11']
        if 'EP_continent_12' in df_cat.columns:
            df_all['EP_continent_12'] = df_cat['EP_continent_12']
        if 'EP_continent_13' in df_cat.columns:
            df_all['EP_continent_13'] = df_cat['EP_continent_13']
        if 'EP_continent_14' in df_cat.columns:
            df_all['EP_continent_14'] = df_cat['EP_continent_14']
        if 'EP_continent_15' in df_cat.columns:
            df_all['EP_continent_15'] = df_cat['EP_continent_15']
        if 'EP_continent_16' in df_cat.columns:
            df_all['EP_continent_16'] = df_cat['EP_continent_16']
        if 'EP_continent_17' in df_cat.columns:
            df_all['EP_continent_17'] = df_cat['EP_continent_17']
        if 'EP_continent_18' in df_cat.columns:
            df_all['EP_continent_18'] = df_cat['EP_continent_18']
        if 'EP_continent_19' in df_cat.columns:
            df_all['EP_continent_19'] = df_cat['EP_continent_19']
        if 'EP_continent_20' in df_cat.columns:
            df_all['EP_continent_20'] = df_cat['EP_continent_20']
        if 'EP_continent_21' in df_cat.columns:
            df_all['EP_continent_21'] = df_cat['EP_continent_21']
        if 'EP_continent_22' in df_cat.columns:
            df_all['EP_continent_22'] = df_cat['EP_continent_22']
        if 'EP_continent_23' in df_cat.columns:
            df_all['EP_continent_23'] = df_cat['EP_continent_23']
        if 'EP_continent_24' in df_cat.columns:
            df_all['EP_continent_24'] = df_cat['EP_continent_24']
        if 'EP_continent_25' in df_cat.columns:
            df_all['EP_continent_25'] = df_cat['EP_continent_25']
        if 'EP_continent_26' in df_cat.columns:
            df_all['EP_continent_26'] = df_cat['EP_continent_26']
        if 'EP_continent_27' in df_cat.columns:
            df_all['EP_continent_27'] = df_cat['EP_continent_27']
        if 'EP_continent_28' in df_cat.columns:
            df_all['EP_continent_28'] = df_cat['EP_continent_28']
        if 'EP_continent_29' in df_cat.columns:
            df_all['EP_continent_29'] = df_cat['EP_continent_29']
        if 'EP_continent_30' in df_cat.columns:
            df_all['EP_continent_30'] = df_cat['EP_continent_30']
        if 'EP_continent_31' in df_cat.columns:
            df_all['EP_continent_31'] = df_cat['EP_continent_31']
        if 'EP_continent_32' in df_cat.columns:
            df_all['EP_continent_32'] = df_cat['EP_continent_32']
        if 'EP_continent_33' in df_cat.columns:
            df_all['EP_continent_33'] = df_cat['EP_continent_33']
        if 'EP_continent_34' in df_cat.columns:
            df_all['EP_continent_34'] = df_cat['EP_continent_34']
        if 'EP_continent_35' in df_cat.columns:
            df_all['EP_continent_35'] = df_cat['EP_continent_35']
        if 'EP_continent_36' in df_cat.columns:
            df_all['EP_continent_36'] = df_cat['EP_continent_36']
        if 'EP_continent_37' in df_cat.columns:
            df_all['EP_continent_37'] = df_cat['EP_continent_37']
        if 'EP_continent_38' in df_cat.columns:
            df_all['EP_continent_38'] = df_cat['EP_continent_38']
        if 'EP_continent_39' in df_cat.columns:
            df_all['EP_continent_39'] = df_cat['EP_continent_39']
        if 'EP_continent_40' in df_cat.columns:
            df_all['EP_continent_40'] = df_cat['EP_continent_40']
        if 'EP_continent_41' in df_cat.columns:
            df_all['EP_continent_41'] = df_cat['EP_continent_41']
        if 'EP_continent_42' in df_cat.columns:
            df_all['EP_continent_42'] = df_cat['EP_continent_42']


        # TARGET POINT
        tp_frontal_84 = [84]
        tp_frontal_92 = [92]
        tp_frontal_105 = [105]
        tp_frontal_106 = [106]
        tp_frontal_119 = [119]
        tp_frontal_120 = [120]
        tp_frontal_125 = [125]
        tp_frontal_126 = [126]
        tp_frontal_137 = [137]
        tp_frontal_138 = [138]
        tp_frontal_141 = [141]
        tp_frontal_142 = [142]
        tp_frontal_147 = [147]
        tp_frontal_148 = [148]
        tp_frontal_153 = [153]
        tp_frontal_154 = [154]
        tp_frontal_163 = [163]
        tp_frontal_164 = [164]
        tp_frontal_179 = [179]
        tp_frontal_180 = [180]
        tp_frontal_191 = [191]
        tp_frontal_192 = [192]
        tp_frontal_193 = [193]
        tp_frontal_194 = [194]
        tp_frontal_205 = [205]
        tp_frontal_206 = [206]
        tp_central_113 = [113]
        tp_central_114 = [114]
        tp_central_149 = [149]
        tp_central_150 = [150]
        tp_central_151 = [151]
        tp_central_152 = [152]
        tp_central_177 = [177]
        tp_central_178 = [178]
        tp_central_183 = [183]
        tp_central_184 = [184]
        tp_temporal_32 = [32]
        tp_temporal_33 = [33]
        tp_temporal_39 = [39]
        tp_temporal_40 = [40]
        tp_temporal_48 = [48]
        tp_temporal_49 = [49]
        tp_temporal_50 = [50]
        tp_temporal_51 = [51]
        tp_temporal_52 = [52]
        tp_temporal_53 = [53]
        tp_temporal_58 = [58]
        tp_temporal_59 = [59]
        tp_temporal_60 = [60]
        tp_temporal_61 = [61]
        tp_temporal_62 = [62]
        tp_temporal_63 = [63]
        tp_temporal_66 = [66]
        tp_temporal_67 = [67]
        tp_temporal_81 = [81]
        tp_temporal_87 = [87]
        tp_temporal_88 = [88]
        tp_temporal_89 = [89]
        tp_temporal_96 = [96]
        tp_temporal_97 = [97]
        tp_temporal_117 = [117]
        tp_temporal_118 = [118]
        tp_temporal_123 = [123]
        tp_temporal_124 = [124]
        tp_temporal_171 = [171]
        tp_temporal_172 = [172]
        tp_temporal_201 = [201]
        tp_temporal_202 = [202]
        tp_temporal_203 = [203]
        tp_temporal_204 = [204]
        tp_parietal_86 = [86]
        tp_parietal_94 = [94]
        tp_parietal_107 = [107]
        tp_parietal_108 = [108]
        tp_parietal_109 = [109]
        tp_parietal_110 = [110]
        tp_parietal_169 = [169]
        tp_parietal_170 = [170]
        tp_parietal_175 = [175]
        tp_parietal_176 = [176]
        tp_occipital_85 = [85]
        tp_occipital_93 = [93]
        tp_occipital_115 = [115]
        tp_occipital_116 = [116]
        tp_occipital_135 = [135]
        tp_occipital_136 = [136]
        tp_insula_45 = [45]
        tp_insula_46 = [46]
        tp_insula_82 = [82]
        tp_insula_90 = [90]
        tp_insula_103 = [103]
        tp_insula_104 = [104]
        tp_insula_173 = [173]
        tp_insula_174 = [174]
        tp_cingulum_2 = [2]
        tp_cingulum_4 = [4]
        tp_cingulum_83 = [83]
        tp_cingulum_91 = [91]
        tp_cingulum_101 = [101]
        tp_cingulum_102 = [102]
        tp_cingulum_139 = [139]
        tp_cingulum_140 = [140]
        tp_cingulum_167 = [167]
        tp_cingulum_168 = [168]
        print("Unique TP_region = ", sorted(df['TP_region'].unique()))
        df['TP_continent'] = df['TP_region']
        df.loc[df['TP_region'].isin(tp_frontal_84), 'TP_continent'] = 0
        df.loc[df['TP_region'].isin(tp_frontal_92), 'TP_continent'] = 1
        df.loc[df['TP_region'].isin(tp_frontal_105), 'TP_continent'] = 2
        df.loc[df['TP_region'].isin(tp_frontal_106), 'TP_continent'] = 3
        df.loc[df['TP_region'].isin(tp_frontal_119), 'TP_continent'] = 4
        df.loc[df['TP_region'].isin(tp_frontal_120), 'TP_continent'] = 5
        df.loc[df['TP_region'].isin(tp_frontal_125), 'TP_continent'] = 6
        df.loc[df['TP_region'].isin(tp_frontal_126), 'TP_continent'] = 7
        df.loc[df['TP_region'].isin(tp_frontal_137), 'TP_continent'] = 8
        df.loc[df['TP_region'].isin(tp_frontal_138), 'TP_continent'] = 9
        df.loc[df['TP_region'].isin(tp_frontal_141), 'TP_continent'] = 10
        df.loc[df['TP_region'].isin(tp_frontal_142), 'TP_continent'] = 11
        df.loc[df['TP_region'].isin(tp_frontal_147), 'TP_continent'] = 12
        df.loc[df['TP_region'].isin(tp_frontal_148), 'TP_continent'] = 13
        df.loc[df['TP_region'].isin(tp_frontal_153), 'TP_continent'] = 14
        df.loc[df['TP_region'].isin(tp_frontal_154), 'TP_continent'] = 15
        df.loc[df['TP_region'].isin(tp_frontal_163), 'TP_continent'] = 16
        df.loc[df['TP_region'].isin(tp_frontal_164), 'TP_continent'] = 17
        df.loc[df['TP_region'].isin(tp_frontal_179), 'TP_continent'] = 18
        df.loc[df['TP_region'].isin(tp_frontal_180), 'TP_continent'] = 19
        df.loc[df['TP_region'].isin(tp_frontal_191), 'TP_continent'] = 20
        df.loc[df['TP_region'].isin(tp_frontal_192), 'TP_continent'] = 21
        df.loc[df['TP_region'].isin(tp_frontal_193), 'TP_continent'] = 22
        df.loc[df['TP_region'].isin(tp_frontal_194), 'TP_continent'] = 23
        df.loc[df['TP_region'].isin(tp_frontal_205), 'TP_continent'] = 24
        df.loc[df['TP_region'].isin(tp_frontal_206), 'TP_continent'] = 25
        df.loc[df['TP_region'].isin(tp_central_113), 'TP_continent'] = 26
        df.loc[df['TP_region'].isin(tp_central_114), 'TP_continent'] = 27
        df.loc[df['TP_region'].isin(tp_central_149), 'TP_continent'] = 28
        df.loc[df['TP_region'].isin(tp_central_150), 'TP_continent'] = 29
        df.loc[df['TP_region'].isin(tp_central_151), 'TP_continent'] = 30
        df.loc[df['TP_region'].isin(tp_central_152), 'TP_continent'] = 31
        df.loc[df['TP_region'].isin(tp_central_177), 'TP_continent'] = 32
        df.loc[df['TP_region'].isin(tp_central_178), 'TP_continent'] = 33
        df.loc[df['TP_region'].isin(tp_central_183), 'TP_continent'] = 34
        df.loc[df['TP_region'].isin(tp_central_184), 'TP_continent'] = 35
        df.loc[df['TP_region'].isin(tp_temporal_32), 'TP_continent'] = 36
        df.loc[df['TP_region'].isin(tp_temporal_33), 'TP_continent'] = 37
        df.loc[df['TP_region'].isin(tp_temporal_39), 'TP_continent'] = 38
        df.loc[df['TP_region'].isin(tp_temporal_40), 'TP_continent'] = 39
        df.loc[df['TP_region'].isin(tp_temporal_48), 'TP_continent'] = 40
        df.loc[df['TP_region'].isin(tp_temporal_49), 'TP_continent'] = 41
        df.loc[df['TP_region'].isin(tp_temporal_50), 'TP_continent'] = 42
        df.loc[df['TP_region'].isin(tp_temporal_51), 'TP_continent'] = 43
        df.loc[df['TP_region'].isin(tp_temporal_52), 'TP_continent'] = 44
        df.loc[df['TP_region'].isin(tp_temporal_53), 'TP_continent'] = 45
        df.loc[df['TP_region'].isin(tp_temporal_58), 'TP_continent'] = 46
        df.loc[df['TP_region'].isin(tp_temporal_59), 'TP_continent'] = 47
        df.loc[df['TP_region'].isin(tp_temporal_60), 'TP_continent'] = 48
        df.loc[df['TP_region'].isin(tp_temporal_61), 'TP_continent'] = 49
        df.loc[df['TP_region'].isin(tp_temporal_62), 'TP_continent'] = 50
        df.loc[df['TP_region'].isin(tp_temporal_63), 'TP_continent'] = 51
        df.loc[df['TP_region'].isin(tp_temporal_66), 'TP_continent'] = 52
        df.loc[df['TP_region'].isin(tp_temporal_67), 'TP_continent'] = 53
        df.loc[df['TP_region'].isin(tp_temporal_81), 'TP_continent'] = 54
        df.loc[df['TP_region'].isin(tp_temporal_87), 'TP_continent'] = 55
        df.loc[df['TP_region'].isin(tp_temporal_88), 'TP_continent'] = 56
        df.loc[df['TP_region'].isin(tp_temporal_89), 'TP_continent'] = 57
        df.loc[df['TP_region'].isin(tp_temporal_96), 'TP_continent'] = 58
        df.loc[df['TP_region'].isin(tp_temporal_97), 'TP_continent'] = 59
        df.loc[df['TP_region'].isin(tp_temporal_117), 'TP_continent'] = 60
        df.loc[df['TP_region'].isin(tp_temporal_118), 'TP_continent'] = 61
        df.loc[df['TP_region'].isin(tp_temporal_123), 'TP_continent'] = 62
        df.loc[df['TP_region'].isin(tp_temporal_124), 'TP_continent'] = 63
        df.loc[df['TP_region'].isin(tp_temporal_171), 'TP_continent'] = 64
        df.loc[df['TP_region'].isin(tp_temporal_172), 'TP_continent'] = 65
        df.loc[df['TP_region'].isin(tp_temporal_201), 'TP_continent'] = 66
        df.loc[df['TP_region'].isin(tp_temporal_202), 'TP_continent'] = 67
        df.loc[df['TP_region'].isin(tp_temporal_203), 'TP_continent'] = 68
        df.loc[df['TP_region'].isin(tp_temporal_204), 'TP_continent'] = 69
        df.loc[df['TP_region'].isin(tp_parietal_86), 'TP_continent'] = 70
        df.loc[df['TP_region'].isin(tp_parietal_94), 'TP_continent'] = 71
        df.loc[df['TP_region'].isin(tp_parietal_107), 'TP_continent'] = 72
        df.loc[df['TP_region'].isin(tp_parietal_108), 'TP_continent'] = 73
        df.loc[df['TP_region'].isin(tp_parietal_109), 'TP_continent'] = 74
        df.loc[df['TP_region'].isin(tp_parietal_110), 'TP_continent'] = 75
        df.loc[df['TP_region'].isin(tp_parietal_169), 'TP_continent'] = 76
        df.loc[df['TP_region'].isin(tp_parietal_170), 'TP_continent'] = 77
        df.loc[df['TP_region'].isin(tp_parietal_175), 'TP_continent'] = 78
        df.loc[df['TP_region'].isin(tp_parietal_176), 'TP_continent'] = 79
        df.loc[df['TP_region'].isin(tp_occipital_85), 'TP_continent'] = 80
        df.loc[df['TP_region'].isin(tp_occipital_93), 'TP_continent'] = 81
        df.loc[df['TP_region'].isin(tp_occipital_115), 'TP_continent'] = 82
        df.loc[df['TP_region'].isin(tp_occipital_116), 'TP_continent'] = 83
        df.loc[df['TP_region'].isin(tp_occipital_135), 'TP_continent'] = 84
        df.loc[df['TP_region'].isin(tp_occipital_136), 'TP_continent'] = 85
        df.loc[df['TP_region'].isin(tp_insula_45), 'TP_continent'] = 86
        df.loc[df['TP_region'].isin(tp_insula_46), 'TP_continent'] = 87
        df.loc[df['TP_region'].isin(tp_insula_82), 'TP_continent'] = 88
        df.loc[df['TP_region'].isin(tp_insula_90), 'TP_continent'] = 89
        df.loc[df['TP_region'].isin(tp_insula_103), 'TP_continent'] = 90
        df.loc[df['TP_region'].isin(tp_insula_104), 'TP_continent'] = 91
        df.loc[df['TP_region'].isin(tp_insula_173), 'TP_continent'] = 92
        df.loc[df['TP_region'].isin(tp_insula_174), 'TP_continent'] = 93
        df.loc[df['TP_region'].isin(tp_cingulum_2), 'TP_continent'] = 94
        df.loc[df['TP_region'].isin(tp_cingulum_4), 'TP_continent'] = 95
        df.loc[df['TP_region'].isin(tp_cingulum_83), 'TP_continent'] = 96
        df.loc[df['TP_region'].isin(tp_cingulum_91), 'TP_continent'] = 97
        df.loc[df['TP_region'].isin(tp_cingulum_101), 'TP_continent'] = 98
        df.loc[df['TP_region'].isin(tp_cingulum_102), 'TP_continent'] = 99
        df.loc[df['TP_region'].isin(tp_cingulum_139), 'TP_continent'] = 100
        df.loc[df['TP_region'].isin(tp_cingulum_140), 'TP_continent'] = 101
        df.loc[df['TP_region'].isin(tp_cingulum_167), 'TP_continent'] = 102
        df.loc[df['TP_region'].isin(tp_cingulum_168), 'TP_continent'] = 103
        print("TP_continent", df['TP_continent'].shape, sorted(df['TP_continent'].unique()))
        cat_cols += ['TP_continent_0', 'TP_continent_1', 'TP_continent_2', 'TP_continent_3', 'TP_continent_4', 'TP_continent_5', 'TP_continent_6', 'TP_continent_7', 'TP_continent_8', 'TP_continent_9',
                     'TP_continent_10', 'TP_continent_11', 'TP_continent_12', 'TP_continent_13', 'TP_continent_14', 'TP_continent_15', 'TP_continent_16', 'TP_continent_17', 'TP_continent_18', 'TP_continent_19',
                     'TP_continent_20', 'TP_continent_21', 'TP_continent_22', 'TP_continent_23', 'TP_continent_24', 'TP_continent_25', 'TP_continent_26', 'TP_continent_27', 'TP_continent_28', 'TP_continent_29',
                     'TP_continent_30', 'TP_continent_31', 'TP_continent_32', 'TP_continent_33', 'TP_continent_34', 'TP_continent_35', 'TP_continent_36', 'TP_continent_37', 'TP_continent_38', 'TP_continent_39',
                     'TP_continent_40', 'TP_continent_41', 'TP_continent_42', 'TP_continent_43', 'TP_continent_44', 'TP_continent_45', 'TP_continent_46', 'TP_continent_47', 'TP_continent_48', 'TP_continent_49',
                     'TP_continent_50', 'TP_continent_51', 'TP_continent_52', 'TP_continent_53', 'TP_continent_54', 'TP_continent_55', 'TP_continent_56', 'TP_continent_57', 'TP_continent_58', 'TP_continent_59',
                     'TP_continent_60', 'TP_continent_61', 'TP_continent_62', 'TP_continent_63', 'TP_continent_64', 'TP_continent_65', 'TP_continent_66', 'TP_continent_67', 'TP_continent_68', 'TP_continent_69',
                     'TP_continent_70', 'TP_continent_71', 'TP_continent_72', 'TP_continent_73', 'TP_continent_74', 'TP_continent_75', 'TP_continent_76', 'TP_continent_77', 'TP_continent_78', 'TP_continent_79',
                     'TP_continent_80', 'TP_continent_81', 'TP_continent_82', 'TP_continent_83', 'TP_continent_84', 'TP_continent_85', 'TP_continent_86', 'TP_continent_87', 'TP_continent_88', 'TP_continent_89',
                     'TP_continent_90', 'TP_continent_91', 'TP_continent_92', 'TP_continent_93', 'TP_continent_94', 'TP_continent_95', 'TP_continent_96', 'TP_continent_97', 'TP_continent_98', 'TP_continent_99',
                     'TP_continent_100', 'TP_continent_101', 'TP_continent_102', 'TP_continent_103']
        df['TP_continent_0'] = 0
        df['TP_continent_1'] = 0
        df['TP_continent_2'] = 0
        df['TP_continent_3'] = 0
        df['TP_continent_4'] = 0
        df['TP_continent_5'] = 0
        df['TP_continent_6'] = 0
        df['TP_continent_7'] = 0
        df['TP_continent_8'] = 0
        df['TP_continent_9'] = 0
        df['TP_continent_10'] = 0
        df['TP_continent_11'] = 0
        df['TP_continent_12'] = 0
        df['TP_continent_13'] = 0
        df['TP_continent_14'] = 0
        df['TP_continent_15'] = 0
        df['TP_continent_16'] = 0
        df['TP_continent_17'] = 0
        df['TP_continent_18'] = 0
        df['TP_continent_19'] = 0
        df['TP_continent_20'] = 0
        df['TP_continent_21'] = 0
        df['TP_continent_22'] = 0
        df['TP_continent_23'] = 0
        df['TP_continent_24'] = 0
        df['TP_continent_25'] = 0
        df['TP_continent_26'] = 0
        df['TP_continent_27'] = 0
        df['TP_continent_28'] = 0
        df['TP_continent_29'] = 0
        df['TP_continent_30'] = 0
        df['TP_continent_31'] = 0
        df['TP_continent_32'] = 0
        df['TP_continent_33'] = 0
        df['TP_continent_34'] = 0
        df['TP_continent_35'] = 0
        df['TP_continent_36'] = 0
        df['TP_continent_37'] = 0
        df['TP_continent_38'] = 0
        df['TP_continent_39'] = 0
        df['TP_continent_40'] = 0
        df['TP_continent_41'] = 0
        df['TP_continent_42'] = 0
        df['TP_continent_43'] = 0
        df['TP_continent_44'] = 0
        df['TP_continent_45'] = 0
        df['TP_continent_46'] = 0
        df['TP_continent_47'] = 0
        df['TP_continent_48'] = 0
        df['TP_continent_49'] = 0
        df['TP_continent_50'] = 0
        df['TP_continent_51'] = 0
        df['TP_continent_52'] = 0
        df['TP_continent_53'] = 0
        df['TP_continent_54'] = 0
        df['TP_continent_55'] = 0
        df['TP_continent_56'] = 0
        df['TP_continent_57'] = 0
        df['TP_continent_58'] = 0
        df['TP_continent_59'] = 0
        df['TP_continent_60'] = 0
        df['TP_continent_61'] = 0
        df['TP_continent_62'] = 0
        df['TP_continent_63'] = 0
        df['TP_continent_64'] = 0
        df['TP_continent_65'] = 0
        df['TP_continent_66'] = 0
        df['TP_continent_67'] = 0
        df['TP_continent_68'] = 0
        df['TP_continent_69'] = 0
        df['TP_continent_70'] = 0
        df['TP_continent_71'] = 0
        df['TP_continent_72'] = 0
        df['TP_continent_73'] = 0
        df['TP_continent_74'] = 0
        df['TP_continent_75'] = 0
        df['TP_continent_76'] = 0
        df['TP_continent_77'] = 0
        df['TP_continent_78'] = 0
        df['TP_continent_79'] = 0
        df['TP_continent_80'] = 0
        df['TP_continent_81'] = 0
        df['TP_continent_82'] = 0
        df['TP_continent_83'] = 0
        df['TP_continent_84'] = 0
        df['TP_continent_85'] = 0
        df['TP_continent_86'] = 0
        df['TP_continent_87'] = 0
        df['TP_continent_88'] = 0
        df['TP_continent_89'] = 0
        df['TP_continent_90'] = 0
        df['TP_continent_91'] = 0
        df['TP_continent_92'] = 0
        df['TP_continent_93'] = 0
        df['TP_continent_94'] = 0
        df['TP_continent_95'] = 0
        df['TP_continent_96'] = 0
        df['TP_continent_97'] = 0
        df['TP_continent_98'] = 0
        df['TP_continent_99'] = 0
        df['TP_continent_100'] = 0
        df['TP_continent_101'] = 0
        df['TP_continent_102'] = 0
        df['TP_continent_103'] = 0
        cols_cat = ['TP_continent']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'TP_continent_0' in df_cat.columns:
            df_all['TP_continent_0'] = df_cat['TP_continent_0']
        if 'TP_continent_1' in df_cat.columns:
            df_all['TP_continent_1'] = df_cat['TP_continent_1']
        if 'TP_continent_2' in df_cat.columns:
            df_all['TP_continent_2'] = df_cat['TP_continent_2']
        if 'TP_continent_3' in df_cat.columns:
            df_all['TP_continent_3'] = df_cat['TP_continent_3']
        if 'TP_continent_4' in df_cat.columns:
            df_all['TP_continent_4'] = df_cat['TP_continent_4']
        if 'TP_continent_5' in df_cat.columns:
            df_all['TP_continent_5'] = df_cat['TP_continent_5']
        if 'TP_continent_6' in df_cat.columns:
            df_all['TP_continent_6'] = df_cat['TP_continent_6']
        if 'TP_continent_7' in df_cat.columns:
            df_all['TP_continent_7'] = df_cat['TP_continent_7']
        if 'TP_continent_8' in df_cat.columns:
            df_all['TP_continent_8'] = df_cat['TP_continent_8']
        if 'TP_continent_9' in df_cat.columns:
            df_all['TP_continent_9'] = df_cat['TP_continent_9']
        if 'TP_continent_10' in df_cat.columns:
            df_all['TP_continent_10'] = df_cat['TP_continent_10']
        if 'TP_continent_11' in df_cat.columns:
            df_all['TP_continent_11'] = df_cat['TP_continent_11']
        if 'TP_continent_12' in df_cat.columns:
            df_all['TP_continent_12'] = df_cat['TP_continent_12']
        if 'TP_continent_13' in df_cat.columns:
            df_all['TP_continent_13'] = df_cat['TP_continent_13']
        if 'TP_continent_14' in df_cat.columns:
            df_all['TP_continent_14'] = df_cat['TP_continent_14']
        if 'TP_continent_15' in df_cat.columns:
            df_all['TP_continent_15'] = df_cat['TP_continent_15']
        if 'TP_continent_16' in df_cat.columns:
            df_all['TP_continent_16'] = df_cat['TP_continent_16']
        if 'TP_continent_17' in df_cat.columns:
            df_all['TP_continent_17'] = df_cat['TP_continent_17']
        if 'TP_continent_18' in df_cat.columns:
            df_all['TP_continent_18'] = df_cat['TP_continent_18']
        if 'TP_continent_19' in df_cat.columns:
            df_all['TP_continent_19'] = df_cat['TP_continent_19']
        if 'TP_continent_20' in df_cat.columns:
            df_all['TP_continent_20'] = df_cat['TP_continent_20']
        if 'TP_continent_21' in df_cat.columns:
            df_all['TP_continent_21'] = df_cat['TP_continent_21']
        if 'TP_continent_22' in df_cat.columns:
            df_all['TP_continent_22'] = df_cat['TP_continent_22']
        if 'TP_continent_23' in df_cat.columns:
            df_all['TP_continent_23'] = df_cat['TP_continent_23']
        if 'TP_continent_24' in df_cat.columns:
            df_all['TP_continent_24'] = df_cat['TP_continent_24']
        if 'TP_continent_25' in df_cat.columns:
            df_all['TP_continent_25'] = df_cat['TP_continent_25']
        if 'TP_continent_26' in df_cat.columns:
            df_all['TP_continent_26'] = df_cat['TP_continent_26']
        if 'TP_continent_27' in df_cat.columns:
            df_all['TP_continent_27'] = df_cat['TP_continent_27']
        if 'TP_continent_28' in df_cat.columns:
            df_all['TP_continent_28'] = df_cat['TP_continent_28']
        if 'TP_continent_29' in df_cat.columns:
            df_all['TP_continent_29'] = df_cat['TP_continent_29']
        if 'TP_continent_30' in df_cat.columns:
            df_all['TP_continent_30'] = df_cat['TP_continent_30']
        if 'TP_continent_31' in df_cat.columns:
            df_all['TP_continent_31'] = df_cat['TP_continent_31']
        if 'TP_continent_32' in df_cat.columns:
            df_all['TP_continent_32'] = df_cat['TP_continent_32']
        if 'TP_continent_33' in df_cat.columns:
            df_all['TP_continent_33'] = df_cat['TP_continent_33']
        if 'TP_continent_34' in df_cat.columns:
            df_all['TP_continent_34'] = df_cat['TP_continent_34']
        if 'TP_continent_35' in df_cat.columns:
            df_all['TP_continent_35'] = df_cat['TP_continent_35']
        if 'TP_continent_36' in df_cat.columns:
            df_all['TP_continent_36'] = df_cat['TP_continent_36']
        if 'TP_continent_37' in df_cat.columns:
            df_all['TP_continent_37'] = df_cat['TP_continent_37']
        if 'TP_continent_38' in df_cat.columns:
            df_all['TP_continent_38'] = df_cat['TP_continent_38']
        if 'TP_continent_39' in df_cat.columns:
            df_all['TP_continent_39'] = df_cat['TP_continent_39']
        if 'TP_continent_40' in df_cat.columns:
            df_all['TP_continent_40'] = df_cat['TP_continent_40']
        if 'TP_continent_41' in df_cat.columns:
            df_all['TP_continent_41'] = df_cat['TP_continent_41']
        if 'TP_continent_42' in df_cat.columns:
            df_all['TP_continent_42'] = df_cat['TP_continent_42']
        if 'TP_continent_43' in df_cat.columns:
            df_all['TP_continent_43'] = df_cat['TP_continent_43']
        if 'TP_continent_44' in df_cat.columns:
            df_all['TP_continent_44'] = df_cat['TP_continent_44']
        if 'TP_continent_45' in df_cat.columns:
            df_all['TP_continent_45'] = df_cat['TP_continent_45']
        if 'TP_continent_46' in df_cat.columns:
            df_all['TP_continent_46'] = df_cat['TP_continent_46']
        if 'TP_continent_47' in df_cat.columns:
            df_all['TP_continent_47'] = df_cat['TP_continent_47']
        if 'TP_continent_48' in df_cat.columns:
            df_all['TP_continent_48'] = df_cat['TP_continent_48']
        if 'TP_continent_49' in df_cat.columns:
            df_all['TP_continent_49'] = df_cat['TP_continent_49']
        if 'TP_continent_50' in df_cat.columns:
            df_all['TP_continent_50'] = df_cat['TP_continent_50']
        if 'TP_continent_51' in df_cat.columns:
            df_all['TP_continent_51'] = df_cat['TP_continent_51']
        if 'TP_continent_52' in df_cat.columns:
            df_all['TP_continent_52'] = df_cat['TP_continent_52']
        if 'TP_continent_53' in df_cat.columns:
            df_all['TP_continent_53'] = df_cat['TP_continent_53']
        if 'TP_continent_54' in df_cat.columns:
            df_all['TP_continent_54'] = df_cat['TP_continent_54']
        if 'TP_continent_55' in df_cat.columns:
            df_all['TP_continent_55'] = df_cat['TP_continent_55']
        if 'TP_continent_56' in df_cat.columns:
            df_all['TP_continent_56'] = df_cat['TP_continent_56']
        if 'TP_continent_57' in df_cat.columns:
            df_all['TP_continent_57'] = df_cat['TP_continent_57']
        if 'TP_continent_58' in df_cat.columns:
            df_all['TP_continent_58'] = df_cat['TP_continent_58']
        if 'TP_continent_59' in df_cat.columns:
            df_all['TP_continent_59'] = df_cat['TP_continent_59']
        if 'TP_continent_60' in df_cat.columns:
            df_all['TP_continent_60'] = df_cat['TP_continent_60']
        if 'TP_continent_61' in df_cat.columns:
            df_all['TP_continent_61'] = df_cat['TP_continent_61']
        if 'TP_continent_62' in df_cat.columns:
            df_all['TP_continent_62'] = df_cat['TP_continent_62']
        if 'TP_continent_63' in df_cat.columns:
            df_all['TP_continent_63'] = df_cat['TP_continent_63']
        if 'TP_continent_64' in df_cat.columns:
            df_all['TP_continent_64'] = df_cat['TP_continent_64']
        if 'TP_continent_65' in df_cat.columns:
            df_all['TP_continent_65'] = df_cat['TP_continent_65']
        if 'TP_continent_66' in df_cat.columns:
            df_all['TP_continent_66'] = df_cat['TP_continent_66']
        if 'TP_continent_67' in df_cat.columns:
            df_all['TP_continent_67'] = df_cat['TP_continent_67']
        if 'TP_continent_68' in df_cat.columns:
            df_all['TP_continent_68'] = df_cat['TP_continent_68']
        if 'TP_continent_69' in df_cat.columns:
            df_all['TP_continent_69'] = df_cat['TP_continent_69']
        if 'TP_continent_70' in df_cat.columns:
            df_all['TP_continent_70'] = df_cat['TP_continent_70']
        if 'TP_continent_71' in df_cat.columns:
            df_all['TP_continent_71'] = df_cat['TP_continent_71']
        if 'TP_continent_72' in df_cat.columns:
            df_all['TP_continent_72'] = df_cat['TP_continent_72']
        if 'TP_continent_73' in df_cat.columns:
            df_all['TP_continent_73'] = df_cat['TP_continent_73']
        if 'TP_continent_74' in df_cat.columns:
            df_all['TP_continent_74'] = df_cat['TP_continent_74']
        if 'TP_continent_75' in df_cat.columns:
            df_all['TP_continent_75'] = df_cat['TP_continent_75']
        if 'TP_continent_76' in df_cat.columns:
            df_all['TP_continent_76'] = df_cat['TP_continent_76']
        if 'TP_continent_77' in df_cat.columns:
            df_all['TP_continent_77'] = df_cat['TP_continent_77']
        if 'TP_continent_78' in df_cat.columns:
            df_all['TP_continent_78'] = df_cat['TP_continent_78']
        if 'TP_continent_79' in df_cat.columns:
            df_all['TP_continent_79'] = df_cat['TP_continent_79']
        if 'TP_continent_80' in df_cat.columns:
            df_all['TP_continent_80'] = df_cat['TP_continent_80']
        if 'TP_continent_81' in df_cat.columns:
            df_all['TP_continent_81'] = df_cat['TP_continent_81']
        if 'TP_continent_82' in df_cat.columns:
            df_all['TP_continent_82'] = df_cat['TP_continent_82']
        if 'TP_continent_83' in df_cat.columns:
            df_all['TP_continent_83'] = df_cat['TP_continent_83']
        if 'TP_continent_84' in df_cat.columns:
            df_all['TP_continent_84'] = df_cat['TP_continent_84']
        if 'TP_continent_85' in df_cat.columns:
            df_all['TP_continent_85'] = df_cat['TP_continent_85']
        if 'TP_continent_86' in df_cat.columns:
            df_all['TP_continent_86'] = df_cat['TP_continent_86']
        if 'TP_continent_87' in df_cat.columns:
            df_all['TP_continent_87'] = df_cat['TP_continent_87']
        if 'TP_continent_88' in df_cat.columns:
            df_all['TP_continent_88'] = df_cat['TP_continent_88']
        if 'TP_continent_89' in df_cat.columns:
            df_all['TP_continent_89'] = df_cat['TP_continent_89']
        if 'TP_continent_90' in df_cat.columns:
            df_all['TP_continent_90'] = df_cat['TP_continent_90']
        if 'TP_continent_91' in df_cat.columns:
            df_all['TP_continent_91'] = df_cat['TP_continent_91']
        if 'TP_continent_92' in df_cat.columns:
            df_all['TP_continent_92'] = df_cat['TP_continent_92']
        if 'TP_continent_93' in df_cat.columns:
            df_all['TP_continent_93'] = df_cat['TP_continent_93']
        if 'TP_continent_94' in df_cat.columns:
            df_all['TP_continent_94'] = df_cat['TP_continent_94']
        if 'TP_continent_95' in df_cat.columns:
            df_all['TP_continent_95'] = df_cat['TP_continent_95']
        if 'TP_continent_96' in df_cat.columns:
            df_all['TP_continent_96'] = df_cat['TP_continent_96']
        if 'TP_continent_97' in df_cat.columns:
            df_all['TP_continent_97'] = df_cat['TP_continent_97']
        if 'TP_continent_98' in df_cat.columns:
            df_all['TP_continent_98'] = df_cat['TP_continent_98']
        if 'TP_continent_99' in df_cat.columns:
            df_all['TP_continent_99'] = df_cat['TP_continent_99']
        if 'TP_continent_100' in df_cat.columns:
            df_all['TP_continent_100'] = df_cat['TP_continent_100']
        if 'TP_continent_101' in df_cat.columns:
            df_all['TP_continent_101'] = df_cat['TP_continent_101']
        if 'TP_continent_102' in df_cat.columns:
            df_all['TP_continent_102'] = df_cat['TP_continent_102']
        if 'TP_continent_103' in df_cat.columns:
            df_all['TP_continent_103'] = df_cat['TP_continent_103']

        # Cortex White Deep
        cat_cols += ['cwd_0', 'cwd_1', 'cwd_2', 'cwd_3']
        df['cwd_0'] = 0
        df['cwd_1'] = 0
        df['cwd_2'] = 0
        df['cwd_3'] = 0
        cols_cat = ['cwd']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'cwd_0' in df_cat.columns:
            df_all['cwd_0'] = df_cat['cwd_0']
        if 'cwd_1' in df_cat.columns:
            df_all['cwd_1'] = df_cat['cwd_1']
        if 'cwd_2' in df_cat.columns:
            df_all['cwd_2'] = df_cat['cwd_2']
        if 'cwd_3' in df_cat.columns:
            df_all['cwd_3'] = df_cat['cwd_3']

        # Stylet
        cat_cols += ['stylet_n', 'stylet_0', 'stylet_1']
        df['stylet_n'] = 0
        df['stylet_0'] = 0
        df['stylet_1'] = 0
        cols_cat = ['stylet']
        df_cat = df[cols_cat]
        df_cat = pd.get_dummies(df_cat, columns=cols_cat)
        if 'stylet_n' in df_cat.columns:
            df_all['stylet_n'] = df_cat['stylet_n']
        if 'stylet_0' in df_cat.columns:
            df_all['stylet_0'] = df_cat['stylet_0']
        if 'stylet_1' in df_cat.columns:
            df_all['stylet_1'] = df_cat['stylet_1']

        return df_all, cat_cols