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

import os
import pickle
import scipy
import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Options:
    data_dir = '..\\Data'
    models_dir = '.\\models'
    space = 'mni_aff'
    labels = 'vector'

options = Options()


def main():
    ''' Load data containing MSE values '''
    # columns=['method', 'label', 'region', 'fold', 'case', 'elec', 'mse_gt', 'mse_plan', 'mse_impl']
    pickle_file = open(os.path.join(options.data_dir, 'cwd-mc-lu-df-col.pkl'), "rb")
    mri_vector_mse_col_df = pickle.load(pickle_file)
    pickle_file.close()
    # columns=['method', 'label', 'region', 'fold', 'case', 'elec', 'type', 'mse']
    pickle_file = open(os.path.join(options.data_dir, 'cwd-mc-lu-df-row.pkl'), "rb")
    mri_vector_mse_row_df = pickle.load(pickle_file)
    pickle_file.close()

    print('df[{}]'.format(len(mri_vector_mse_col_df)))
    print('     number of pred elec within MSE=1mm: {}'.format(len(mri_vector_mse_col_df[mri_vector_mse_col_df.mse_impl<=1.0])))
    # print('df [asc by gt]', mri_vector_mse_col_df.sort_values(by=['mse_gt'], ascending=True))

    # split dataframes
    over_df = mri_vector_mse_col_df[(mri_vector_mse_col_df.mse_plan > mri_vector_mse_col_df.mse_gt) &
                                    (mri_vector_mse_col_df.mse_impl < mri_vector_mse_col_df.mse_plan)]
    under_df = mri_vector_mse_col_df[(mri_vector_mse_col_df.mse_plan <= mri_vector_mse_col_df.mse_gt) &
                                     (mri_vector_mse_col_df.mse_impl <= mri_vector_mse_col_df.mse_gt)]
    wrong_df = mri_vector_mse_col_df[(mri_vector_mse_col_df.mse_impl > mri_vector_mse_col_df.mse_gt) &
                                     (mri_vector_mse_col_df.mse_impl > mri_vector_mse_col_df.mse_plan)]
    print('Split dataframe by mse performance {}={}+{}+{}'.format(len(mri_vector_mse_col_df), len(over_df), len(under_df), len(wrong_df)))
    print('     over_df[{}]\n{}'.format(len(over_df), over_df))
    print('     under_df[{}]\n{}'.format(len(under_df), under_df))
    print('     wrong_df[{}]\n{}'.format(len(wrong_df), under_df))

    # split dataframes per group
    sfg_df = mri_vector_mse_col_df[mri_vector_mse_col_df.region == 'sfg']
    mfg_df = mri_vector_mse_col_df[mri_vector_mse_col_df.region == 'mfg']
    ifog_df = mri_vector_mse_col_df[mri_vector_mse_col_df.region == 'ifog']
    tg_df = mri_vector_mse_col_df[mri_vector_mse_col_df.region == 'tg']
    apcg_df = mri_vector_mse_col_df[mri_vector_mse_col_df.region == 'apcg']
    po_df = mri_vector_mse_col_df[mri_vector_mse_col_df.region == 'po']
    print('Split dataframe by group {}={}+{}+{}+{}+{}+{}'.format(len(mri_vector_mse_col_df),
                                                                 len(sfg_df), len(mfg_df), len(ifog_df),
                                                                 len(tg_df), len(apcg_df), len(po_df)))
    print('     sfg_df[{}]\n{}'.format(len(sfg_df), sfg_df))
    print('     mfg_df[{}]\n{}'.format(len(mfg_df), mfg_df))
    print('     ifog_df[{}]\n{}'.format(len(ifog_df), ifog_df))
    print('     tg_df[{}]\n{}'.format(len(tg_df), tg_df))
    print('     apcg_df[{}]\n{}'.format(len(apcg_df), apcg_df))
    print('     po_df[{}]\n{}'.format(len(po_df), po_df))

    # print('under_df', under_df)
    # print('over_df [ascending]', over_df.sort_values(by=['mse_impl'], ascending=True))
    # print('wrong_df [ascending]', wrong_df.sort_values(by=['mse_plan'], ascending=True))

    hue_order = ['sfg', 'mfg', 'ifog', 'tg', 'apcg', 'po']

    # ''' show all observations '''
    # sns.set_theme(style="white")
    # sns.set(rc={'figure.figsize': (6, 4)})
    # sns.despine(bottom=True, left=True)
    # sns.stripplot(x="type", y="mse", hue="region", data=mri_vector_mse_row_df, dodge=True, alpha=.35, size=10.0, zorder=1, hue_order=hue_order)
    # sns.pointplot(x="type", y="mse", hue="region", data=mri_vector_mse_row_df, dodge=.675, join=False, palette="dark",
    #               markers="d", scale=.75, ci=95, legend=False, hue_order=hue_order)
    # plt.show()

    ''' show correlation between length and MSE '''
    r_sfg = np.corrcoef(sfg_df.points, sfg_df.mse_impl)
    r_mfg = np.corrcoef(mfg_df.points, mfg_df.mse_impl)
    r_ifog = np.corrcoef(ifog_df.points, ifog_df.mse_impl)
    r_tg = np.corrcoef(tg_df.points, tg_df.mse_impl)
    r_apcg = np.corrcoef(apcg_df.points, apcg_df.mse_impl)
    r_po = np.corrcoef(po_df.points, po_df.mse_impl)
    b_sfg, m_sfg = polyfit(sfg_df.points, sfg_df.mse_impl, 1)
    b_mfg, m_mfg = polyfit(mfg_df.points, mfg_df.mse_impl, 1)
    b_ifog, m_ifog = polyfit(ifog_df.points, ifog_df.mse_impl, 1)
    b_tg, m_tg = polyfit(tg_df.points, tg_df.mse_impl, 1)
    b_apcg, m_apcg = polyfit(apcg_df.points, apcg_df.mse_impl, 1)
    b_po, m_po = polyfit(po_df.points, po_df.mse_impl, 1)
    p_sfg = scipy.stats.pearsonr(sfg_df.points, sfg_df.mse_impl)
    p_mfg = scipy.stats.pearsonr(mfg_df.points, mfg_df.mse_impl)
    p_ifog = scipy.stats.pearsonr(ifog_df.points, ifog_df.mse_impl)
    p_tg = scipy.stats.pearsonr(tg_df.points, tg_df.mse_impl)
    p_apcg = scipy.stats.pearsonr(apcg_df.points, apcg_df.mse_impl)
    p_po = scipy.stats.pearsonr(po_df.points, po_df.mse_impl)
    print('Correlation length vs mse_impl')
    print('\nsfg=\n{} \nmfg=\n{} \nifog=\n{} \ntg=\n{} \napcg=\n{} \npo=\n{}'.format(r_sfg, r_mfg, r_ifog, r_tg, r_apcg, r_po))
    print('Pearson correlation length vs mse_impl')
    print('\nsfg={} \nmfg={} \nifog={} \ntg={} \napcg={} \npo={}'.format(p_sfg, p_mfg, p_ifog, p_tg, p_apcg, p_po))

    sns.set_theme(style="white")
    sns.set(rc={'figure.figsize': (6, 4)})
    sns.despine(bottom=True, left=True)
    sns.set_theme(style="white")
    # ax = sns.stripplot(x="mse_gt", y="mse_impl", hue="region", data=over_df, dodge=True, alpha=.35, size=10.0, zorder=1)
    ax = sns.scatterplot(x="points", y="mse_impl", hue="region", data=mri_vector_mse_col_df, alpha=.35, s=150.0, zorder=1)
    ax.axhline(1.0, ls='--', alpha=0.5, c='black')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
    # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('length vs mse_impl')
    plt.ylim(-0.5, 10)
    # plt.xticks(np.arange(0.0, max(mri_vector_mse_col_df.mse_gt)+0.1, 0.1))
    plt.show()

    ''' show only impl '''
    sns.set_theme(style="white")
    sns.set(rc={'figure.figsize': (6, 4)})
    sns.despine(bottom=True, left=True)
    sns.set_theme(style="white")
    ax1 = sns.stripplot(x="region", y="mse_impl", data=mri_vector_mse_col_df, dodge=True, alpha=.35, size=10.0, zorder=1, hue_order=hue_order)
    ax2 = sns.pointplot(x="region", y="mse_impl", data=mri_vector_mse_col_df, dodge=.675, join=False, palette="dark",
                  markers="d", scale=.75, ci=95, legend=False, hue_order=hue_order)
    ax1.axhline(1.0, ls='--', alpha=0.5, c='black')
    #ax1.get_legend().remove()
    # ax2.get_legend().remove()
    plt.ylim(-0.5, 10)
    plt.show()

    ''' show only impl (per fold) '''
    sns.set_theme(style="white")
    sns.set(rc={'figure.figsize': (6, 4)})
    sns.despine(bottom=True, left=True)
    sns.set_theme(style="white")
    ax1 = sns.stripplot(x="fold", y="mse_impl", hue="region", data=mri_vector_mse_col_df, dodge=True, alpha=.35, size=10.0, zorder=1, hue_order=hue_order)
    ax2 = sns.pointplot(x="fold", y="mse_impl", hue="region", data=mri_vector_mse_col_df, dodge=.675, join=False, palette="dark",
                  markers="d", scale=.75, ci=95, legend=False, hue_order=hue_order)
    ax1.axhline(1.0, ls='--')
    ax1.get_legend().remove()
    #ax2.get_legend().remove()
    plt.ylim(-0.5, 10)
    plt.show()

    ''' show over, under, wrong '''
    sns.set_theme(style="white")
    sns.set(rc={'figure.figsize': (6, 4)})
    sns.despine(bottom=True, left=True)
    sns.set_theme(style="white")
    # ax = sns.stripplot(x="mse_gt", y="mse_impl", hue="region", data=over_df, dodge=True, alpha=.35, size=10.0, zorder=1)
    ax = sns.scatterplot(x="mse_gt", y="mse_impl", hue="region", data=over_df, alpha=.35, s=150.0, zorder=1, hue_order=hue_order)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(.5))
    # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.title('over')
    # plt.xticks(np.arange(0.0, max(mri_vector_mse_col_df.mse_gt)+0.1, 0.1))
    plt.show()

    sns.set_theme(style="white")
    # ax = sns.stripplot(x="mse_gt", y="mse_impl", hue="region", data=under_df, dodge=True, alpha=.25, zorder=1)
    ax = sns.scatterplot(x="mse_gt", y="mse_impl", hue="region", data=under_df, alpha=.35, s=150.0, zorder=1, hue_order=hue_order)
    plt.title('under')
    plt.show()

    sns.set_theme(style="white")
    #ax = sns.stripplot(x="mse_gt", y="mse_plan", hue="region", data=wrong_df, dodge=True, alpha=.25, zorder=1)
    ax = sns.scatterplot(x="mse_gt", y="mse_impl", hue="region", data=wrong_df, alpha=.35, s=150.0, zorder=1, hue_order=hue_order)
    plt.title('wrong')
    plt.show()


if __name__ == '__main__':
    main()