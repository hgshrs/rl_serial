import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats
import scipy.optimize
import sys
import os
import pickle
from tqdm import tqdm
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"

def color_violinplot(parts, c='black'):
    for pc in parts['bodies']:
        pc.set_facecolor(c)
        pc.set_edgecolor(None)
    parts['cmeans'].set_color(c)
    parts['cmaxes'].set_color(c)
    parts['cmins'].set_color(c)
    parts['cbars'].set_color(c)

def diff_orientations(a, b): # a and b should be in a range of 0--180
    return np.abs(signed_diff_orientations(a, b))

def signed_diff_orientations(a, b): # a and b should be in a range of 0--180
    if type(a) in [type([0, 1]), type(np.array([0, 1]))]:
        difference = np.zeros(len(a))
        for ii in range(len(a)):
            difference[ii] = signed_diff_orientations(a[ii], b[ii])
    elif type(a) == type(pd.Series()):
        a = np.array(a)
        b = np.array(b)
        difference = signed_diff_orientations(a, b)
    else:
        difference = b - a
        if difference > 90:
            difference -= 180
        elif difference < -90:
            difference += 180
    return difference

def devgauss(x, loc, scale, height):
    g = scipy.stats.norm.pdf(x, loc, scale) # loc: mean, scale: standard deviation
    dg = - (x - loc) / scale**2 * g
    return dg * height

def fun_fit_devgauss(variables, x, y):
    loc = variables[0]
    scale = variables[1]
    height = variables[2]
    y_est = devgauss(x, loc, scale, height)
    d = y - y_est
    return np.sqrt(np.dot(d, d))

def fit_devgauss(x, y, both='two_side'):
    if both == 'two_side':
        res1 = scipy.optimize.minimize(fun_fit_devgauss, [0, 10, 1000], args=(x, y), bounds=[(-0, 0), (.1, 200), (-10000, 10000)])
        res2 = scipy.optimize.minimize(fun_fit_devgauss, [0, 10, -1000], args=(x, y), bounds=[(-0, 0), (.1, 200), (-10000, 10000)])
        if res1.fun < res2.fun:
            res = res1
        else:
            res = res2
    elif both == 'data':
        if y[np.where(x > 0)].mean() < 0: # might be attractive
            res1 = scipy.optimize.minimize(fun_fit_devgauss, [0, 10, -1000], args=(x, y), bounds=[(-0, 0), (.1, 200), (-10000, 10000)])
        else: # might be repulsive
            res1 = scipy.optimize.minimize(fun_fit_devgauss, [0, 10, 1000], args=(x, y), bounds=[(-0, 0), (.1, 200), (-10000, 10000)])
    elif both == 'random':
        if np.random.choice([True, False]):
            res = scipy.optimize.minimize(fun_fit_devgauss, [0, 10, 1000], args=(x, y), bounds=[(-0, 0), (.1, 90), (-10000, 10000)])
        else:
            res = scipy.optimize.minimize(fun_fit_devgauss, [0, 10, -1000], args=(x, y), bounds=[(-0, 0), (.1, 90), (-10000, 10000)])
    else:
        res = scipy.optimize.minimize(fun_fit_devgauss, [0, 10, 0.], args=(x, y), bounds=[(-0, 0), (.1, 90), (-10000, 10000)])
    return res.x[0], res.x[1], res.x[2]

def out_residue_fit_devgauss(x, y, both=False):
    loc, scale, height = fit_devgauss(x, y, both)
    y_ = devgauss(x, loc, scale, height)
    residual = y - y_
    return np.sum(residual ** 2)

def plot_xy_fit(ax, x, y, label='', c=None, ms=2, verbose=False):
    l = ax.plot(x, y, 'o', c=c, alpha=.1, ms=ms, mec=None)
    loc, scale, height = fit_devgauss(x, y)
    degrange = np.arange(-90, 90)
    val_devgauss = devgauss(degrange, loc, scale, height)
    peak = np.abs(val_devgauss).max()
    ax.plot(degrange, val_devgauss, c=l[0].get_color(), label=label, lw=4)
    if verbose:
        print('loc: {:.3f}, scale: {:.3f}, height: {:.3f}, peak: {:.3f} deg'.format(loc, scale, height, peak))
    return l, [loc, scale, height], peak

def out_diff_xy(df, cmbx, cmby, cnd_queries='', wise='trial', verbose=True):
    if len(cnd_queries) > 0:
        rcnd_queries = cnd_queries + ' & reject == False'
    else:
        rcnd_queries = 'reject == False'
    df = df.query(rcnd_queries)
    df = df.assign(x = signed_diff_orientations(df[cmbx[0]], df[cmbx[1]]))
    df = df.assign(y = signed_diff_orientations(df[cmby[0]], df[cmby[1]]))
    n_samples = len(df)
    bin_width = 10
    # bin_width = 5
    centers = np.arange(-90, 90 + 1, bin_width)
    if wise == 'trial':
        x = df['x']
        y = df['y']
    elif wise == 'block':
        x = []
        y = []
        for cent in centers:
            for blk in df['blk'].unique():
                _df = df.query('blk == {} & x >= {} & x < = {}'.format(blk, cent - .5 * bin_width, cent + .5 * bin_width))
                if len(_df) > 0:
                    x.append(cent)
                    y.append(_df['y'].mean())
    elif wise == 'session':
        x = []
        y = []
        for cent in centers:
            for sess in df['session'].unique():
                if type(sess) == type('a'):
                    _df = df.query("session == '{}' & x >= {} & x < = {}".format(sess , cent - .5 * bin_width, cent + .5 * bin_width))
                else:
                    _df = df.query('session == {} & x >= {} & x < = {}'.format(sess , cent - .5 * bin_width, cent + .5 * bin_width))
                if len(_df) > 0:
                    x.append(cent)
                    y.append(_df['y'].mean())
    x = np.array(x)
    y = np.array(y)
    return x, y, n_samples

def test_w_bootstrap(x, y):
    _y = np.random.permutation(y)
    _x = np.random.permutation(x)
    res_perm = out_residue_fit_devgauss(_x, _y, both='random')
    return res_perm

if __name__=='__main__':
    np.random.seed(0)

    df_ori = pd.read_csv('tmp/bhv.csv', index_col=0) # created by unite_bhv.py

    thrs_deg_quantile = .85
    thrs_rt = [1.5, 10]
    # thrs_rt = [df_ori['rt'].quantile(.1) / 1.5, df_ori['rt'].quantile(.9) * 1.5]
    thrs_rejected = .8
    removing_blks = [0,]
    thrs_lik = .6
    thrs_phase_ratio = .01
    thrs_ave_rew = 1.75
    thrs_bw_ratio = .0
    n_bootstrap = int(1e1)
    fix_bonferroni = True
    print('==================\nParameters\n==================')
    print('thrs_deg_quantile:\t{}'.format(thrs_deg_quantile))
    print('thrs_rt:\t{}'.format(thrs_rt))
    print('thrs_rejected:\t{:.0%}'.format(thrs_rejected))
    print('thrs_lik:\t{}'.format(thrs_lik))
    print('thrs_phase:\t{:.0%}'.format(thrs_phase_ratio))
    print('thrs_ave_rew:\t{}'.format(thrs_ave_rew))
    print('n_bootstrap:\t{:,d}'.format(n_bootstrap))
    print('fix_bonferroni:\t{}'.format(fix_bonferroni))

    print('==================\nRejecting samples\n==================')
    # idxs = np.abs(df_ori['adj_err']) > thrs_deg
    thrs_deg = np.abs(df_ori['adj_err']).quantile(thrs_deg_quantile)
    print('thrs_deg:\t{}'.format(thrs_deg))
    idxs = np.abs(df_ori['adj_err']) > thrs_deg
    df_ori.loc[idxs, 'reject'] = True
    print('Reject {:,}/{:,} samples for AE < thrs_deg = {}'.format(idxs.sum(), len(df_ori), thrs_deg))

    idxs = df_ori['rt'] < thrs_rt[0]
    df_ori.loc[idxs, 'reject'] = True
    print('Reject {:,}/{:,} samples for RT < {:.2f}'.format(idxs.sum(), len(df_ori), thrs_rt[0]))

    idxs = df_ori['rt'] > thrs_rt[1]
    df_ori.loc[idxs, 'reject'] = True
    print('Reject {:,}/{:,} samples for RT > {:.2f}'.format(idxs.sum(), len(df_ori), thrs_rt[1]))

    print('#rejected {:.1%} samples: {:,}/{:,}'.format(df_ori['reject'].sum(), df_ori['reject'].sum()/len(df_ori), len(df_ori)))

    print('==================\nRejecting participants\n==================')
    n_rmid = 0
    for pid in df_ori['session'].unique():
        _df = df_ori.query("session == '{}'".format(pid,))
        n_rejected = _df['reject'].sum()
        if n_rejected / len(_df) > thrs_rejected:
            df_ori.loc[df_ori['session'] == pid, 'reject_p'] = True
            n_rmid += 1
    print('Reject {} participants with {:.0%} rejected samples.'.format(n_rmid, thrs_rejected))

    df_ori1 = df_ori[df_ori['reject'] == False]

    for bidx in removing_blks:
        idxs = df_ori1['blk_wi_sess'] == bidx
        df_ori1.loc[idxs, 'reject'] = True
    print('Reject {}/{} blocks for blocks {}.'.format(len(removing_blks), len(df_ori1['blk_wi_sess'].unique()), removing_blks))

    n_rmid = 0
    for pid in df_ori1['session'].unique():
        _df = df_ori1.query("session == '{}'".format(pid,))
        n_exploi = (_df['lik_trg'] >= thrs_lik).sum()
        n_explor = (_df['lik_trg'] < thrs_lik).sum()
        r = n_exploi / (n_exploi + n_explor)
        r = np.min([r, 1 - r])
        if  r < thrs_phase_ratio:
            df_ori1.loc[df_ori1['session'] == pid, 'reject_p'] = True
            n_rmid += 1
    print('Reject {} participants for phase ratio < thrs_phase_ratio = {:.0%}'.format(n_rmid, thrs_phase_ratio))

    n_rmid = 0
    for pid in df_ori1['session'].unique():
        _df = df_ori1.query("session == '{}'".format(pid,))
        ave_rew = _df['rew'].mean()
        if  ave_rew < thrs_ave_rew:
            df_ori1.loc[df_ori1['session'] == pid, 'reject_p'] = True
            n_rmid += 1
    print('Reject {} participants for total reward < thrs_ave_rew = {}'.format(n_rmid, thrs_ave_rew))

    n_rmid = 0
    for pid in df_ori1['session'].unique():
        _df = df_ori1.query("session == '{}'".format(pid,))
        na1 = len(_df[_df['action'] == 1.])
        na2 = len(_df[_df['action'] == 2.])
        bw_ratio = np.min(np.array([na1, na2]) / (na1 + na2))
        if  bw_ratio < thrs_bw_ratio:
            df_ori1.loc[df_ori1['session'] == pid, 'reject_p'] = True
            n_rmid += 1
    print('Reject {} participants for black/white ratio < thrs_bw_ratio = {:.0%}'.format(n_rmid, thrs_bw_ratio))

    df = df_ori1[df_ori1['reject_p'] == False]
    print('#participants: {:,}/{:,}'.format(len(df['session'].unique()), len(df_ori['session'].unique())))
    print('#blocks: {}'.format(len(df['blk'].unique())))
    print('df with {:,} samples of {} participants.'.format(len(df), len(df['session'].unique())))


    plt.figure(0).clf()
    _df = df_ori[df_ori['reject_p'] == False]
    print('AE before rejecting: {:.2f}+-{:.2f}'.format(_df['adj_err'].mean(), _df['adj_err'].std()))
    print('AE after rejecting: {:.2f}+-{:.2f}'.format(df['adj_err'].mean(), df['adj_err'].std()))
    bins = np.arange(-75.5, 75.5, 1.)
    plt.hist(df_ori['adj_err'], bins=bins, color='k')
    ylim = plt.gca().get_ylim()
    plt.plot([thrs_deg, thrs_deg], ylim, 'k--')
    plt.plot([-thrs_deg, -thrs_deg], ylim, 'k--')
    plt.xlabel('AE [deg]')
    plt.ylabel('Number of trials')
    plt.savefig('figs2/stats/adj_err.pdf', bbox_inches='tight', transparent=True)

    plt.figure(0).clf()
    bins = np.arange(-thrs_deg - .5, thrs_deg + .5, 1.)
    df1 = df[df['aa'] == True]
    df2 = df[df['aa'] == False]
    plt.hist(df1['adj_err'], bins=bins, alpha=1.)
    plt.hist(df2['adj_err'], bins=bins, alpha=.5)
    plt.legend(['Same', 'Diff'])
    res = scipy.stats.ttest_ind(np.abs(df1['adj_err']), np.abs(df2['adj_err']))
    print('AE btw the target change (ab) and same (aa): t({:,}): {:.3f}, pvalue: {:.3f}'.format(int(res.df), res.statistic, res.pvalue))

    rdiff_range = range(-10, 11)
    na1 = np.zeros(len(rdiff_range), dtype=int)
    na2 = np.zeros(len(rdiff_range), dtype=int)
    na1s = np.zeros([len(df['session'].unique()), len(rdiff_range)], dtype=int)
    na2s = np.zeros([len(df['session'].unique()), len(rdiff_range)], dtype=int)
    n = 0
    for rr, rdiff in enumerate(rdiff_range):
        df1 = df[(df['rew1'] - df['rew2']).to_numpy() == rdiff]
        na1[rr] = (df1['action'] == 1.).sum()
        na2[rr] = (df1['action'] == 2.).sum()
        for ss, session in enumerate(df1['session'].unique()):
            df2 = df1[df1['session'] == session]
            na1s[ss, rr] = (df2['action'] == 1.).sum()
            na2s[ss, rr] = (df2['action'] == 2.).sum()
    plt.figure(0).clf()
    plt.plot(rdiff_range, na1 / (na1 + na2) * 100, 'k--o', mfc='k', mec='k', label='Black (K)')
    plt.plot(rdiff_range, na2 / (na1 + na2) * 100, 'k-o', mfc='w', mec='k', label='White (W)')
    plt.ylabel('Proportion of the target arm [%]')
    plt.xlabel('Diff in max rewards, ' + r'$\Omega_{\rm K} - \Omega_{\rm W}$')
    plt.legend(loc='upper right')
    plt.savefig('figs2/stats/prop_action.pdf', bbox_inches='tight', transparent=True)

    ########################################
    # Black vs white
    na1 = np.zeros(len(df['session'].unique()), dtype=int)
    na2 = np.zeros(len(df['session'].unique()), dtype=int)
    ae1 = np.zeros(len(df['session'].unique()))
    ae2 = np.zeros(len(df['session'].unique()))
    rt1 = np.zeros(len(df['session'].unique()))
    rt2 = np.zeros(len(df['session'].unique()))
    for ss, session in enumerate(df['session'].unique()):
        df1 = df[df['session'] == session]
        na1[ss] = len(df1[df1['action'] == 1.])
        na2[ss] = len(df1[df1['action'] == 2.])
        ae1[ss] = np.abs(df1[df1['action'] == 1.]['adj_err']).mean()
        ae2[ss] = np.abs(df1[df1['action'] == 2.]['adj_err']).mean()
        rt1[ss] = np.abs(df1[df1['action'] == 1.]['rt']).mean()
        rt2[ss] = np.abs(df1[df1['action'] == 2.]['rt']).mean()

    plt.figure(0).clf()
    plt.subplot(121)
    parts = plt.violinplot([na1, na2], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Black', 'White'])
    plt.xlabel('Target arm')
    plt.ylabel('Number of trials')
    plt.savefig('figs2/stats/actions_n.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_rel(na1, na2)
    print('Number of trials btw black vs white: t({:,}): {:.3f}, pvalue: {:.3f}'.format(int(res.df), res.statistic, res.pvalue))

    plt.figure(0).clf()
    plt.subplot(121)
    parts = plt.violinplot([ae1, ae2], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Black', 'White'])
    plt.xlabel('Target arm')
    plt.ylabel(r'$|$AE$|$ [deg]')
    plt.savefig('figs2/stats/actions_ae.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_rel(ae1, ae2)
    print('|AE| btw black vs white: t({:,}): {:.3f}, pvalue: {:.3f}'.format(int(res.df), res.statistic, res.pvalue))

    plt.figure(0).clf()
    plt.subplot(121)
    parts = plt.violinplot([rt1, rt2], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Black', 'White'])
    plt.xlabel('Target arm')
    plt.ylabel('RT [s]')
    plt.savefig('figs2/stats/actions_rt.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_rel(rt1, rt2)
    print('RT btw black vs white: t({:,}): {:.3f}, pvalue: {:.3f}'.format(int(res.df), res.statistic, res.pvalue))

    ########################################
    # Exolor vs exploi
    nar = np.zeros(len(df['session'].unique()), dtype=int)
    nai = np.zeros(len(df['session'].unique()), dtype=int)
    aer = np.zeros(len(df['session'].unique()))
    aei = np.zeros(len(df['session'].unique()))
    rtr = np.zeros(len(df['session'].unique()))
    rti = np.zeros(len(df['session'].unique()))
    for ss, session in enumerate(df['session'].unique()):
        df1 = df[df['session'] == session]
        nar[ss] = len(df1[df1['lik_trg'] < thrs_lik])
        nai[ss] = len(df1[df1['lik_trg'] >= thrs_lik])
        aer[ss] = np.abs(df1[df1['lik_trg'] < thrs_lik]['adj_err']).mean()
        aei[ss] = np.abs(df1[df1['lik_trg'] >= thrs_lik]['adj_err']).mean()
        rtr[ss] = np.abs(df1[df1['lik_trg'] < thrs_lik]['rt']).mean()
        rti[ss] = np.abs(df1[df1['lik_trg'] >= thrs_lik]['rt']).mean()

    plt.figure(0).clf()
    plt.subplot(121)
    parts = plt.violinplot([nai, nar], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Exploi', 'Explor'])
    plt.xlabel('Phase')
    plt.ylabel('Number of trials')
    plt.savefig('figs2/stats/phases_n.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_rel(nai, nar)
    print('Number of trials btw Exploi vs Explor: t({:,}): {:.3f}, pvalue: {:.3f}'.format(int(res.df), res.statistic, res.pvalue))

    plt.figure(0).clf()
    plt.subplot(121)
    parts = plt.violinplot([aei, aer], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Exploi', 'Explor'])
    plt.xlabel('Phase')
    plt.ylabel(r'$|$AE$|$ [deg]')
    plt.savefig('figs2/stats/phases_ae.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_rel(aei, aer)
    print('|AE| btw Exploi vs Explor: t({:,}): {:.3f}, pvalue: {:.3f}'.format(int(res.df), res.statistic, res.pvalue))

    plt.figure(0).clf()
    plt.subplot(121)
    parts = plt.violinplot([rti, rtr], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2], ['Exploi', 'Explor'])
    plt.xlabel('Phase')
    plt.ylabel('RT [s]')
    plt.savefig('figs2/stats/phases_rt.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.ttest_rel(rti, rtr)
    print('RT btw Exploi vs Explor: t({:,}): {:.3f}, pvalue: {:.3f}'.format(int(res.df), res.statistic, res.pvalue))

    ########################################
    # Exolor vs exploi in the preceding/current trial
    liktypes = {'preceding':'pre_lik', 'current':'lik_trg'}
    if fix_bonferroni:
        pfix = len(liktypes)
    else:
        pfix = 1
    for _liktype in liktypes.keys():
        liktype = liktypes[_liktype]
        lik_range = np.arange(0, 1.1, .1)
        ae = np.zeros([len(df['session'].unique()), len(lik_range) - 1])
        ae_ave = np.zeros(len(lik_range) - 1)
        n = 0
        for ll, lik1 in enumerate(lik_range[:-1]):
            df1 = df[df['{}'.format(liktype)] > lik1]
            df2 = df1[df1['{}'.format(liktype)] <= lik_range[ll + 1]]
            for ss, session in enumerate(df['session'].unique()):
                df3 = df2[df2['session'] == session]
                if len(df3) > 0:
                    _ae = np.abs(df3['adj_err']).mean()
                    ae[ss, ll] = _ae
                    # if _ae == 0.:
                        # print(df3['adj_err'])
                else:
                    ae[ss, ll] = np.nan
            ae_ave[ll] = ae[:, ll][np.logical_not(np.isnan(ae[:, ll]))].mean()
        lik_center = lik_range[:-1] + .05
        y = ae.ravel()
        x = np.repeat(lik_center[np.newaxis], len(df['session'].unique()), 0).ravel()[np.logical_not(np.isnan(y))]
        y = y[np.logical_not(np.isnan(y))]
        plt.figure(0).clf()
        res = scipy.stats.pearsonr(x, y)
        pvalue = np.min([res[1] * pfix, 1])
        print('Corr btw {} lik and |AE|: {:.3f} (df:{}), pvalue: {:.3f}'.format(liktype, res[0], len(y) - 2, pvalue))
        rl = np.polyfit(x, y, 1)
        _y = np.poly1d(rl)([x.min(), x.max()])
        plt.plot([x.min(), x.max()], _y, 'k')
        plt.plot(x + np.random.uniform(-.01, .01, len(x)), y, 'ko', alpha=.1)
        plt.plot(lik_center, ae_ave, 'ko', mfc='w')
        plt.xlabel('Likelihood ({})'.format(_liktype))
        plt.ylabel(r'$|$AE$|$ [deg]')
        plt.savefig('figs2/stats/cor_{}_ae.pdf'.format(liktype), bbox_inches='tight', transparent=True)

    ########################################
    # SD analysis by DoG fitting
    ########################################
    print('==================\nAnalysis for SD by DoG fitting\n==================')
    cmby = ['acted_cue', 'res']
    set_cmb_queries = [
            ['KN',
                {   
                    'Explor': {'cmbx': ['acted_cue', 'pre_nonacted_cue'], 'cmby': cmby, 'cnd':'pre_lik < {} & aa == True'.format(thrs_lik)},
                    'Exploi': {'cmbx': ['acted_cue', 'pre_nonacted_cue'], 'cmby': cmby, 'cnd':'pre_lik >= {} & aa == True'.format(thrs_lik)},
                    }],
            ['KT',
                {   
                    'Explor': {'cmbx': ['acted_cue', 'pre_acted_cue'], 'cmby': cmby, 'cnd':'pre_lik < {} & aa == True'.format(thrs_lik)},
                    'Exploi': {'cmbx': ['acted_cue', 'pre_acted_cue'], 'cmby': cmby, 'cnd':'pre_lik >= {} & aa == True'.format(thrs_lik)},
                    }],
            ['CN',
                {   
                    'Explor': {'cmbx': ['acted_cue', 'pre_nonacted_cue'], 'cmby': cmby, 'cnd':'pre_lik < {} & aa == False'.format(thrs_lik)},
                    'Exploi': {'cmbx': ['acted_cue', 'pre_nonacted_cue'], 'cmby': cmby, 'cnd':'pre_lik >= {} & aa == False'.format(thrs_lik)},
                    }],
            ['CT',
                {   
                    'Explor': {'cmbx': ['acted_cue', 'pre_acted_cue'], 'cmby': cmby, 'cnd':'pre_lik < {} & aa == False'.format(thrs_lik)},
                    'Exploi': {'cmbx': ['acted_cue', 'pre_acted_cue'], 'cmby': cmby, 'cnd':'pre_lik >= {} & aa == False'.format(thrs_lik)},
                    }],
            ]
    n_columns = 2
    n_rows = int(np.ceil(len(set_cmb_queries) / n_columns))
    wise = 'session'
    if fix_bonferroni:
        pfix = len(set_cmb_queries)
    else:
        pfix = 1

    fk = plt.figure(1); fk.clf()
    sdpeaks = []
    conditions = []
    for ll, cmb_queries in enumerate(set_cmb_queries):
        ptext = ''
        labels = []
        ax = fk.add_subplot(n_rows, n_columns, ll + 1)
        for cc, label in enumerate(cmb_queries[1].keys()):
            cqs = cmb_queries[1][label]
            x, y, n_samples = out_diff_xy(df, cmbx=cqs['cmbx'], cmby=cqs['cmby'], cnd_queries=cqs['cnd'], wise=wise, verbose=False)
            res_eval = out_residue_fit_devgauss(x, y, both='two_side')
            import joblib
            res_perm = joblib.Parallel(n_jobs=-1)(joblib.delayed(test_w_bootstrap)(x, y) for bb in range(n_bootstrap))
            p = len(np.where(np.sort(res_perm) <= res_eval)[0]) / n_bootstrap * pfix
            p = np.min([p, 1.])
            mk = ''
            if p < 0.05:
                mk = 's'
            ptext += r'{}, {:.3f}{}'.format(len(x), p, mk)

            plt.figure(0).clf()
            ax0 = plt.subplot(111)
            l, fp, sdpeak = plot_xy_fit(ax0, x, y, c='k', ms=5)
            ax0.set_ylim(-20, 20)
            if cmb_queries[0][1] == 'T':
                plt.xlabel(r'$\Delta$Target')
            elif cmb_queries[0][1] == 'N':
                plt.xlabel(r'$\Delta$Non-target')
            elif cmb_queries[0][1] == 'A':
                plt.xlabel(r'$\Delta$Adjustment')
            if label == 'Explor':
                sl = 'R'
            elif label == 'Exploi':
                sl = 'I'
            plt.ylabel('AE [deg]')
            plt.savefig('figs2/sd/{}{}.pdf'.format(cmb_queries[0], sl), bbox_inches='tight', transparent=True)
            plt.pause(.1)

            print('{}{} (#samples: {:,})'.format(cmb_queries[0], sl, n_samples))
            print('\tloc: {:.3f}, scale: {:.3f}, height: {:.3f}, peak: {:.2f}, pvalue: {:.3f}{}'.format(*fp, sdpeak, p, mk))
            sdpeaks.append(sdpeak)
            conditions.append(cmb_queries[0] + sl)
        ax.set_title(cmb_queries[0])
        ax.text(-90, ax.get_ylim()[1], ptext, va='top')

        plt.figure(0).clf()
        plt.bar(range(len(sdpeaks))[0::2], sdpeaks[0::2], color='k', label='Exploration')
        plt.bar(range(len(sdpeaks))[1::2], sdpeaks[1::2], color='grey', label='Exploitation')
        plt.xticks([])
        plt.ylabel('Orientation shift [deg]')
        plt.legend()
        plt.savefig('figs2/sd/peaks.pdf'.format(), bbox_inches='tight', transparent=True)
