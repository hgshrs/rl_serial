import importlib
import pandas as pd
import matplotlib.pylab as plt
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
import numpy as np
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api
import statsmodels.stats
import itertools
import sys
import scipy.stats

def color_violinplot(parts, c='black'):
    for pc in parts['bodies']:
        pc.set_facecolor(c)
        pc.set_edgecolor(None)
    parts['cmeans'].set_color(c)
    parts['cmaxes'].set_color(c)
    parts['cmins'].set_color(c)
    parts['cbars'].set_color(c)

def sessionwise_out(df, averaging_param_name, wise_param_name, values=[]):
    if len(values) == 0:
        values = df[wise_param_name].unique()
    out = pd.DataFrame(index=[], columns=[])
    for sess in df['session'].unique():
        dfs = df[df['session'] == sess]
        for value in values:
            dfsc = dfs[dfs[wise_param_name] == value]
            if len(dfsc) > 0:
                exval = dfsc[averaging_param_name].mean()
            else:
                exval = np.nan
            new_data = pd.DataFrame(
                    data = [{
                        'value': exval,
                        'condition': value,
                        'session': sess,
                        'participant': dfs.iloc[0]['participant'],
                        }])
            out = pd.concat([out, new_data], axis=0)
    return out

def deg2rad_von(x):
    return np.deg2rad(x) - np.pi

if __name__=='__main__':

    threshold_angle_quantile = .9
    threshold_n_trials = 1
    threshold_lik = .75
    threshold_over_angle_trials = 10
    data_path = 'data_exp1.csv'

    target_participants = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    target_sessions = [0, 1, 2, 3, 4, 5, 6]
    sess_queries = 'participant in {} & session_within_participant in {}'.format(target_participants, target_sessions)
    df_org = pd.read_csv(data_path, index_col=0).query(sess_queries)
    df_all = df_org.copy()

    threshold_angle = df_all['angle_diff'].quantile(threshold_angle_quantile)
    print('Outliers > {:.4f} deg'.format(threshold_angle))

    for ss in df_all.session.unique():
        if len(df_all.query('angle_diff > {} & session == {}'.format(threshold_angle, ss))) > threshold_over_angle_trials:
            print('Session #{} rejected'.format(ss))
            df_all = df_all.query('session != {}'.format(ss))

    # ==============================
    # individual session
    # ==============================
    # for ss in df_all.session.unique():
    for ss in [82]:
        df = df_all.query('session == {}'.format(ss))

        bandit_labels = ['K', 'W']
        bandit_names = ['Black', 'White']
        bandit_color = ['k', 'w']
        lss = ['-', '--']

        plt.figure(1); plt.clf()
        plt.subplot(2, 1, 1)
        kt = np.zeros(len(df))
        wt = np.zeros(len(df))
        for tt in range(len(df)):
            plt.plot([tt, tt], [0, df['angle_diff'][tt]], ls='-', c='k')
            if df['angle_keep'][tt] == 1:
                plt.plot(tt, df['angle_diff'][tt], 'o', c='w', mec='k', ms=9)
            if df['target_dot'][tt] == 0:
                kt[tt] = df['angle_diff'][tt]
                wt[tt] = np.nan
            else:
                wt[tt] = df['angle_diff'][tt]
                kt[tt] = np.nan
        plt.plot(range(len(df)), kt, 'o', c='k', mec='k', label='Black (K)')
        plt.plot(range(len(df)), wt, 'o', c='w', mec='k', label='White (W)')
        plt.ylim([0, threshold_angle])
        plt.legend(loc='upper right')
        plt.xlabel('Trial')
        plt.ylabel('$|$AE$|$ [deg]')
        plt.savefig('figs1/session/dir_p{:02d}s{:02d}.pdf'.format(df.participant.loc[0], df.session_within_participant.loc[0]), bbox_inches='tight', transparent=True)

        plt.figure(2); plt.clf()
        for bb in range(2):
            plt.subplot(2, 1, bb + 1)
            label_u = r'$\Omega_' r'{\rm ' + '{}'.format(bandit_labels[bb]) + '}$'
            lu = plt.plot(df['rew_max{}'.format(bb)], c='k', ls='--', label=label_u)
            label_q ='$Q_' r'{\rm ' + '{}'.format(bandit_labels[bb]) + '}$'
            lq = plt.plot(df['q{}'.format(bb)], ls='-', c='k', label=label_q)
            for tt in range(len(df)):
                if df['target_dot'][tt] == bb:
                    bp = plt.plot(tt, df['rew'][tt], 'o', c=bandit_color[bb], mec='k')
            bp[0].set_label('Rew')
            plt.ylim([-1, 11])
            plt.ylabel('{} arm'.format(bandit_names[bb]))
            plt.legend(loc='upper right')
            if bb < 1:
                plt.xticks([])
            else:
                plt.xlabel('Trial')
        plt.savefig('figs1/session/uqrew_p{:02d}s{:02d}.pdf'.format(df.participant.loc[0], df.session_within_participant.loc[0]), bbox_inches='tight', transparent=True)

        plt.figure(3); plt.clf()
        ax = plt.subplot(2, 1, 1)
        for tt in range(len(df)):
            if df['target_lik'][tt] < threshold_lik:
                ax.axvspan(tt - .5, tt + .5, color='gray', ec=None, alpha=.5)
        for bb in range(2):
            label_p ='$P(' r'{\rm ' + '{}'.format(bandit_labels[bb]) + '})$'
            lp = plt.plot(df['lik{}'.format(bb)], c='k', ls=lss[bb], label=label_p)
        plt.ylim([-.1, 1.1])
        plt.legend(loc='upper right')
        plt.ylabel('Likelihood')
        plt.xlabel('Trial')
        plt.savefig('figs1/session/lk_p{:02d}s{:02d}.pdf'.format(df.participant.loc[0], df.session_within_participant.loc[0]), bbox_inches='tight', transparent=True)

        plt.figure(4); plt.clf()
        plt.subplot(311)
        l0 = plt.plot(df['rew_max0'], ls='--', label='r0')
        l1 = plt.plot(df['rew_max1'], ls='--', label='r1')
        for tt in range(len(df)):
            if df['target_dot'][tt] == 0:
                c = l0[0].get_color()
            else:
                c = l1[0].get_color()
            plt.plot(tt, df['rew'][tt], 'o', c=c, mec='k')

    # ==============================
    # grand analysis
    # ==============================
    dir_err = df_all['angle_diff']
    plt.figure(11).clf()
    plt.hist(dir_err, bins=100, color='k')
    ylim = plt.gca().get_ylim()
    plt.plot([threshold_angle, threshold_angle], ylim, 'k--')
    plt.plot(threshold_angle, ylim[1], 'ko')
    plt.plot([dir_err.mean(), dir_err.mean()], ylim, 'k--')
    plt.plot(dir_err.mean(), ylim[1], 'ks')
    plt.plot([dir_err.median(), dir_err.median()], ylim, 'k--')
    plt.plot(dir_err.median(), ylim[1], 'kv')
    plt.xlabel('$|$AE$|$ [deg]')
    plt.ylabel('Number of trials')
    plt.savefig('figs1/angle_diff.pdf', bbox_inches='tight', transparent=True)
    print('\n=========================')
    print('Mean dir error:\t{}+-{}'.format(dir_err.mean(), dir_err.std()))

    rew_kappa = 5.
    mu = 0
    nu = dir_err.mean()
    max_p = scipy.stats.vonmises.pdf(deg2rad_von(mu), rew_kappa, loc=deg2rad_von(mu))
    reduced_rew_proportion = scipy.stats.vonmises.pdf(deg2rad_von(nu), rew_kappa, loc=deg2rad_von(mu)) / max_p
    print('Max reward was reduced by {:.2%} on average'.format(reduced_rew_proportion))

    # sys.exit()

    # ==============================
    base_queries = 'angle_diff <= {} & participant in {} & session_within_participant in {}'.format(
            threshold_angle, target_participants, target_sessions)
    df_all1 = df_all.query(base_queries)

    n_removed_trials = len(df_org) - len(df_all1)
    n_removed_sessions = len(df_org['session'].unique()) - len(df_all1['session'].unique())
    n_removed_participants = len(df_org['participant'].unique()) - len(df_all1['participant'].unique())
    print('Removed: {} samples {:.1%}'.format(n_removed_trials, n_removed_trials/len(df_org)))
    print('Removed: {} sessions {:.1%}'.format(n_removed_sessions, n_removed_sessions/len(df_org['session'].unique())))
    print('Removed: {} participant {:.1%}'.format(n_removed_participants, n_removed_participants/len(df_org['participant'].unique())))
    # ==============================

    print('\n=========================')
    print('Propotion of the target arm (K vs W)')
    rew_diff = np.zeros(len(df_all1))
    rew_diff[:] = df_all1['rew_max1'] - df_all1['rew_max0']
    rew_diff = np.round(rew_diff).astype(int)
    dfrd = pd.concat([df_all1.reset_index(), pd.Series(rew_diff, name='rew_diff')], axis=1)
    dfw = sessionwise_out(dfrd, 'target_dot', 'rew_diff')
    dfw = dfw[np.logical_not(np.isnan(dfw['value']))]
    ls = []; vs = []; es = []
    for rd in np.sort(dfw.condition.unique()):
        ls.append(rd)
        vs.append(dfw[dfw['condition'] == rd]['value'].mean())
        es.append(dfw[dfw['condition'] == rd]['value'].std())
    plt.figure(1).clf()
    # plt.plot(np.array(ls), np.array(vs) * 100, 'k-')
    plt.errorbar(np.array(ls), np.array(vs), yerr=np.array(es), capsize=2, fmt='o', markersize=10, ecolor='k', mec='k', c='k')
    rl = np.polyfit(dfw['condition'], dfw['value'], 1)
    y = np.poly1d(rl)([dfw['condition'].min(), dfw['condition'].max()])
    plt.plot([dfw['condition'].min(), dfw['condition'].max()], y, 'k--')
    plt.ylabel('Proportion that\nblack arm was target')
    plt.xlabel('Diff in max rewards, ' + r'$\Omega_{\rm K} - \Omega_{\rm W}$')
    plt.yticks([0, .5, 1])
    plt.savefig('figs1/prop_action.pdf', bbox_inches='tight', transparent=True)

    res = scipy.stats.pearsonr(dfw['condition'], dfw['value'])
    print('Correlation between reward difference and propotion of the target arm')
    print('Corr({}):\t{}'.format(len(dfw) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))

    # =================================
    # Test AE by ANOVA
    # =================================
    print('\n=========================')
    query_set = {
                'AH': 'angle_keep == 1',
                'AN': 'angle_keep == 0',
                'TD': 'target_change == 1',
                'TS': 'target_change == 0',
            }
    query_cmbs = [['AH', 'AN'], ['TD', 'TS']]
    dfw = pd.DataFrame(index=[], columns=[])
    for ss, session in enumerate(df_all1.session.unique()):
        participant = df_all1.query('session == {}'.format(session))['participant'].unique()[0]
        for qq in itertools.product(*query_cmbs):
            _df = df_all1.query('session == {}'.format(session))
            data = {}
            for q in qq:
                data[q[0]] = q[1]
                _df = _df.query(query_set[q])
            data['value'] = _df['angle_diff'].mean()
            data['participant'] = participant
            data['session'] = session
            dfw = pd.concat([dfw, pd.DataFrame(data=[data])], axis=0)
    dfw = dfw[np.logical_not(np.isnan(dfw['value']))]
    print('AE for angle_keep X target_change by ANOVA ({} samples)'.format(len(dfw)))
    print('Hold vs No: {:.3f}+-{:.3f} vs {:.3f}+-{:.3f}'.format(
        dfw[dfw['A'] == 'H']['value'].mean(), dfw[dfw['A'] == 'H']['value'].std(),
        dfw[dfw['A'] == 'N']['value'].mean(), dfw[dfw['A'] == 'N']['value'].std(),
        ))
    print('Diff vs Same: {:.3f}+-{:.3f} vs {:.3f}+-{:.3f}'.format(
        dfw[dfw['T'] == 'D']['value'].mean(), dfw[dfw['T'] == 'D']['value'].std(),
        dfw[dfw['T'] == 'S']['value'].mean(), dfw[dfw['T'] == 'S']['value'].std(),
        ))

    f = 'value ~ C(A) + C(T) + C(A)*C(T) + C(participant) + C(session)'
    model = statsmodels.formula.api.ols(f, dfw).fit()
    # print(model.summary())
    aov_table = sm.stats.anova_lm(model, type=2)
    print(aov_table)

    plt.figure(4).clf()
    parts = plt.violinplot([
        dfw.query("A == 'H' & T == 'D'")['value'],
        dfw.query("A == 'N' & T == 'D'")['value'],
        dfw.query("A == 'H' & T == 'S'")['value'],
        dfw.query("A == 'N' & T == 'S'")['value'],
        ], showmeans=True)
    color_violinplot(parts)
    plt.xticks([1, 2, 3, 4], ['Hold\nChange', 'No-hold\nChange', 'Hold\nKeep', 'No-hold\nKeep'])
    plt.xlabel('Trial condition/Target continuity')
    plt.ylabel('$|$AE$|$ [deg]')
    # plt.ylim([3, 25])
    plt.savefig('figs1/dir_err_trial_condition_continuity.pdf', bbox_inches='tight', transparent=True)


    # =================================
    # Test AE in correlation
    # =================================
    rew_max = np.zeros(len(df_all1))
    rew_max[df_all1['target_dot'] == 0] = df_all1[df_all1['target_dot'] == 0]['rew_max0']
    rew_max[df_all1['target_dot'] == 1] = df_all1[df_all1['target_dot'] == 1]['rew_max1']
    rew_max = np.round(rew_max).astype(int)
    dfr = pd.concat([df_all1.reset_index(), pd.Series(rew_max, name='rew_max')], axis=1)
    dfw = sessionwise_out(dfr, 'angle_diff', 'rew_max')
    dfw = dfw[np.logical_not(np.isnan(dfw['value']))]
    ls = []; vs = []; es = []
    for rm in np.sort(dfw.condition.unique()):
        ls.append(rm)
        vs.append(dfw[dfw['condition'] == rm]['value'].mean())
        es.append(dfw[dfw['condition'] == rm]['value'].std())

    plt.figure(3).clf()
    # plt.plot(dfw['condition'], dfw['value'], 'ko', alpha=.2)
    plt.errorbar(np.array(ls), np.array(vs), yerr=np.array(es), capsize=2, fmt='o', markersize=10, ecolor='k', mec='k', c='k')
    rl = np.polyfit(dfw['condition'], dfw['value'], 1)
    y = np.poly1d(rl)([dfw['condition'].min(), dfw['condition'].max()])
    plt.plot([dfw['condition'].min(), dfw['condition'].max()], y, 'k--')
    plt.ylabel('$|$AE$|$ [deg]')
    plt.xlabel('Max reward on the target arm, ' + r'$\Omega_{s}$')
    plt.savefig('figs1/corr_err_rew_target.pdf', bbox_inches='tight', transparent=True)
    res = scipy.stats.pearsonr(dfw['condition'], dfw['value'])
    print('\n=========================')
    print('Correlation between max reward on the target arm and dir error')
    print('Corr({}):\t{}'.format(len(dfw) - 2, res[0]))
    print('pvalue:\t{}'.format(res[1]))


    # ==============================
    # Conditioned analysis
    # ==============================
    df_all = pd.read_csv(data_path, index_col=0)
    base_queries = 'prior_angle_diff <= {} & angle_diff <= {} \
            & participant in {} & session_within_participant in {} \
            & target_lik < {} & target_change == 1'.format(
            threshold_angle, threshold_angle,
            target_participants, target_sessions,
            threshold_lik)
    df_all = df_all.query(base_queries)

    queries = {
            'RH': "prior_target_lik < {} & angle_keep == 1".format(threshold_lik),
            'IH': "prior_target_lik >= {} & angle_keep == 1".format(threshold_lik),
            'RN': "prior_target_lik < {} & angle_keep == 0".format(threshold_lik),
            'IN': "prior_target_lik >= {} & angle_keep == 0".format(threshold_lik),
            }
    abbr2sta = { 'RH': 'Explor-\nHold', 'IH': 'Exploi-\nHold', 'RN': 'Explor-\nNohold', 'IN': 'Exploi-\nNohold', }
    dfk = pd.DataFrame(index=[], columns=[])
    for ss, session in enumerate(df_all.session.unique()):
        participant = df_all.query('session == {}'.format(session))['participant'].unique()[0]
        _dfk = pd.DataFrame(index=[], columns=[])
        for qq, condition in enumerate(queries.keys()):
            _df = df_all.query(queries[condition] + '& session == {}'.format(session))
            if len(_df) >= threshold_n_trials:
                _dfk0 = pd.DataFrame(
                        data = [{
                            'mdee': _df['angle_diff'].mean(),
                            'phase': condition[0],
                            'condition': condition[1],
                            'participant': participant,
                            'session': session,
                            'phasecond': condition[0] + condition[1],
                            'lik': _df['target_lik'].mean(),
                            }])
                _dfk = pd.concat([_dfk, _dfk0], axis=0)
        if len(_dfk) == len(queries):
            dfk = pd.concat([dfk, _dfk], axis=0)
    dfk.reset_index()
    n_participants = len(dfk['participant'].unique())
    n_org_participants = len(df_org['participant'].unique())
    n_sessions = len(dfk['session'].unique())
    n_org_sessions = len(df_org['session'].unique())
    print('#sessions: {} ({} ({:.1%}) removed)'.format(n_sessions, n_org_sessions - n_sessions, (n_org_sessions - n_sessions)/n_org_sessions))
    print('#participants: {} ({} ({:.1%}) removed)'.format(n_participants, n_org_participants - n_participants, (n_org_participants - n_participants)/n_org_participants))


    # =================================
    # Test AE for state
    # =================================
    print('\n=========================')
    print('AE at state')
    f = 'mdee ~ phase + condition + C(participant) + C(session) + phase*condition'
    model = statsmodels.formula.api.ols(f, dfk).fit()
    # print(model.summary())
    aov_table = sm.stats.anova_lm(model, type=2)
    print(aov_table)

    for c0, c1 in itertools.combinations(queries.keys(), 2):
        mdee0 = dfk.query("phase == '{}' & condition == '{}'".format(c0[0], c0[1]))['mdee']
        mdee1 = dfk.query("phase == '{}' & condition == '{}'".format(c1[0], c1[1]))['mdee']
        tres = scipy.stats.ttest_rel(mdee0, mdee1, alternative='less')
        print('{} vs {}: {}'.format(c0, c1, tres))

    plt.figure(0).clf()
    # plt.subplot(121)
    vals = np.zeros([n_sessions, len(queries)])
    xs = np.zeros([n_sessions, len(queries)])
    labels = []
    for qq, condition in enumerate(queries.keys()):
        vals[:, qq] = dfk.query("phase == '{}' & condition == '{}'".format(condition[0], condition[1]))['mdee']
        xs[:, qq] = qq + 1 + np.random.uniform(-.2, .2, size=n_sessions)
        # plt.plot(qq + 1 + np.random.uniform(-.2, .2, size=n_sessions), vals[:, qq], 'ko', alpha=.2)
        labels.append(abbr2sta[condition])
    plt.plot(xs.T, vals.T, 'k--', alpha=.1, lw=1)
    plt.plot(xs, vals, 'ko', alpha=.2)
    vparts = plt.violinplot(vals, showmeans=True)
    color_violinplot(vparts)
    plt.xticks(np.arange(len(queries)) + 1, labels)
    plt.ylabel('$|$AE$|$ [deg]')
    plt.xlabel('State')
    plt.savefig('figs1/mdee_states.pdf'.format(), bbox_inches='tight', transparent=True)

    # =================================
    # Test likelihood for state
    # =================================
    print('\n=========================')
    print('Likelihood at state')
    f = 'lik ~ phase + condition + C(participant) + C(session) + phase*condition'
    model = statsmodels.formula.api.ols(f, dfk).fit()
    aov_table = sm.stats.anova_lm(model, type=2)
    print(aov_table)

    plt.figure(1).clf()
    vals = np.zeros([n_sessions, len(queries)])
    xs = np.zeros([n_sessions, len(queries)])
    for qq, condition in enumerate(queries.keys()):
        vals[:, qq] = dfk.query("phase == '{}' & condition == '{}'".format(condition[0], condition[1]))['lik']
        xs[:, qq] = qq + 1 + np.random.uniform(-.2, .2, size=n_sessions)
    plt.plot(xs.T, vals.T, 'k--', alpha=.1, lw=1)
    plt.plot(xs, vals, 'ko', alpha=.2)
    vparts = plt.violinplot(vals, showmeans=True)
    color_violinplot(vparts)
    plt.xticks(np.arange(len(queries)) + 1, labels)
    plt.ylabel('Likelihood')
    plt.xlabel('State')
    plt.savefig('figs1/lik_states.pdf'.format(), bbox_inches='tight', transparent=True)

    # =================================
    # Test likelihood x AE
    # =================================
    print('\n=========================')
    print('Correlation between AE and likelihood')
    res = scipy.stats.pearsonr(dfk['lik'], dfk['mdee'])
    print('Corr(df: {}):\t{:.3f}'.format(len(dfk) - 2, res[0]))
    print('pvalue:\t{:.3f}'.format(res[1]))
    plt.figure(2).clf()
    markers = ['o', 'o', '^', '^']
    mfcs = ['k', 'w', 'k', 'w']
    abbr2sta = { 'RH': 'Explor-Hold', 'IH': 'Exploi-Hold', 'RN': 'Explor-Nohold', 'IN': 'Exploi-Nohold', }
    for ss, state in enumerate(dfk['phasecond'].unique()):
        plt.plot(dfk[dfk['phasecond'] == state]['lik'], dfk[dfk['phasecond'] == state]['mdee'], c='k', mfc=mfcs[ss], marker=markers[ss], ls='None', label=abbr2sta[state])
    plt.legend()
    rl = np.polyfit(dfk['lik'], dfk['mdee'], 1)
    y = np.poly1d(rl)([dfk['lik'].min(), dfk['lik'].max()])
    plt.plot([dfk['lik'].min(), dfk['lik'].max()], y, 'k')
    plt.xlabel('Likelihood')
    plt.ylabel('$|$AE$|$ [deg]')
    plt.savefig('figs1/lik_mdee.pdf'.format(), bbox_inches='tight', transparent=True)
    
    plt.show()
