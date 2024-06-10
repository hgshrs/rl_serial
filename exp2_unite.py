import importlib
import exp2 as ab
importlib.reload(ab)
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.optimize

class rl_agent():
    def __init__(self, alpha, gamma, beta, init_q=0):
        self.alpha = alpha # learning rate
        self.gamma = gamma # forgetting factor
        self.beta = beta # inverse temperature
        self.init_q = init_q # initial value
        self.reset()

    def update_q(self, arm, reward):
        # initialize q function
        try:
            a = self.q[arm]
        except:
            self.q[arm] = self.init_q

        # update with learning rate. Eq [6.1]
        err = self.q[arm] - reward
        self.q[arm] = self.q[arm] - self.alpha * err

        # forget with forgetting factor
        arm_symbols = list(self.q.keys())
        if len(arm_symbols) > 1:
            arm_symbols.remove(arm)
            for dd in arm_symbols:
                # self.q[dd] = self.init_q + (1 - self.gamma) * (self.q[dd] - self.init_q)
                self.q[dd] = self.init_q + (1 - self.gamma) * (self.q[dd] - self.init_q)

    def make_decision(self, current_arms):
        # initialize q function
        for arm in current_arms:
            try:
                a = self.q[arm]
            except:
                self.q[arm] = self.init_q

        # Compute soft max with Q. Eq [6.2]
        m = np.zeros(len(current_arms))
        for aa, arm in enumerate(current_arms):
            m[aa] = np.exp(self.beta * self.q[arm])
        if m.sum() == 0.:
            m[:] = 1 / len(current_arms)
        else:
            m /= m.sum()

        likelihood = {}
        for aa, arm in enumerate(current_arms):
            likelihood[arm] = m[aa]

        return np.random.choice(list(likelihood.keys()), p=list(likelihood.values())), likelihood

    def reset(self):
        self.q = {
                1: self.init_q,
                2: self.init_q,
                }

def out_nllik(alpha, gamma, beta, init_q, df):
    agent = rl_agent(alpha=alpha, gamma=gamma, beta=beta, init_q=init_q)
    n_eps = len(df)
    nllik = 0.
    for ee in range(n_eps):
        act = df.iloc[ee]['action']
        rew = df.iloc[ee]['rew']
        _, lik = agent.make_decision([1, 2])
        nllik += -1 * np.log(lik[act])
        agent.update_q(act, rew)
    return nllik

def out_nllik_ovs(alpha, gamma, beta, init_q, dfs): # over sessions
    n_sess = len(dfs)
    nllik = 0.
    for ss in range(n_sess):
        _nllik = out_nllik(alpha, gamma, beta, init_q, dfs[ss])
        nllik += _nllik
    return nllik

def optimize_fun(x, dfs):
    alpha = x[0]
    gamma = x[1]
    beta = x[2]
    init_q = x[3]
    return out_nllik_ovs(alpha, gamma, beta, init_q, dfs)

def pickle_dump(var, path):
    with open(path, 'wb') as f:
        pickle.dump(var, f)

def pickle_load(path):
    with open(path, 'rb') as f:
        out = pickle.load(f)
    return out

if __name__=='__main__':
    data_dir = 'data_exp2'
    data_paths = [f.name for f in os.scandir(data_dir) if not f.name.startswith('.')]
    dfk = []
    blk_wi_sess = []
    blk_session = []
    for pp, dp in enumerate(tqdm(data_paths)):
        participant = dp[-28:-4]
        if dp[-3:] == 'csv' and dp[:3] == '202':
            df = pd.read_csv(data_dir + '/' + dp)
            df_blk = pd.DataFrame()
            blk_idx = 0
            ee = 0
            while ee < len(df):
                if df.iloc[ee]['key_resp_5.keys'] == 'space':
                    df_blk = pd.concat([df_blk, df.iloc[ee:ee+1]], axis=0)
                else:
                    if df.iloc[ee]['break_blk.started'] > 0:
                        blk_wi_sess.append(blk_idx)
                        blk_session.append(participant)
                        dfk.append(df_blk)
                        df_blk = pd.DataFrame()
                        blk_idx += 1
                ee += 1

    df = pd.DataFrame()
    removing_blks = [0, ]
    for bb, df_blk in enumerate(tqdm(dfk)):
        n_eps = len(df_blk)
        for ee in range(n_eps):
            reject = False
            if ee > 0:
                pc = df_blk.iloc[ee - 1]['ori{:.0f}'.format(df_blk.iloc[ee - 1]['action'])]
                pc_ = df_blk.iloc[ee - 1]['ori{:.0f}'.format(({1, 2} - {df_blk.iloc[ee - 1]['action']}).pop())]
                pr = df_blk.iloc[ee - 1]['res_ori']
                aa = (df_blk.iloc[ee - 1]['action'] == df_blk.iloc[ee]['action']) # taking the same action for ee-1 and ee
                prew = df_blk.iloc[ee - 1]['rew']
                if df_blk.iloc[ee - 1]['key_resp_5.rt'] > 10:
                    reject = True
            else:
                pc = np.nan
                pc_ = np.nan
                pr = np.nan
                aa = False
                prew = np.nan
            if ee > 1:
                pprew = df_blk.iloc[ee - 2]['rew']
            else:
                pprew = np.nan
            cc = df_blk.iloc[ee]['ori{:.0f}'.format(df_blk.iloc[ee]['action'])]
            cc_ = df_blk.iloc[ee]['ori{:.0f}'.format(({1, 2} - {df_blk.iloc[ee]['action']}).pop())]
            cr = df_blk.iloc[ee]['res_ori']
            adj_err = ab.signed_diff_orientations(cc, cr)
            _df = pd.DataFrame()
            _df['session'] = [blk_session[bb]]
            _df['blk'] = bb
            _df['blk_wi_sess'] = blk_wi_sess[bb]
            _df['ep'] = ee
            _df['ori1'] = df_blk.iloc[ee]['ori1']
            _df['ori2'] = df_blk.iloc[ee]['ori2']
            _df['ori'] = df_blk.iloc[ee]['res_ori']
            _df['rew1'] = df_blk.iloc[ee]['rew1']
            _df['rew2'] = df_blk.iloc[ee]['rew2']
            _df['action'] = df_blk.iloc[ee]['action']
            _df['rew'] = df_blk.iloc[ee]['rew']
            _df['rt'] = df_blk.iloc[ee]['key_resp_5.rt']
            _df['pre_rew'] = prew
            _df['ppre_rew'] = pprew
            _df['reject'] = reject
            _df['adj_err'] = adj_err
            _df['pre_acted_cue'] = pr
            _df['pre_nonacted_cue'] = pc_
            _df['pre_res'] = pr
            _df['acted_cue'] = cc
            _df['nonacted_cue'] = cc_
            _df['res'] = cr
            _df['aa'] = aa
            df = pd.concat([df, _df], axis=0)
    df = df.reset_index(drop=True)
    df.to_csv('tmp/bhv.csv')
    df = pd.read_csv('tmp/bhv.csv', index_col=0)

    n_participants = len(df['session'].unique())
    n_blocks = len(df['blk'].unique())
    print('#participant: {}'.format(n_participants))
    print('#blocks: {}'.format(n_blocks))
    print('#blocks/participant: {}'.format(n_blocks / n_participants))

    print('==================\nParticipant analysis\n==================')
    rew_p = []
    adj_err_p = []
    pids = []
    rejecting_participants = []
    df['reject_p'] = False
    bonus_p = []
    for pid in df['session'].unique():
        if type(pid) == type('a'):
            dfp = df.query("session == '{}'".format(pid))
        else:
            dfp = df.query("session == {}".format(pid))
        n_samples = len(dfp)
        pids.append(pid)
        rew_p.append(dfp['rew'].mean())
        bonus_p.append(dfp['rew'].mean())
        adj_err_p.append(np.abs(dfp['adj_err']).mean())
        ep_opt_act = np.ones(n_samples, dtype=int)
        ep_opt_act[dfp['rew2'] > dfp['rew1']] = 2
        opt_ratio = (ep_opt_act == dfp['action']).sum() / n_samples 

        ave_rand_rew = 0.
        if rew_p[-1] < ave_rand_rew:
            rejecting_participants.append(pid)
            df.loc[df['session'] == pid, 'reject_p'] = True
            bonus_p[-1] = 0.

        print('ID: ...{}, Rew: {:.2f}, Rand_rew: {:.2f}, ERR: {:.2f}, RT Ave: {:.2f}, RT Med: {:.2f}, OptR: {:.2%}'.format(
            pid[-4:], rew_p[-1], ave_rand_rew, adj_err_p[-1], dfp['rt'].mean(), dfp['rt'].median(), opt_ratio))
    my_rew_ave = 4.07
    my_rew_std = 0.46
    my_err_ave = 9.21
    my_err_std = 0.31
    print('ON: Rew: {:.2f}+-{:.2f}, ERR: {:.2f}+-{:.2f}'.format(np.mean(rew_p), np.std(rew_p), np.mean(adj_err_p), np.std(adj_err_p)))
    print('MY: Rew: {:.2f}+-{:.2f}, ERR: {:.2f}+-{:.2f}'.format(my_rew_ave, my_rew_std, my_err_ave, my_err_std))
    print('#rejected participant: {}/{}'.format(len(rejecting_participants), len(df['session'].unique())))
    print('Reject IDs: {}'.format(rejecting_participants))

    # bonus caclculation
    df_bonus = pd.DataFrame()
    df_bonus['pid'] = pids
    bonus_p = np.array(bonus_p)
    bonus_amount = 400 # pond
    df_bonus['val'] = np.floor(np.round(bonus_amount / bonus_p.sum() * bonus_p, decimals=3) * 100) / 100
    df_bonus = df_bonus[df_bonus['val'] > 0.]
    print('Amount of bonus: {} ({})'.format(df_bonus['val'].sum(), bonus_amount))
    df_bonus.to_csv('tmp/bonus.csv', header=False, index=False)

    df.to_csv('tmp/bhv.csv')
    df = pd.read_csv('tmp/bhv.csv', index_col=0)

    ########################################
    # RL analysis
    ########################################
    df_rl = pd.DataFrame(columns=['alpha', 'gamma', 'beta', 'init_q'])
    for pp, pid in enumerate(df['session'].unique()):
        dfs = []
        _dfs = df[df['session'] == pid]
        for bb, blk_idx in enumerate(_dfs['blk'].unique()):
            _df = df[df['blk'] == blk_idx]
            dfs.append(_df)
        res = scipy.optimize.minimize(
                optimize_fun,
                x0=[.5, .5, 1, 0],
                args=(dfs),
                bounds=[(0.1, .9999), (0, .9), (.1, 10), (0, 9)],
                )
        print('ID: ...{}, alpha: {:.3f}, gamma: {:.3f}, beta: {:.3f}, init_q: {:.3f}'.format(pid[-4:], *res.x))
        df_rl.loc[pid] = res.x
    df_rl.to_csv('tmp/rl_params.csv')
    df_rl = pd.read_csv('tmp/rl_params.csv', index_col=0)

    print('Add RL-model estimated values to dataframe...')
    for bb, blk_idx in enumerate(tqdm(df['blk'].unique())):
        dfb = df[df['blk'] == blk_idx]
        pid = dfb.iloc[0]['session']
        alpha = df_rl
        ag = rl_agent(df_rl.loc[pid].alpha, df_rl.loc[pid].gamma, df_rl.loc[pid].beta, df_rl.loc[pid].init_q)
        n_eps = len(dfb)
        lik_trg = .5
        for ee in range(len(dfb)):
            _df = dfb.loc[dfb['ep'] == ee]
            act = int(_df['action'].iloc[0])
            loc_index = (df['blk'] == blk_idx) & (df['ep'] == ee)
            df.loc[loc_index, 'q1'] = ag.q[1]
            df.loc[loc_index, 'q2'] = ag.q[2]
            df.loc[loc_index, 'pre_lik'] = lik_trg
            _, lik = ag.make_decision([1, 2])
            df.loc[loc_index, 'lik1'] = lik[1]
            df.loc[loc_index, 'lik2'] = lik[2]
            df.loc[loc_index, 'lik_trg'] = lik[act]
            lik_trg = lik[act]
            rew = _df['rew'].iloc[0]
            ag.update_q(act, rew)
    df.to_csv('tmp/bhv.csv')
