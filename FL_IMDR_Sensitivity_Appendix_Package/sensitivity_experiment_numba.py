from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np
import pandas as pd
from numba import njit

N_PATIENTS = 100
N_INSURERS = 4
T = 120
D = 11
N_SEEDS = 30
LAST_WINDOW = 10

@dataclass(frozen=True)
class Config:
    tau: float = 0.10
    coverage_target: float = 0.90
    fl_interval: int = 10
    dp_sigma: float = 0.05
    imitation_weight: float = 1.0
    gap_beta: float = 3.0
    clip_C: float = 1.0
    local_lr: float = 0.055
    local_steps: int = 2
    switch_prob: float = 0.50
    force_step_pct: float = 0.05
    max_voucher: float = 450.0
    claim_scale: float = 850.0

BASELINE = Config()
SWEEPS: Dict[str, List[float]] = {
    'tau': [0.05, 0.075, 0.10, 0.125, 0.15],
    'coverage_target': [0.80, 0.85, 0.90, 0.95, 0.98],
    'fl_interval': [2, 5, 10, 20, 30],
    'dp_sigma': [0.00, 0.025, 0.05, 0.075, 0.10],
    'imitation_weight': [0.00, 0.50, 1.00, 1.50, 2.00],
    'gap_beta': [1.0, 2.0, 3.0, 4.0, 5.0],
}
PARAM_SYMBOLS = {
    'tau': r'$\\tau$',
    'coverage_target': r'$\\eth_{\\mathrm{target}}$',
    'fl_interval': r'$K$',
    'dp_sigma': r'$\\sigma_{\\mathrm{DP}}$',
    'imitation_weight': r'$\\omega_{\\mathrm{imit}}$',
    'gap_beta': r'$\\beta_g$',
}
PARAM_LABELS = {
    'tau': 'Tax/cash-back rate',
    'coverage_target': 'Coverage target',
    'fl_interval': 'Federation interval',
    'dp_sigma': 'DP noise scale',
    'imitation_weight': 'Imitation weight',
    'gap_beta': 'Coverage-gap gain',
}
METRIC_NAMES = [
    'insured_pct', 'effective_price', 'coverage', 'profit_per_insurer',
    'subsidy_per_patient', 'risk_mse', 'hhi', 'coverage_disparity',
    'goal_score', 'insurer_mse_disparity'
]
METRIC_LABELS = {
    'insured_pct': 'Insured percentage',
    'effective_price': 'Accepted effective premium',
    'coverage': 'Accepted coverage',
    'profit_per_insurer': 'Profit per insurer',
    'subsidy_per_patient': 'Subsidy per patient',
    'risk_mse': 'Risk-prediction MSE',
    'hhi': 'Market concentration (HHI)',
    'coverage_disparity': 'Risk-group coverage disparity',
    'goal_score': 'Regulator goal score',
    'insurer_mse_disparity': 'Insurer MSE disparity',
}
PRIMARY_METRICS = METRIC_NAMES[:-1]


def make_random_tape(seed: int):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 70, size=N_PATIENTS).astype(np.float64)
    income = np.clip(rng.normal(70_000, 25_000, size=N_PATIENTS), 10_000, 200_000).astype(np.float64)
    credit = rng.uniform(300, 850, size=N_PATIENTS).astype(np.float64)
    health = rng.beta(4, 2, size=N_PATIENTS).astype(np.float64)
    location = rng.uniform(0, 1, size=N_PATIENTS).astype(np.float64)
    wanted = rng.uniform(0.10, 0.90, size=N_PATIENTS).astype(np.float64)
    budget_frac = rng.uniform(0.07, 0.20, size=N_PATIENTS).astype(np.float64)
    return (
        age, income, credit, health, location, wanted, budget_frac,
        rng.normal(size=(T, N_PATIENTS)).astype(np.float64),
        rng.normal(size=(T, N_PATIENTS)).astype(np.float64),
        rng.normal(size=(T, N_PATIENTS)).astype(np.float64),
        rng.normal(size=(T, N_PATIENTS)).astype(np.float64),
        rng.normal(size=(T, N_PATIENTS, N_INSURERS)).astype(np.float64),
        rng.normal(size=(T, N_PATIENTS, N_INSURERS)).astype(np.float64),
        rng.random(size=(T, N_PATIENTS)).astype(np.float64),
        rng.random(size=(T, N_PATIENTS)).astype(np.float64),
        rng.random(size=(T, N_PATIENTS)).astype(np.float64),
        rng.lognormal(mean=-0.5 * 0.25**2, sigma=0.25, size=(T, N_PATIENTS)).astype(np.float64),
        rng.normal(size=(T, D)).astype(np.float64),
        rng.normal(0.0, 0.02, size=D).astype(np.float64),
        rng.integers(0, N_INSURERS, size=N_PATIENTS).astype(np.int64),
    )


def cfg_array(cfg: Config):
    return np.array([
        cfg.tau, cfg.coverage_target, float(cfg.fl_interval), cfg.dp_sigma,
        cfg.imitation_weight, cfg.gap_beta, cfg.clip_C, cfg.local_lr,
        float(cfg.local_steps), cfg.switch_prob, cfg.force_step_pct,
        cfg.max_voucher, cfg.claim_scale
    ], dtype=np.float64)


@njit(cache=True)
def sigmoid(x):
    if x > 30.0:
        return 1.0
    if x < -30.0:
        return 0.0
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def calc_risk(age, income, credit, health, noise, out):
    for i in range(N_PATIENTS):
        h = min(1.0, max(0.0, health[i]))
        c = min(1.0, max(0.0, credit[i] / 850.0))
        a = min(1.0, max(0.0, age[i] / 100.0))
        y = min(1.0, max(0.0, income[i] / 120000.0))
        base = (1.0 - 0.2 * y) * np.exp(-3.0 * h) * (0.5 * (1.0-c)**2 + 0.5 * (a*a + 0.2*a))
        r = base * (1.0 + 0.05 * noise[i])
        out[i] = min(1.0, max(0.0, r))


@njit(cache=True)
def build_features(age, income, credit, health, location, X):
    for i in range(N_PATIENTS):
        a = min(1.5, max(0.0, age[i] / 70.0))
        y = min(1.7, max(0.0, income[i] / 120000.0))
        c = min(1.0, max(0.0, credit[i] / 850.0))
        h = min(1.0, max(0.0, health[i]))
        l = min(1.0, max(0.0, location[i]))
        X[i,0]=1.0; X[i,1]=a; X[i,2]=y; X[i,3]=c; X[i,4]=h; X[i,5]=l
        X[i,6]=a*a; X[i,7]=(1.0-c)**2; X[i,8]=h*h; X[i,9]=a*h; X[i,10]=y*h


@njit(cache=True)
def dot_clip(X, w, i):
    s = 0.0
    for d in range(D):
        s += X[i,d] * w[d]
    if s < 0.0:
        return 0.0
    if s > 1.0:
        return 1.0
    return s


@njit(cache=True)
def run_one_numba(
    age0, income0, credit0, health0, location, wanted, budget_frac,
    state_income, state_health, state_credit, risk_noise,
    offer_price_noise, offer_cov_noise, accept_u, voucher_accept_u, switch_u,
    claim_noise, dp_noise, init_w, init_assign, params
):
    tau=params[0]; target=params[1]; fl_interval=int(params[2]); dp_sigma=params[3]
    imitation=params[4]; gap_beta=params[5]; clip_C=params[6]; lr=params[7]
    local_steps=int(params[8]); switch_prob=params[9]; force_step=params[10]
    max_voucher=params[11]; claim_scale=params[12]

    age=age0.copy(); income=income0.copy(); credit=credit0.copy(); health=health0.copy()
    base_income=income0.copy(); base_credit=credit0.copy(); base_health=health0.copy()
    assignment=init_assign.copy()
    insured=np.zeros(N_PATIENTS, dtype=np.uint8)
    for i in range(10): insured[i]=1
    current_price=np.zeros(N_PATIENTS); current_eff=np.zeros(N_PATIENTS); current_cov=np.zeros(N_PATIENTS)
    w_global=init_w.copy(); w_local=np.empty((N_INSURERS,D))
    for j in range(N_INSURERS):
        for d in range(D): w_local[j,d]=w_global[d]

    margin_price=np.array([-0.05,0.0,0.04,0.08])
    risk_load_price=np.array([0.25,0.45,-0.10,0.65])
    margin_cov=np.array([0.05,0.0,-0.03,0.03])
    risk_load_cov=np.array([0.15,0.25,-0.05,0.35])

    y=np.zeros(N_PATIENTS); X=np.zeros((N_PATIENTS,D)); monthly_budget=np.zeros(N_PATIENTS)
    value_per_cov=np.zeros(N_PATIENTS); pred=np.zeros((N_PATIENTS,N_INSURERS))
    price=np.zeros((N_PATIENTS,N_INSURERS)); cov=np.zeros((N_PATIENTS,N_INSURERS)); util=np.zeros((N_PATIENTS,N_INSURERS))
    best=np.zeros(N_PATIENTS,dtype=np.int64); best_u=np.zeros(N_PATIENTS)
    subsidy_paid=np.zeros(N_PATIENTS); claim=np.zeros(N_PATIENTS)
    metrics=np.zeros((T,10))

    # fixed baseline risk quartiles by rank
    calc_risk(age, income, credit, health, risk_noise[0], y)
    order=np.argsort(y); group=np.zeros(N_PATIENTS,dtype=np.int64)
    for rank in range(N_PATIENTS): group[order[rank]]=min(3,(4*rank)//N_PATIENTS)

    for t in range(T):
        if t>0:
            for i in range(N_PATIENTS):
                income[i] += 0.05*(base_income[i]-income[i]) + 0.05*base_income[i]*state_income[t,i]
                health[i] += 0.05*(base_health[i]-health[i]) + 0.05*state_health[t,i]
                credit[i] += 0.05*(base_credit[i]-credit[i]) + 0.03*base_credit[i]*state_credit[t,i]
                age[i]=min(age[i]+1.0,100.0)
                income[i]=min(200000.0,max(10000.0,income[i]))
                health[i]=min(1.0,max(0.0,health[i])); credit[i]=min(850.0,max(300.0,credit[i]))

        calc_risk(age,income,credit,health,risk_noise[t],y); build_features(age,income,credit,health,location,X)
        for i in range(N_PATIENTS):
            monthly_budget[i]=min(1000.0,max(130.0,income[i]*budget_frac[i]/36.0))
            value_per_cov[i]=monthly_budget[i]/max(wanted[i],1e-3)
            best_u[i]=-1e30; best[i]=-1
            for j in range(N_INSURERS):
                pr=dot_clip(X,w_local[j],i); pred[i,j]=pr
                p=205.0*(1.0+margin_price[j]+risk_load_price[j]*(pr-0.20))*(1.0+0.025*offer_price_noise[t,i,j])
                p=min(700.0,max(150.0,p)); price[i,j]=p
                c=0.72+margin_cov[j]-risk_load_cov[j]*(pr-0.20)+0.025*offer_cov_noise[t,i,j]
                c=min(1.0,max(0.10,c)); cov[i,j]=c
                u=value_per_cov[i]*c-p-0.10*max(p-monthly_budget[i],0.0)-0.10*value_per_cov[i]*max(wanted[i]-c,0.0)
                util[i,j]=u
                # Patient selection follows the manuscript's affordability and
                # minimum-coverage constraints.
                if p<=monthly_budget[i] and c>=wanted[i] and u>best_u[i]:
                    best_u[i]=u; best[i]=j

        for i in range(N_PATIENTS):
            can_act = insured[i]==0 or switch_u[t,i]<switch_prob
            feasible = best[i] >= 0
            pacc=sigmoid(5.0*best_u[i]/max(monthly_budget[i],1.0)) if feasible else 0.0
            accept=can_act and feasible and accept_u[t,i]<pacc
            if t<5 and insured[i]==0: accept=False
            if accept:
                j=best[i]; assignment[i]=j; insured[i]=1
                current_price[i]=price[i,j]; current_eff[i]=price[i,j]; current_cov[i]=cov[i,j]

        client_sizes=np.zeros(N_INSURERS,dtype=np.int64)
        # local gradient steps
        for j in range(N_INSURERS):
            for i in range(N_PATIENTS):
                if insured[i]==1 and assignment[i]==j: client_sizes[j]+=1
            if client_sizes[j]>0:
                for step in range(local_steps):
                    grad=np.zeros(D)
                    for i in range(N_PATIENTS):
                        if insured[i]==1 and assignment[i]==j:
                            raw=0.0
                            for d in range(D): raw += X[i,d]*w_local[j,d]
                            err=raw-y[i]
                            for d in range(D): grad[d]+=2.0*X[i,d]*err/client_sizes[j]
                    for d in range(D): w_local[j,d]-=lr*grad[d]

        if imitation>0.0:
            eta=min(0.08,0.025*imitation)
            mean_w=np.zeros(D)
            for d in range(D):
                for j in range(N_INSURERS): mean_w[d]+=w_local[j,d]/N_INSURERS
            for j in range(N_INSURERS):
                for d in range(D): w_local[j,d]+=eta*(mean_w[d]-w_local[j,d])

        preliminary_profit=np.zeros(N_INSURERS)
        for i in range(N_PATIENTS):
            subsidy_paid[i]=0.0
            if insured[i]==1:
                claim[i]=y[i]*current_cov[i]*claim_scale*claim_noise[t,i]
                preliminary_profit[assignment[i]] += current_price[i]-claim[i]
            else: claim[i]=0.0
        count_ins=0
        for i in range(N_PATIENTS): count_ins+=insured[i]
        insured_pre=count_ins/N_PATIENTS; gap=max(0.0,target-insured_pre)
        gamma_gap=(1.0+gap_beta*gap/max(target,1e-8))**2
        positive_profit=0.0
        for j in range(N_INSURERS): positive_profit+=max(preliminary_profit[j],0.0)
        pool=tau*gamma_gap*positive_profit
        uninsured_count=N_PATIENTS-count_ins
        if t>=5 and uninsured_count>0 and pool>0.0:
            voucher=min(max_voucher,pool/uninsured_count)
            for i in range(N_PATIENTS):
                if insured[i]==0:
                    j=-1; min_p=1e30
                    for jj in range(N_INSURERS):
                        if cov[i,jj]>=wanted[i] and price[i,jj]<min_p:
                            min_p=price[i,jj]; j=jj
                    if j>=0:
                        eff=max(price[i,j]-voucher,0.0)
                        uv=value_per_cov[i]*cov[i,j]-eff-0.10*max(eff-monthly_budget[i],0.0)-0.10*value_per_cov[i]*max(wanted[i]-cov[i,j],0.0)
                        if eff<=monthly_budget[i] and voucher_accept_u[t,i] < sigmoid(5.0*uv/max(monthly_budget[i],1.0)):
                            assignment[i]=j; insured[i]=1; current_price[i]=price[i,j]; current_eff[i]=eff; current_cov[i]=cov[i,j]
                            subsidy_paid[i]=min(voucher,current_price[i])

        count_ins=0
        for i in range(N_PATIENTS): count_ins+=insured[i]
        target_count=int(np.ceil(target*N_PATIENTS)); needed=max(0,target_count-count_ins)
        if t>=5 and needed>0:
            allowed=min(needed,max(1,int(np.ceil(force_step*N_PATIENTS*(1.0+gap)))))
            for rep in range(allowed):
                chosen=-1; chosen_j=0; chosen_p=1e30
                for i in range(N_PATIENTS):
                    if insured[i]==0:
                        local_j=-1; local_p=1e30
                        for j in range(N_INSURERS):
                            if cov[i,j]>=wanted[i] and price[i,j]<local_p:
                                local_p=price[i,j]; local_j=j
                        if local_j<0:
                            local_j=0; local_p=price[i,0]
                            for j in range(1,N_INSURERS):
                                if price[i,j]<local_p: local_p=price[i,j]; local_j=j
                        if local_p<chosen_p: chosen=i; chosen_j=local_j; chosen_p=local_p
                if chosen<0: break
                insured[chosen]=1; assignment[chosen]=chosen_j; current_price[chosen]=price[chosen,chosen_j]
                current_cov[chosen]=cov[chosen,chosen_j]; v=min(max_voucher,current_price[chosen])
                current_eff[chosen]=max(current_price[chosen]-v,0.0); subsidy_paid[chosen]=v

        profits=np.zeros(N_INSURERS); shares=np.zeros(N_INSURERS); local_mses=np.zeros(N_INSURERS)
        for j in range(N_INSURERS):
            n_j=0; mse_j=0.0
            for i in range(N_PATIENTS):
                if insured[i]==1 and assignment[i]==j:
                    cl=y[i]*current_cov[i]*claim_scale*claim_noise[t,i]
                    profits[j]+=current_price[i]-cl; n_j+=1
                pr=dot_clip(X,w_local[j],i); mse_j+=(pr-y[i])**2/N_PATIENTS
            shares[j]=n_j/N_PATIENTS; local_mses[j]=mse_j

        if (t+1)%fl_interval==0:
            weights=np.zeros(N_INSURERS); wsum=0.0
            for j in range(N_INSURERS): weights[j]=max(client_sizes[j],1); wsum+=weights[j]
            for j in range(N_INSURERS): weights[j]/=wsum
            avg_delta=np.zeros(D)
            for j in range(N_INSURERS):
                norm=0.0
                for d in range(D): norm+=(w_local[j,d]-w_global[d])**2
                norm=np.sqrt(norm); scale=min(1.0,clip_C/max(norm,1e-12))
                for d in range(D): avg_delta[d]+=weights[j]*(w_local[j,d]-w_global[d])*scale
            for d in range(D): w_global[d]+=avg_delta[d]+dp_sigma*dp_noise[t,d]
            for j in range(N_INSURERS):
                for d in range(D): w_local[j,d]=w_global[d]

        risk_mse=0.0
        for i in range(N_PATIENTS):
            pg=dot_clip(X,w_global,i); risk_mse+=(pg-y[i])**2/N_PATIENTS
        group_cov=np.zeros(4); group_n=np.zeros(4)
        count_ins=0; sum_eff=0.0; sum_cov=0.0; subsidy_total=0.0
        for i in range(N_PATIENTS):
            g=group[i]; group_n[g]+=1.0; group_cov[g]+=insured[i]
            if insured[i]==1: count_ins+=1; sum_eff+=current_eff[i]; sum_cov+=current_cov[i]
            subsidy_total+=subsidy_paid[i]
        for g in range(4): group_cov[g]/=max(group_n[g],1.0)
        cov_disp=np.max(group_cov)-np.min(group_cov); insured_pct=count_ins/N_PATIENTS
        eff_price=sum_eff/max(count_ins,1); acc_cov=sum_cov/max(count_ins,1)
        hhi=0.0; profit_mean=0.0
        for j in range(N_INSURERS): hhi+=shares[j]**2; profit_mean+=profits[j]/N_INSURERS
        mean_budget=0.0
        for i in range(N_PATIENTS): mean_budget+=monthly_budget[i]/N_PATIENTS
        afford=1.0-min(1.0,max(0.0,eff_price/max(mean_budget,1.0)))
        goal=0.5*insured_pct+0.3*acc_cov+0.2*afford
        max_mse=local_mses[0]; min_mse=local_mses[0]
        for j in range(1,N_INSURERS): max_mse=max(max_mse,local_mses[j]); min_mse=min(min_mse,local_mses[j])
        metrics[t,0]=insured_pct; metrics[t,1]=eff_price; metrics[t,2]=acc_cov; metrics[t,3]=profit_mean
        metrics[t,4]=subsidy_total/N_PATIENTS; metrics[t,5]=risk_mse; metrics[t,6]=hhi
        metrics[t,7]=cov_disp; metrics[t,8]=goal; metrics[t,9]=max_mse-min_mse
    return metrics


def all_settings():
    for param, values in SWEEPS.items():
        for value in values:
            cfg = replace(BASELINE, **{param: int(value) if param=='fl_interval' else float(value)})
            yield param, float(value), cfg


def convergence_month(curve):
    curve=np.asarray(curve,float); smooth=pd.Series(curve).rolling(5,min_periods=1).mean().to_numpy()
    final=float(np.mean(smooth[-10:])); initial=float(np.mean(smooth[:5]))
    if initial>final: threshold=final+0.10*(initial-final); candidates=np.where(smooth<=threshold)[0]
    else: candidates=np.where(np.abs(smooth-final)<=0.05*max(abs(final),1e-8))[0]
    return int(candidates[0]+1) if len(candidates) else T


def run_sensitivity(n_seeds=N_SEEDS):
    tapes=[make_random_tape(s) for s in range(n_seeds)]
    # warm-up compilation
    _=run_one_numba(*tapes[0],cfg_array(BASELINE))
    rows=[]; total=sum(len(v) for v in SWEEPS.values())
    for idx,(param,value,cfg) in enumerate(all_settings(),1):
        print(f'[{idx:02d}/{total}] {param}={value}',flush=True)
        p=cfg_array(cfg)
        for seed,tape in enumerate(tapes):
            arr=run_one_numba(*tape,p)
            for t in range(T):
                row={'parameter':param,'value':value,'seed':seed,'month':t+1}
                for k,name in enumerate(METRIC_NAMES): row[name]=float(arr[t,k])
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_sensitivity(raw):
    per=[]
    for (param,value,seed),g in raw.groupby(['parameter','value','seed'],sort=False):
        tail=g.tail(LAST_WINDOW); row={'parameter':param,'value':value,'seed':seed}
        for metric in PRIMARY_METRICS: row[metric]=float(tail[metric].mean())
        row['convergence_month']=convergence_month(g['risk_mse'].to_numpy()); per.append(row)
    per_seed=pd.DataFrame(per)
    agg=per_seed.groupby(['parameter','value'],sort=False).agg(['mean','std','count']).reset_index()
    agg.columns=['parameter','value']+[f'{a}_{b}' for a,b in agg.columns.tolist()[2:]]
    erows=[]
    for param,values in SWEEPS.items():
        p0=float(getattr(BASELINE,param)); base=per_seed[(per_seed.parameter==param)&np.isclose(per_seed.value,p0)]
        bmeans=base[PRIMARY_METRICS].mean()
        for value in values:
            if np.isclose(value,p0): continue
            cur=per_seed[(per_seed.parameter==param)&np.isclose(per_seed.value,float(value))]
            cmeans=cur[PRIMARY_METRICS].mean(); denom=(float(value)-p0)/p0
            for metric in PRIMARY_METRICS:
                m0=float(bmeans[metric]); m=float(cmeans[metric]); e=((m-m0)/m0)/denom if abs(m0)>1e-12 else np.nan
                erows.append({'parameter':param,'value':float(value),'metric':metric,'elasticity':e,'abs_elasticity':abs(e)})
    return per_seed,agg,pd.DataFrame(erows)


def paired_tests(per_seed):
    from scipy import stats
    rows=[]
    for param,values in SWEEPS.items():
        p0=float(getattr(BASELINE,param)); base=per_seed[(per_seed.parameter==param)&np.isclose(per_seed.value,p0)].sort_values('seed')
        for metric in PRIMARY_METRICS:
            local=[]
            for value in values:
                if np.isclose(value,p0): continue
                cur=per_seed[(per_seed.parameter==param)&np.isclose(per_seed.value,float(value))].sort_values('seed')
                x=cur[metric].to_numpy(); y=base[metric].to_numpy()
                try: stat,p=stats.wilcoxon(x,y,zero_method='wilcox')
                except ValueError: stat,p=0.0,1.0
                local.append({'parameter':param,'value':float(value),'metric':metric,'W':float(stat),'p_raw':float(p)})
            pvals=np.array([r['p_raw'] for r in local]); order=np.argsort(pvals); adj=np.empty_like(pvals); running=0.0; m=len(pvals)
            for rank,ii in enumerate(order): running=max(running,min(1.0,(m-rank)*pvals[ii])); adj[ii]=running
            for r,a in zip(local,adj): r['p_holm']=float(a); rows.append(r)
    return pd.DataFrame(rows)


def build_key_summary(agg,elasticity):
    rows=[]
    for param,values in SWEEPS.items():
        p0=float(getattr(BASELINE,param)); e=elasticity[elasticity.parameter==param]
        by=e.groupby('metric')['abs_elasticity'].max().sort_values(ascending=False); metric=by.index[0]
        a=agg[agg.parameter==param]
        b=float(a[np.isclose(a.value,p0)][f'{metric}_mean'].iloc[0]); lo=float(a[np.isclose(a.value,float(values[0]))][f'{metric}_mean'].iloc[0]); hi=float(a[np.isclose(a.value,float(values[-1]))][f'{metric}_mean'].iloc[0])
        rows.append({'parameter':param,'symbol':PARAM_SYMBOLS[param],'description':PARAM_LABELS[param],'baseline':p0,'tested_min':float(values[0]),'tested_max':float(values[-1]),'most_sensitive_metric':metric,'metric_label':METRIC_LABELS[metric],'max_abs_elasticity':float(by.iloc[0]),'low_endpoint_change_pct':100*(lo-b)/b,'high_endpoint_change_pct':100*(hi-b)/b})
    return pd.DataFrame(rows)

if __name__=='__main__':
    out=Path('/mnt/data/fl_imdr_sensitivity_appendix'); out.mkdir(parents=True,exist_ok=True)
    raw=run_sensitivity(N_SEEDS); per_seed,agg,elasticity=summarize_sensitivity(raw); tests=paired_tests(per_seed); key=build_key_summary(agg,elasticity)
    raw.to_csv(out/'sensitivity_raw_timeseries.csv',index=False); per_seed.to_csv(out/'sensitivity_per_seed.csv',index=False); agg.to_csv(out/'sensitivity_summary.csv',index=False); elasticity.to_csv(out/'sensitivity_elasticity.csv',index=False); tests.to_csv(out/'sensitivity_paired_tests.csv',index=False); key.to_csv(out/'sensitivity_key_summary.csv',index=False)
    print(key.to_string(index=False))
