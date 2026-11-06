"""Q1 - Case (e): offline meter-reading tablet. Send-immediately vs store-and-forward.
Link = two-state Markov chain (up/down) so outages are bursty (realistic dead zones).
1 step = 1 minute; shift = 480 min; one reading every 5 min (96 updates);
mean outage 30 min; queue capacity 20 readings; on link-up the queue drains within the minute.
1000 independent trials per availability, seeds 0..999."""
import numpy as np, pandas as pd
from common import *
H=480; EVERY=5; MEAN_DOWN=30; QCAP=20; TRIALS=1000
def link_trace(a,rng):
    p_du=1/MEAN_DOWN; p_ud=p_du*(1-a)/a          # stationary P(up)=a
    s=rng.random()<a; out=np.empty(H,bool)
    for t in range(H):
        out[t]=s
        s = (rng.random()>=p_ud) if s else (rng.random()<p_du)
    return out
def run(a,seed):
    rng=np.random.default_rng(seed); up=link_trace(a,rng)
    gen=list(range(0,H,EVERY)); n=len(gen)
    # Strategy A: send immediately, fail if disconnected
    A_del=int(sum(up[t] for t in gen)); A_lost=n-A_del
    # Strategy B: store & forward, bounded FIFO queue (drop-oldest on overflow)
    q=[]; delays=[]; lost=0; imm=0
    for t in range(H):
        if t%EVERY==0:
            if up[t]: imm+=1; delays.append(0)
            else:
                if len(q)>=QCAP: q.pop(0); lost+=1
                q.append(t)
        if up[t] and q:
            delays += [t-g for g in q]; q=[]
    B_del_delayed=len(delays)-imm
    return dict(avail=a,seed=seed,generated=n,uptime=up.mean(),
        A_delivered=A_del,A_lost=A_lost,
        B_immediate=imm,B_delayed=B_del_delayed,B_lost_overflow=lost,B_pending_end=len(q),
        B_mean_delay_min=float(np.mean(delays)) if delays else np.nan,
        B_max_delay_min=max(delays) if delays else np.nan)
rows=[run(a,s) for a in (0.9,0.6,0.3) for s in range(TRIALS)]
df=pd.DataFrame(rows); df.to_csv(f"{DATA}/q1_trials.csv",index=False)
g=df.groupby("avail").mean(numeric_only=True).drop(columns="seed"); g.to_csv(f"{DATA}/q1_summary.csv")
print(g.round(2).T)
save_json("q1.json",g.round(3).reset_index().to_dict("records"))
fig,ax=plt.subplots(1,2,figsize=(8,3))
av=[90,60,30]; w=.35; x=np.arange(3)
ax[0].bar(x-w/2,g.A_delivered/g.generated*100,w,label="Send immediately: delivered",color="#c0504d")
ax[0].bar(x+w/2,(g.B_immediate+g.B_delayed)/g.generated*100,w,label="Store-and-forward: delivered",color="#4f81bd")
ax[0].set_xticks(x);ax[0].set_xticklabels([f"{a}%" for a in av]);ax[0].set_xlabel("link availability");ax[0].set_ylabel("% of 96 updates delivered");ax[0].legend(fontsize=7);ax[0].set_ylim(0,110)
ax[1].bar(x,g.B_mean_delay_min,color="#4f81bd");ax[1].set_xticks(x);ax[1].set_xticklabels([f"{a}%" for a in av]);ax[1].set_xlabel("link availability");ax[1].set_ylabel("mean delivery delay (min)");ax[1].set_title("Store-and-forward delay cost",fontsize=9)
plt.tight_layout();plt.savefig(f"{FIGS}/q1_sim.png");
