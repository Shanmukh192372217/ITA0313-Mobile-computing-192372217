"""Q7 - Diagnose/redesign/validate: dedicated FDMA/TDMA channels idle under bursty (ON/OFF) traffic.
Traffic: each device ON ~ Exp(2 s), OFF ~ Exp(18 s)  (duty 10%).  Channel pool = 76 (usable FDMA channels from Q2).
ORIGINAL : dedicated channel per device for its whole session (first 76 devices admitted, rest blocked).
REDESIGN : burst-level on-demand channel (PRMA-like): request on a slotted-ALOHA RACH (1 slot / 4.615 ms GSM frame, ideal 1/n backoff),
           grant if a channel is free, release at end of burst after hold timer T_hold (T_hold=0 => release immediately).
PREDICTION (before running): (1) devices served at <=1% burst blocking rises from 76 to ~8x (Erlang-B: ~64 E / 0.1 duty ~ 640);
 (2) held-channel utilisation rises from ~10% to >90% at T_hold=0; (3) signalling grows from ~0 to ~3-4 msgs/burst (>=1 request + grant + release);
 (4) T_hold>0 does NOT help here (OFF gap 18 s >> hold) -> raises idle-hold, little signalling saving;
 (5) the RACH (~0.37 x 217 slots/s ~ 80 req/s) becomes the new bottleneck around N ~ 1,500 devices."""
import numpy as np, pandas as pd, time
from common import *
DT=0.004615; CH=76; ON_M=2.0; OFF_M=18.0; TSIM=300; WARM=30
def sim(N,mode,thold,seed):
    rng=np.random.default_rng(seed); nfr=int(TSIM/DT); wf=int(WARM/DT)
    phase=np.zeros(N,np.int8); tmr=np.zeros(N); rs=np.zeros(N,int)
    if mode=="dedicated": 
        phase[:CH]=3; tmr[:CH]=np.inf
    used=int((phase>=2).sum()); wake_p=DT/OFF_M
    started=blocked=served=msgs=0; on_f=held_f=0; delays=[]
    for f in range(nfr):
        m=f>=wf
        # wake-ups from OFF(0) or HOLD(3)
        w=((phase==0)|(phase==3))&(rng.random(N)<wake_p)
        if w.any():
            from_hold=w&(phase==3); from_off=w&(phase==0)
            tmr[from_hold]=rng.exponential(ON_M,from_hold.sum()); phase[from_hold]=2
            if m: started+=int(w.sum()); served+=int(from_hold.sum())
            if mode=="dedicated":
                if m: blocked+=int(from_off.sum())
            else: phase[from_off]=1; rs[from_off]=f
        # timers
        act=(phase==2)|(phase==3); tmr[act]-=DT
        fin=(phase==2)&(tmr<=0)
        if fin.any():
            if mode=="dedicated" : phase[fin]=3; tmr[fin]=np.inf
            elif thold>0: phase[fin]=3; tmr[fin]=thold
            else:
                phase[fin]=0; used-=int(fin.sum()); 
                if m: msgs+=int(fin.sum())
        ex=(phase==3)&(tmr<=0)
        if ex.any(): phase[ex]=0; used-=int(ex.sum()); msgs+= int(ex.sum()) if m else 0
        # RACH
        if mode!="dedicated":
            rq=np.flatnonzero(phase==1)
            if len(rq):
                tx=rng.random(len(rq))<1/len(rq); nt=int(tx.sum())
                if m: msgs+=nt
                if nt==1:
                    d=rq[tx][0]
                    if m: msgs+=1
                    if used<CH:
                        phase[d]=2; tmr[d]=rng.exponential(ON_M); used+=1
                        if m: served+=1; delays.append((f-rs[d])*DT)
                    else:
                        phase[d]=0
                        if m: blocked+=1; started+=0
        if m:
            on_f+=int((phase==2).sum()); held_f+=int(((phase==2)|(phase==3)).sum())
    # bursts that were requested: started includes wake events; blocked counted at decision time
    att=started if mode=="dedicated" else (served+blocked)   # decided bursts
    T=(nfr-wf)*DT
    return dict(N=N,mode=mode,thold=thold,bursts=att,blocking=blocked/max(att,1),held_util=on_f/max(held_f,1),
        chan_busy=held_f*DT/(CH*T),payload_chan=on_f*DT/(CH*T),msgs_per_burst=msgs/max(att,1),mean_delay_ms=1000*float(np.mean(delays)) if delays else 0.0,
        p95_delay_ms=1000*float(np.percentile(delays,95)) if delays else 0.0)
def erlangB(E,m):
    b=1.0
    for k in range(1,m+1): b=E*b/(k+E*b)
    return b
rows=[]; t0=time.time()
for N in (76,200,400,640,800,1000,1500):
    rows.append(sim(N,"dedicated",0,1))
    for th in (0,2.0):
        rows.append(sim(N,"ondemand",th,1))
    print(N,round(time.time()-t0),flush=True)
df=pd.DataFrame(rows); df["erlangB_block"]=[erlangB(r.N*0.1,CH) if r["mode"]=="ondemand" and r.thold==0 else np.nan for _,r in df.iterrows()]
df.to_csv(f"{DATA}/q7_results.csv",index=False); print(df.round(3).to_string())
# capacity at 1% GoS (Erlang-B) vs duty
cap=[]
for duty in (0.02,0.05,0.1,0.3,0.5,0.9):
    E=max(e for e in np.arange(1,200,.25) if erlangB(e,CH)<=0.01); cap.append(dict(duty=duty,erlangB_E=E,devices_ondemand=E/duty,devices_dedicated=CH,gain=E/duty/CH))
cd=pd.DataFrame(cap); cd.to_csv(f"{DATA}/q7_duty_capacity.csv",index=False); print(cd.round(2))
save_json("q7.json",dict(rows=rows,cap=cap))
# figure
fig,ax=plt.subplots(1,3,figsize=(10,3))
for lab,sel,c in (("Dedicated (original)",(df["mode"]=="dedicated"),"#c0504d"),("On-demand, T_hold=0",((df["mode"]=="ondemand")&(df.thold==0)),"#2e8b57"),("On-demand, T_hold=2 s",((df["mode"]=="ondemand")&(df.thold==2)),"#8064a2")):
    d=df[sel]
    ax[0].plot(d.N,d.blocking*100,"o-",label=lab,color=c); ax[1].plot(d.N,d.held_util*100,"o-",color=c); ax[2].plot(d.N,d.msgs_per_burst,"o-",color=c)
ax[0].axhline(1,ls="--",c="k",lw=.8);ax[0].set_ylabel("burst/device blocking (%)");ax[1].set_ylabel("payload share of held channel-time (%)");ax[2].set_ylabel("signalling msgs per burst")
for a in ax: a.set_xlabel("devices N")
ax[0].legend(fontsize=6);plt.tight_layout();plt.savefig(f"{FIGS}/q7_results.png")
