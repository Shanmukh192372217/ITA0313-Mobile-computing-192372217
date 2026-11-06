"""Q2 - FDMA (25 kHz) vs TDMA (200 kHz x 8 slots) with dedicated per-call channel allocation.
Parameters & assumptions are listed in PARAMS and in the report (sources: 3GPP TS 45.001/45.002 for GSM burst & multiframe)."""
import numpy as np, pandas as pd, heapq
from common import *
# ---------- capacity budget ----------
BW=2000.0
fd=dict(nominal=BW/25, edge_guard=2, control=2)            # 1 guard channel each band edge, 2 signalling channels
fd["usable"]=fd["nominal"]-fd["edge_guard"]-fd["control"]
fd["payload_eff"]=0.99                                       # 1% preamble/sync per continuous channel (assumed)
td=dict(carriers=BW/200); td["nominal"]=td["carriers"]*8
td["edge_guard_slots"]=8                                     # 100 kHz guard each edge = 1 carrier (8 slots)
td["control"]=2                                              # TS0 BCCH/CCCH + 1 SDCCH slot
td["usable"]=td["nominal"]-td["edge_guard_slots"]-td["control"]
burst_eff=114/156.25                                         # 114 data bits per 156.25-bit slot (tail 6, training 26, flags 2, guard 8.25)
mf_eff=24/26                                                 # 26-multiframe: 24 TCH, 1 SACCH, 1 idle
td["payload_eff"]=burst_eff*mf_eff
cap=dict(FDMA=fd,TDMA=td,burst_eff=burst_eff,mf_eff=mf_eff)
for k in ("FDMA","TDMA"):
    cap[k]["eff_channels"]=cap[k]["usable"]*cap[k]["payload_eff"]
cap["FDMA"]["payload_kbps"]=cap["FDMA"]["usable"]*25*cap["FDMA"]["payload_eff"]    # assume 1 bit/s/Hz => 25 kbps/ch
cap["TDMA"]["payload_kbps"]=cap["TDMA"]["usable"]*(270.833/8)*td["payload_eff"]
print(cap)
SETUP={"FDMA":0.4,"TDMA":0.6}      # s of channel held before payload flows (assigned: retune+handshake; TDMA adds timing-advance/frame sync)
NOM=80; TSIM=7200; WARM=600
def gen(ds,rng):
    t=0; calls=[]
    if ds=="D4_sync_short":
        while t<TSIM:
            for _ in range(90): calls.append((t,rng.exponential(3.0),1.0))
            t+=10
        return calls
    if ds=="D1_voice": lam=70/120; mix=[(1.0,120,1.0)]
    if ds=="D2_bursty": lam=70/300; mix=[(1.0,300,0.1)]
    if ds=="D3_mix70_30": lam=70/174; mix=[(0.7,120,1.0),(0.3,300,0.1)]
    while True:
        t+=rng.exponential(1/lam)
        if t>=TSIM: break
        u=rng.random(); c=0
        for p,h,d in mix:
            c+=p
            if u<=c: break
        calls.append((t,rng.exponential(h),d))
    return calls
def simulate(calls,scheme):
    usable=cap[scheme]["usable"]; eta=cap[scheme]["payload_eff"]; s=SETUP[scheme]
    heap=[]; n=blocked=0; payload=held=0.0
    for t,h,d in sorted(calls):
        while heap and heap[0]<=t: heapq.heappop(heap)
        if t<WARM: 
            if len(heap)<usable: heapq.heappush(heap,t+h+s)
            continue
        n+=1
        if len(heap)>=usable: blocked+=1; continue
        heapq.heappush(heap,t+h+s)
        payload+=h*d*eta; held+=h+s
    T=TSIM-WARM
    util=payload/(NOM*T); occ=held/(usable*T)
    wasted=(held-payload)/(NOM*T)           # channel-time held but carrying no payload (idle gaps, setup, overhead)
    unusable=1-usable/NOM                   # permanently lost to guard/control
    return dict(calls=n,blocking=blocked/n,utilisation=util,occupancy=occ,wasted_held=wasted,unusable_static=unusable)
rows=[]
for ds in ("D1_voice","D2_bursty","D3_mix70_30","D4_sync_short"):
    for sch in ("FDMA","TDMA"):
        res=[simulate(gen(ds,np.random.default_rng(100+r)),sch) for r in range(5)]
        m={k:float(np.mean([x[k] for x in res])) for k in res[0]}
        rows.append(dict(dataset=ds,scheme=sch,**m))
df=pd.DataFrame(rows); df.to_csv(f"{DATA}/q2_results.csv",index=False); print(df.round(3))
# Erlang-B check
def erlangB(E,m):
    b=1.0
    for k in range(1,m+1): b=E*b/(k+E*b)
    return b
chk=dict(D1_FDMA=erlangB(70*(120.4/120),76),D1_TDMA=erlangB(70*(120.6/120),70),D2_FDMA=erlangB(70*(300.4/300),76),D2_TDMA=erlangB(70*(300.6/300),70))
print({k:round(v,3) for k,v in chk.items()})
# telemetry design decision numbers
tele=dict(report_bits=200*8,period_s=60)
slot_rate=270.833/8*burst_eff*mf_eff*1000    # payload bit/s of one TDMA slot
tele["slot_payload_bps"]=slot_rate; tele["airtime_per_report_s"]=1600/slot_rate; tele["duty"]=tele["airtime_per_report_s"]/60
tele["devices_dedicated_max"]=cap["FDMA"]["usable"]
tele["agg_bps_per_device"]=1600/60
tele["devices_per_slot_ideal"]=slot_rate/ (1600/60)
tele["devices_per_slot_aloha"]=0.368*slot_rate/(1600/60)
tele["devices_per_slot_poll"]=0.8*slot_rate/(1600/60)
print(tele)
save_json("q2.json",dict(cap=cap,rows=rows,erlangB=chk,tele=tele))
# figure
fig,ax=plt.subplots(1,3,figsize=(9,2.9))
ds=df.dataset.unique(); x=np.arange(len(ds)); w=.38
for i,(col,t) in enumerate((("utilisation","Payload utilisation of nominal 80 ch"),("blocking","Call blocking probability"),("wasted_held","Held-but-unused capacity (share of nominal)"))):
    for j,(sc,cl) in enumerate((("FDMA","#4f81bd"),("TDMA","#e08a2c"))):
        ax[i].bar(x+(j-.5)*w,df[df.scheme==sc][col].values,w,label=sc,color=cl)
    ax[i].set_xticks(x);ax[i].set_xticklabels(["D1\nvoice","D2\nbursty","D3\n70/30","D4\nsync-short"],fontsize=7);ax[i].set_title(t,fontsize=8)
ax[0].legend(fontsize=7);plt.tight_layout();plt.savefig(f"{FIGS}/q2_results.png")
