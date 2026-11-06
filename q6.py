"""Q6 - weighted decision matrix + sensitivity (+/-20 %). Ratings 1-5 (5 best) traceable to Q2-Q5/Q7 evidence (see EVID)."""
import numpy as np, pandas as pd
from common import *
T=["FDMA","TDMA","CDMA","OFDMA","SDMA"]
CR=["Efficiency (traffic-specific)","Peak rate","Interference/jam robustness","Many-user / small-packet scalability","Simplicity, cost & power"]
W={"1 Wide-area cellular voice":[30,10,20,15,25],"2 Jam-resistant tactical link":[10,10,50,10,20],
   "3 High-rate stadium downlink":[25,45,10,5,15],"4 5,000 low-rate sensors":[20,5,10,35,30]}
# ratings[deployment] = 5 criteria x 5 techniques (order FDMA,TDMA,CDMA,OFDMA,SDMA)
J=[2,2,5,3,3]; X=[4,3,2,2,1]; R=[2,3,3,5,4]
E={"1":[4,3,3,3,4],"2":[2,3,3,3,2],"3":[2,2,2,5,3],"4":[1,2,2,3,1]}
S={"1":[2,3,4,3,2],"2":[2,3,3,2,1],"3":[1,2,2,4,2],"4":[1,4,3,3,1]}
rat={d:np.array([E[d[0]],R,J,S[d[0]],X]) for d in W}
sc={}; out=[]
for d,w in W.items():
    w=np.array(w,float)/100; s=w@rat[d]; sc[d]=s
    order=np.argsort(-s); best=order[0]; margin=s[best]-s[order[1]]
    rng=np.random.default_rng(11); keep=0; flips=[]
    for ci in range(5):
        for f in (1.2,.8):
            ww=w.copy(); ww[ci]*=f; ww/=ww.sum(); ss=ww@rat[d]
            if ss.argmax()!=best: flips.append(f"{CR[ci][:14]} x{f}: {T[ss.argmax()]}")
    M=5000; mc=0
    for _ in range(M):
        ww=w*rng.uniform(.8,1.2,5); ww/=ww.sum(); mc+= (ww@rat[d]).argmax()==best
    out.append(dict(deployment=d,**{t:round(float(x),2) for t,x in zip(T,s)},winner=T[best],runner_up=T[order[1]],margin=round(float(margin),2),OAT_flips="; ".join(flips) or "none",MC_same_winner_pct=100*mc/M))
df=pd.DataFrame(out); df.to_csv(f"{DATA}/q6_scores.csv",index=False); print(df.to_string())
save_json("q6.json",dict(out=out,W=W,CR=CR,T=T,rat={d:rat[d].tolist() for d in rat}))
