"""Q4 - OFDMA: 8 users, 64 subcarriers, 20 MHz (802.11a/g-style numerology: df=312.5 kHz, Tu=3.2 us, CP=0.8 us, Ts=4 us).
PREDICTIONS (before running): max-sum best-subcarrier > interleaved > contiguous in sum throughput (multiuser diversity; contiguous blocks suffer
when a block sits in a deep fade); fairness: max-sum worst (weak-SNR users get ~0), interleaved ~ contiguous, both fair in access but not in rate
(SNR spread). Max-min / min-rate-aware allocation gives best Jain with small sum-rate loss."""
import numpy as np, pandas as pd
from common import *
U,K,R=8,64,1000; TS=4e-6; DF=312.5e3; TU=3.2e-6; CP=0.8e-6
SNRDB=np.array([24,21,18,15,12,9,6,3.])      # mean SNR per user (path-loss spread)
RMIN_MBPS=2.0                               # minimum rate per student (720p lecture stream)
# ---- orthogonality & CP verification ----
n=np.arange(64); t=n/64
subs=[3,4,5]; S=np.array([np.exp(2j*np.pi*k*t) for k in subs])
G=S.conj()@S.T/64
leak=lambda eps:abs(np.vdot(np.exp(2j*np.pi*3*t),np.exp(2j*np.pi*(4+eps)*t))/64)
orth=dict(gram_abs=np.abs(G).round(12).tolist(),max_offdiag=float(np.abs(G-np.eye(3)).max()),leak_df_offset_0p1=float(leak(0.1)),leak_df_offset_0p3=float(leak(0.3)))
print(orth)
def ofdm_evm(cp,rng,ntaps=6,nsym=2000):
    ev=[]
    for _ in range(nsym):
        h=(rng.normal(size=ntaps)+1j*rng.normal(size=ntaps))*np.exp(-np.arange(ntaps)/2)/np.sqrt(2); h/=np.linalg.norm(h)
        X=(rng.choice([-1,1],64)+1j*rng.choice([-1,1],64))/np.sqrt(2)
        x=np.fft.ifft(X)*8; xs=np.concatenate([x[64-cp:],x]) if cp>0 else x
        y=np.convolve(xs,h)[:len(xs)]
        y=y[cp:cp+64]; Y=np.fft.fft(y)/8; H=np.fft.fft(h,64)
        ev.append(np.mean(np.abs(Y/H-X)**2))
    return float(np.mean(ev))
rng=np.random.default_rng(1)
cpres={cp:ofdm_evm(cp,rng) for cp in (0,2,4,5,8,16)}; print("EVM(CP):",cpres)
# ---- channel realisations: frequency-selective Rayleigh via 16-tap exponential PDP (fixed seed) ----
def gains(U_,R_,seed,snr_db):
    rg=np.random.default_rng(seed); L=16
    pdp=np.exp(-np.arange(L)/3.0); pdp/=pdp.sum()
    h=(rg.normal(size=(R_,U_,L))+1j*rg.normal(size=(R_,U_,L)))/np.sqrt(2)*np.sqrt(pdp)
    Hf=np.fft.fft(h,64,axis=-1); g=np.abs(Hf)**2                      # E[g]=1
    return g*(10**(np.asarray(snr_db)/10))[None,:,None]                  # linear SNR per subcarrier
def rate(snr): return np.log2(1+snr)                                      # bits/subcarrier/symbol (Shannon, ideal adaptive MCS)
def alloc_contig(r): 
    a=np.repeat(np.arange(U),K//U); return np.broadcast_to(a,(r.shape[0],K))
def alloc_inter(r):
    a=np.arange(K)%U; return np.broadcast_to(a,(r.shape[0],K))
def alloc_best(r): return r.argmax(axis=1)
def alloc_maxmin(r):
    Rr,U_,K_=r.shape; out=np.empty((Rr,K_),int)
    for i in range(Rr):
        tot=np.zeros(U_); free=np.ones(K_,bool); ri=r[i]
        for _ in range(K_):
            u=int(tot.argmin()); cand=np.where(free,ri[u],-1); k=int(cand.argmax())
            out[i,k]=u; free[k]=False; tot[u]+=ri[u,k]
    return out
def alloc_minrate(r,rmin_bits):
    """2-phase: (1) neediest user below rmin takes its best free subcarrier until all >= rmin (or no free carriers); (2) rest to best user."""
    Rr,U_,K_=r.shape; out=np.empty((Rr,K_),int)
    for i in range(Rr):
        tot=np.zeros(U_); free=np.ones(K_,bool); ri=r[i]
        while free.any():
            need=np.where(tot<rmin_bits)[0]
            if len(need)==0: break
            u=need[tot[need].argmin()]; cand=np.where(free,ri[u],-1); k=int(cand.argmax())
            out[i,k]=u; free[k]=False; tot[u]+=ri[u,k]
        for k in np.where(free)[0]:
            u=int(ri[:,k].argmax()); out[i,k]=u
    return out
def per_user(r,a):
    Rr,U_,K_=r.shape; th=np.zeros((Rr,U_))
    for u in range(U_): th[:,u]=(r[:,u,:]*(a==u)).sum(axis=1)
    return th/TS/1e6                                                       # Mbps
def jain(x): return (x.sum(-1)**2)/(x.shape[-1]*(x**2).sum(-1)+1e-300)
def evaluate(r,a,U_):
    th=per_user(r,a); mean_u=th.mean(0)
    return dict(sum_Mbps=float(th.sum(1).mean()),jain_longterm=float(jain(mean_u)),jain_inst=float(jain(th).mean()),
        min_user_Mbps=float(mean_u.min()),all_ge_rmin=float(np.mean((th>=RMIN_MBPS).all(1))),per_user=mean_u.round(2).tolist(),
        weakest_outage=float(np.mean(th[:,-1]<RMIN_MBPS)),users_below_rmin=float(np.mean((th<RMIN_MBPS).sum(1))))
g=gains(U,R,42,SNRDB); r=rate(g)
rmin_bits=RMIN_MBPS*1e6*TS
schemes={"Contiguous blocks":alloc_contig(r),"Interleaved RR":alloc_inter(r),"Best-subcarrier (max-sum)":alloc_best(r),
         "Max-min greedy":alloc_maxmin(r),"Min-rate + max-sum (proposed)":alloc_minrate(r,rmin_bits)}
res={k:evaluate(r,a,U) for k,a in schemes.items()}
df=pd.DataFrame([{**{"scheme":k},**{kk:v for kk,v in d.items() if kk!="per_user"},**{f"U{i+1}":x for i,x in enumerate(d["per_user"])}} for k,d in res.items()])
df.to_csv(f"{DATA}/q4_allocation.csv",index=False); print(df.round(3).to_string())
# ---- number of users effect (equal mean SNR 12 dB) ----
nu=[]
for Un in (1,2,4,8,16,32):
    gg=gains(Un,R,7,[12]*Un); rr=rate(gg)
    if Un==1: s_b=s_i=per_user(rr,np.zeros((R,K),int)).sum(1).mean()
    else:
        s_i=per_user(rr,alloc_inter_n(rr,Un) if False else np.broadcast_to(np.arange(K)%Un,(R,K))).sum(1).mean()
        s_b=per_user(rr,rr.argmax(axis=1)).sum(1).mean()
    nu.append(dict(users=Un,interleaved_Mbps=float(s_i),best_Mbps=float(s_b),gain_pct=float((s_b/s_i-1)*100)))
print(pd.DataFrame(nu).round(2)); pd.DataFrame(nu).to_csv(f"{DATA}/q4_users.csv",index=False)
save_json("q4.json",dict(orth=orth,evm_cp=cpres,res=res,nusers=nu,rmin=RMIN_MBPS))
# figures
fig,ax=plt.subplots(1,3,figsize=(10,3))
names=list(res); cols=["#7f7f7f","#4f81bd","#c0504d","#8064a2","#2e8b57"]
w=.16
for i,nm in enumerate(names): ax[0].bar(np.arange(U)+(i-2)*w,res[nm]["per_user"],w,color=cols[i],label=nm)
ax[0].axhline(RMIN_MBPS,color="k",ls="--",lw=1);ax[0].set_xticks(range(U));ax[0].set_xticklabels([f"U{i+1}\n{int(s)}dB" for i,s in enumerate(SNRDB)],fontsize=6);ax[0].set_ylabel("mean per-user throughput (Mbps)");ax[0].legend(fontsize=5)
ax[1].bar(range(5),[res[n]["sum_Mbps"] for n in names],color=cols);ax[1].set_xticks(range(5));ax[1].set_xticklabels(["Contig","Interl.","Best","MaxMin","MinRate+"],fontsize=7);ax[1].set_ylabel("sum throughput (Mbps)")
for i,n in enumerate(names): ax[1].text(i,res[n]["sum_Mbps"]+.5,f"J={res[n]['jain_longterm']:.2f}",ha="center",fontsize=7)
d=pd.DataFrame(nu);ax[2].plot(d.users,d.gain_pct,"o-");ax[2].set_xscale("log",base=2);ax[2].set_xlabel("number of users (equal 12 dB mean SNR)");ax[2].set_ylabel("best-subcarrier gain over interleaved (%)")
plt.tight_layout();plt.savefig(f"{FIGS}/q4_alloc.png")
# CP / orthogonality figure
fig,ax=plt.subplots(1,2,figsize=(7.5,2.8))
tt=np.linspace(0,1,400)
for k in (3,4,5): ax[0].plot(tt,np.cos(2*np.pi*k*tt),label=f"k={k}")
ax[0].set_title("Three subcarriers over one symbol Tu",fontsize=8);ax[0].legend(fontsize=6);ax[0].set_xlabel("t / Tu")
ax[1].semilogy(list(cpres),list(cpres.values()),"o-");ax[1].set_xlabel("CP length (samples; channel has 6 taps)");ax[1].set_ylabel("EVM (MSE after 1-tap equaliser)");ax[1].set_title("CP removes ISI/ICI once CP >= taps-1",fontsize=8)
plt.tight_layout();plt.savefig(f"{FIGS}/q4_orth.png")
