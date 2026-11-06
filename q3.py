"""Q3 - DS-CDMA uplink (BPSK data, +/-1 spreading, matched-filter/correlator receiver).
PREDICTIONS (written before running):
 (i)   Walsh, synchronous, equal power: zero MAI -> BER = single-user BER Q(sqrt(2Eb/N0)) ~ 4e-6 at 10 dB -> 0 errors in 1e5 bits, flat vs K.
 (ii)  Random codes: MAI ~ (K-1)/N so BER rises with load K/N, nearly independent of N; Gaussian approx BER=Q(1/sqrt((K-1)/N+N0/2Eb)):
       ~0.06 at K/N=0.5, ~0.1 at K/N=1.
 (iii) Walsh, synchronous, one user 10x power: orthogonality is exact so *still* ~0 errors (near-far needs non-ideal alignment to bite).
 (iv)  Walsh, chip-asynchronous, equal power: orthogonality lost -> BER similar to / slightly worse than random codes.
 (v)   Walsh async + one 10x-power user: strongest MAI (interferer power x10) -> BER >> (iv), error floor even at K=2.
Eb/N0 = 10 dB. Bits per user per run L=10000."""
import numpy as np, pandas as pd
from scipy.linalg import hadamard
from scipy.stats import norm
from common import *
L=10000; EBN0=10.0; BOOST_POW=10.0
def run(N,K,cond,seed,noise=True):
    rng=np.random.default_rng(seed)
    if cond.startswith("walsh"): C=hadamard(N)[:K].astype(float)
    else: C=rng.choice([-1.,1.],size=(K,N))
    amp=np.ones(K)
    boost=cond.endswith("10x")
    if boost and K>1: amp[-1]=np.sqrt(BOOST_POW)
    asyn="async" in cond
    tau=np.zeros(K,int)
    if asyn: tau=rng.integers(1,N,size=K); tau[0]=0
    B=rng.choice([-1.,1.],size=(K,L))
    r=np.zeros(L*N)
    for k in range(K):
        x=(B[k][:,None]*C[k][None,:]*amp[k]).ravel()
        r+=np.roll(x,tau[k])
    if noise: r=r+rng.normal(0,np.sqrt(N/(2*10**(EBN0/10))),L*N)
    measured=[j for j in range(K) if not (boost and K>1 and j==K-1)]
    if asyn: measured=measured[:6]
    errs=0;tot=0
    for j in measured:
        z=(np.roll(r,-tau[j]).reshape(L,N)@C[j])/N
        errs+=int(np.sum(np.sign(z)!=B[j])); tot+=L
    return errs,tot
conds={"C1_walsh_sync":"walsh_sync","C2_random_sync":"random_sync","C3_walsh_sync_10x":"walsh_sync_10x",
       "C4_walsh_async":"walsh_async","C5_walsh_async_10x":"walsh_async_10x","C6_random_sync_10x":"random_sync_10x"}
rows=[]
for N in (8,16,64):
    for frac in (1/8,1/4,1/2,3/4,1):
        K=max(2,int(round(frac*N)))
        for cn,c in conds.items():
            e,t=run(N,K,c,seed=1000+N*100+K)
            th=norm.sf(1/np.sqrt((K-1)/N+1/(2*10**(EBN0/10))))
            rows.append(dict(N=N,K=K,load=K/N,cond=cn,bits=t,errors=e,BER=e/t,gauss_theory=th,PG_dB=10*np.log10(N)))
df=pd.DataFrame(rows); df.to_csv(f"{DATA}/q3_ber.csv",index=False)
piv=df.pivot_table(index=["N","K"],columns="cond",values="BER"); print(piv.to_string(float_format=lambda v:f"{v:.4f}"))
# noiseless check of orthogonality + correlations
chk=[]
for N in (8,16,64):
    H=hadamard(N).astype(float)
    G=H@H.T/N; off=np.abs(G-np.eye(N)).max()
    rng=np.random.default_rng(5); R=rng.choice([-1.,1.],size=(N,N)); GR=R@R.T/N; offr=np.abs(GR-np.eye(N))
    # async Walsh periodic cross-correlation worst case over shifts
    worst=0
    for a in range(min(N,16)):
        for b in range(a+1,min(N,16)):
            for s in range(1,N): worst=max(worst,abs(np.dot(H[a],np.roll(H[b],s))/N))
    e,t=run(N,N,"walsh_sync",7,noise=False); e2,t2=run(N,N,"walsh_async_10x",7,noise=False)
    chk.append(dict(N=N,PG=N,PG_dB=10*np.log10(N),walsh_max_offdiag=float(off),random_max_abs_corr=float(offr.max()),random_rms_corr=float(np.sqrt((offr**2).sum()/(N*N-N))),theory_rms=1/np.sqrt(N),walsh_async_worst_corr=float(worst),noiseless_walsh_sync_errors=e,noiseless_walsh_async10x_errors=e2,bits=t))
print(pd.DataFrame(chk).round(4).to_string())
# capacity estimate: max K meeting BER target for random-code/async (Gaussian approx validated by sim) 
TARGET=1e-2
def kmax(N,ebn0_db=10.0):
    n0=1/(2*10**(ebn0_db/10)); ks=[k for k in range(1,4*N) if norm.sf(1/np.sqrt((k-1)/N+n0))<=TARGET]; return max(ks)
cap={N:kmax(N) for N in (8,16,64,128)}; print("Kmax @BER<=1e-2:",cap)
# simulation-based Kmax (random sync codes)
simk={}
for N in (8,16,64):
    best=0
    for K in range(2,2*N):
        e,t=run(N,K,"random_sync",seed=9000+K); 
        if e/t<=TARGET: best=K
        else: break
    simk[N]=best
print("sim Kmax",simk)
save_json("q3.json",dict(chk=chk,kmax_theory=cap,kmax_sim=simk,target=TARGET))
# figure
fig,ax=plt.subplots(1,3,figsize=(9.5,3),sharey=True)
mk={"C1_walsh_sync":("o","#2e8b57"),"C2_random_sync":("s","#4f81bd"),"C3_walsh_sync_10x":("^","#8064a2"),"C4_walsh_async":("D","#e08a2c"),"C5_walsh_async_10x":("v","#c0504d"),"C6_random_sync_10x":("x","#7f7f7f")}
for i,N in enumerate((8,16,64)):
    d=df[df.N==N]
    for cn,(m,c) in mk.items():
        s=d[d.cond==cn]; ax[i].semilogy(s.load,np.maximum(s.BER,3e-5),marker=m,color=c,label=cn,lw=1)
    ax[i].set_title(f"N = {N} (PG = {10*np.log10(N):.1f} dB)",fontsize=9);ax[i].set_xlabel("load K/N");ax[i].set_ylim(2e-5,.6)
ax[0].set_ylabel("BER (floor 3e-5 = zero errors)");ax[2].legend(fontsize=6)
plt.tight_layout();plt.savefig(f"{FIGS}/q3_ber.png")
