# ITA03-MP Mobile Computing - Assignment 2 (CO2): simulations, data and report

All experiments are in plain Python 3 (numpy, scipy, pandas, matplotlib). Every random generator is seeded; re-running reproduces `data/*.csv` exactly.

| File | Question | Output |
|---|---|---|
| `q1.py` | Q1 store-and-forward vs send-immediately (Markov link, 1000 trials/availability, seeds 0..999) | `data/q1_trials.csv`, `q1_summary.csv`, `figs/q1_sim.png` |
| `q2.py` | Q2 FDMA vs TDMA capacity budget + dedicated-channel call simulation (5 replications, seeds 100..104) | `data/q2_results.csv`, `q2.json` |
| `q3.py` | Q3 DS-CDMA (Walsh / random, sync / async, 10x power), N = 8, 16, 64 (seeds 1000+N*100+K) | `data/q3_ber.csv`, `q3.json` |
| `q4.py` | Q4 OFDMA orthogonality, CP test, 5 allocations x 1000 realisations (seed 42; user-count study seed 7) | `data/q4_allocation.csv`, `q4_users.csv` |
| `q5.py` | Q5 ULA patterns, SIR vs separation, angular-spread Monte Carlo (seed 2024) | `data/q5_*.csv` |
| `q6.py` | Q6 weighted decision matrix, +/-20 % sensitivity (seed 11) | `data/q6_scores.csv` |
| `q7.py` | Q7 dedicated vs on-demand burst allocation (seed 1) | `data/q7_results.csv`, `q7_duty_capacity.csv` |
| `diagrams.py` | block diagrams / flowcharts | `figs/*.png` |
| `build_report.py` | assembles the PDF | `Assignment2_CO2_Report.pdf` |

Run everything: `bash run_all.sh` (about 3 minutes on one core).

Assumed (not standard) values are marked "assumed" in the report: FDMA 1 bit/s/Hz, call set-up occupancy 0.4 s / 0.6 s, RACH = 1 slot per GSM frame, SNR spread of the 8 OFDMA users, R_min = 2 Mbps, decision-matrix weights and ratings.
