# Defense plan (about 10 minutes)
| Part | Time | Who / what |
|---|---|---|
| Problem statement and objectives | 2 min | slides 2–3 |
| Live demo | 3–5 min | `python study_planner.py`, `--ask` (w w w c c vs c c w w w), website |
| Three AI topics | 2 min | slide 4: Bayes, HMM/Viterbi, MDP (+ hill climbing) |
| Questions | up to 3 min | every member answers individually |

## Questions to prepare (each member should answer all)
1. Why do we need an HMM if Bayes already gives P(K)? — Bayes ignores order; the HMM models a level that changes over time.
2. Viterbi vs forward? — Viterbi: single best state path (max). Forward: current belief (sum).
3. Why average Q over the belief? — the true state is hidden; expected value under uncertainty.
4. Is the MDP fully solved? — no, one Bellman backup with a fixed V; full value iteration is future work.
5. Where do the numbers come from? — labeled assumptions of a toy model; they could be learned from data.
6. Can hill climbing get stuck? — generally yes (local maxima); here the score is unimodal along the sorted list, and the test checks it against the global best.
7. Why is Dana "Ready" with P(K)=0.058? — Bayes counts the early wrong answers; the HMM sees recent improvement.
8. How do you know it works? — 22 tests, hand-computed values, synthetic evaluation (75.9% vs 70.2% and 57.6%), with its limitation.
9. Which part was AI-assisted? — see the AI usage declaration; we can explain every line.
