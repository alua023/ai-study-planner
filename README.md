# AI Study Planner — Mini Adaptive Quiz Coach

Final Project, **AI Fundamentals** (Topic 01: *Mini Adaptive Quiz Coach*).
A student answers a short Python quiz. The system estimates what the student knows, infers their hidden level over time,
decides between **Review** and **Next topic**, and recommends one concrete exercise.

## Three required AI topics + one extension (all taught in the course)
| # | Algorithm | Course topic | Role in the project | Function |
|---|---|---|---|---|
| 1 | Bayesian inference | Probability & Bayesian Networks | P(student knows topic), updated after every answer | `bayes_update` |
| 2 | HMM: Viterbi + forward | Lecture 7, Hidden Markov Models | hidden level Beginner/Ready from the answer sequence | `viterbi`, `forward_belief` |
| 3 | MDP: Bellman backup | MDP & Reinforcement Learning | Review vs Next topic from a 2-state reward table | `q_value`, `mdp_decide` |
| + | Hill climbing (extension) | Local Search | picks the exercise whose success chance is closest to 70% | `hill_climbing` |

```
answers --> Bayes ----------------------------> P(K) -----+
   |                                                       v
   +--> HMM (Viterbi, forward) --> belief --> MDP --> action --> Hill climbing --> exercise
```
Setup matches the topic: 5 questions, 2 hidden states (Beginner, Ready), 3 observations (Wrong, Partial, Correct), 2 actions (Review, NextTopic).
Output: knowledge probability, most likely state sequence, recommended action (+ exercise).

## Run (Python 3.8+, no dependencies)
```
python study_planner.py          # demo of 3 students + built-in checks + evaluation
python study_planner.py --ask    # type your own 5 answers: w = Wrong, p = Partial, c = Correct
python -m unittest discover -s tests -v    # 22 unit tests
```
Web version (no install, runs in the browser): `docs/website_en.html` (English), `docs/website_ru.html` (Russian).

## Sample output
| Student | Answers | P(K) | Viterbi levels | Action | Exercise |
|---|---|---|---|---|---|
| Aida | W,W,P,W,P | 0.0002 (prints 0.000) | B→B→B→B→B | Review | R1 print 1..10 with for |
| Dana | W,P,W,C,C | 0.058 | B→B→B→R→R | NextTopic | N1 define a simple function |
| Aibek | C,C,P,C,C | 0.947 | R→R→R→R→R | NextTopic | N5 recursion: Fibonacci |

Full console output: `sample_output.txt`. Unit-test log: `test_results.txt`.

## Testing and results
- 22 unit tests (`tests/test_study_planner.py`): exact Bayes posteriors (0.34/0.49 and 0.06/0.51), hand-computed Q-values
  (8.5, −2.1, 9.1, 13.2), Viterbi extremes and a 3-answer case, belief sums to 1, MDP switches action exactly once as P(Ready) grows,
  hill climbing equals the global best for 101 values of P(K), all 3^5 = 243 answer combinations run end to end.
- Evaluation on 2000 synthetic students (seed 42): system **75.9%**, always-NextTopic 70.2%, "≥3 Correct" rule 57.6%.
  The data come from the same HMM, so this checks the decision logic, not real-world accuracy.

## Assumptions (toy model, labeled)
Bayes: P(K)=0.40, P(correct|K)=0.85, P(correct|not K)=0.25 (Partial counts as not correct here).
HMM: start (B 0.7, R 0.3); B→B 0.7, B→R 0.3, R→B 0.1, R→R 0.9; emissions B (W .5, P .3, C .2), R (W .1, P .3, C .6).
MDP: γ=0.9, V(B)=0, V(R)=10; rewards B: Review +4, Next −3; R: Review +1, Next +6.
Exercises: five per action with assumed P(success|knows) / P(success|not); target success 0.70.

## Repository layout
```
study_planner.py        application (console)
tests/                  unit tests
docs/                   technical report, AI usage declaration, team contribution, demo script, defense plan, web versions
sample_output.txt       console output of the demo
test_results.txt        unit-test output
```

## Future development
Learn parameters from real quiz data (Baum–Welch), Bayesian network over many topics, value iteration / Q-learning for a
multi-week plan, simulated annealing for larger exercise banks, Telegram or web front end connected to the Python code.

*AI assistance was used; see `docs/AI_USAGE_DECLARATION.md`.*
