# Technical Report — AI Study Planner (Mini Adaptive Quiz Coach)
AI Fundamentals · Final Project · Team: [names]

## 1. Problem statement and objectives
After a short quiz, a student wants to know what they actually know and what to study next. Raw scores are misleading: a correct answer can be
a lucky guess, a wrong one a slip, and the student's level changes during the quiz. **Goal:** turn five quiz answers into one explainable
recommendation. **Objectives:** (1) estimate the probability the student knows the topic; (2) infer the hidden level (Beginner/Ready) over time;
(3) choose between Review and Next topic; (4) as an extension, pick the exercise with about 70% success chance.

## 2. System overview
Input: five answers, each Wrong, Partial or Correct. Output: P(knows topic), most likely level sequence, recommended action, recommended exercise.
Setup: 5 questions, 2 hidden states, 3 observations, 2 actions. Each algorithm uses the output of the previous one, so removing any breaks the chain.

```
answers -> Bayes -> P(K) ----------------------------+
   +-> HMM (Viterbi, forward) -> belief -> MDP -> action -> Hill climbing -> exercise
```

## 3. Algorithms: what, why, how
### 3.1 Bayesian inference (Probability & Bayesian Networks)
Rule (P(e) is the total probability of the evidence): P(K|e) = P(e|K)·P(K) / [P(e|K)·P(K) + P(e|¬K)·P(¬K)]. Assumptions: P(K)=0.40, P(correct|K)=0.85, P(correct|¬K)=0.25.
Example, one wrong answer: 0.15·0.40 = 0.060; 0.75·0.60 = 0.450; P(K|wrong) = 0.060/0.510 = 0.118. The posterior becomes the next prior, so
five answers give a running belief. *Why:* answers are noisy evidence. *Limit:* Bayes treats answers as exchangeable; order is ignored.
Implementation: `bayes_update(prior, correct)`.

### 3.2 Hidden Markov Model with Viterbi (Lecture 7)
Hidden states Beginner and Ready, observations Wrong/Partial/Correct. Start (0.7, 0.3); transitions B→B 0.7, B→R 0.3, R→B 0.1, R→R 0.9;
emissions B (0.5, 0.3, 0.2), R (0.1, 0.3, 0.6). **Viterbi** keeps, per state, the best path score δt(h) = max over h′ of [ δt−1(h′)·A(h′→h) ] · B(h, ot) and a back pointer,
then traces back. For three answers W,P,C the best path is Beginner→Ready→Ready (probability 0.017). For Dana (W,P,W,C,C) it is B→B→B→R→R.
The **forward algorithm** sums instead of maximizing and gives the current belief P(Ready): 0.876 for Dana. *Why:* the level is hidden and changes in time; order matters:
W,W,W,C,C gives P(Ready)=0.86, C,C,W,W,W gives 0.13. Implementation: `viterbi`, `forward_belief`.

### 3.3 Markov Decision Process (MDP & RL)
Actions Review and NextTopic. Transitions: B/Review {B .5, R .5}, B/Next {B .9, R .1}, R/Review {B .1, R .9}, R/Next {B .2, R .8}.
Rewards: B: Review +4, Next −3; R: Review +1, Next +6. γ=0.9, V(B)=0, V(R)=10. One Bellman backup:
Q(s,a) = Σs′ T(s′|s,a)·[R(s,a)+γV(s′)], giving Q(B,Review)=8.5, Q(B,Next)=−2.1, Q(R,Review)=9.1, Q(R,Next)=13.2.
The state is uncertain, so expected Q is averaged over the HMM belief. Dana (P(Ready)=0.876): Review 9.03, Next 11.30, so NextTopic.
*Why:* choose by long-term reward, not a fixed threshold. *Simplification:* one backup with a fixed V, not full value iteration. Implementation: `q_value`, `mdp_decide`.

### 3.4 Hill climbing, extension (Local Search)
Exercises are sorted from easy to hard; neighbours are one step easier/harder. Success by total probability P(S)=P(S|K)P(K)+P(S|¬K)P(¬K);
objective 1−|P(S)−0.70|. Aibek (P(K)=0.947) climbs N1→N5 (success 0.929→0.686); Dana stays at N1 (0.573). The score is unimodal along the sorted list, so hill climbing reaches the global best
(tested for 101 values of P(K)). Implementation: `hill_climbing`.

## 4. Implementation
Python 3, standard library only (~13 KB, `study_planner.py`). Console demo with printed steps (input, key calculation, output, one-line reason), `--ask` mode for custom answers,
and a browser version (`docs/website_en.html`) that ports the same algorithms to JavaScript and was checked against the Python output.

## 5. Testing and results
| Check | Result |
|---|---|
| 22 unit tests (`python -m unittest discover -s tests -v`) | all pass |
| Exact values: Bayes 0.34/0.49, 0.06/0.51; Q = 8.5, −2.1, 9.1, 13.2 | pass |
| Viterbi extremes (all C → all Ready; all W → all Beginner), 3-answer case | pass |
| Belief sums to 1; MDP switches action once as P(Ready) grows | pass |
| Hill climbing = global best, 101 values of P(K), both banks | pass |
| All 243 answer combinations run without error | pass |

| Student | P(K) | Levels | Action | Exercise |
|---|---|---|---|---|
| Aida W,W,P,W,P | 0.0002 | B→B→B→B→B | Review | R1 |
| Dana W,P,W,C,C | 0.058 | B→B→B→R→R | NextTopic | N1 |
| Aibek C,C,P,C,C | 0.947 | R→R→R→R→R | NextTopic | N5 |

Evaluation, 2000 synthetic students (seed 42), correct action = Review if the true current level is Beginner: system 75.9%, always NextTopic 70.2%, "≥3 Correct" rule 57.6%.
**Limitation:** the students are generated from the same HMM, so this validates the decision logic, not real-world accuracy.

## 6. Limitations
All probabilities and rewards are labeled assumptions (no dataset). Bayes treats Partial as not correct. The MDP uses one Bellman backup. Evaluation is synthetic.

## 7. Future development
Learn parameters from real quizzes (Baum–Welch); Bayesian network over many topics; value iteration or Q-learning for multi-week plans; simulated annealing for bigger exercise banks;
Telegram/web front end; use as a base for a larger intelligent tutoring system.

## 8. Mapping to the assessment criteria
| Criterion (points) | Where |
|---|---|
| Three AI topics (30) | §3: Bayes, HMM/Viterbi, MDP (+ hill climbing) |
| Functionality and stability (20) | demo, `--ask`, all 243 inputs tested |
| Testing and results (15) | §5, `tests/`, `test_results.txt`, evaluation |
| UI/UX (10) | step-by-step console output, interactive website with demo, lab and games |
| Code quality, GitHub, docs (10) | README, comments, tests, this report |
| Originality and practical value (5) | adaptive tutoring; integrated pipeline; hill-climbing extension |
| Defense (10) | `docs/DEFENSE_PLAN.md` |
