"""
AI Study Planner  (Final Project, AI Fundamentals)  -  version 2

Real-world problem: a student takes a short Python quiz. The system must
  (a) estimate what the student knows and the chance of success,
  (b) decide WHAT TYPE of study comes next (Review or New Topic),
  (c) pick the SPECIFIC exercise that fits the student best.

Four course algorithms, one pipeline (each output feeds the next step):
  1. Bayesian inference      -> P(student knows topic)               [Probability & Bayes]
  2. HMM (forward + Viterbi) -> hidden level Beginner/Ready over time [Lecture 7: HMM]
  3. MDP (Bellman backup)    -> action: Review or NextTopic           [MDP & RL]
  4. Hill climbing           -> best exercise among 5 candidates      [Local Search]

   answers --> Bayes --------------------------> P(K) ------+
      |                                                     v
      +-----> HMM --> belief --> MDP --> action --> Hill climbing --> exercise

All probabilities / rewards are labelled assumptions (toy model, no dataset).

Usage:
    python study_planner.py          demo (3 students) + tests + evaluation
    python study_planner.py --ask    you type the 5 answers yourself
"""
import math
import random
import sys

# ============================================================ 1. BAYES
PRIOR_KNOW = 0.40            # P(K): student knows the topic, before any answer
P_CORRECT_GIVEN_KNOW = 0.85  # P(correct | K)      (15% slip)
P_CORRECT_GIVEN_NOT = 0.25   # P(correct | not K)  (25% lucky guess)


def bayes_update(prior, answer_correct):
    """P(K|e) = P(e|K)P(K) / [P(e|K)P(K) + P(e|~K)P(~K)]"""
    like_k = P_CORRECT_GIVEN_KNOW if answer_correct else 1 - P_CORRECT_GIVEN_KNOW
    like_n = P_CORRECT_GIVEN_NOT if answer_correct else 1 - P_CORRECT_GIVEN_NOT
    num = like_k * prior
    return num / (num + like_n * (1 - prior)), like_k, like_n


# ============================================================== 2. HMM
H = ("Beginner", "Ready")                 # hidden states
O = ("Wrong", "Partial", "Correct")       # observations
PI = {"Beginner": 0.7, "Ready": 0.3}
A = {"Beginner": {"Beginner": 0.7, "Ready": 0.3},      # P(h' | h)
     "Ready":    {"Beginner": 0.1, "Ready": 0.9}}
B = {"Beginner": {"Wrong": 0.5, "Partial": 0.3, "Correct": 0.2},   # P(o | h)
     "Ready":    {"Wrong": 0.1, "Partial": 0.3, "Correct": 0.6}}


def viterbi(obs, verbose=True):
    """delta_t(h) = max_h' delta_{t-1}(h') * A[h'][h] * B[h][o_t]"""
    delta = [{h: PI[h] * B[h][obs[0]] for h in H}]
    back = [{}]
    for t in range(1, len(obs)):
        d, bp = {}, {}
        for h in H:
            best_prev = max(H, key=lambda p: delta[t - 1][p] * A[p][h])
            bp[h] = best_prev
            d[h] = delta[t - 1][best_prev] * A[best_prev][h] * B[h][obs[t]]
        delta.append(d)
        back.append(bp)
    if verbose:
        for t, o in enumerate(obs, 1):
            print(f"    t={t} obs={o:7s} " +
                  "  ".join(f"delta({h})={delta[t-1][h]:.5f}" for h in H))
    last = max(H, key=lambda h: delta[-1][h])
    path = [last]
    for t in range(len(obs) - 1, 0, -1):
        path.append(back[t][path[-1]])
    return path[::-1], delta[-1][last]


def forward_belief(obs):
    """Filtering: P(h_T | o_1..o_T)."""
    alpha = {h: PI[h] * B[h][obs[0]] for h in H}
    for o in obs[1:]:
        alpha = {h: sum(alpha[p] * A[p][h] for p in H) * B[h][o] for h in H}
    z = sum(alpha.values())
    return {h: alpha[h] / z for h in H}


# ============================================================== 3. MDP
GAMMA = 0.9
ACTIONS = ("Review", "NextTopic")
T = {  # P(s' | s, a)
    "Beginner": {"Review":    {"Beginner": 0.5, "Ready": 0.5},
                 "NextTopic": {"Beginner": 0.9, "Ready": 0.1}},
    "Ready":    {"Review":    {"Beginner": 0.1, "Ready": 0.9},
                 "NextTopic": {"Beginner": 0.2, "Ready": 0.8}},
}
R = {  # R(s, a)
    "Beginner": {"Review": +4, "NextTopic": -3},   # new material is too hard
    "Ready":    {"Review": +1, "NextTopic": +6},   # review is redundant
}
V0 = {"Beginner": 0.0, "Ready": 10.0}


def q_value(s, a):
    """One Bellman backup: Q(s,a) = sum_s' P(s'|s,a) [R(s,a) + gamma V(s')]"""
    return sum(p * (R[s][a] + GAMMA * V0[s2]) for s2, p in T[s][a].items())


def mdp_decide(belief):
    q = {a: sum(belief[s] * q_value(s, a) for s in H) for a in ACTIONS}
    return max(q, key=q.get), q


# ================================================== 4. LOCAL SEARCH
# Exercise banks, sorted from easiest to hardest.
# (name, P(success | knows), P(success | doesn't know))   <- assumptions
EXERCISES = {
    "Review": [
        ("R1 print 1..10 with for",        0.98, 0.75),
        ("R2 sum of a list with for",      0.95, 0.60),
        ("R3 loop + if (even numbers)",    0.92, 0.45),
        ("R4 nested loops (pattern)",      0.88, 0.30),
        ("R5 while loop with break",       0.85, 0.20),
    ],
    "NextTopic": [
        ("N1 define a simple function",    0.95, 0.55),
        ("N2 function with parameters",    0.92, 0.40),
        ("N3 function returning a value",  0.88, 0.28),
        ("N4 recursion: factorial",        0.80, 0.15),
        ("N5 recursion: Fibonacci",        0.72, 0.08),
    ],
}
TARGET_SUCCESS = 0.70   # a good exercise is challenging but doable (~70%)


def exercise_success(ex, p_know):
    """Total probability: P(success) = P(S|K)P(K) + P(S|~K)P(~K)"""
    return ex[1] * p_know + ex[2] * (1 - p_know)


def exercise_score(ex, p_know):
    """Objective (maximize): 1 - |P(success) - 0.70|."""
    return 1 - abs(exercise_success(ex, p_know) - TARGET_SUCCESS)


def hill_climbing(candidates, p_know, start=0, verbose=True):
    """Neighbours of exercise i are i-1 and i+1 (slightly easier / harder).
    Move to a better neighbour until none improves."""
    cur = start
    while True:
        s_cur = exercise_score(candidates[cur], p_know)
        if verbose:
            print(f"    at {candidates[cur][0]:32s} "
                  f"P(success)={exercise_success(candidates[cur], p_know):.3f} score={s_cur:.3f}")
        nbrs = [i for i in (cur - 1, cur + 1) if 0 <= i < len(candidates)]
        best = max(nbrs, key=lambda i: exercise_score(candidates[i], p_know))
        if exercise_score(candidates[best], p_know) <= s_cur:
            return cur
        cur = best


# ========================================================== PIPELINE
def run_student(name, answers, verbose=True):
    print("=" * 70)
    print(f"STUDENT: {name}   answers = {answers}")
    print("=" * 70)

    # 1. Bayes ("Partial" counts as incorrect in this binary test)
    print(f"\n[1] BAYES  input: P(K)={PRIOR_KNOW}, P(correct|K)={P_CORRECT_GIVEN_KNOW}, "
          f"P(correct|~K)={P_CORRECT_GIVEN_NOT}")
    p = PRIOR_KNOW
    for i, a in enumerate(answers, 1):
        correct = (a == "Correct")
        new, lk, ln = bayes_update(p, correct)
        print(f"    answer {i}: {'correct' if correct else 'incorrect':9s} "
              f"P(K) {p:.3f} -> {new:.3f}   ({lk:.2f}*{p:.3f}) / "
              f"({lk:.2f}*{p:.3f} + {ln:.2f}*{1-p:.3f})")
        p = new
    print(f"    output: P(student knows topic) = {p:.3f}")
    print("    Why: answers are noisy evidence (guess / slip), so we update a belief.")

    # 2. HMM
    print(f"\n[2] HMM  input: hidden {H}, observations = the {len(answers)} answers")
    path, prob = viterbi(answers)
    belief = forward_belief(answers)
    print(f"    most likely levels (Viterbi): {' -> '.join(path)}  (prob {prob:.5f})")
    print("    belief now: " + ", ".join(f"P({h})={belief[h]:.3f}" for h in H))
    print("    Why: the level is hidden and changes over time; HMM uses answer ORDER.")

    # 3. MDP
    print(f"\n[3] MDP  input: belief from HMM, actions {ACTIONS}, gamma={GAMMA}")
    for s in H:
        print("    " + "   ".join(f"Q({s},{a})={q_value(s, a):6.2f}" for a in ACTIONS))
    action, q = mdp_decide(belief)
    for a in ACTIONS:
        print(f"    expected Q({a}) = {q[a]:.3f}")
    print(f"    output: best action = {action}")
    print("    Why: Bellman backup = reward now + discounted future, averaged over the belief.")

    # 4. Hill climbing
    print(f"\n[4] HILL CLIMBING  input: 5 '{action}' exercises, P(K)={p:.3f}, "
          f"target success {TARGET_SUCCESS}")
    bank = EXERCISES[action]
    idx = hill_climbing(bank, p)
    print(f"    output: recommended exercise = {bank[idx][0]}")
    print("    Why: the neighbourhood is 'one step easier/harder'; we climb to the best fit.")

    print("\n[OUTPUT]")
    print(f"    P(K)={p:.3f} | levels={'->'.join(path)} | action={action} | exercise={bank[idx][0]}")
    return p, path, action, bank[idx][0]


# ========================================================= EVALUATION
def sample_student(n_answers, rng):
    """Generate a synthetic student from the HMM: true levels + answers."""
    def draw(dist):
        r, c = rng.random(), 0.0
        for k, v in dist.items():
            c += v
            if r < c:
                return k
        return k
    h = draw(PI)
    levels, answers = [], []
    for _ in range(n_answers):
        levels.append(h)
        answers.append(draw(B[h]))
        h = draw(A[h])
    return levels, answers


def evaluate(n_students=2000, n_answers=5, seed=42):
    """Compare the system with two baselines on synthetic students.
    'Correct' action = Review if the true CURRENT level is Beginner else NextTopic.
    NOTE: data come from the same model, so this is a consistency check of the
    decision logic, not proof of real-world accuracy."""
    rng = random.Random(seed)
    hits = {"system": 0, "always NextTopic": 0, "count rule (>=3 Correct)": 0}
    for _ in range(n_students):
        levels, answers = sample_student(n_answers, rng)
        truth = "Review" if levels[-1] == "Beginner" else "NextTopic"
        hits["system"] += mdp_decide(forward_belief(answers))[0] == truth
        hits["always NextTopic"] += "NextTopic" == truth
        rule = "NextTopic" if answers.count("Correct") >= 3 else "Review"
        hits["count rule (>=3 Correct)"] += rule == truth
    print("=" * 70)
    print(f"EVALUATION: {n_students} synthetic students, {n_answers} answers each (seed {seed})")
    print("=" * 70)
    for k, v in hits.items():
        print(f"  {k:26s} accuracy = {v / n_students:.1%}")
    print("  (data generated from the same HMM -> checks the logic, not real-world accuracy)\n")
    return {k: v / n_students for k, v in hits.items()}


# ============================================================== TESTS
def run_tests():
    print("=" * 70 + "\nTESTS\n" + "=" * 70)
    p, *_ = bayes_update(PRIOR_KNOW, True)
    assert abs(p - 0.34 / 0.49) < 1e-9
    assert bayes_update(0.4, True)[0] > 0.4 > bayes_update(0.4, False)[0]
    print("  Bayes: exact posterior 0.34/0.49; correct raises, wrong lowers belief   OK")

    assert viterbi(["Correct"] * 5, verbose=False)[0] == ["Ready"] * 5
    assert viterbi(["Wrong"] * 5, verbose=False)[0] == ["Beginner"] * 5
    assert abs(sum(forward_belief(["Wrong", "Partial", "Correct"]).values()) - 1) < 1e-9
    print("  HMM: all correct -> Ready, all wrong -> Beginner, belief sums to 1      OK")

    assert mdp_decide({"Beginner": 1.0, "Ready": 0.0})[0] == "Review"
    assert mdp_decide({"Beginner": 0.0, "Ready": 1.0})[0] == "NextTopic"
    print("  MDP: certain Beginner -> Review, certain Ready -> NextTopic             OK")

    for action, bank in EXERCISES.items():
        for pk in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
            i = hill_climbing(bank, pk, verbose=False)
            best = max(range(len(bank)), key=lambda j: exercise_score(bank[j], pk))
            assert i == best, (action, pk)
    assert (hill_climbing(EXERCISES["Review"], 1.0, verbose=False)
            >= hill_climbing(EXERCISES["Review"], 0.0, verbose=False))
    print("  Hill climbing: finds the global best exercise for every P(K) tested;")
    print("                 a stronger student gets a harder exercise                OK")
    print("  ALL TESTS PASSED\n")


def ask_mode():
    print("Answer 5 quiz questions: w = Wrong, p = Partial, c = Correct")
    mapping = {"w": "Wrong", "p": "Partial", "c": "Correct"}
    answers = []
    for i in range(1, 6):
        while True:
            x = input(f"Answer {i} (w/p/c): ").strip().lower()
            if x in mapping:
                answers.append(mapping[x])
                break
            print("  please type w, p or c")
    run_student("You", answers)


if __name__ == "__main__":
    if "--ask" in sys.argv:
        ask_mode()
    else:
        run_student("Aida (struggling)",   ["Wrong", "Wrong", "Partial", "Wrong", "Partial"])
        print()
        run_student("Dana (improving)",    ["Wrong", "Partial", "Wrong", "Correct", "Correct"])
        print()
        run_student("Aibek (strong)",      ["Correct", "Correct", "Partial", "Correct", "Correct"])
        print()
        run_tests()
        evaluate()
