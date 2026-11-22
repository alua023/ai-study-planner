"""Unit tests for AI Study Planner.  Run:  python -m unittest -v   (from the repo root)"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import study_planner as sp  # noqa: E402

W, P, C = "Wrong", "Partial", "Correct"


class TestBayes(unittest.TestCase):
    def test_one_correct_answer_exact(self):
        post, _, _ = sp.bayes_update(0.40, True)
        self.assertAlmostEqual(post, 0.34 / 0.49, places=9)      # 0.694

    def test_one_wrong_answer_exact(self):
        post, _, _ = sp.bayes_update(0.40, False)
        self.assertAlmostEqual(post, 0.06 / 0.51, places=9)      # 0.118

    def test_direction_and_bounds(self):
        for prior in (0.01, 0.2, 0.5, 0.8, 0.99):
            self.assertGreater(sp.bayes_update(prior, True)[0], prior)
            self.assertLess(sp.bayes_update(prior, False)[0], prior)
            self.assertTrue(0 < sp.bayes_update(prior, True)[0] < 1)

    def test_order_does_not_matter_for_bayes(self):
        def run(seq):
            p = sp.PRIOR_KNOW
            for x in seq:
                p = sp.bayes_update(p, x)[0]
            return p
        self.assertAlmostEqual(run([True, False, True]), run([True, True, False]), places=12)


class TestHMM(unittest.TestCase):
    def test_extremes(self):
        self.assertEqual(sp.viterbi([C] * 5, verbose=False)[0], ["Ready"] * 5)
        self.assertEqual(sp.viterbi([W] * 5, verbose=False)[0], ["Beginner"] * 5)

    def test_three_answers_like_the_topic_spec(self):
        path, prob = sp.viterbi([W, P, C], verbose=False)
        self.assertEqual(len(path), 3)
        self.assertTrue(0 < prob < 1)

    def test_dana_path(self):
        path, _ = sp.viterbi([W, P, W, C, C], verbose=False)
        self.assertEqual(path, ["Beginner", "Beginner", "Beginner", "Ready", "Ready"])

    def test_belief_is_distribution(self):
        b = sp.forward_belief([W, P, C, C, W])
        self.assertAlmostEqual(sum(b.values()), 1.0, places=12)
        self.assertTrue(all(0 <= v <= 1 for v in b.values()))

    def test_dana_belief_ready(self):
        self.assertAlmostEqual(sp.forward_belief([W, P, W, C, C])["Ready"], 0.876, places=3)

    def test_order_matters(self):
        late_good = sp.forward_belief([W, W, W, C, C])["Ready"]
        late_bad = sp.forward_belief([C, C, W, W, W])["Ready"]
        self.assertGreater(late_good, 0.8)
        self.assertLess(late_bad, 0.2)


class TestMDP(unittest.TestCase):
    def test_hand_computed_q_values(self):
        self.assertAlmostEqual(sp.q_value("Beginner", "Review"), 8.5)
        self.assertAlmostEqual(sp.q_value("Beginner", "NextTopic"), -2.1)
        self.assertAlmostEqual(sp.q_value("Ready", "Review"), 9.1)
        self.assertAlmostEqual(sp.q_value("Ready", "NextTopic"), 13.2)

    def test_certain_states(self):
        self.assertEqual(sp.mdp_decide({"Beginner": 1.0, "Ready": 0.0})[0], "Review")
        self.assertEqual(sp.mdp_decide({"Beginner": 0.0, "Ready": 1.0})[0], "NextTopic")

    def test_dana_expected_q(self):
        action, q = sp.mdp_decide(sp.forward_belief([W, P, W, C, C]))
        self.assertEqual(action, "NextTopic")
        self.assertAlmostEqual(q["Review"], 9.03, delta=0.01)
        self.assertAlmostEqual(q["NextTopic"], 11.30, delta=0.01)

    def test_monotone_in_belief(self):
        chosen = [sp.mdp_decide({"Beginner": 1 - r, "Ready": r})[0] for r in [i / 20 for i in range(21)]]
        switches = sum(a != b for a, b in zip(chosen, chosen[1:]))
        self.assertEqual(switches, 1)       # exactly one switch Review -> NextTopic


class TestHillClimbing(unittest.TestCase):
    def test_success_is_total_probability(self):
        ex = sp.EXERCISES["NextTopic"][4]    # N5: 0.72 / 0.08
        self.assertAlmostEqual(sp.exercise_success(ex, 0.947), 0.72 * 0.947 + 0.08 * 0.053)

    def test_finds_global_best_everywhere(self):
        for bank in sp.EXERCISES.values():
            for k in range(0, 101):
                pk = k / 100
                i = sp.hill_climbing(bank, pk, verbose=False)
                best = max(range(len(bank)), key=lambda j: sp.exercise_score(bank[j], pk))
                self.assertEqual(i, best, (pk,))

    def test_stronger_student_harder_exercise(self):
        bank = sp.EXERCISES["NextTopic"]
        self.assertLessEqual(sp.hill_climbing(bank, 0.05, verbose=False),
                             sp.hill_climbing(bank, 0.95, verbose=False))

    def test_aibek_gets_n5(self):
        idx = sp.hill_climbing(sp.EXERCISES["NextTopic"], 0.947, verbose=False)
        self.assertTrue(sp.EXERCISES["NextTopic"][idx][0].startswith("N5"))


class TestIntegration(unittest.TestCase):
    def _run(self, answers):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return sp.run_student("t", answers)

    def test_three_students(self):
        p, path, action, ex = self._run([W, W, P, W, P])
        self.assertEqual((action, ex[:2]), ("Review", "R1"))
        p, path, action, ex = self._run([W, P, W, C, C])
        self.assertEqual((action, ex[:2]), ("NextTopic", "N1"))
        self.assertAlmostEqual(p, 0.058, places=3)
        p, path, action, ex = self._run([C, C, P, C, C])
        self.assertEqual((action, ex[:2]), ("NextTopic", "N5"))
        self.assertAlmostEqual(p, 0.947, places=3)

    def test_every_answer_combination_runs(self):
        import itertools
        for combo in itertools.product((W, P, C), repeat=5):     # all 243 inputs
            p, path, action, ex = self._run(list(combo))
            self.assertTrue(0 <= p <= 1)
            self.assertEqual(len(path), 5)
            self.assertIn(action, sp.ACTIONS)
            self.assertTrue(ex)

    def test_evaluation_beats_baselines(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            r = sp.evaluate(n_students=2000, seed=42)
        self.assertGreater(r["system"], r["always NextTopic"])
        self.assertGreater(r["system"], r["count rule (>=3 Correct)"])

    def test_synthetic_student_is_reproducible(self):
        a = sp.sample_student(5, random.Random(1))
        b = sp.sample_student(5, random.Random(1))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
