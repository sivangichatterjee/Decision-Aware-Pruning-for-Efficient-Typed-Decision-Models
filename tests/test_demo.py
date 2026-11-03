"""Check presentation semantics and that user text reaches the real API shape."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from omnijev_demo import classify_text, save_run, summarize_answers


class DemoTests(unittest.TestCase):
    def test_abstention_is_preserved_even_when_upstream_names_an_option(self):
        run = {
            "questions": {"category": {"type": "choice"}, "statement": {"type": "noul"}},
            "answers": {
                "category": {
                    "choice": "billing", "probabilities": {"billing": 0.2, "account": 0.1},
                    "abstain": 0.7, "valid": False,
                },
                "statement": {"noul": 0.8},
            },
        }
        original = copy.deepcopy(run)
        table = summarize_answers(run)
        choice = table[table.question == "category"]
        self.assertEqual(choice.loc[choice.probability.idxmax(), "option"], "[abstain]")
        self.assertAlmostEqual(choice.probability.sum(), 1.0)
        self.assertAlmostEqual(table[table.question == "statement"].probability.sum(), 1.0)
        self.assertEqual(run, original)

    def test_context_is_in_instructions_and_run_can_be_saved(self):
        calls = []
        answers = {"category": {"choice": "billing", "probabilities": {"billing": 0.8, "account": 0.1}, "abstain": 0.1}}
        with tempfile.TemporaryDirectory() as directory:
            def infer(state, questions):
                calls.append((state, questions))
                return answers
            model = SimpleNamespace(
                _demo_root=Path(directory), _demo_runtime={"device": "cpu"},
                dev=SimpleNamespace(type="cpu"), system_one=infer,
            )
            run = classify_text(model, "Refund the duplicate charge.", {"billing": "Payments", "account": "Login"})
            self.assertEqual(len(calls), 1)
            state, questions = calls[0]
            self.assertTrue(Path(state["images"][0]).is_file())
            self.assertIn("Refund the duplicate charge.", questions["category"]["instructions"])
            self.assertEqual(questions["category"]["criteria"], {"billing": "Payments", "account": "Login"})
            self.assertEqual(run["answers"], answers)
            path = save_run(run, Path(directory) / "results" / "run.json")
            self.assertEqual(json.loads(path.read_text()), run)

    def test_incomplete_probabilities_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = SimpleNamespace(
                _demo_root=Path(directory), _demo_runtime={"device": "cpu"},
                dev=SimpleNamespace(type="cpu"),
                system_one=lambda *_: {"category": {"probabilities": {"a": 0.3, "b": 0.1}, "abstain": 0.1}},
            )
            with self.assertRaisesRegex(RuntimeError, "Probability mass"):
                classify_text(model, "Example", {"a": "First", "b": "Second"})


if __name__ == "__main__":
    unittest.main()
