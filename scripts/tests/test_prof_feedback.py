"""Offline unit tests for guarded campaign parsing and symbolic operators."""
from __future__ import annotations

import unittest
import numpy as np

from scripts.analysis.prof_feedback_prepare import parse_facts, query_for_retrieval, rank_pipeline, rank_retrieval

from scripts.lib.prof_feedback import (
    Candidate,
    QueryPlan,
    deterministic_candidates,
    canonical_boolean,
    execute_plan,
    local_history_complete,
    normalized_historical,
    parse_fact_template,
    parse_and_validate_candidates,
    approved_evaluation_prompt_and_schema,
    build_approved_evaluation_request,
    request_cache_key,
    strict_integer,
    subem,
)
from scripts.lib.evaluation_scorer import parse_boolean, parse_final_number, score_answer


def plan(**updates):
    base = {
        "intent": "aggregation", "entity": "Alpha", "property_scope": "specific",
        "property_name": "language", "temporal_scope": "any", "ordinal_offset": None,
        "aggregation_mode": "count", "boolean_target": None, "negated": False,
        "supported": True, "reason": "explicit count of languages",
    }
    base.update(updates)
    return QueryPlan.from_object(base)


class QueryPlanTests(unittest.TestCase):
    def test_rejects_extra_schema_keys_and_invalid_scope(self):
        value = plan().as_dict()
        value["unexpected"] = 1
        with self.assertRaises(ValueError):
            QueryPlan.from_object(value)
        value = plan().as_dict()
        value["property_scope"] = "guess"
        with self.assertRaises(ValueError):
            QueryPlan.from_object(value)

    def test_rejects_ambiguous_plan_marked_supported(self):
        with self.assertRaises(ValueError):
            plan(property_scope="ambiguous", property_name=None)


class CandidateTests(unittest.TestCase):
    def test_approved_prompt_builder_is_evaluation_only_and_strict(self):
        prompt, schema = approved_evaluation_prompt_and_schema()
        self.assertIn("Return exactly one JSON object", prompt)
        self.assertIn("names no relation", prompt)
        self.assertIn("primordial", prompt)
        self.assertFalse(schema["additionalProperties"])
        request = build_approved_evaluation_request(
            "What is Alpha's language?",
            [{"fact_idx": 3, "text": "Alpha has the language of English."}],
        )
        body = request["messages"][0]["content"]
        self.assertIn("What is Alpha's language?", body)
        self.assertIn("3. Alpha has the language of English.", body)
        self.assertEqual(request["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual(request["max_completion_tokens"], 1024)
        self.assertEqual(schema["required"], ["query_plan"])
        self.assertNotIn("candidates", schema["properties"])

    def test_validates_serial_and_exact_retrieved_text(self):
        retrieved = [{"fact_idx": 4, "text": "Alpha has the language of English."}]
        parsed = parse_and_validate_candidates({"candidates": [{
            "serial": 4, "fact_text": "Alpha has the language of English.",
            "answer_entity": "English",
        }]}, retrieved)
        self.assertEqual(parsed, [Candidate(4, "Alpha has the language of English.", "English")])
        with self.assertRaises(ValueError):
            parse_and_validate_candidates({"candidates": [{
                "serial": 4, "fact_text": "invented", "answer_entity": "English",
            }]}, retrieved)

    def test_deduplicates_and_rejects_answer_not_stated_in_fact(self):
        retrieved = [{"fact_idx": 4, "text": "Alpha has the language of English."}]
        row = {"serial": 4, "fact_text": "Alpha has the language of English.", "answer_entity": "English"}
        parsed = parse_and_validate_candidates({"candidates": [row, row]}, retrieved)
        self.assertEqual(len(parsed), 1)
        fabricated = {**row, "answer_entity": "French"}
        with self.assertRaises(ValueError):
            parse_and_validate_candidates({"candidates": [fabricated]}, retrieved)


class OperatorTests(unittest.TestCase):
    def test_template_extraction_handles_spoken_language_and_deterministic_scope(self):
        self.assertEqual(
            parse_fact_template("Alpha speaks the language of English."),
            ("Alpha", "English"),
        )
        retrieved = [
            {"fact_idx": 1, "text": "Alpha speaks the language of English."},
            {"fact_idx": 2, "text": "Alpha is affiliated with the religion of Buddhism."},
            {"fact_idx": 3, "text": "Beta speaks the language of French."},
        ]
        selected = deterministic_candidates(plan(), retrieved)
        self.assertEqual([(row.serial, row.answer_entity) for row in selected], [(1, "English")])

    def test_entity_wide_plan_selects_relations_without_inventing_property(self):
        generic = plan(property_scope="entity_wide_values", property_name=None)
        selected = deterministic_candidates(generic, [
            {"fact_idx": 1, "text": "Alpha speaks the language of English."},
            {"fact_idx": 2, "text": "Alpha is affiliated with the religion of Buddhism."},
        ])
        self.assertEqual([row.answer_entity for row in selected], ["English", "Buddhism"])

    def test_boolean_positive_evidence_answers_true_without_complete_history(self):
        boolean = plan(intent="boolean", property_name="language", temporal_scope="any",
                       aggregation_mode="none", boolean_target="English")
        retrieved = [{"fact_idx": 1, "text": "Alpha speaks the language of English."}]
        candidates = deterministic_candidates(boolean, retrieved)
        self.assertFalse(local_history_complete(boolean, retrieved, {"Alpha": 2}))
        self.assertIs(execute_plan(boolean, candidates, complete_history=False)["answer"], True)

    def test_boolean_negative_abstains_until_local_entity_count_is_complete(self):
        boolean = plan(intent="boolean", property_name="language", temporal_scope="any",
                       aggregation_mode="none", boolean_target="Klingon")
        one_of_two = [{"fact_idx": 1, "text": "Alpha speaks the language of English."}]
        candidates = deterministic_candidates(boolean, one_of_two)
        self.assertFalse(local_history_complete(boolean, one_of_two, {"Alpha": 2}))
        self.assertIsNone(execute_plan(boolean, candidates, complete_history=False)["answer"])
        complete = [
            {"fact_idx": 1, "text": "Alpha speaks the language of English."},
            {"fact_idx": 2, "text": "Alpha is affiliated with the religion of Buddhism."},
        ]
        self.assertTrue(local_history_complete(boolean, complete, {"Alpha": 2}))
        result = execute_plan(boolean, deterministic_candidates(boolean, complete), complete_history=True)
        self.assertIs(result["answer"], False)

    def test_aggregation_deduplicates_normalized_values_and_requires_coverage(self):
        candidates = [
            Candidate(1, "Alpha has the language of English.", "English"),
            Candidate(2, "Alpha has the language of english.", "english"),
            Candidate(3, "Alpha has the language of French.", "French"),
        ]
        self.assertEqual(execute_plan(plan(), candidates)["status"], "insufficient_evidence")
        result = execute_plan(plan(), candidates, complete_history=True)
        self.assertEqual(result["answer"], 2)

    def test_historical_previous_collapses_repeated_states(self):
        candidates = [
            Candidate(1, "Alpha has the language of English.", "English"),
            Candidate(2, "Alpha has the language of English.", "English"),
            Candidate(3, "Alpha has the language of French.", "French"),
        ]
        historical = plan(intent="historical", property_name="language", temporal_scope="previous",
                          ordinal_offset=None, aggregation_mode="none")
        self.assertEqual(execute_plan(historical, candidates)["status"], "insufficient_history")
        result = execute_plan(historical, candidates, complete_history=True)
        self.assertEqual(result["answer"], "English")
        self.assertEqual(result["serial"], 2)

    def test_historical_ordinal_and_out_of_range_are_explicit(self):
        candidates = [
            Candidate(1, "Alpha has the language of English.", "English"),
            Candidate(2, "Alpha has the language of French.", "French"),
            Candidate(3, "Alpha has the language of Spanish.", "Spanish"),
        ]
        ordinal = plan(intent="historical", property_name="language", temporal_scope="ordinal",
                       ordinal_offset=2, aggregation_mode="none")
        result = execute_plan(ordinal, candidates, complete_history=True)
        self.assertEqual(result["answer"], "English")
        out_of_range = plan(intent="historical", property_name="language", temporal_scope="ordinal",
                            ordinal_offset=5, aggregation_mode="none")
        self.assertEqual(execute_plan(out_of_range, candidates, complete_history=True)["status"], "insufficient_history")

    def test_aggregation_list_preserves_first_seen_normalized_unique_values(self):
        candidates = [
            Candidate(1, "Alpha has the language of English.", "English"),
            Candidate(2, "Alpha has the language of english.", "english"),
            Candidate(3, "Alpha has the language of French.", "French"),
        ]
        listing = plan(aggregation_mode="list")
        self.assertEqual(execute_plan(listing, candidates, complete_history=True)["answer"], ["English", "French"])

    def test_boolean_never_does_not_infer_absence_from_top_k(self):
        boolean = plan(intent="boolean", property_name="language", temporal_scope="any",
                       aggregation_mode="none", boolean_target="Klingon", negated=True)
        self.assertEqual(execute_plan(boolean, [])["status"], "insufficient_evidence")
        result = execute_plan(boolean, [], complete_history=True)
        self.assertIs(result["answer"], True)
        matching = [Candidate(1, "Alpha has the language of Klingon.", "Klingon")]
        self.assertIs(execute_plan(boolean, matching)["answer"], False)

    def test_boolean_any_and_latest_have_distinct_scopes(self):
        candidates = [
            Candidate(1, "Alpha has the language of English.", "English"),
            Candidate(2, "Alpha has the language of French.", "French"),
        ]
        any_plan = plan(intent="boolean", property_name="language", temporal_scope="any",
                        aggregation_mode="none", boolean_target="English")
        self.assertIs(execute_plan(any_plan, candidates)["answer"], True)
        latest_plan = plan(intent="boolean", property_name="language", temporal_scope="latest",
                           aggregation_mode="none", boolean_target="English", negated=True)
        self.assertIs(execute_plan(latest_plan, candidates)["answer"], False)
        self.assertIs(execute_plan(latest_plan, candidates, complete_history=True)["answer"], False)


class ScoringAndCacheTests(unittest.TestCase):
    def test_intent_specific_scorers(self):
        self.assertTrue(subem("English is spoken", "english"))
        self.assertTrue(strict_integer(" 3 ", "3"))
        self.assertFalse(strict_integer("3.0", "3"))
        self.assertTrue(canonical_boolean("TRUE", "true"))
        self.assertFalse(canonical_boolean("yes", "true"))
        self.assertTrue(normalized_historical("The-Beatles", "the Beatles"))
        self.assertTrue(subem("15", "5"), "legacy SubEM has a known substring collision")
        self.assertFalse(strict_integer("15", "5"))

    def test_fixed_campaign_scorer_normalizes_boolean_aliases(self):
        self.assertTrue(score_answer("boolean", "Yes, that is correct.", "True"))
        self.assertTrue(score_answer("boolean", "No.", "False"))
        self.assertFalse(score_answer("boolean", "No.", "True"))
        self.assertIs(parse_boolean("The answer is yes, correct."), True)

    def test_fixed_campaign_aggregation_scorer_uses_final_exact_number(self):
        self.assertEqual(parse_final_number("There are 5 values. Final answer: 4."), 4)
        self.assertEqual(parse_final_number("The answer is four."), 4)
        self.assertTrue(score_answer("aggregation", "There are 15 values.", "5") is False)
        self.assertTrue(score_answer("aggregation", "There are 5 values.", "5"))

    def test_cache_key_changes_with_request_parameters(self):
        args = dict(model="model", messages=[{"role": "user", "content": "q"}], temperature=0,
                    seed=42, max_output_tokens=20, response_format={"type": "json_object"},
                    adapter_version="v1")
        first = request_cache_key(**args)
        self.assertEqual(first, request_cache_key(**args))
        self.assertNotEqual(first, request_cache_key(**{**args, "temperature": 0.1}))


class RetrievalAuditTests(unittest.TestCase):
    def test_line_start_parser_does_not_split_u2_or_other_embedded_numbers(self):
        context = "1. U2 has the language of English.\n2. Another fact is clear."
        facts = parse_facts(context)
        self.assertEqual([fact["serial"] for fact in facts], [1, 2])
        self.assertEqual(facts[0]["text"], "U2 has the language of English")

    def test_cleaned_queries_and_stable_tie_breaking(self):
        self.assertEqual(query_for_retrieval("How many different languages are on file?", "aggregation"), "languages are on file?")
        self.assertEqual(rank_retrieval(np.array([1.0, 1.0, 0.0]), 3), [0, 1, 2])
        self.assertEqual(rank_pipeline(np.array([0.0, 0.0, 0.0]), 3), [2, 1, 0])


if __name__ == "__main__":
    unittest.main()
