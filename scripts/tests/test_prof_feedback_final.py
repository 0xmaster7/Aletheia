"""Offline tests for evaluation-only final-v3 overrides and Boolean semantics."""
from __future__ import annotations

import unittest

from scripts.lib.prof_feedback import QueryPlan
from scripts.lib.prof_feedback_final import (
    COUNT_CUES,
    KNOWN_RELATION_LIST,
    TEMPORAL_CUE_MAP,
    execute_final_plan,
    override_query_plan,
)


def make_plan(**updates):
    row = {
        "intent": "current_value", "entity": "Alpha", "property_scope": "specific",
        "property_name": "country", "temporal_scope": "unspecified",
        "ordinal_offset": None, "aggregation_mode": "none", "boolean_target": None,
        "negated": False, "supported": True, "reason": "",
    }
    row.update(updates)
    return QueryPlan.from_object(row)


class BooleanFinalTests(unittest.TestCase):
    def test_positive_target_search_ignores_named_property(self):
        p = make_plan(intent="boolean", property_name="country", temporal_scope="any",
                      boolean_target="Buddhism", supported=True)
        retrieved = [
            {"fact_idx": 1, "text": "Alpha has the country of Canada."},
            {"fact_idx": 2, "text": "Alpha has the religion of Buddhism."},
        ]
        plan, result, complete = execute_final_plan("Was Buddhism ever a fact about Alpha?", p,
                                                     retrieved, {"Alpha": 3})
        self.assertFalse(complete)
        self.assertIs(result["answer"], True)
        self.assertEqual(plan.property_name, None)  # question names no known relation
        # A separately named relation keeps specific plan scope, but the
        # Boolean target operator must still search every subject fact.
        named = make_plan(intent="boolean", property_name="country", temporal_scope="any",
                          boolean_target="Buddhism", supported=True)
        _, result, _ = execute_final_plan("Was Buddhism ever recorded as the language for Alpha?",
                                          named, retrieved, {"Alpha": 3})
        self.assertIs(result["answer"], True)

    def test_false_waits_for_verified_complete_entity_history(self):
        p = make_plan(intent="boolean", property_name="country", temporal_scope="any",
                      boolean_target="Klingon", supported=True)
        partial = [{"fact_idx": 1, "text": "Alpha has the country of Canada."}]
        _, result, complete = execute_final_plan("Was Klingon ever a fact about Alpha?", p,
                                                  partial, {"Alpha": 2})
        self.assertFalse(complete)
        self.assertIsNone(result["answer"])
        complete_rows = partial + [{"fact_idx": 2, "text": "Alpha has the religion of Buddhism."}]
        _, result, complete = execute_final_plan("Was Klingon ever a fact about Alpha?", p,
                                                  complete_rows, {"Alpha": 2})
        self.assertTrue(complete)
        self.assertIs(result["answer"], False)


class PlanOverrideTests(unittest.TestCase):
    def test_known_relation_list_is_generator_relation_list(self):
        self.assertEqual(KNOWN_RELATION_LIST, (
            "country", "genre", "location", "language", "religion", "nationality",
            "ethnicity", "occupation", "creator",
        ))

    def test_no_named_relation_forces_entity_wide_and_discards_model_property(self):
        p = make_plan(property_name="archival value")
        out = override_query_plan("What value was first recorded for Alpha?", p)
        self.assertEqual(out.property_scope, "entity_wide_values")
        self.assertIsNone(out.property_name)
        self.assertEqual(out.intent, "historical")
        self.assertEqual(out.temporal_scope, "earliest")

    def test_named_relation_preserves_specific_scope(self):
        p = make_plan(property_name="language")
        out = override_query_plan("What was the initial language for Alpha?", p)
        self.assertEqual(out.property_scope, "specific")
        self.assertEqual(out.property_name, "language")

    def test_temporal_cues_and_source_template_synonyms(self):
        for cue in ("first", "earliest", "initial", "oldest", "original", "beginning",
                    "inaugural", "founding", "ancestral", "primordial", "genesis",
                    "foundational", "debut", "preliminary", "embryonic", "seminal", "initially"):
            out = override_query_plan(f"What {cue} value applied to Alpha?", make_plan())
            self.assertEqual((out.intent, out.temporal_scope), ("historical", "earliest"), cue)
        for cue in ("previous", "prior", "earlier", "before"):
            out = override_query_plan(f"What value was {cue} for Alpha?", make_plan())
            self.assertEqual((out.intent, out.temporal_scope), ("historical", "previous"), cue)
        self.assertEqual(set(TEMPORAL_CUE_MAP), {
            "first", "earliest", "initial", "oldest", "original", "previous", "prior",
            "earlier", "before", "beginning", "inaugural", "founding", "ancestral",
            "primordial", "genesis", "foundational", "debut", "preliminary", "embryonic",
            "seminal", "initially",
        })

    def test_count_wording_forces_aggregation_count(self):
        for question in ("How many values for Alpha?", "What cardinality for Alpha?",
                         "Give the count of values for Alpha.",
                         "What is the number of values for Alpha?"):
            out = override_query_plan(question, make_plan(intent="historical", temporal_scope="earliest"))
            self.assertEqual((out.intent, out.aggregation_mode), ("aggregation", "count"), question)

    def test_before_any_updates_is_earliest_not_previous(self):
        for wording in ("Prior to any updates", "Before any modifications"):
            out = override_query_plan(f"{wording}, what value applied to Alpha?", make_plan())
            self.assertEqual((out.intent, out.temporal_scope), ("historical", "earliest"))


if __name__ == "__main__":
    unittest.main()
