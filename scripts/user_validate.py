import json
from collections import defaultdict

file_path = "data/synthetic_benchmark.json"

try:
    with open(file_path, "r") as f:
        data = json.load(f)
        
    print(f"Total objects loaded: {len(data)}")
    assert len(data) == 13425, f"Expected 13425 questions, got {len(data)}"

    entities = defaultdict(list)
    bool_counts = {"True": 0, "False": 0}
    
    for idx, obj in enumerate(data):
        # 4. Check required keys and emptiness
        assert all(k in obj for k in ["entity", "intent", "question_text", "ground_truth_answer"]), f"Missing keys at index {idx}!"
        assert str(obj["question_text"]).strip() != "", f"Empty question_text at index {idx}!"
        assert str(obj["ground_truth_answer"]).strip() != "", f"Empty ground_truth_answer at index {idx}!"
        
        intent = obj["intent"]
        gt_answer = str(obj["ground_truth_answer"]).strip()
        
        # 1. Intent values checked
        assert intent in ["historical", "aggregation", "boolean"], f"Invalid intent '{intent}' at index {idx}"
        
        # 2. Boolean answers strictly checked
        if intent == "boolean":
            assert gt_answer in ["True", "False"], f"Invalid boolean answer '{gt_answer}' at index {idx}"
            bool_counts[gt_answer] += 1
            
        # 3. Aggregation answers parseable as positive integers
        if intent == "aggregation":
            assert gt_answer.isdigit() and int(gt_answer) > 0, f"Invalid aggregation integer '{gt_answer}' at index {idx}"

        entities[obj["entity"]].append(intent)

    print(f"Total distinct entities: {len(entities)}")
    assert len(entities) == 4475, f"Expected 4475 entities, got {len(entities)}"

    # 5. Duplicate or missing intents caught cleanly
    for ent, intents in entities.items():
        assert sorted(intents) == ["aggregation", "boolean", "historical"], f"Intent mismatch or duplicate for entity: {ent}. Found: {intents}"

    # 6. Distribution report
    print("\n" + "="*30)
    print("📊 DATASET DISTRIBUTION REPORT")
    print("="*30)
    print(f"Historical queries:  {len(entities)}")
    print(f"Aggregation queries: {len(entities)}")
    print(f"Boolean queries:     {len(entities)}")
    print(f"  ↳ Boolean True:    {bool_counts['True']}")
    print(f"  ↳ Boolean False:   {bool_counts['False']}")
    print("="*30)
    print("✅ DATASET IS 100% VALIDATED AND PAPER-READY.")

except Exception as e:
    print(f"\n❌ Validation failed: {e}")
