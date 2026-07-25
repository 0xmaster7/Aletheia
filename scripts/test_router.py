from semantic_router import Route
from semantic_router import SemanticRouter  # Updated import
from semantic_router.encoders import HuggingFaceEncoder

# 1. Define Routes and Anchor Utterances
current_val = Route(
    name="current_value",
    utterances=[
        "What is the current state of X?",
        "What operating system is currently used?",
        "Where is the user living right now?",
    ],
)

historical = Route(
    name="historical",
    utterances=[
        "What was used before this?",
        "What was the previous configuration?",
        "Which operating system was used prior to switching?",
    ],
)

boolean = Route(
    name="boolean",
    utterances=[
        "Is the user still using Ubuntu?",
        "Does the project support feature X?",
        "Is this record still active?",
    ],
)

aggregation = Route(
    name="aggregation",
    utterances=[
        "How many total projects are there?",
        "List all unique languages used.",
        "How many times has the configuration changed?",
    ],
)

# 2. Load Local Embedding Model
print("Loading local embedding model...")
encoder = HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2")

# 3. Initialize the Router (Updated class name)
rl = SemanticRouter(encoder=encoder, routes=[current_val, historical, boolean, aggregation])

# 4. Test Routing Precision
test_queries = [
    "What OS is installed right now?",
    "What did he use before Fedora?",
    "Is Ubuntu still running?",
    "How many projects has he worked on?"
]

print("\n--- Routing Results ---")
for q in test_queries:
    route = rl(q)
    print(f"Query: '{q}'  -->  Route: {route.name}")