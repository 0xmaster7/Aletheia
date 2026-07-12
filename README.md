This repository provides an extended implementation of the architecture introduced in the 2026 arXiv preprint: ["Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution"](https://arxiv.org/pdf/2606.01435)

This repository extends the baseline project by implementing an Embedding-Based Semantic Router and an Adaptive Operator Layer to resolve major logic limitations found in the original implementation.

## Getting Stared

1. **Clone the Repository**

   Clone this repository and navigate into the project root:
   ```
   git clone https://github.com//memory-conflict-resolution.git
   cd memory-conflict-resolution
   ```

2. **Environment Setup**

   **Option A: Using Conda (Recommended)**

   Create an isolated environment named Aletheia, activate it, and install all required dependencies:
   ```
   conda create --name Aletheia python=3.10 -y
   conda activate Aletheia
   pip install -r requirements.txt
   ```

   **Option B: Using Standard Pip venv (Alternative)**

   If you do not have Conda installed, set up a standard Python virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure API Credentials**

   Copy the environment template file to `.env`:
   ```
   cp .env.example .env
   ```
   Open the `.env` file in your text editor and add your OpenAI API Key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## The Architecture: Limitations & Solutions

While the original paper's max(serial) approach effectively eliminates hallucination rates in basic key-value updates, its reliance on a single mathematical primitive causes it to fail on more complex conversational queries.

To overcome this, this repository introduces a Semantic Routing Layer right at the entry point in `scripts/_pipeline.py`. When a query arrives, it is converted into a vector representation using a local embedding model. The system computes the cosine similarity between the incoming query vector and pre-defined semantic anchor points to immediately direct the question to one of four specialized execution pathways.

Below are the 3 distinct query tasks handled by this architecture:

1. **Historical / Temporal Queries**
   - **The Problem:** The original implementation blindly executes max(serial), meaning it will always return the newest fact, completely failing if a user explicitly asks about the past.
   - **The Adaptive Solution:** For queries tracking historical sequence, the pipeline switches to a sort(serial) routine. The extracted facts are arranged chronologically, and a mathematical index offset is applied to fetch the exact historical record preceding the latest state.

2. **Boolean (Yes/No) Queries**
   - **The Problem:** The deterministic primitive outputs raw text/values rather than evaluating a logical statement, making it incapable of answering binary verification checks.
   - **The Adaptive Solution:** When a true/false query is routed, the system runs the standard max(serial) search to find the latest valid truth state. It then passes both the query target and the current state into a hard-coded Python logic gate. If they match, it outputs True; if they mismatch, it outputs False.

3. **Aggregation & Counting Queries**
   - **The Problem:** A single max() tracking operation isolates only one record, dropping the rest of the historical timeline and causing the model to fail when asked to count or list historical actions.
   - **The Adaptive Solution:** For queries requiring compilation, the pipeline extracts target entries across the entire historical memory stream. It passes these values into a mathematical Python set() object to instantly deduplicate identical historical entries. The system then applies a len(set) function to return an absolute count, or formats the raw unique set as an array if a list is requested.

## Implementing the Semantic Routing Layer

To build this semantic layer, you do not need to gather massive datasets or build anything from scratch. The architecture relies on three core components that are readily available:

1. **The Routing Library**

   We use a lightweight, open-source Python package called semantic-router. It is an industry-standard framework designed specifically to build fast decision-making layers using vector embeddings rather than slow LLM generations. You can add it directly to your project using a simple command:
   ```
   pip install semantic-router
   ```

2. **The Embedding Model**

   To compare the semantic meaning of the questions, the router needs an encoder model to convert the text into numerical vectors. While semantic-router supports paid API endpoints, you can bypass those costs entirely by running it locally. Hooking up a small embedding model via the library's local HuggingFace encoder keeps your entire routing layer private, offline, and completely free.

3. **The Anchor Data (Utterances)**

   You do not need to download or curate a large training dataset. The semantic router relies on a small, hardcoded list of example phrases, called utterances, defined right inside your pipeline script.

You simply define a Route object for each of our four pathways and provide about 5 to 10 example sentences. For instance, for the Boolean Route, you supply utterances like "Is the system still running Ubuntu?" or "Does the application support this feature?". For the Aggregation Route, you supply utterances like "How many projects are there?" or "List all previous operating systems."

When a new query hits the system, the router converts it into an embedding, calculates the similarity against your predefined utterances, and instantly triggers the matching Python function.