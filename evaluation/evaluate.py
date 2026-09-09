import os
import sys
import warnings
import pandas as pd
from dotenv import load_dotenv

# Silence Ragas deprecation warnings in the terminal
warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datasets import Dataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

from retrieval.retriever import Retriever
from generation.llm import LLMGenerator
from indexing.embedding import get_embedding_model


def run_benchmark():
    print("=== Initializing Ragas Evaluation for BRICS Pipeline ===")

    retriever = Retriever(top_k=3, collection_name="brics_rag")
    generator = LLMGenerator()

    # Evaluator model: temperature=0, single candidate response
    raw_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.0,
        max_retries=6
    )
    raw_embeddings = get_embedding_model()

    evaluator_llm = LangchainLLMWrapper(raw_llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(raw_embeddings)

    # Gemini only supports candidate_count=1 (strictness=1 prevents 400 InvalidArgument error)
    if hasattr(answer_relevancy, "strictness"):
        answer_relevancy.strictness = 1

    test_cases = [
        {
            "question": "What was China's coal production in 2025 according to the report?",
            "ground_truth": "China's coal production in 2025 was 4,849 million tons."
        },
        {
            "question": "What was India's coal production in 2025 according to the report?",
            "ground_truth": "India's coal production in 2025 was 1,040 million tons (provisional)."
        },
        {
            "question": "What major cereals and meat types are highlighted for South Africa in agricultural tables?",
            "ground_truth": "The report highlights major cereals and major meat types for South Africa along with livestock notes."
        },
        {
            "question": "What is the projected commercial drone market revenue for South Africa in 2026?",
            "ground_truth": "The report does not contain any information or projections regarding commercial drone market revenue."
        }
    ]

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    print("\nExecuting live queries against pipeline...")
    for idx, case in enumerate(test_cases, 1):
        q = case["question"]
        gt = case["ground_truth"]
        print(f"[{idx}/{len(test_cases)}] Query: {q}")

        retrieved_docs = retriever.retrieve(q)
        retrieved_texts = [d.page_content for d in retrieved_docs] if retrieved_docs else ["No context retrieved."]
        formatted_context = retriever.format_context(retrieved_docs) if retrieved_docs else "No context available."

        answer = generator.generate_response(
            query=q,
            context=formatted_context,
            chat_history=[]
        )

        questions.append(q)
        ground_truths.append(gt)
        answers.append(answer)
        contexts.append(retrieved_texts)

    if hasattr(retriever.vector_store, "close"):
        retriever.vector_store.close()

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\nCalculating Ragas scores with single-worker rate limiting...")
    # Serial execution to strictly respect Google free-tier limits
    run_cfg = RunConfig(max_workers=1, timeout=180, max_retries=6)

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_cfg
    )

    print("\n=== Benchmark Summary Scores ===")
    print(results)

    df = getattr(results, "to_pandas")()
    output_path = os.path.join(PROJECT_ROOT, "evaluation_report.csv")
    df.to_csv(output_path, index=False)
    print(f"\nDetailed evaluation results saved to: {output_path}")


if __name__ == "__main__":
    run_benchmark()