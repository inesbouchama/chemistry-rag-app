import pandas as pd
import itertools
import re
import json
from tqdm import tqdm

# ✅ Correct relative import
from src.RAG.rag import ChemistryRAG


# def extract_answer(text):
#     # Look for patterns like: Answer: d or Answer:\n d
#     match = re.search(r"Answer:\s*([a-dA-D])\b", text)
#     if match:
#         return match.group(1).lower()
#     else:
#         print("⚠️ Invalid answer format:", text[:200])  # Print only the first 200 chars for debugging
#         return None


if __name__ == "__main__":
    # Initialize the RAG system
    rag = ChemistryRAG(
        train_caption_dir="data/4m_data/english/train/caption",
        train_image_dir="data/4m_data/english/train/rgb",
        test_caption_dir="data/4m_data/english/test/caption",
        test_image_dir="data/4m_data/english/test/rgb",
        save_dir="./src/RAG/saved_indexes",
        load_saved=True
    )

    # Step 1: Compute RAG accuracy and save misclassified results
    print("\n🔍 Computing RAG accuracy...")
    rag.evaluate_rag_accuracy(
        benchmark_path="./src/benchmark/mcq_labelled.csv",
        output_path="./src/benchmark/misclassified-rag.csv"
    )
    print("✅ RAG accuracy computation completed. Misclassified results saved to './src/benchmark/misclassified-rag.csv'.")

    # Step 2: Perform grid search (already implemented)
    temp = 0.3
    top_p = 0.9
    hybrid_weight = 0.3

    df = pd.read_csv('./src/benchmark/mcq_labelled.csv')
    search_results = []

    
    print(f"\n🔍 Running config: temp={temp}, top_p={top_p}, hybrid_weight={hybrid_weight}")
    config_raw_outputs = []
    correct = 0
    total = 0

    for index, row in tqdm(df.iterrows(), total=len(df)):
        question = row['question']
        a, b, c, d = row['a'], row['b'], row['c'], row['d']
        true_label = row['label']

        try:
            retrievals, raw_answer = rag.answer_question(
                query=question,
                a=a, b=b, c=c, d=d,
                top_k=2,
                hybrid_weight=hybrid_weight,
                min_score_threshold=0
            )

            # ✅ Override sampling params dynamically
            rag.llm.generation_config.temperature = temp
            rag.llm.generation_config.top_p = top_p

        except Exception as e:
            print(f"❌ Error at row {index}: {e}")
            continue

        predicted = extract_answer(raw_answer)
        is_correct = predicted == true_label
        total += 1
        correct += int(is_correct)

        config_raw_outputs.append({
            "id": row['id'],
            "temperature": temp,
            "top_p": top_p,
            "hybrid_weight": hybrid_weight,
            "question": question,
            "a": a, "b": b, "c": c, "d": d,
            "true_label": true_label,
            "raw_generation": raw_answer,
            "extracted_answer": predicted,
            "correct": is_correct
        })

    acc = correct / total if total > 0 else 0.0
    print(f"✅ Config Result: Accuracy = {acc:.2f}")
    search_results.append({
        "temperature": temp,
        "top_p": top_p,
        "hybrid_weight": hybrid_weight,
        "accuracy": acc,
        "total": total,
        "correct": correct
    })

    config_df = pd.DataFrame(config_raw_outputs)
    config_name = f"temp{temp}_top{top_p}_hybrid{hybrid_weight}".replace(".", "")
    config_df.to_csv(f"./src/benchmark/raw_outputs_{config_name}.csv", index=False)

summary_df = pd.DataFrame(search_results)
#summary_df.to_csv("./src/benchmark/grid_search_summary.csv", index=False)
#print("📁 Grid search completed and results saved.")