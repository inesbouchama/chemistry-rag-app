import pandas as pd
import re
import glob
from sklearn.metrics import accuracy_score

# 🔁 Define your updated extractor
def extract_answer(text):
    # Look for patterns like: Answer: d or Answer:\n d
    match = re.search(r"Answer:\s*([a-dA-D])\b", text)
    if match:
        return match.group(1).lower()
    else:
        print("⚠️ Invalid answer format:", text[:200])  # Print only the first 200 chars for debugging
        return None

# 🔍 Load all raw output CSVs
files = glob.glob("src/benchmark/raw_outputs_temp*_top*_hybrid*.csv")

summary = []

for path in files:
    df = pd.read_csv(path)

    # Apply new extraction
    df['reprocessed_answer'] = df['raw_generation'].apply(extract_answer)
    df['reprocessed_correct'] = df['reprocessed_answer'] == df['true_label']

    # Accuracy
    reprocessed_acc = df['reprocessed_correct'].mean()
    original_acc = df['correct'].mean()

    # Extract config from filename
    config = path.split("/")[-1].replace("raw_outputs_", "").replace(".csv", "")

    # Save reprocessed version
    new_path = path.replace(".csv", "_reprocessed.csv")
    df.to_csv(new_path, index=False)

    summary.append({
        "config": config,
        "original_accuracy": round(original_acc, 3),
        "reprocessed_accuracy": round(reprocessed_acc, 3),
        "delta": round(reprocessed_acc - original_acc, 3),
        "file": new_path
    })

# 🔢 Show comparison
summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(by="reprocessed_accuracy", ascending=False)

print("\n🔍 Accuracy Comparison Summary:")
print(summary_df.to_string(index=False))

# Save
summary_df.to_csv("src/benchmark/reprocessed_summary.csv", index=False)
