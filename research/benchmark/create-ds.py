import re
import pandas as pd

def parse_mcq_txt_to_csv(txt_path: str, csv_path: str):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    data = []
    q_id = 1
    i = 0
    current_speaker = None

    while i < len(lines):
        line = lines[i].strip()

        # Detect speaker
        if line.startswith("#"):
            current_speaker = line[1:].strip()
            i += 1
            continue

        # Detect timestamp (start of a question block)
        if re.match(r"^t\.", line):
            timestamp = line.strip()
            i += 1

            # Gather question lines until option 'a.' appears
            question_lines = []
            while i < len(lines) and not lines[i].strip().startswith("a."):
                question_lines.append(lines[i].strip())
                i += 1
            question = " ".join(question_lines)

            # Extract options
            a = lines[i].replace("a. ", "").strip(); i += 1
            b = lines[i].replace("b. ", "").strip(); i += 1
            c = lines[i].replace("c. ", "").strip(); i += 1
            d = lines[i].replace("d. ", "").strip(); i += 1

            # Save question data
            data.append({
                "id": q_id,
                "speaker": current_speaker,
                "timestamp": timestamp,
                "question": question,
                "a": a,
                "b": b,
                "c": c,
                "d": d,
                "label": ""
            })
            q_id += 1
        else:
            i += 1

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV file saved to: {csv_path}")

# Example usage:
parse_mcq_txt_to_csv("./benchmark/questions&context.txt", "./benchmark/ds.csv")
