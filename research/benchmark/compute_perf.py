import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load your CSV file
df = pd.read_csv('./benchmark/results.csv')  # Replace with your actual file path

# Convert both 'True_Label' and 'Generated_Answer' columns to strings
df['True_Label'] = df['True_Label'].astype(str)
df['Generated_Answer'] = df['Generated_Answer'].astype(str)

# Drop rows with missing values in 'True_Label' or 'Generated_Answer', or where 'Generated_Answer' is empty
df = df.dropna(subset=['True_Label', 'Generated_Answer'])
df = df[df['Generated_Answer'].str.strip() != '']  # Remove rows where 'Generated_Answer' is empty

# Extract the True_Label and Generated_Answer columns
true_labels = df['True_Label']
generated_answers = df['Generated_Answer']

# Compute accuracy
accuracy = accuracy_score(true_labels, generated_answers)

# Compute precision, recall, and F1-score for multi-class or binary classification
precision = precision_score(true_labels, generated_answers, average='weighted')  # Change average as needed (micro, macro, weighted)
recall = recall_score(true_labels, generated_answers, average='weighted')
f1 = f1_score(true_labels, generated_answers, average='weighted')

# Print the results
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Optional: Display a detailed classification report
print("\nClassification Report:")
print(classification_report(true_labels, generated_answers))
