import pandas as pd

# Define the metrics to compare
METRICS = [
    "Problem Understanding (Comprehension)",
    "Clarity and Step-by-Step Explanation",
    "Accuracy of Process (Correctness of Steps)",
    "Correctness of Final Answer",
    "Learning Appropriateness (Is the Explanation Suitable for Learners?)",
    "Generalization (Can the Learner Apply This Method to Similar Problems?)",
    "Technical Terms Explanation",
    "Addressing Common Errors",
]

# File paths for the three languages (replace with your actual file paths)
files = {
    "en": "3_topic_areas_evaluations.csv",
    "ar": "3_topic_areas_evaluations_ar.csv",
    "de": "3_topic_areas_evaluations_de.csv",
}

def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)

def calculate_metric_averages(df):
    """Calculate the average values for the specified metrics."""
    return df[METRICS].mean()

def compare_results(files):
    """Compare evaluation results across languages."""
    # Load data for each language
    results = {}
    for lang, file_path in files.items():
        df = load_csv(file_path)
        results[lang] = calculate_metric_averages(df)

    # Combine results into a single DataFrame for comparison
    comparison_df = pd.DataFrame(results)

    # Save the comparison to a CSV file
    comparison_df.T.to_csv("4_comparison_results.csv", index=True)

    # Print the comparison
    print("\n=== Comparison of Metrics Across Languages ===\n")
    print(comparison_df.T)  # Transpose for better readability

    # Identify the language with the highest average for each metric
    highest_avg = comparison_df.idxmax(axis=1)
    print("\n=== Language with Highest Average for Each Metric ===\n")
    print(highest_avg)

    # Save the highest averages to a separate CSV file
    highest_avg.to_csv("4_highest_averages.csv", header=["Language"], index_label="Metric")

if __name__ == "__main__":
    compare_results(files)