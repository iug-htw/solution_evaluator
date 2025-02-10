import pandas as pd

# Define the metrics to compare
METRICS = [
    "Problem Understanding (Comprehension)",
    "Clarity and Step-by-Step Explanation",
    "Accuracy of Process (Correctness of Steps)",
    "Correctness of Final Answer",
    "Learning Appropriateness",
    "Generalization",
    "Technical Terms Explanation",
    "Addressing Common Errors",
    "Appropriateness Based on Progress Level",
]

# File paths for the three languages
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
    df[METRICS] = df[METRICS].apply(pd.to_numeric, errors='coerce')  # Coerce non-numeric to NaN
    return df[METRICS].mean()

def compare_results(files, output_dir=""):
    """Compare evaluation results across languages."""
    # Load data for each language
    results = {}
    for lang, file_path in files.items():
        df = load_csv(file_path)
        results[lang] = calculate_metric_averages(df)

    # Combine results into a single DataFrame for comparison
    comparison_df = pd.DataFrame(results)

    # Save the comparison to a CSV file
    comparison_df.T.to_csv(f"{output_dir}comparison_results.csv", index=True)

    # Identify the language with the highest average for each metric
    highest_avg = comparison_df.idxmax(axis=1)
    print("\n=== Language with Highest Average for Each Metric ===\n")
    print(highest_avg)

    # Save the highest averages to a separate CSV file
    highest_avg.to_csv(f"{output_dir}highest_averages.csv", header=["Language"], index_label="Metric")

if __name__ == "__main__":
    compare_results(files)