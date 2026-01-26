"""
_5_pairwise_results_clean.py

This script cleans and aggregates the raw pairwise evaluation results into
majority-vote rankings, then visualizes the distribution of rankings
across languages with a heatmap.

Pipeline role:
- Consumes raw outputs from `_4_pairwise_evaluation.py`
- Parses ranking strings into structured dictionaries
- Applies majority voting across judge models
- Produces a cleaned CSV with aggregated best/mid/worst rankings
- Generates a heatmap to visualize ranking distributions
"""

import pandas as pd
import ast
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# NOTE: update following array to extend to more llm judges.
# The fuction uses this array to look up different judges responses in the input csv file,
# so add the names of your judges here.
JUDGES = ['gpt-4o-mini', 'gemini-2.5-flash', 'qwen-plus', 'claude-3-5-haiku']

def parse_ranking(ranking_str):
    """ Safely parses a ranking string into a dictionary. """
    try:
        return ast.literal_eval(ranking_str)
    except (ValueError, SyntaxError):
        return {}

def pairwise_results_clean(file_dir="", solving_model=""):
    """
    Cleans pairwise evaluation results and generates summary outputs.

    Parameters
    ----------
    file_dir : str, optional
        Directory containing the input file `judge_pairwise_evaluation.csv`.
        Default is current directory.
    solving_model : str, optional
        Name of the solving model being evaluated (for heatmap title).
        Example: "gpt-4o-mini".

    Behavior
    --------
    Step 1. Load and sort data  
        - Reads raw pairwise evaluation results.  
        - Sorts rows by exercise index.  

    Step 2. Parse rankings  
        - Converts string-formatted rankings into dictionaries for each judge model.  

    Step 3. Majority voting  
        - Aggregates rankings across models.  
        - Determines "Best", "Mid", and "Worst" languages per exercise.  
        - Labels ties explicitly as "TIE".  

    Step 4. Save processed results  
        - Writes a cleaned CSV `pairwise_results_cleaned.csv` containing:  
          [Exercise Index, Progress Level, Rankings per model, Best, Mid, Worst].  

    Step 5. Visualization  
        - Counts occurrences of each language in best/mid/worst categories.  
        - Produces a heatmap showing ranking distributions across languages.  

    Output
    ------
    - CSV file: `pairwise_results_cleaned.csv`
    - Heatmap: Majority vote rankings by language
    """

    # Step 1: Load and sort the data
    in_file = os.path.join(file_dir, "judge_pairwise_evaluation.csv")
    out_file = os.path.join(file_dir, "pairwise_results_cleaned.csv")
    df = pd.read_csv(in_file).sort_values(by="Exercise Index")

    # Step 2: Parse ranking strings into dictionaries
    for model in JUDGES:
        df[f'{model} Ranking'] = df[f'{model} Ranking'].apply(parse_ranking)

    # --- Infer languages from the rankings across all judges/rows ---
    inferred_langs = set()
    for model in JUDGES:
        col = f"{model} Ranking"
        inferred_langs |= set().union(*df[col].dropna().apply(lambda d: d.keys() if isinstance(d, dict) else []))

    # Keep deterministic order
    inferred_langs = sorted(inferred_langs)

    best_list, mid_list, worst_list = [], [], []

    for _, row in df.iterrows():
        rankings = [row[f'{model} Ranking'] for model in JUDGES]

        # --- NEW: dynamic aggregator ---
        rank_aggregate = {lang: [] for lang in inferred_langs}

        for ranking in rankings:
            if not ranking:
                continue
            for lang, rank in ranking.items():
                # In case a judge returns a language you haven't seen elsewhere
                if lang not in rank_aggregate:
                    rank_aggregate[lang] = []
                rank_aggregate[lang].append(rank)
        
        # Determine majority rank for each position
        best_langs = [lang for lang, ranks in rank_aggregate.items() if ranks.count(1) > 1]
        mid_langs = [lang for lang, ranks in rank_aggregate.items() if ranks.count(2) > 1]
        worst_langs = [lang for lang, ranks in rank_aggregate.items() if ranks.count(3) > 1]
        
        best_list.append(best_langs[0] if len(best_langs) == 1 else 'TIE')
        mid_list.append(mid_langs[0] if len(mid_langs) == 1 else 'TIE')
        worst_list.append(worst_langs[0] if len(worst_langs) == 1 else 'TIE')

    df['Best'] = best_list
    df['Mid'] = mid_list
    df['Worst'] = worst_list

    # Step 4: Save the processed data
    base_cols = ['Exercise Index', 'Progress Level']
    ranking_cols = [f"{judge} Ranking" for judge in JUDGES]
    outcome_cols = ['Best', 'Mid', 'Worst']

    df_to_save = df[base_cols + ranking_cols + outcome_cols]
    df_to_save.to_csv(out_file, index=False)

    # Step 5: Visualize with a heatmap
    rank_counts = {'Best': Counter(df['Best']), 'Mid': Counter(df['Mid']), 'Worst': Counter(df['Worst'])}
    rank_df = pd.DataFrame(rank_counts).fillna(0).astype(int)

    heatmap_order = inferred_langs + (["TIE"] if "TIE" in rank_df.index else [])
    rank_df = rank_df.reindex(heatmap_order).fillna(0).astype(int)

    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_df, annot=True, cmap='coolwarm', cbar=False, fmt='d')
    plt.title(f"{solving_model}{' ' if solving_model else ''}Majority Vote Ranking Heatmap")
    plt.xlabel('Ranking Position')
    plt.ylabel('Language')
    plt.show()

if __name__ == "__main__":
    pairwise_results_clean(file_dir="gpt_4o_mini", solving_model="gpt-4o-mini")