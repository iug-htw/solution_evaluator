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

def parse_ranking(ranking_str):
    """
    Safely parses a ranking string into a dictionary.

    Parameters
    ----------
    ranking_str : str
        A string representation of a ranking dictionary,
        e.g. "{'en': 1, 'de': 2, 'ar': 3}"

    Returns
    -------
    dict
        Parsed dictionary of language → rank mappings.
        Returns an empty dict if parsing fails.
    """
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
    df = pd.read_csv(in_file)
    df = df.sort_values(by='Exercise Index')

    # Step 2: Parse ranking strings into dictionaries
    for model in ['gpt-4o-mini', 'gemini-2.5-flash', 'qwen-plus', 'claude-3-5-haiku']:
        df[f'{model} Ranking'] = df[f'{model} Ranking'].apply(parse_ranking)

    # Step 3: Determine majority rankings
    best_list, mid_list, worst_list = [], [], []

    for _, row in df.iterrows():
        rankings = [row[f'{model} Ranking'] for model in ['gpt-4o-mini', 'gemini-2.5-flash', 'qwen-plus', 'claude-3-5-haiku']]
        
        # Aggregate ranks for each language
        rank_aggregate = {'en': [], 'de': [], 'ar': []}
        for ranking in rankings:
            if not ranking:
                continue
            for lang, rank in ranking.items():
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
    df_to_save = df[['Exercise Index', 'Progress Level',
                     'gpt-4o-mini Ranking', 'gemini-2.5-flash Ranking',
                     'qwen-plus Ranking', 'claude-3-5-haiku Ranking',
                     'Best', 'Mid', 'Worst']]
    df_to_save.to_csv(out_file, index=False)

    # Step 5: Visualize with a heatmap
    rank_counts = {'Best': Counter(df['Best']), 'Mid': Counter(df['Mid']), 'Worst': Counter(df['Worst'])}
    rank_df = pd.DataFrame(rank_counts).fillna(0).astype(int)

    # Reorder rows for consistent heatmap display
    rank_df = rank_df.reindex(['en', 'de', 'ar', 'TIE'])

    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_df, annot=True, cmap='coolwarm', cbar=False, fmt='d')
    plt.title(f"{solving_model}{' ' if solving_model else ''}Majority Vote Ranking Heatmap")
    plt.xlabel('Ranking Position')
    plt.ylabel('Language')
    plt.show()
