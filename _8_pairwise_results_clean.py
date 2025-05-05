import pandas as pd
import ast
import os
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

def parse_ranking(ranking_str):
    try:
        return ast.literal_eval(ranking_str)
    except (ValueError, SyntaxError):
        return {}

def pairwise_results_clean(file_dir="", solving_model=""):
    # Step 1: Load and sort the data
    in_file = os.path.join(file_dir, "judge_pairwise_evaluation.csv")
    out_file = os.path.join(file_dir, "pairwise_results_cleaned.csv")
    df = pd.read_csv(in_file)
    df = df.sort_values(by='Exercise Index')

    # Step 2: Parse ranking strings into dictionaries
    for model in ['gpt-4o-mini', 'gemini-1.5-flash', 'qwen-plus']:
        df[f'{model} Ranking'] = df[f'{model} Ranking'].apply(parse_ranking)

    # Step 3: Determine majority rankings
    best_list, mid_list, worst_list = [], [], []

    for _, row in df.iterrows():
        rankings = [row[f'{model} Ranking'] for model in ['gpt-4o-mini', 'gemini-1.5-flash', 'qwen-plus']]
        
        # Aggregate ranks for each language
        rank_aggregate = {'en': [], 'de': [], 'ar': []}
        for ranking in rankings:
            for lang, rank in ranking.items():
                rank_aggregate[lang].append(rank)
        
        # Determine majority rank for each position
        best_langs = [lang for lang, ranks in rank_aggregate.items() if ranks.count(1) > 1]
        mid_langs = [lang for lang, ranks in rank_aggregate.items() if ranks.count(2) > 1]
        worst_langs = [lang for lang, ranks in rank_aggregate.items() if ranks.count(3) > 1]
        
        if len(best_langs) == 1:
            best_list.append(best_langs[0])
        else:
            best_list.append('TIE')
        
        if len(mid_langs) == 1:
            mid_list.append(mid_langs[0])
        else:
            mid_list.append('TIE')
        
        if len(worst_langs) == 1:
            worst_list.append(worst_langs[0])
        else:
            worst_list.append('TIE')

    df['Best'] = best_list
    df['Mid'] = mid_list
    df['Worst'] = worst_list

    # Step 4: Save the processed data
    df_to_save = df[['Exercise Index', 'Progress Level', 'gpt-4o-mini Ranking', 'gemini-1.5-flash Ranking', 'qwen-plus Ranking', 'Best', 'Mid', 'Worst']]
    df_to_save.to_csv(out_file, index=False)

    # Step 5: Visualize with a heatmap
    # Count occurrences of each language in each rank
    rank_counts = {'Best': Counter(df['Best']), 'Mid': Counter(df['Mid']), 'Worst': Counter(df['Worst'])}
    rank_df = pd.DataFrame(rank_counts).fillna(0).astype(int)

    # Reorder rows for consistent heatmap display
    rank_df = rank_df.reindex(['en', 'de', 'ar', 'TIE'])

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(rank_df, annot=True, cmap='coolwarm', cbar=False, fmt='d')
    plt.title(f'{solving_model}{' ' if solving_model else ''}Majority Vote Ranking Heatmap')
    plt.xlabel('Ranking Position')
    plt.ylabel('Language')
    plt.show()
