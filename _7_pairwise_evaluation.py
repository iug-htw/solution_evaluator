import openai
import google.generativeai as genai
import pandas as pd
import os
import re
import csv
import random
import time
from dotenv import load_dotenv
from collections import Counter

progress_levels = {
    "B": "2nd grade (7yo)",
    "C": "4th grade (9yo)",
    "D": "6th grade (11yo)", 
    "E": "7th grade (12 yo)",
    "F": "8th grade (13yo)",
    "G": "9th grade (14yo)",
    "H": "10th grade (15yo)"
}

LLM_MODELS = {
    "gpt-4o-mini": "openai",
    "gemini-1.5-flash": "google",
    "qwen-plus": "openai"
}

def get_llm_client(model_name):
    """Returns the appropriate client for each LLM."""
    load_dotenv()
    
    if model_name == "gpt-4o-mini":
        api_key = os.getenv("OPENAI_API_KEY")
        return openai.OpenAI(api_key=api_key)

    elif model_name == "gemini-1.5-flash":
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")

    elif model_name == "qwen-plus":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        return openai.OpenAI(
            api_key=api_key, 
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

def rank_solutions(ex_index, solutions, shuffled_langs, progress_level, exercise_terms, model):
    """
    Gets ranking evaluation from a specific LLM, ensuring explanations properly use and define technical terms.

    Args:
    - ex_index (int): Exercise index.
    - solutions (list): List of solutions for each language.
    - langs (list): List of language names corresponding to solutions.
    - progress_level (str): Progress level of the exercise.
    - model (str): LLM model name.
    - exercise_terms (dict): Dictionary mapping each language to its relevant technical terms.

    Returns:
    - str: LLM response containing the ranking.
    """

    client = get_llm_client(model)
    progress_text = progress_levels.get(progress_level, "Unknown grade level")

    prompt = f"""
    You are an expert teacher trainer evaluating and ranking three math solutions, each explaining how to solve the same problem.

    **Exercise Index:** {ex_index}
    **Progress Level:** {progress_text}

    **Solution 1 ({shuffled_langs[0]}):**
    {solutions[0]}

    **Solution 2 ({shuffled_langs[1]}):**
    {solutions[1]}

    **Solution 3 ({shuffled_langs[2]}):**
    {solutions[2]}

    **Technical Terms Required for Understanding:**
    - {exercise_terms[shuffled_langs[0]]} (for {shuffled_langs[0]})
    - {exercise_terms[shuffled_langs[1]]} (for {shuffled_langs[1]})
    - {exercise_terms[shuffled_langs[2]]} (for {shuffled_langs[2]})

    **Evaluation Criteria**:
    - Which explanation shows the best problem understanding?
    - Which explanation is the clearest for students?
    - Which solution provides the best step-by-step breakdown?
    - Which one uses the best math terminology?
    - Which explanation provides the most accurate final answer?
    - Which one avoids common mistakes and explains them well?
    - Which explanation is best suited for learning?
    - Which explanation is most generalizable to similar problems?
    - Which explanation is the most appropriate for the given progress level?
    - Which explanation best incorporates and explains the required technical terms?

    **Ranking Instructions**:
    - Rank the solutions from **1st (best) to 3rd (worst)**.
    - Format your response strictly as follows:
    **Ranking:** [{shuffled_langs[0]}: X, {shuffled_langs[1]}: Y, {shuffled_langs[2]}: Z]  
    **Justification:** [Short explanation]  
    """

    try:
        if model == "gemini-1.5-flash":
            response = client.generate_content(prompt)
            result = response.text.strip()

        elif model == "gpt-4o-mini" or model == "qwen-plus":
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            result = completion.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unknown model: {model}")

        return result

    except Exception as e:
        print(f"Error processing exercise {ex_index} with {model}: {e}")
        return "Error"

def ordinal_to_int(ordinal_str):
    """Convert ordinal numbers (1st, 2nd, 3rd) into integers."""
    match = re.match(r"(\d+)", ordinal_str.strip())  # Extract leading digits
    return int(match.group(1)) if match else None  # Convert to int

def evaluate_explanations(files, technical_terms_files, current_model="gpt-4o-mini", output_dir=""):
    """
    Compares LLM-generated math solutions using ranking-based evaluation with majority voting.
    Writes results incrementally to CSV after each row.

    Args:
    - files (dict): Dictionary of language keys and their corresponding CSV file paths.
    - technical_terms_files (dict): Dictionary of language keys and their corresponding technical terms CSV file paths.
    - output_dir (str): Directory path where results should be saved.

    Returns:
    - pd.DataFrame: DataFrame containing ranking-based evaluation results.
    """
    
    output_file = os.path.join(output_dir, "judge_pairwise_evaluation.csv")

    # Load solution and technical terms data
    dfs = {lang: pd.read_csv(file) for lang, file in files.items()}
    terms_dfs = {lang: pd.read_csv(file) for lang, file in technical_terms_files.items()} 

    # Ensure all dataframes have the same number of rows
    min_length = min(len(df) for df in dfs.values())
    for lang in dfs:
        dfs[lang] = dfs[lang].iloc[:min_length]
        terms_dfs[lang] = terms_dfs[lang].iloc[:min_length]  

    # Define CSV headers
    fieldnames = [
        "Exercise Index",
        "Solution 1 Language",
        "Solution 2 Language",
        "Solution 3 Language",
        "Progress Level",
        "Best Explanation",
        "Worst Explanation",
        "gpt-4o-mini Ranking",
        "gemini-1.5-flash Ranking",
        "qwen-plus Ranking",
        "Majority Vote Ranking",
        "Justification gpt-4o-mini",
        "Justification gemini-1.5-flash",
        "Justification qwen-plus"
    ]

    # Check if output file exists to determine whether to write headers
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write headers only if file is new

        for ex_index in range(min_length):
            if ex_index not in [274, 276, 279, 282, 285, 288, 291, 294, 297, 267, 17,  21,  24,  30,  31,  33,  36,  39,  42,  45, 49,  51,  53,  55,  56,  58,  63,  66,  84,  85, 86,  87,  88,  90, 100, 105, 108, 117, 132, 135, 138, 159, 162, 168, 174, 205, 260, 224, 228, 191]:
                continue         
            try:
                print(f"Evaluating exercise {ex_index}...", end=" ")

                # Shuffle language order for this exercise
                shuffled_langs = list(files.keys())
                random.shuffle(shuffled_langs)  # Randomize the order for each exercise

                # Retrieve solutions in the shuffled order
                solutions = [dfs[lang].iloc[ex_index][f"{current_model} solution"] for lang in shuffled_langs]
                progress_level = dfs[shuffled_langs[0]].iloc[ex_index]["Progress Level"]

                # Extract relevant technical terms for the current exercise
                exercise_terms = {
                    lang: terms_dfs[lang].iloc[ex_index]["Technical Terms"]
                    if ex_index < len(terms_dfs[lang]) else "No specific terms"
                    for lang in shuffled_langs
                }

                # Get evaluations from all models
                rankings = {}
                justifications = {}

                for model in LLM_MODELS.keys():
                    judge_response = rank_solutions(ex_index, solutions, shuffled_langs, progress_level, exercise_terms, model)

                    if judge_response != "Error":
                        response_lines = judge_response.split("\n")
                        ranking = response_lines[0].replace("**Ranking:** ", "").strip()
                        justification = response_lines[1].replace("**Justification:** ", "").strip()

                        ranking_dict = {pair.split(":")[0].strip(): ordinal_to_int(pair.split(":")[1].strip()) for pair in ranking.strip("[]").split(",")}
                        rankings[model] = ranking_dict
                        justifications[model] = justification

                # Reverse map rankings to the original language names
                reverse_map = {shuffled_langs[i]: list(files.keys())[i] for i in range(len(shuffled_langs))}
                mapped_rankings = {
                    model: {reverse_map[lang]: rank for lang, rank in rankings[model].items()} for model in rankings
                }

                # Majority voting for best & worst explanation
                best_votes = Counter([min(mapped_rankings[model], key=mapped_rankings[model].get) for model in mapped_rankings])
                worst_votes = Counter([max(mapped_rankings[model], key=mapped_rankings[model].get) for model in mapped_rankings])

                best_explanation = best_votes.most_common(1)[0][0] if best_votes else "N/A"
                worst_explanation = worst_votes.most_common(1)[0][0] if worst_votes else "N/A"

                row_data = {
                    "Exercise Index": ex_index,
                    "Solution 1 Language": shuffled_langs[0],
                    "Solution 2 Language": shuffled_langs[1],
                    "Solution 3 Language": shuffled_langs[2],
                    "Progress Level": progress_levels.get(progress_level, "Unknown"),
                    "Best Explanation": best_explanation,
                    "Worst Explanation": worst_explanation,
                    "gpt-4o-mini Ranking": mapped_rankings.get("gpt-4o-mini", {}),
                    "gemini-1.5-flash Ranking": mapped_rankings.get("gemini-1.5-flash", {}),
                    "qwen-plus Ranking": mapped_rankings.get("qwen-plus", {}),
                    "Majority Vote Ranking": best_explanation,
                    "Justification gpt-4o-mini": justifications.get("gpt-4o-mini", ""),
                    "Justification gemini-1.5-flash": justifications.get("gemini-1.5-flash", ""),
                    "Justification qwen-plus": justifications.get("qwen-plus", "")
                }

                # Append new row to CSV file
                writer.writerow(row_data)
                print(f"✅ Saved exercise {ex_index} to CSV.")

            except Exception as e:
                print(f"⚠️ Skipping exercise {ex_index} due to error: {e}")

            time.sleep(2)

    print(f"Ranking-based evaluation completed. Results saved to {output_file}")
    return pd.read_csv(output_file)
