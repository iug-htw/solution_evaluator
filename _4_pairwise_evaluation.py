"""
_4_pairwise_evaluation.py

This script performs **pairwise evaluation** of multilingual math solutions
using a panel of LLM judges. For each exercise, three solutions
(English, German, Arabic) are ranked by different LLMs according to pedagogical
criteria. Rankings are aggregated with **majority voting** to identify the best
and worst explanations.

Judges:
- GPT-4o-mini (OpenAI)
- Gemini-2.5-Flash (Google)
- Qwen-Plus (Alibaba Cloud / Dashscope)
- Claude 3.5 Haiku (Anthropic)
"""

import openai
import google.generativeai as genai
import anthropic
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
    "gemini-2.5-flash": "google",
    "qwen-plus": "openai",
    "claude-3-5-haiku": "anthropic"
}


# -----------------------------------------------------------------------------------
# Helper Function
# -----------------------------------------------------------------------------------

def get_llm_client(model_name):
    """
    Returns the appropriate client for each LLM.

    NOTE: Extend to more API clients by updating this function
    """

    load_dotenv()
    
    if model_name == "gpt-4o-mini":
        api_key = os.getenv("OPENAI_API_KEY")
        return openai.OpenAI(api_key=api_key)

    elif model_name == "gemini-2.5-flash":
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")

    elif model_name == "qwen-plus":
        api_key = os.getenv("DASHSCOPE_API_KEY")
        return openai.OpenAI(
            api_key=api_key, 
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    
    elif model_name == "claude-3-5-haiku":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return anthropic.Anthropic(api_key=api_key)

    else:
        raise ValueError(f"Unknown model: {model_name}")

# NOTE: This function currently supports 3 solutions/languages comparison.
# In case of more languages added, update the prompt used here accordingly
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
    - Provide a short justification for your ranking avoiding bullet points.
    - Format your response strictly as follows:
    **Ranking:** [{shuffled_langs[0]}: X, {shuffled_langs[1]}: Y, {shuffled_langs[2]}: Z]  
    **Justification:** [Short explanation]  
    """

    try:
        if model == "gemini-2.5-flash":
            response = client.generate_content(prompt)
            result = response.text.strip()

        elif model == "claude-3-5-haiku":
            completion = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=8192,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            result = "".join([part.text for part in completion.content]).strip()

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


# -----------------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------------

def evaluate_explanations(files, technical_terms_files, current_model="gpt-4o-mini", output_dir="", tasks_indices=None):
    """
    Run a held-out, pairwise-style evaluation of multilingual solutions using multiple LLM judges,
    and write the results to a cumulative CSV file.

    For each exercise index, this function:
    - loads the pre-generated solutions for each language (from `files`)
    - loads the extracted technical terms per exercise (from `technical_terms_files`)
    - randomizes the order of languages per exercise to reduce position/order bias
    - asks a panel of judge models (all models in `LLM_MODELS` except `current_model`, plus
      `claude-3-5-haiku`) to rank the three solutions and provide a short justification
    - aggregates judge rankings via majority vote for “best” and “worst”
    - appends a row to `judge_pairwise_evaluation.csv` under `output_dir`

    The output file is appended to if it already exists, allowing incremental runs. If it does
    not exist, a header row is created.

    Parameters
    ----------
    files : dict[str, str]
        Mapping from language label to CSV path containing solutions for that language.
        Example:
            {
              "en": "solutions_en.csv",
              "de": "solutions_de.csv",
              "ar": "solutions_ar.csv"
            }
        NOTE: to extend to more languages, you simply add a new entry to the files dict
              Example: { "french": "solutions_fr.csv" }

    technical_terms_files : dict[str, str]
        Mapping from language label to CSV path containing the extracted technical terms
        corresponding to the same exercises as `files`.
        Each CSV is expected to contain a "Technical Terms" column.

    current_model : str, optional
        The model whose solutions are being evaluated. This model is excluded from the judge
        panel to avoid “self-judging”.
        Defaults to "gpt-4o-mini".

    output_dir : str, optional
        Directory where the output CSV will be created/appended. The file name is fixed:
        "judge_pairwise_evaluation.csv".
        Defaults to "" (current working directory).

    tasks_indices : iterable[int] or None, optional
        Zero-based exercise indices (after header) to evaluate. If None, all exercises up to the
        shortest dataframe length across languages are evaluated.

    Output format
    -------------
    The CSV contains (among others) the following columns:
    - Exercise Index
    - Solution 1/2/3 Language (shuffled per exercise)
    - Progress Level (mapped via `progress_levels`)
    - Best Explanation / Worst Explanation (majority vote; "TIE" if all judges disagree)
    - Per-judge rankings for: gpt-4o-mini, gemini-2.5-flash, qwen-plus, claude-3-5-haiku
    - Per-judge justifications for the same set
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
        "gemini-2.5-flash Ranking",
        "qwen-plus Ranking",
        "claude-3-5-haiku Ranking",
        "Majority Vote Ranking",
        "Justification gpt-4o-mini",
        "Justification gemini-2.5-flash",
        "Justification qwen-plus",
        "Justification claude-3-5-haiku"
    ]

    # Check if output file exists to determine whether to write headers
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode="a", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write headers only if file is new

        for ex_index in range(min_length):       
            if tasks_indices is not None and ex_index not in tasks_indices:
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

                judge_models = [m for m in LLM_MODELS if m != current_model]
                if "claude-3-5-haiku" not in judge_models:
                    judge_models.append("claude-3-5-haiku")

                for model in judge_models:
                    judge_response = rank_solutions(ex_index, solutions, shuffled_langs, progress_level, exercise_terms, model)

                    ranking = None
                    justification = None

                    if judge_response != "Error":
                        response_lines = judge_response.split("\n")
                        for line in response_lines:
                            if line.strip().startswith("**Ranking:**"):
                                ranking = line.replace("**Ranking:**", "").strip()
                            elif "**justification:**" in line.lower():
                                justification_start = response_lines.index(line)
                                justification_raw = "\n".join(response_lines[justification_start:]).split(":", 1)[-1].strip()

                                justification = (
                                    justification_raw
                                    .lstrip("*–- ") 
                                    .replace('"', '""') 
                                    .replace('""""', '""') 
                                    .strip()
                                )
                                break

                        ranking_dict = {pair.split(":")[0].strip(): ordinal_to_int(pair.split(":")[1].strip()) for pair in ranking.strip("[]").split(",")}
                        rankings[model] = ranking_dict
                        justifications[model] = justification

                # Reverse map rankings to the original language names
                mapped_rankings = {
                    model: rankings[model] for model in rankings
                }

                # Majority voting for best & worst explanation
                best_votes = Counter([min(mapped_rankings[model], key=mapped_rankings[model].get) for model in mapped_rankings])
                worst_votes = Counter([max(mapped_rankings[model], key=mapped_rankings[model].get) for model in mapped_rankings])

                # Check for ties (if all models voted differently)
                best_explanation = "TIE" if len(best_votes) == 3 else best_votes.most_common(1)[0][0]
                worst_explanation = "TIE" if len(worst_votes) == 3 else worst_votes.most_common(1)[0][0]

                row_data = {
                    "Exercise Index": ex_index,
                    "Solution 1 Language": shuffled_langs[0],
                    "Solution 2 Language": shuffled_langs[1],
                    "Solution 3 Language": shuffled_langs[2],
                    "Progress Level": progress_levels.get(progress_level, "Unknown"),
                    "Best Explanation": best_explanation,
                    "Worst Explanation": worst_explanation,
                    "gpt-4o-mini Ranking": rankings.get("gpt-4o-mini", {}),
                    "gemini-2.5-flash Ranking": rankings.get("gemini-2.5-flash", {}),
                    "qwen-plus Ranking": rankings.get("qwen-plus", {}),
                    "claude-3-5-haiku Ranking": rankings.get("claude-3-5-haiku", {}),
                    "Majority Vote Ranking": best_explanation,
                    "Justification gpt-4o-mini": justifications.get("gpt-4o-mini", ""),
                    "Justification gemini-2.5-flash": justifications.get("gemini-2.5-flash", ""),
                    "Justification qwen-plus": justifications.get("qwen-plus", ""),
                    "Justification claude-3-5-haiku": justifications.get("claude-3-5-haiku", "")
                }

                # Append new row to CSV file
                writer.writerow(row_data)
                print(f"✅ Saved exercise {ex_index} to CSV.")

            except Exception as e:
                print(f"⚠️ Skipping exercise {ex_index} due to error: {e}")

            time.sleep(2)

    print(f"Ranking-based evaluation completed. Results saved to {output_file}")
    return pd.read_csv(output_file)

if __name__ == "__main__":
    solution_files = {
        "en": "gpt_4o_mini/solutions_en.csv",
        "de": "gpt_4o_mini/solutions_de.csv",
        "ar": "gpt_4o_mini/solutions_ar.csv",
    }

    technical_terms_files = {
        "en": "technical_terms_en.csv",
        "de": "technical_terms_de.csv",
        "ar": "technical_terms_ar.csv",
    }

    evaluate_explanations(solution_files, technical_terms_files, output_dir="gpt_4o_mini", current_model="gpt-4o-mini")