import openai
import pandas as pd
import os
from dotenv import load_dotenv

progress_levels = {
    "B": "2nd grade (7yo)",
    "C": "4th grade (9yo)",
    "D": "6th grade (11yo)", 
    "E": "7th grade (12yo)",
    "F": "8th grade (13yo)",
    "G": "9th grade (14yo)",
    "H": "10th grade (15yo)"
}

def evaluate_explanations(files, output_file="ranked_explanation_evaluation.csv", model="gpt-4o-mini"):
    """
    Compares LLM-generated math solutions in different languages using ranking-based evaluation.

    Args:
    - files (dict): Dictionary of language keys and their corresponding CSV file paths.
    - output_file (str): Name of the CSV file to store evaluation results.
    - model (str): Name of the model used for generating solutions.

    Returns:
    - pd.DataFrame: DataFrame containing ranking-based evaluation results.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    
    # Load solutions into dataframes
    dfs = {lang: pd.read_csv(file) for lang, file in files.items()}

    # Ensure all dataframes have the same number of rows
    min_length = min(len(df) for df in dfs.values())  # Find the smallest dataframe
    for lang in dfs:
        dfs[lang] = dfs[lang].iloc[:min_length]  # Trim all dataframes to match

    # Function to rank three solutions using an LLM judge
    def rank_solutions(ex_index, solutions, langs, progress_level):
        progress_text = progress_levels.get(progress_level, "Unknown grade level")

        prompt = f"""
        You are an expert teacher trainer evaluating and ranking three math solutions, each explaining how to solve the same problem.

        **Exercise Index:** {ex_index}
        **Progress Level:** {progress_text}

        **Solution 1 ({langs[0]}):**
        {solutions[0]}

        **Solution 2 ({langs[1]}):**
        {solutions[1]}

        **Solution 3 ({langs[2]}):**
        {solutions[2]}

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

        **Ranking Instructions**:
        - Rank the solutions from **1st (best) to 3rd (worst)**.
        - Format your response strictly as follows:
          **Ranking:** [Lang1: X, Lang2: Y, Lang3: Z]  
          **Justification:** [Short explanation]  
        """

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.choices[0].message.content.strip()
            return result
        except Exception as e:
            print(f"Error processing exercise {ex_index}: {e}")
            return "Error"

    # Store evaluation results
    evaluation_results = []

    for ex_index in range(min_length):  # Iterate by row index, not exercise text
        try:
            solutions = [dfs[lang].iloc[ex_index][f"{model} solution"] for lang in files.keys()]
            langs = list(files.keys())
            progress_level = dfs[langs[0]].iloc[ex_index]["Progress Level"]

            # Get evaluation from LLM judge
            judge_response = rank_solutions(ex_index, solutions, langs, progress_level)

            if judge_response != "Error":
                # Extract ranking and justification
                response_lines = judge_response.split("\n")
                ranking = response_lines[0].replace("**Ranking:** ", "").strip()
                justification = response_lines[1].replace("**Justification:** ", "").strip()

                # Parse ranking into separate columns
                ranking_dict = {pair.split(":")[0].strip(): int(pair.split(":")[1].strip()) for pair in ranking.strip("[]").split(",")}

                evaluation_results.append({
                    "Exercise Index": ex_index,
                    "Progress Level": progress_levels.get(progress_level, "Unknown"),
                    "Best Explanation": min(ranking_dict, key=ranking_dict.get),
                    "Worst Explanation": max(ranking_dict, key=ranking_dict.get),
                    "Ranking": ranking,
                    "Justification": justification
                })
        except Exception as e:
            print(f"Skipping exercise {ex_index} due to error: {e}")

    # Convert results to dataframe and save
    results_df = pd.DataFrame(evaluation_results)
    results_df.to_csv(output_file, index=False)

    print(f"Ranking-based evaluation completed. Results saved to {output_file}")
    
    return results_df
