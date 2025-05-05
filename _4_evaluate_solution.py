from openai import OpenAI
import csv
import os
import time
from dotenv import load_dotenv

def evaluate_solutions(input_file, terms_file, output_file, model="gpt-4o-mini"):
    """
    Evaluates AI-generated math solutions using an expanded scoring system (1-10 scale) 
    across multiple criteria.
    
    Args:
    - input_file (str): Path to the CSV file containing solutions.
    - terms_file (str): Path to the CSV file containing technical terms.
    - output_file (str): Path to the CSV file where evaluations will be saved.
    - model (str): LLM model to use for evaluation.

    Returns:
    - None (Writes results to CSV file)
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    # Define evaluation criteria
    criteria = [
        "Problem Understanding (Comprehension)",
        "Clarity and Step-by-Step Explanation",
        "Accuracy of Process (Correctness of Steps)",
        "Correctness of Final Answer",
        "Learning Appropriateness",
        "Generalization",
        "Technical Terms Explanation",
        "Addressing Common Errors",
        "Appropriateness Based on Progress Level",
        "Explanation Clarity & Syntax"
    ]

    # Define progress levels
    progress_levels = {
        "B": "2nd grade (7yo)",
        "C": "4th grade (9yo)",
        "D": "6th grade (11yo)", 
        "E": "7th grade (12 yo)",
        "F": "8th grade (13yo)",
        "G": "9th grade (14yo)",
        "H": "10th grade (15yo)"
    }
    
    # Read technical terms from the terms file
    technical_terms_dict = {}
    with open(terms_file, mode='r', encoding='utf-8') as termsfile:
        terms_reader = csv.reader(termsfile)
        next(terms_reader)  # Skip header
        for row in terms_reader:
            key = (row[0], row[1], row[2], row[3])  # (Topic Area, Topic, Progress Level, Exercise)
            technical_terms_dict[key] = row[4]
    
    # Open files
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new evaluation columns
        header = next(reader)
        writer.writerow(header[:-1] + criteria + ["Justification"])  # Exclude the solution column
        
        for i, row in enumerate(reader):
            topic_area, topic, progress_level, exercise, solution = row
            if i >= 400:
                break
            
            # Fetch relevant technical terms
            technical_terms = technical_terms_dict.get((topic_area, topic, progress_level, exercise), "")

            # Construct the evaluation prompt
            prompt = (
                f"You are a **very critical teacher trainer**. Critically evaluate the solution '{solution}' by a tutor "
                f"explaining how to solve the **math problem** '{exercise}' for students in level **{progress_levels[progress_level]}**. "
                f"Use the following **10 criteria**, scoring each on a scale from **1 (very poor) to 10 (excellent)**:\n\n"

                "**Evaluation Criteria:**\n"
                "1) **Problem Understanding (Comprehension)**\t- Does the solution correctly interpret the problem?\n"
                "- Are key terms/concepts properly introduced?\t0: Misunderstands or omits key elements\n"
                "10: Fully understands and clearly explains the problem\n"
                "2) **Clarity and Step-by-Step Explanation**\t- Are steps clearly outlined in a logical order?\n"
                "- Are any steps skipped or rushed?\t0: Steps unclear or missing\n"
                "10: Well-structured, detailed, and easy to follow\n"
                "3) **Accuracy of Process (Correctness of Steps)**\t- Is the method followed correctly?\n"
                "- Are all calculations/logical steps correct?\t0: Major mistakes in steps\n"
                "10: Fully correct and complete\n"
                "4) **Correctness of Final Answer**\t- Does the solution arrive at the correct final answer?\t0: Incorrect answer\n"
                "10: Correct answer with clear justification\n"
                "5) **Learning Appropriateness (Is the Explanation Suitable for Learners?)**\t- Is the explanation engaging, accessible, and structured for learning?\n"
                "- Does it anticipate common mistakes and clarify them?\t0: Too technical or difficult to follow\n"
                "10: Well-adapted for learners with helpful explanations\n"
                "6) **Generalization (Can the Learner Apply This Method to Similar Problems?)**\t- Does the solution give a generalizable method?\n"
                "- Does it explain the reasoning behind the approach?\t0: Too problem-specific, not generalizable\n"
                "10: Strongly generalizable method with clear reasoning\n"
                "7) **Technical Terms Explanation**\t- Are the technical terms '{technical_terms}' fully explained and correctly used?\t0: Terms not explained or used incorrectly\n"
                "10: All terms fully explained and correctly used\n"
                "8) **Addressing Common Errors**\t- Does the solution address common errors?\t0: Does not address common errors\n"
                "10: Fully addresses common errors with clear explanations\n"
                "9) **Appropriateness Based on Progress Level (Grade)**\t- Is the explanation appropriate for the given grade level (2nd-10th grade)?\n"
                "- Does the solution take into account the grade level of the exercise and adjust its language and depth accordingly?\t0: Explanation too advanced or too simple for the grade level\n"
                "10: Perfectly suited to the grade level with the right amount of detail and complexity\n"
                "10) **Explanation Clarity & Syntax**\t- Is the explanation grammatically correct and uses the relevant terminology of the target language?\t"
                "- Does the explanation use clear and concise language?\t0: Poor grammar, non-relevant terminology\n"
                "10: Perfect grammar, relevant terminology, and clear language\n\n"
                
                "**Scoring Instructions:**\n"
                "Provide a **single numeric score (1-10) per criterion**, separated by commas.\n"
                "Example response format: `1,2,5,2,3,3,4,1,4,5`\n\n"
                
                "**Justification:**\n"
                "Briefly justify the scores in **under 20 words** after a double line break."
            )

            print(f"Evaluating solution for task {i+1}: {exercise}")
            retry_count = 0
            while retry_count < 5:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    break
                except openai.error.RateLimitError:
                    retry_count += 1
                    print(f"Rate limit exceeded. Retrying in {retry_count * 10} seconds...")
                    time.sleep(retry_count * 10)
            else:
                print("Failed to get a response after multiple retries.")
                continue

            # Extract LLM-generated scores & justification
            evaluation = completion.choices[0].message.content.strip()
            try:
                scores, explanation = map(str.strip, evaluation.split('\n\n', 1))
                scores = scores.split(',')
                if len(scores) != 10:  # Ensure we get all 10 scores
                    raise ValueError("Invalid scoring format")
            except Exception as e:
                print(f"Error processing scores for task {i+1}: {e}")
                continue  # Skip this row if formatting fails

            writer.writerow(row[:-1] + scores + [explanation])  # Exclude the solution column

if __name__ == "__main__":
    input_file = '1_topic_areas_solutions.csv'
    terms_file = '2_topic_areas_technical_terms.csv'
    output_file = '3_topic_areas_evaluations.csv'
    evaluate_solutions(input_file, terms_file, output_file)
