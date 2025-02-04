from openai import OpenAI
import csv
import os
from dotenv import load_dotenv

def evaluate_solutions(input_file, output_file, model="gpt-4o"):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    criteria = [
        "Problem Understanding (Comprehension)",
        "Clarity and Step-by-Step Explanation",
        "Accuracy of Process (Correctness of Steps)",
        "Correctness of Final Answer",
        "Learning Appropriateness (Is the Explanation Suitable for Learners?)",
        "Generalization (Can the Learner Apply This Method to Similar Problems?)",
        "Explanation"
    ]
    
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new columns for evaluation criteria
        header = next(reader)
        writer.writerow(header[:-1] + criteria)  # Exclude the solution column
        
        for i, row in enumerate(reader):
            topic_area, topic, progress_level, exercise, solution = row
            prompt = (
                f"You are a very demanding teacher trainer. Critically evaluate the solution '{solution}' by a tutor to the problem '{exercise}' using the following 6 criteria:\n"
                "Criterion\tKey Aspects\tScoring (0-2 points per criterion)\n"
                "1) Problem Understanding (Comprehension)\t- Does the solution correctly interpret the problem?\n"
                "- Are key terms/concepts properly introduced?\t0: Misunderstands or omits key elements\n"
                "1: Understands the problem but lacks explanation\n"
                "2: Fully understands and clearly explains the problem\n"
                "2) Clarity and Step-by-Step Explanation\t- Are steps clearly outlined in a logical order?\n"
                "- Are any steps skipped or rushed?\t0: Steps unclear or missing\n"
                "1: Steps provided but could be clearer\n"
                "2: Well-structured, detailed, and easy to follow\n"
                "3) Accuracy of Process (Correctness of Steps)\t- Is the method followed correctly?\n"
                "- Are all calculations/logical steps correct?\t0: Major mistakes in steps\n"
                "1: Minor mistakes or skipped steps\n"
                "2: Fully correct and complete\n"
                "4) Correctness of Final Answer\t- Does the solution arrive at the correct final answer?\t0: Incorrect answer\n"
                "1: Correct answer but with unclear justification\n"
                "2: Correct answer with clear justification\n"
                "5) Learning Appropriateness (Is the Explanation Suitable for Learners?)\t- Is the explanation engaging, accessible, and structured for learning?\n"
                "- Does it anticipate common mistakes and clarify them?\t0: Too technical or difficult to follow\n"
                "1: Understandable but could be clearer\n"
                "2: Well-adapted for learners with helpful explanations\n"
                "6) Generalization (Can the Learner Apply This Method to Similar Problems?)\t- Does the solution give a generalizable method?\n"
                "- Does it explain the reasoning behind the approach?\t0: Too problem-specific, not generalizable\n"
                "1: Some generalizability, but limited explanation\n"
                "2: Strongly generalizable method with clear reasoning\n"
                "Provide a score from 0 to 2 for each criterion, separated by commas (e.g., 2,2,2,2,2,2)\n"
                "Provide a very short text (not more than 20 words) at the end to justify your scores."
            )
            print(f"Evaluating solution for task {i+1}: {exercise}")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            evaluation = completion.choices[0].message.content.strip()
            scores, explanation = evaluation.split('\n')[-1].split(' ', 1)
            scores = scores.split(',')
            writer.writerow(row[:-1] + scores + [explanation])  # Exclude the solution column

if __name__ == "__main__":
    input_file = '1_topic_areas_solutions.csv'
    output_file = '3_topic_areas_evaluations.csv'
    evaluate_solutions(input_file, output_file)
