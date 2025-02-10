import google.generativeai as genai
import csv
import os
import time
from dotenv import load_dotenv

def evaluate_solutions(input_file, terms_file, output_file):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel("gemini-1.5-flash")
    
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
        "Explanation"
    ]

    progress_levels = {
        "B": "2nd grade (7yo)",
        "C": "4th grade (9yo)",
        "D": "6th grade (11yo)", 
        "E": "7th grade (12 yo)",
        "F": "8th grade (13yo)",
        "G": "9th grade (14yo)",
        "H": "10th grade (15yo)"
    }
    
    # Read technical terms from terms_file
    technical_terms_dict = {}
    with open(terms_file, mode='r', encoding='utf-8') as termsfile:
        terms_reader = csv.reader(termsfile)
        next(terms_reader)  # Skip header
        for row in terms_reader:
            key = (row[0], row[1], row[2], row[3])  # (Topic Area, Topic, Progress Level, Exercise)
            technical_terms_dict[key] = row[4]
    
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new columns for evaluation criteria
        header = next(reader)
        writer.writerow(header[:-1] + criteria)  # Exclude the solution column
        
        for i, row in enumerate(reader):
            if i >= 400:
                break
            topic_area, topic, progress_level, exercise, solution = row
            technical_terms = technical_terms_dict.get((topic_area, topic, progress_level, exercise), "")
            
            prompt = (
                f"You are a very critical teacher trainer. Critically evaluate the solution '{solution}' to the math problem '{exercise}' "
                f"for progress level '{progress_level}' ({progress_levels[progress_level]}) using these criteria:\n"
                "1. Problem Understanding (0-2 points)\n"
                "2. Clarity and Step-by-Step Explanation (0-2 points)\n"
                "3. Accuracy of Process (0-2 points)\n"
                "4. Correctness of Final Answer (0-2 points)\n"
                "5. Learning Appropriateness (0-2 points)\n"
                "6. Generalization (0-2 points)\n"
                f"7. Technical Terms Explanation (for terms: {technical_terms}) (0-2 points)\n"
                "8. Addressing Common Errors (0-2 points)\n"
                "9. Grade Level Appropriateness (0-2 points)\n\n"
                "Provide scores as comma-separated numbers (e.g., 2,2,2,2,2,2,2,2,2)\n"
                "Then provide a brief explanation (max 20 words) separated by a double line break."
            )
            
            print(f"Evaluating solution for task {i+1}: {exercise}")
            try:
                response = client.generate_content(prompt)
                evaluation = response.text.strip()
                scores, explanation = map(str.strip, evaluation.split('\n\n', 1))
                scores = scores.split(',')
                writer.writerow(row[:-1] + scores + [explanation])  # Exclude the solution column
            except Exception as e:
                print(f"Error evaluating task {i+1}: {e}")
                continue
            
            # Add a delay of 5 seconds between requests
            time.sleep(5)

if __name__ == "__main__":
    input_file = 'gemini/topic_areas_solutions.csv'
    terms_file = 'gemini/topic_areas_technical_terms.csv'
    output_file = 'gemini/topic_areas_evaluations.csv'
    evaluate_solutions(input_file, terms_file, output_file) 