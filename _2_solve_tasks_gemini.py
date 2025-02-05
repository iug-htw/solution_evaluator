import google.generativeai as genai
import csv
import os
import time
from dotenv import load_dotenv

def solve_tasks(input_file, output_file, model="gemini-1.5-flash"):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel("gemini-1.5-flash")
    
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new column for solutions
        header = next(reader)
        writer.writerow(header + [f"{model} solution"])
        
        for i, row in enumerate(reader):
            if i > 10:
                break
            topic_area, topic, progress_level, exercise = row
            prompt = f"Explain how to solve this task: {exercise}"
            print(f"Solving task {i+1}: {exercise}")
    
            response = client.generate_content(prompt)

            solution = response.text.strip()
            writer.writerow(row + [solution])

if __name__ == "__main__":
    input_file = 'topic_areas_cleaned.csv'
    output_file = 'gemini/topic_areas_solutions_en.csv'
    solve_tasks(input_file, output_file)