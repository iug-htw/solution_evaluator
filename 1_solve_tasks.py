from openai import OpenAI
import csv
import os
from dotenv import load_dotenv

def solve_tasks(input_file, output_file, model="gpt-4o"):
    #Alias gpt-4o points to gpt-4o-2024-08-06
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new column for solutions
        header = next(reader)
        writer.writerow(header + ["gpt-4o solution"])
        
        for i, row in enumerate(reader):
            if i >= 10:
                break
            topic_area, topic, progress_level, exercise = row
            prompt = f"Explain how to solve this task: {exercise}"
            print(f"Solving task {i+1}: {exercise}")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            solution = completion.choices[0].message.content.strip()
            writer.writerow(row + [solution])

if __name__ == "__main__": 
    input_file = 'topic_areas_cleaned.csv'
    output_file = '1_topic_areas_solutions.csv'
    solve_tasks(input_file, output_file)