from openai import OpenAI
import csv
import os
from dotenv import load_dotenv

def extract_technical_terms(input_file, output_file, target_language='en', model="gpt-4o"):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new column for technical terms
        header = next(reader)
        writer.writerow(header + ["Technical Terms"])
        
        for i, row in enumerate(reader):
            if i >= 400:
                break
            topic_area, topic, progress_level, exercise = row
            prompt = f"Which technical terms need to be understood to solve the following problem: {exercise} Provide only a list of the terms, no further text."
            if target_language != 'en':
                prompt += f"The technical terms should be in {target_language}."
            
            print(f"Extracting technical terms for task {i+1}: {exercise}")
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            technical_terms = completion.choices[0].message.content.strip()
            writer.writerow(row + [technical_terms])

if __name__ == "__main__":
    input_file = 'topic_areas_cleaned.csv'
    output_file = '2_topic_areas_technical_terms.csv'
    extract_technical_terms(input_file, output_file)
