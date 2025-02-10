import google.generativeai as genai
import csv
import os
import time
from dotenv import load_dotenv

def extract_technical_terms(input_file, output_file, target_language='en'):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel("gemini-1.5-flash")
    
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new column for technical terms
        header = next(reader)
        writer.writerow(header + ["Technical Terms"])
        
        for i, row in enumerate(reader):
            if i >= 50:
                break
            topic_area, topic, progress_level, exercise = row
            prompt = f"Which technical terms need to be understood to solve the following problem: {exercise} Provide only a list of the terms, no further text."
            if target_language != 'en':
                prompt += f" The technical terms should be in {target_language}."
            
            print(f"Extracting technical terms for task {i+1}: {exercise}")
            response = client.generate_content(prompt)
            technical_terms = response.text.strip()
            writer.writerow(row + [technical_terms])

            # Add a delay of 5 seconds between requests
            time.sleep(5)

if __name__ == "__main__":
    input_file = 'topic_areas_cleaned.csv'
    output_file = 'gemini/topic_areas_technical_terms.csv'
    extract_technical_terms(input_file, output_file) 