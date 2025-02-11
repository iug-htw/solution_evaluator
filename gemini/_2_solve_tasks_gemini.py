import google.generativeai as genai
import csv
import os
import time
from dotenv import load_dotenv

def solve_tasks(input_file, output_file, model="gemini-1.5-flash", prompt_prefix="Explain to me how I can solve this task"):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel("gemini-1.5-flash")

    # Load progress cache
    language = input_file.split(".")[0].split("_").pop()
    if language != "de" and language != "ar":
        language = "en"
    language_name = {"en": "English", "de": "German", "ar": "Arabic"}	

    # if the file does not exist, create it with progress 0
    if not os.path.exists(f"gemini/progress_cache_{model}_{language}.txt"):
        with open(f"gemini/progress_cache_{model}_{language}.txt", "w") as f:
            f.write("Progress: 0")
    
    with open(f"gemini/progress_cache_{model}_{language}.txt", "r") as f:
        content = f.read()
        if content == "":
            progress = 0
        else:
            progress = int(content.split(":")[1].strip())

    # Open output file in appropriate mode
    file_mode = "a" if progress > 0 else "w"
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode=file_mode, encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        if progress == 0:           
            # Write header with new column for solutions
            header = next(reader)
            writer.writerow(header + [f"{model} solution"])
        
        # start from the last solved task
        for i, row in enumerate(reader):
            if i < progress:
                continue
            if i >= 50:
                break
            topic_area, topic, progress_level, exercise = row
            prompt = f"{prompt_prefix}: {exercise}"
            print(f"Solving task {i+1} in {language_name[language]}: {exercise}")

            response = client.generate_content(prompt)

            solution = response.text.replace(',','').strip()
            writer.writerow(row + [solution])

            with open(f"gemini/progress_cache_{model}_{language}.txt", "w") as f:
                f.write(f"Progress: {i+1}")

            # Add a delay of 5 seconds between requests
            time.sleep(5)

if __name__ == "__main__":
    input_file = 'topic_areas_cleaned.csv'
    output_file = 'gemini/topic_areas_solutions_en.csv'
    solve_tasks(input_file, output_file)