from openai import OpenAI, RateLimitError
import csv
import os
import time
from dotenv import load_dotenv

def solve_tasks(input_file, output_file, model="qwen-plus", prompt_prefix="Explain to me how I can solve this task"):
    load_dotenv()
    # Initialize the OpenAI client with your API key and the base URL for DashScope
    api_key = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(
        api_key=api_key, 
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    # Load progress cache
    language = input_file.split(".")[0].split("_").pop()
    if language != "de" and language != "ar":
        language = "en"
    language_name = {"en": "English", "de": "German", "ar": "Arabic"}	

    # if the file does not exist, create it with progress 0
    cache_file = f"qwen_plus/progress_cache_{model}_{language}.txt"
    if not os.path.exists(cache_file):
        with open(cache_file, "w") as f:
            f.write("Progress: 0")
    
    with open(cache_file, "r") as f:
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
            # if i >= 50:
            #     break
            topic_area, topic, progress_level, exercise = row
            prompt = f"{prompt_prefix}: {exercise}"
            print(f"Solving task {i+1} in {language_name[language]}: {exercise}")
            retry_count = 0
            while retry_count < 5:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt},
                        ]
                    )
                    break
                except RateLimitError:
                    retry_count += 1
                    print(f"Rate limit exceeded. Retrying in {retry_count * 10} seconds...")
                    time.sleep(retry_count * 10)
                except Exception as e:
                    retry_count += 1
                    # Code to handle any exception
                    print(f"An error occurred: {e}")
                    time.sleep(retry_count * 10)
            else:
                print("Failed to get a response after multiple retries.")
                continue

            solution = completion.choices[0].message.content.strip()
            writer.writerow(row + [solution])

            with open(cache_file, "w") as f:
                f.write(f"Progress: {i}")

            # Add a delay of 3 seconds between requests
            time.sleep(3)

if __name__ == "__main__":
    input_file = 'topic_areas_cleaned.csv'
    output_file = 'qwen_plus/topic_areas_solutions_en.csv'
    solve_tasks(input_file, output_file)