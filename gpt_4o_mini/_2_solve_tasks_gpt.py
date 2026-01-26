from openai import OpenAI, RateLimitError
import csv
import os
import time
from dotenv import load_dotenv

def solve_tasks(input_file, output_file, model="gpt-4o-mini", prompt_prefix="Explain to me how I can solve this task", tasks_indices=None):
    """
    Generate step-by-step solutions for math exercises using an OpenAI llm model
    and store the results in a new CSV file.

    This function reads a CSV file containing math exercises, sends each exercise
    to an OpenAI chat model with a configurable prompt, and writes the model’s
    generated solution as an additional column in an output CSV file. It supports
    optional filtering to solve only a subset of tasks and includes basic retry
    logic to handle rate limits and transient API errors.

    The OpenAI API key is loaded from the environment using `python-dotenv`
    (via the `OPENAI_API_KEY` variable).

    Parameters
    ----------
    input_file : str
        Path to the input CSV file. The file is expected to contain rows with the
        following columns (in order):
        - topic_area
        - topic
        - progress_level
        - exercise

    output_file : str
        Path to the output CSV file. A new column named
        "<model> solution" is appended to the original header and contains
        the generated solutions.

    model : str, optional
        The OpenAI chat model to use for solution generation.
        Defaults to "gpt-4o-mini". Helps the function
        keep track of the name of the model in use.

    prompt_prefix : str, optional
        Text prepended to each exercise to form the user prompt sent to the model.
        The final prompt has the form:
        "<prompt_prefix>: <exercise>"
        Defaults to "Explain to me how I can solve this task" (English prompt).
        Different languages should have different prompt prefixes corresponding
        to target langauge. Update this parameter with a transalted version of the prompt.

    tasks_indices : iterable of int or None, optional
        Zero-based indices of rows (excluding the header) to process.
        If None, all tasks in the input file are processed.
        This is useful for debugging, batching, or partial reruns.

    Behavior
    --------
    - Reads the input CSV row by row.
    - For each selected task, sends a single user message to the OpenAI Chat API.
    - Retries up to 5 times per task in case of rate limits or other exceptions,
      with an increasing wait time between retries.
    - Writes the original row plus the generated solution to the output CSV.
    - Skips tasks that fail after all retries.
    """

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Write header with new column for solutions
        header = next(reader)
        writer.writerow(header + [f"{model} solution"])
        
        for i, row in enumerate(reader):
            # if i >= 50:
            #     break

            if tasks_indices is not None and i not in tasks_indices:
                continue

            topic_area, topic, progress_level, exercise = row
            prompt = f"{prompt_prefix}: {exercise}"
            print(f"Solving task {i+1}: {exercise}")
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

if __name__ == "__main__":
    input_file = 'topic_areas_cleaned.csv'
    output_file = '2_topic_areas_solutions.csv'
    solve_tasks(input_file, output_file)