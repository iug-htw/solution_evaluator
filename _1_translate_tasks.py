"""
_1_translate_tasks.py

This script uses GPT-4o-mini to automatically translate math exercises from English
into a target language (e.g., Arabic, German). The translations are optimized to
sound natural and pedagogically appropriate for math workbooks, rather than literal
word-for-word translations.
"""

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

def translate_csv(input_csv, output_csv, target_language, model="gpt-4o-mini"):
    """
    Translates the 'Exercise' column of a CSV file from English into a target language
    using GPT-4o-mini.

    Parameters
    ----------
    input_csv : str
        Path to the input CSV file containing English exercises.
    output_csv : str
        Path where the translated CSV will be saved.
    target_language : str
        Target language for translation (e.g., "Arabic", "German").
    model : str, optional
        The LLM model used for translation. Default is "gpt-4o-mini".

    Behavior
    --------
    - Loads exercises from the input file
    - Sends each exercise to the LLM with a translation prompt designed for
      pedagogical clarity
    - Replaces the English exercise text with the translated version
    - Saves the updated dataset to the specified output file
    """

    print(f"Translating exercises from English to {target_language}...")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    def translate_text(text):
        prompt = f"""
        You are a math educator fluent in {target_language}.
        Your task is to write a math exercise in {target_language} that teaches or assesses the same skill or concept as the exercise below.
        Use natural phrasing and terminology appropriate for math workbooks written for students in {target_language}.
        Focus on making the exercise feel like it was originally written in {target_language}, not adapted from another language. Use the same numerical values as the original exercise.
        Avoid using markdown formatting. Reply with the {target_language} exercise text only, without any additional commentary, explanation, or formatting.


        English exercise:
        {text}

        Now write a new exercise in {target_language} with the same objective.
        """

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content.strip()

    # Load the CSV file
    df = pd.read_csv(input_csv)

    column_name = 'Exercise'
    df[column_name] = df[column_name].apply(lambda x: translate_text(x) if isinstance(x, str) else x)

    # Export the translated DataFrame to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Translation to {target_language} complete. CSV file saved as: {output_csv}.")

if __name__ == "__main__":
    translate_csv('topic_areas_cleaned.csv', 'topic_areas_cleaned_ar.csv', 'Arabic')
    print('--------------------------------')
    translate_csv('topic_areas_cleaned.csv', 'topic_areas_cleaned_de.csv', 'German')
