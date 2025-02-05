import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

def translate_csv(input_csv, output_csv, target_language, model="gpt-4o"):
    print(f"Translating exercises from English to {target_language}...")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    def translate_text(text):
        prompt = f"Translate the following exercise description from English to {target_language}: {text}"
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ]
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