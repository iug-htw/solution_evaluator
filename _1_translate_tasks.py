import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

def translate_csv(input_csv, output_csv, target_language, model="gpt-4o-mini"):
    print(f"Translating exercises from English to {target_language}...")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    examples = {
        "Arabic": {
            "input": "Rationalize: 1⁄(1 + √7)",
            "output": "بسّط الكسر التالي بحيث لا يحتوي المقام على جذر: ١ / (١ + √٧)" 
        },
        "German": {
            "input": "Rationalize: 1⁄(1 + √7)",
            "output": "Rationalisiere den Bruch: 1 / (1 + √7)"
        }
    }

    def translate_text(text):
        prompt = f"""
        You are a math educator fluent in both English and {target_language}. Your task is to rewrite the following English math exercise into the target language, using accurate mathematical terminology and phrasing that matches how math problems appear in textbooks or classroom worksheets in that language.
        Do not translate literally. Instead, use math-education-specific vocabulary, especially for instructional verbs like:
        - "simplify"
        - "rationalize"
        - "factor"
        - "solve"
        - "circle"
        - "complete the square"
        Translate these using their equivalent meaning in math pedagogy as commonly used in schools that teach in {target_language}. 
        Write the translated instruction as a student would realistically see it in a math workbook.
        Avoid using markdown formatting. Reply with the translated text only, without any additional commentary, explanation, or formatting.

        example:
        Input: {examples[target_language]['input']}
        Output: {examples[target_language]['output']}

        Now, translate the following exercise into {target_language}: {text}"""
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