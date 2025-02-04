
from openai import OpenAI
import os
from dotenv import load_dotenv




if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        store=True,
        messages=[
            {"role": "user", "content": "Explain how to solve this task. Express 3.75 as a fraction."},
        ]
    )
    print(completion.choices[0].message.content)