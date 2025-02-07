from openai import OpenAI
import csv
import os
import time
from dotenv import load_dotenv


progress_levels = {
    "B": "2nd grade (7yo)",
    "C": "4th grade (9yo)",
    "D": "6th grade (11yo)", 
    "E": "7th grade (12 yo)",
    "F": "8th grade (13yo)",
    "G": "9th grade (14yo)",
    "H": "10th grade (15yo)"
}

def generate_exercises(exercise_objectives_file, output_file, model="gpt-4o-mini"):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    with open(exercise_objectives_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Skip header row from exercise_objectives file
        header = next(reader)
        
        # Write header for output file
        writer.writerow(['Topic Area', 'Topic', 'Progress Level', 'Exercise'])
        
        print("updated...!")
        for row in reader:
            topic_area, topic, progress_level, objective = row
            
            prompt = (
                f"You are a math workbook author. Create 3 clear, concise math exercises for students "
                f"based on this information:\n"
                f"Topic Area: {topic_area}\n"
                f"Topic: {topic}\n"
                f"Progress Level: {progress_levels[progress_level]}\n"
                f"Exercise Objective: {objective}\n\n"
                f"Generate 3 different exercises that would appear in a math workbook. "
                f"Each exercise should be a single sentence. "
                f"try to make each exercise with different goal and wording from the others"
                f"Make them practical and appropriate for the progress level. "
                f"Return only the 3 exercises, one per line, no numbering or extra text."
            )
            
            print(f"Generating exercises for: {objective}")
            
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                exercises = completion.choices[0].message.content.strip().split('\n')
                
                # Write each exercise as a separate row
                print(f"generated {len(exercises)} exercises")
                for exercise in exercises:
                    exercise = exercise.replace(',', '')
                    writer.writerow([topic_area, topic, progress_level, exercise.strip()])
                
                # Add a small delay to avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating exercises for {objective}: {str(e)}")
                continue

if __name__ == "__main__":
    exercise_objectives_file = "exercise_objectives.csv"
    output_file = "topic_areas.csv"
    generate_exercises(exercise_objectives_file, output_file)
