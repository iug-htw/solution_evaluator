from openai import OpenAI
import csv
import os
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

def extract_technical_terms(input_file, output_file, target_language='en', model="gpt-4o-mini"):
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
            topic_area, topic, progress_level, exercise = row

            if target_language == 'en':
                prompt = f"""Which key mathematical terms are necessary for a {progress_levels[progress_level]} student to understand before solving the following math problem? 
                Those are mathematical terms that should be explained first by the teacher or tutor to ensure the student can effectively solve the problem.
                Avoid using terms that are too basic or self-explanatory, and do not include instructional terms like 'solve', 'circle', 'find', etc.
                Avoid listing terms beyond the student's grade level. You can respond with an empty list if the exercise is self-explanatory.
                Reply with a concise list of only genuinely non-obvious terms critical for solving the problem, without any additional commentary, explanation, or formatting.
                
                Topic Area: {topic_area}
                Topic: {topic}
                Exercise: {exercise}
                """

            elif target_language == 'de':
                prompt = f"""Welche grundlegenden mathematischen Begriffe sollte ein Schüler der {progress_levels[progress_level]} verstehen, bevor er die folgende Mathematikaufgabe löst? 
                Diese Begriffe sollten zuerst vom Lehrer oder Nachhilfelehrer erklärt werden, um sicherzustellen, dass der Schüler die Aufgabe effektiv lösen kann.
                Vermeiden Sie zu einfache oder selbsterklärende Begriffe und schließen Sie keine Anweisungsbegriffe wie 'berechne', 'kreise ein', 'finde' usw. ein.
                Vermeiden Sie es, Begriffe aufzulisten, die über das Niveau des Schülers hinausgehen. Sie können mit einer leeren Liste antworten, wenn die Aufgabe selbsterklärend ist.
                Antworten Sie mit einer kurzen Liste von nur wirklich nicht offensichtlichen Begriffen, die für das Lösen der Aufgabe entscheidend sind, ohne zusätzliche Kommentare, Erklärungen oder Formatierungen.

                Themenbereich: {topic_area}
                Thema: {topic}
                Aufgabe: {exercise}
                """
            else:
                prompt = f"""ما هي المصطلحات الرياضية الأساسية التي يحتاج طالب في الصف {progress_levels[progress_level]} إلى فهمها قبل حل المسألة الرياضية التالية؟
                هذه هي المصطلحات اللتي يجب أن يشرحها المعلم أو المدرس الخصوصي أولاً لضمان قدرة الطالب على حل المسألة بشكل فعّال.
                تجنب استخدام المصطلحات البسيطة جداً أو الواضحة، ولا تدرج كلمات تعليمية مثل "احسب"، "ضع دائرة"، "اوجد"، وما شابه.
                تجنب أيضاً إدراج المصطلحات التي تتجاوز مستوى الصف الدراسي للطالب. يمكنك الرد بقائمة فارغة إذا كانت المسألة مفهومة بذاتها.
                أجب بقائمة مختصرة تحتوي فقط على المصطلحات غير البديهية الضرورية لحل المسألة، من دون أي تعليق إضافي أو شرح أو تنسيق.

                المجال: {topic_area}
                الموضوع: {topic}
                المسألة: {exercise}
                """
            
            # if target_language != 'en':
            #     prompt += f"The technical terms should be in {target_language}."
            
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
