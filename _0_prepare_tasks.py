import csv
import argparse

def clean_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        header = next(reader)

        # Update the header for the output file
        updated_header = header[:3] + ['Exercise']
        writer.writerow(updated_header)
        
        for row in reader:
            if not any(row):
                continue  # Skip empty rows
            topic_area, topic, progress_level, exercise = row
            if exercise.strip() != '':
                exercise = exercise.replace('Hier ist die Übersetzung der Übungsbeschreibung ins Deutsche:', '')
                writer.writerow([topic_area, topic, progress_level, exercise.strip()])

if __name__ == "__main__":
    clean_csv('topic_areas.csv','topic_areas_cleaned.csv')
    
