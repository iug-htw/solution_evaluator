import csv
import argparse

def clean_csv(input_file, output_file):
    with open(input_file, mode='r', encoding='utf-8') as infile, open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # Write header
        writer.writerow(next(reader))
        
        for row in reader:
            if not any(row):
                continue  # Skip empty rows
            topic_area, topic, progress_level, exercise1, exercise2 = row
            if exercise1:
                writer.writerow([topic_area, topic, progress_level, exercise1])
            if exercise2:
                writer.writerow([topic_area, topic, progress_level, exercise2])

if __name__ == "__main__":
    clean_csv('topic_areas.csv','topic_areas_cleaned.csv')
