import pandas as pd

def find_differently_performing_exercises(files):
    # Initialize an empty list to hold preprocessed DataFrames
    dataframes = []

    # Iterate through files or DataFrames
    for i, file in enumerate(files):
        if isinstance(file, pd.DataFrame):
            df = file
        else:
            df = pd.read_csv(file)
        
        # Rename columns to match expected format
        df = df.rename(columns={"Exercise": "exercise"})
        df['exercise'] = [f"exercise_{index + 1}" for index in range(len(df))]

        
        # Add a 'language' column if it doesn't exist (e.g., based on source of the file)
        if "language" not in df.columns:
            df["language"] = f"language_{i+1}"  # Replace this with the actual language
        
        # Compute an overall 'score' as the mean of the evaluation metrics
        score_columns = [
            "Problem Understanding (Comprehension)",
            "Clarity and Step-by-Step Explanation",
            "Accuracy of Process (Correctness of Steps)",
            "Correctness of Final Answer",
            "Learning Appropriateness",
            "Generalization",
            "Technical Terms Explanation",
            "Addressing Common Errors",
            "Appropriateness Based on Progress Level",
        ]
        
        for col in df[score_columns]:
            for score in df[col]:
                # check if the value can be converted to a float
                try:
                    float_score = float(score)
                    if float_score < 0 or float_score > 2:
                        raise ValueError(f"Invalid score: {float_score}")
                    
                except ValueError:
                    print(f"Invalid value: {score}")
        df["score"] = df[score_columns].mean(axis=1)
        
        # Keep only the necessary columns
        df = df[["exercise", "language", "score"]]
        
        dataframes.append(df)
    
    # Concatenate all DataFrames
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Pivot the DataFrame to have exercises as rows and languages as columns
    pivoted_df = combined_df.pivot_table(
        index="exercise",
        columns="language",
        values="score",
        aggfunc="mean"
    )

    # Replace NaN values with 0 or another placeholder if necessary
    pivoted_df = pivoted_df.fillna(0)  # Replace NaN with 0 for calculations

    # Calculate the total difference for each exercise
    pivoted_df["Total_Difference"] = pivoted_df.max(axis=1) - pivoted_df.min(axis=1)

    # Sort the exercises by the total difference in descending order
    sorted_df = pivoted_df.sort_values(by="Total_Difference", ascending=False)

    # Reset the index for better readability and add the original index as a column
    sorted_df = sorted_df.reset_index()

    return sorted_df
