import pandas as pd

def load_eco_scores(csv_path="emission_dataset.csv"):

    df = pd.read_csv(csv_path)

    eco_scores = {}

    for _, row in df.iterrows():
        score = float(row["relative_intensity"]) * 100
        eco_scores[row["class"]] = score

    return eco_scores


eco_scores = load_eco_scores()

print(eco_scores)