import pandas as pd

def load_and_preprocess(csv_path):
    data = pd.read_csv(csv_path)
    
    # Number of records and basic stats
    print("Total records:", len(data))
    num_over_50k = (data['income'] == '>50K').sum()
    num_under_50k = (data['income'] == '<=50K').sum()
    percent_over_50k = num_over_50k / len(data) * 100
    print(f"Income >50K: {num_over_50k}, Income <=50K: {num_under_50k}, Percentage >50K: {percent_over_50k:.2f}%")
    
    # One-hot encoding categorical features
    data_encoded = pd.get_dummies(data.drop('income', axis=1))
    data_encoded['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    return data_encoded

if __name__ == "__main__":
    processed_data = load_and_preprocess("data/census_data.csv")
    processed_data.to_csv("data/processed_data.csv", index=False)
