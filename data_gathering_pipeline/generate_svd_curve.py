import json
import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import os

def main():
    # Load feedback data
    data = []
    with open('data/feedback_data.jsonl', 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    data.append({
                        'user_id': d['user_id'],
                        'model_id': d['model_id'],
                        'rating': d.get('rating', 3)
                    })
                except json.JSONDecodeError:
                    pass

    df = pd.DataFrame(data)

    # Surprise Reader
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(df[['user_id', 'model_id', 'rating']], reader)

    # Define k values to test
    k_values = [1, 2, 5, 10, 20, 50, 100, 150]
    rmse_results = []

    print(f"Total ratings: {len(df)}")
    print(f"Unique Users: {df['user_id'].nunique()}")
    print(f"Unique Models: {df['model_id'].nunique()}")

    for k in k_values:
        print(f"Testing k={k}...")
        # FunkSVD in surprise is called SVD
        algo = SVD(n_factors=k, random_state=42)
        # 5-fold cross validation
        cv_results = cross_validate(algo, dataset, measures=['RMSE'], cv=5, verbose=False)
        mean_rmse = cv_results['test_rmse'].mean()
        rmse_results.append(mean_rmse)
        print(f"  RMSE: {mean_rmse:.4f}")

    best_k = k_values[rmse_results.index(min(rmse_results))]
    print(f"Best k: {best_k} with RMSE: {min(rmse_results):.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, rmse_results, marker='o', linestyle='-', color='b')
    plt.title('SVD Validation Error Curve (Finding optimal k)')
    plt.xlabel('Number of Latent Factors (k)')
    plt.ylabel('Cross-Validation RMSE')
    plt.grid(True)
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Optimal k = {best_k}')
    plt.legend()

    # Save the plot 
    output_path = 'svd_curve.png'
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
