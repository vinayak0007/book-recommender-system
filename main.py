# main.py

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import SAMPLE_USERS, TOP_K, RANDOM_STATE
from src.data_loader import load_data
from src.content_model import ContentHybridModel
from src.next_chapter import build_next_chapter_recommender
from src.evaluation import recall_at_k, ndcg_at_k


# Reproducibility
np.random.seed(RANDOM_STATE)


# Load data
interactions, chapters = load_data()


# Sample users for faster evaluation
unique_users = interactions['user_id'].unique()

sample_users = np.random.choice(
    unique_users,
    size=min(SAMPLE_USERS, len(unique_users)),
    replace=False
)

interactions = interactions[
    interactions['user_id'].isin(sample_users)
]


# Leave-one-out split
train_list, test_list = [], []

for user, group in interactions.groupby("user_id"):

    if len(group) < 2:
        continue

    test_row = group.sample(1, random_state=RANDOM_STATE)
    train_rows = group.drop(test_row.index)

    train_list.append(train_rows)
    test_list.append(test_row)

train_df = pd.concat(train_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)


# Train model
model = ContentHybridModel()
model.fit(train_df, chapters)


# Evaluate
recall_scores, ndcg_scores = [], []

for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    user = row['user_id']
    test_book = row['book_id']

    recs = model.recommend(user, train_df, TOP_K)

    recall_scores.append(recall_at_k(recs, test_book))
    ndcg_scores.append(ndcg_at_k(recs, test_book))

mean_recall = np.mean(recall_scores)
mean_ndcg = np.mean(ndcg_scores)

print("\nFinal Results")
print(f"Recall@{TOP_K}: {mean_recall:.6f}")
print(f"NDCG@{TOP_K}: {mean_ndcg:.6f}")


# Random baseline and lift
num_books = len(model.book_ids)
random_recall = TOP_K / num_books
lift = mean_recall / random_recall if random_recall > 0 else 0

print("\nRandom Baseline")
print(f"Random Recall@{TOP_K}: {random_recall:.6f}")

print("\nLift Over Random")
print(f"Lift: {lift:.2f}x")

print(f"\nInterpretation: Model performs {lift:.2f}x better than random ranking.")


# Next Chapter Sample
next_chapter_df = build_next_chapter_recommender(train_df, chapters)

print("\nNext Chapter Sample")
print(next_chapter_df.head())
