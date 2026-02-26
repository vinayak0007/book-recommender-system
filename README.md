# Book Recommendation System

This project implements a recommendation system for a digital reading platform.  
The system is designed to:

1. Recommend the **next chapter** within books a user is currently reading  
2. Recommend **new books** based on historical reading behaviour  
3. Handle **cold-start scenarios** for users and books  

The goal of this project was not only to build a model, but to evaluate multiple paradigms and understand how dataset structure impacts recommender performance.

---

# Dataset Overview

| Metric | Value |
|--------|-------|
| Total Interactions | ~1,000,000 |
| Users | ~150,000 |
| Books | ~9,575 |
| Chapters | ~50,000 |
| Avg Books per User | ~6.6 |

### Dataset Characteristics

- Implicit feedback only (chapter read = 1 interaction)
- No ratings
- No timestamps
- Large candidate space (9575 books)
- Sparse user to book interaction matrix

This sparsity significantly affects collaborative filtering performance.

---

# Project Structure

```
book-recommender/
│
├── README.md
├── config.py
├── data/
│   ├── chapters.csv
│   └── interactions.csv
├── main.py
├── requirements.txt
└── src/
    ├── content_model.py
    ├── data_loader.py
    ├── evaluation.py
    ├── next_chapter.py
    └── popularity.py
```


---

# Next Chapter Recommendation

For each `(user, book)` pair:

1. Identify highest `chapter_sequence_no` read  
2. Recommend next chapter:

\[
NextChapter = max(sequence) + 1
\]

This deterministic progression-based approach:
- Requires no ML
- Is highly reliable
- Does not depend on collaborative overlap
- Has no cold-start issue for existing books

---

# Book Recommendation Models Evaluated

Multiple approaches were tested:

### 1] Popularity Baseline

```
Score(book) = interaction_count / total_interactions
```

**Recall@10 ≈ 0.0087**

Popularity emerged as a strong baseline.

---

### 2] Matrix Factorization (ALS)

- Implicit feedback
- Leave-one-out validation

**Recall@10 ≈ 0.005**

Underperformed popularity due to sparse interactions and weak collaborative overlap.

---

### 3] Item–Item Collaborative Filtering

Cosine similarity based on user co-read patterns:

```
Sim(i, j) = |Users(i) ∩ Users(j)| / sqrt( |Users(i)| × |Users(j)| )
```

**Recall@10 ≈ 0.0005**

Collapsed due to extreme sparsity.

---

### 4] Content + Popularity Hybrid (Final Model)

- TF-IDF on genre tags + author token  
- User profile = mean vector of books read  
- Hybrid scoring:

```
Score = 0.6 × ContentSim + 0.4 × Popularity
```

**Recall@10 ≈ 0.0085–0.0087**  
**Lift over Random ≈ 8×**

This performed comparably to popularity baseline and was selected for robustness.

---

#  Evaluation Methodology

- Leave-One-Out validation
- 20,000 sampled users
- Deterministic random seed
- Metrics:
  - Recall@10
  - NDCG@10
  - Lift over Random baseline

### Random Baseline

```
10 / 9575 ≈ 0.001
```

Final model achieves ~8× improvement over random ranking.

Given full-catalog ranking and sparse interaction history (~6 books/user), this lift is statistically meaningful.

---

#  Key Insights

1. The dataset is **popularity-dominated**
2. Sparse per-user interaction limits collaborative signal
3. Complex collaborative models do not outperform simple baselines
4. Dataset structure should guide model choice

Rather than forcing sophisticated models, selecting an approach aligned with data characteristics leads to more robust systems.

---

#  Cold Start Handling

| Scenario | Strategy |
|----------|----------|
| New User | Recommend popular books |
| New Book | Content similarity |
| Next Chapter | Deterministic progression |

---

#  How to Run

Install dependencies:


pip install -r requirements.txt


Run:


python main.py


The script will:

- Load data
- Perform leave-one-out split
- Train hybrid model
- Evaluate Recall@10, NDCG@10, and Lift
- Display next chapter recommendations

---

# Conclusion

This project demonstrates that:

- Recommender performance is highly dependent on dataset structure
- Sparse interaction settings favor popularity-driven systems
- Empirical comparison across paradigms is essential
- Simple models can outperform complex ones when aligned with data characteristics

The final production-ready architecture combines:

- Popularity-based candidate generation  
- Content-based personalization  
- Deterministic next-chapter recommendation  

---

# Author

Vinayak Pushkar  
