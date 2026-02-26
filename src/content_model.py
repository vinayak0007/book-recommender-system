# src/content_model.py

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

class ContentHybridModel:

    def fit(self, interactions, chapters):

        chapters['tags'] = chapters['tags'].fillna("")
        chapters['author_token'] = "author_" + chapters['author_id'].astype(str)

        chapters['combined'] = (
            chapters['tags'].str.replace("|"," ", regex=False)
            + " "
            + chapters['author_token']
        )

        book_content = (
            chapters.groupby("book_id")['combined']
            .apply(lambda x: " ".join(x))
            .reset_index()
        )

        self.vectorizer = TfidfVectorizer(min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(book_content['combined'])

        self.book_ids = book_content['book_id'].values
        self.book_id_to_idx = {
            b:i for i,b in enumerate(self.book_ids)
        }

        # Popularity
        book_pop = interactions.groupby("book_id").size()
        book_pop = book_pop / book_pop.sum()

        scaler = MinMaxScaler()
        self.pop_scores = scaler.fit_transform(
            book_pop.reindex(self.book_ids, fill_value=0)
            .values.reshape(-1,1)
        ).flatten()

    def recommend(self, user_id, train_df, top_k=10):

        user_books = train_df[
            train_df['user_id']==user_id
        ]['book_id'].unique()

        if len(user_books) == 0:
            top_indices = np.argsort(-self.pop_scores)[:top_k]
            return self.book_ids[top_indices]

        indices = [
            self.book_id_to_idx[b]
            for b in user_books
            if b in self.book_id_to_idx
        ]

        if len(indices) == 0:
            top_indices = np.argsort(-self.pop_scores)[:top_k]
            return self.book_ids[top_indices]

        user_profile = self.tfidf_matrix[indices].mean(axis=0)
        user_profile = np.asarray(user_profile).ravel()

        sims = user_profile @ self.tfidf_matrix.T
        sims = np.asarray(sims).ravel()

        final_scores = 0.6*sims + 0.4*self.pop_scores

        for b in user_books:
            if b in self.book_id_to_idx:
                final_scores[self.book_id_to_idx[b]] = -1

        top_indices = np.argsort(-final_scores)[:top_k]

        return self.book_ids[top_indices]
