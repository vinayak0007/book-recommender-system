# src/popularity.py

class PopularityModel:

    def fit(self, interactions):
        book_pop = interactions.groupby("book_id").size()
        self.ranked_books = (
            book_pop.sort_values(ascending=False).index.tolist()
        )

    def recommend(self, top_k=10):
        return self.ranked_books[:top_k]
