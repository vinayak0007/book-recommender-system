# src/data_loader.py

import pandas as pd
from config import INTERACTIONS_FILE, CHAPTERS_FILE

def load_data():
    interactions = pd.read_csv(INTERACTIONS_FILE)
    chapters = pd.read_csv(CHAPTERS_FILE)

    interactions = interactions[['user_id', 'book_id', 'chapter_id']]
    return interactions, chapters
