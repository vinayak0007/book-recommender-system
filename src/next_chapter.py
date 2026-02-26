# src/next_chapter.py

def build_next_chapter_recommender(interactions, chapters):

    interaction_with_seq = interactions.merge(
        chapters[['chapter_id','book_id','chapter_sequence_no']],
        on=['chapter_id','book_id'],
        how='left'
    )

    user_book_progress = (
        interaction_with_seq
        .groupby(['user_id','book_id'])['chapter_sequence_no']
        .max()
        .reset_index()
    )

    user_book_progress['next_sequence'] = (
        user_book_progress['chapter_sequence_no'] + 1
    )

    next_chapter = user_book_progress.merge(
        chapters,
        left_on=['book_id','next_sequence'],
        right_on=['book_id','chapter_sequence_no'],
        how='left'
    )[["user_id","book_id","chapter_id"]]

    return next_chapter.dropna()
