import numpy as np
from scipy.sparse import csr_matrix
from collections import Counter


def transform_cosine_score(hidden_tracks: csr_matrix, predicted_tracks: csr_matrix, transform_matrix: csr_matrix):
    """
    Computes cosine similarity between predicted and hidden track sets after projecting them 
    into a shared embedding space.

    Parameters:
    ----------
    hidden_tracks : csr_matrix
        Sparse matrix (n, m) of ground truth track sets.
    predicted_tracks : csr_matrix
        Sparse matrix (n, m) of predicted track sets.
    transform_matrix : csr_matrix
        Sparse matrix (m, d) for projecting tracks into d-dimensional space.

    Returns:
    -------
    np.ndarray
        Array (n, 1) of cosine similarities between each predicted/hidden pair.
    """

    # Project hidden and predicted tracks into song co-occurrence space
    predicted_sim = predicted_tracks @ transform_matrix.T
    hidden_sim = hidden_tracks @ transform_matrix.T

    # Compute L2 norms of the projected rows
    norm_pred = np.sqrt(predicted_sim.multiply(predicted_sim).sum(axis=1)).A  # shape (n, 1)
    norm_hidden = np.sqrt(hidden_sim.multiply(hidden_sim).sum(axis=1)).A      # shape (n, 1)

    # Compute denominator of cosine similarity; avoid divide-by-zero
    denom = norm_pred * norm_hidden
    denom[denom == 0] = 1.0

    # Compute dot product (numerator) of corresponding rows
    dot = predicted_sim.multiply(hidden_sim).sum(axis=1).A  # shape (n, 1)

    # Final cosine similarity for each row pair
    return dot / denom


def prediction_accuracy_score(hidden_tracks: csr_matrix, predicted_tracks: csr_matrix):
    """
    Returns the precision per playlist: the fraction of predicted tracks
    that appear in the hidden tracks. Precision is 0 if no predictions.
    
    Parameters:
        hidden_tracks (csr_matrix): Ground-truth tracks per playlist.
        predicted_tracks (csr_matrix): Predicted tracks per playlist.
    
    Returns:
        np.ndarray: Precision scores, shape (num_playlists, 1).
    """
    num_rows,num_cols = hidden_tracks.shape

    prob = np.zeros(shape= (num_rows,1), dtype=np.float32)
    for r in range(num_rows):
        
        start = hidden_tracks.indptr[r]
        end = hidden_tracks.indptr[r+1]
        h = set(hidden_tracks.indices[start:end])

        start = predicted_tracks.indptr[r]
        end = predicted_tracks.indptr[r+1]
        p = set(predicted_tracks.indices[start:end])

        prob[r][0] = len(h.intersection(p))/len(p)
    return prob


def popularity_bias(predicted_tracks: csr_matrix, playlist_tracks_matrix: csr_matrix):
    """
    Computes a Counter over popularity ranks of predicted songs, skipping specified playlists.

    Parameters:
    - predicted_tracks: csr_matrix of shape (num_playlists, num_songs)
        Binary matrix where rows are playlists and columns are predicted songs.
    - playlist_tracks_matrix: csr_matrix of shape (num_playlists, num_songs)
        The full matrix used to calculate global song popularity.
    - playlists_to_skip: set of playlist indices to ignore

    Returns:
    - pop_count: Counter where keys are popularity ranks and values are frequencies
    """

    pop_rank = (np.argsort(playlist_tracks_matrix.sum(axis = 0).A1))[::-1]
    rank_lookup = {song_id: rank for rank, song_id in enumerate(pop_rank)}
    
    num_rows = predicted_tracks.shape[0]         
    pop_count = Counter()
    for r in range(num_rows):
        start = predicted_tracks.indptr[r]
        end = predicted_tracks.indptr[r+1]
        songs = predicted_tracks.indices[start:end]

        pop_count = pop_count + Counter({rank_lookup[s]:1 for s in songs})

    
    return pop_count
