import time as time
import numpy as np
import spotify_tools as tools
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
from sklearn.decomposition import TruncatedSVD
from multiprocessing.dummy import Pool as ThreadPool


def predict_single_playlist(args):
    """
    Predict k songs for a single playlist using distance-based scoring with penalty for low support.
    
    Parameters:
    - playlist_row: sparse row of song indices in the playlist
    - k: number of songs to predict
    - distance_matrix: csr_matrix of song-to-song distances
    - num_songs: total number of songs

    Returns:
    - predicted_songs: list of k predicted song indices
    """
    playlist_row, k, distance_matrix, num_songs = args
    playlist_songs = playlist_row.indices.tolist()


    predicted_songs = np.zeros(k,dtype=int)
    s = np.zeros(num_songs, dtype=np.float32)
    c = np.zeros(num_songs, dtype=np.float32)
    c2 = np.zeros(num_songs, dtype=np.float32)
    score = np.full(num_songs, np.inf, dtype=np.float32)
    
    for i in playlist_songs:
        start = distance_matrix.indptr[i]
        end = distance_matrix.indptr[i+1]
        indices = distance_matrix.indices[start:end]
        data = distance_matrix.data[start:end]

        s[indices] += data
        c[indices] += 1
        c2[indices] = c[indices]**2

    for i in range(k):

        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(s, c2, out=score, where= c > 0)

        score[playlist_songs] = np.inf  # mask existing songs
        min_song = np.argmin(score)
        predicted_songs[i] = min_song
        playlist_songs.append(min_song)

        # update cumulative score and count
        start = distance_matrix.indptr[min_song]
        end = distance_matrix.indptr[min_song+1]
        indices = distance_matrix.indices[start:end]
        data = distance_matrix.data[start:end]

        s[indices] += data
        c[indices] += 1
        c2[indices] = c[indices]**2

    return predicted_songs

def predict_single_playlist_random(args):
    """
    Predict k songs for a single playlist using distance-based scoring with penalty for low support.
    
    Parameters:
    - playlist_row: sparse row of song indices in the playlist
    - k: number of songs to predict
    - distance_matrix: csr_matrix of song-to-song distances
    - num_songs: total number of songs

    Returns:
    - predicted_songs: list of k predicted song indices
    """
    playlist_row, knbr, adjacency_matrix, num_songs, num_max_songs = args
    playlist_songs = playlist_row.indices.tolist()

    predicted_songs = np.zeros(knbr, dtype=int)
    scores = np.zeros(num_songs, dtype=np.float32)

    # Initial cumulative score from existing playlist
    for song_idx in playlist_songs:
        start = adjacency_matrix.indptr[song_idx]
        end = adjacency_matrix.indptr[song_idx + 1]
        neighbors = adjacency_matrix.indices[start:end]
        scores[neighbors] += 1

    for i in range(knbr):
        scores[playlist_songs] = -np.inf  # Mask out already used songs

    
        nonzero_indices = np.flatnonzero(scores>0)
        nonzero_values = scores[nonzero_indices]
        k = min(num_max_songs, len(nonzero_indices))

        top_k_local = np.argpartition(nonzero_values, -k)[-k:]
        candidate_pool = nonzero_indices[top_k_local]

        if len(candidate_pool) == 0:
            return predicted_songs
        
        selected_song = np.random.choice(candidate_pool)
        predicted_songs[i] = selected_song
        playlist_songs.append(selected_song)

        # Update cumulative score with the new song
        start = adjacency_matrix.indptr[selected_song]
        end = adjacency_matrix.indptr[selected_song + 1]
        neighbors = adjacency_matrix.indices[start:end]
        scores[neighbors] += 1

    return predicted_songs


class Spotify_Random():
    def __init__(self):
        pass
        
    def fit(self, num_songs): 
        self.num_songs = num_songs

    def predict(self, num_playlists, num_to_predict):
        all_songs = np.arange(self.num_songs)
        all_predictions = np.random.choice(all_songs, (num_playlists, num_to_predict), replace=False).tolist()

        return tools.create_sparse_indicator_matrix(all_predictions, self.num_songs)

class Spotify_KNN():
    def __init__(self):
        self.distance_matrix = None
        self.song_co_occurrence = None
        self.num_songs = 0
        self.num_playlists = 0
        
    def fit(self, P: csr_matrix, f): 
        """
        Fits the model using a playlist-song 0-1 sparse matrix.

        Parameters:
            P: Sparse matrix of shape (num_playlists, num_songs).
            f (callable): Function to convert counts to distances.

        Computes a song-song distance matrix and builds the song graph.
        """
        
        self.num_playlists, self.num_songs = P.shape
        self.song_co_occurrence = P.T @ P
        self.distance_matrix = self.song_co_occurrence.copy()
        self.distance_matrix.data = f(self.song_co_occurrence.data)

    def predict(self, playlists, num_predict):
        """
        Predict k songs for each playlist using parallel processing.

        Parameters:
        - playlists: csr_matrix of shape (num_playlists, num_songs)
        - k: number of songs to predict per playlist
        - beta: the exponent on 1/n**beta used in averaging neighbors of a playlist to find nearest neighbor.  When beta = 1 we get the traditional average.  

        Returns:
        - csr_matrix of shape (num_playlists, num_songs) with 1s at predicted song position
        """
        num_playlists = playlists.shape[0]
        
        args = [
            (playlists.getrow(i), num_predict, self.distance_matrix, self.num_songs) for i in range(num_playlists)
        ]

        #ThreadPool maintains order so predictions in same order as playlists
        with ThreadPool() as pool:
            all_predictions = pool.map(predict_single_playlist, args)


        return tools.create_sparse_indicator_matrix(all_predictions, self.num_songs)

class Spotify_Spectral_KNN():
    """
    Spectral k-NN model for song recommendation using SVD of the song co-occurrence matrix.
    """

    def __init__(self):
        self.song_co_occurrence = None
        self.laten_songs = None
        self.num_songs = 0
        self.num_playlists = 0
        
        


    def fit(self, P: csr_matrix, k: int): 
        """
        Factorizes the song co-occurrence matrix using top-k eigenvectors.
        
        Parameters:
        - P: 0/1 csr_matrix of shape (num playlists, num songs)
        - k: int, number of dimensions to project onto
        """

        self.num_playlists, self.num_songs = P.shape
    
        P = P.astype(np.float32)
        svd = TruncatedSVD(n_components=k, algorithm='randomized',n_iter=2)
        self.laten_songs = svd.fit_transform(P.T)
        
        
    
    def predict(self, playlists: csr_matrix, k: int):
        """
        Predicts k songs for each playlist based on score (cosine similarity).

        Parameters:
        - playlists: csr_matrix of shape (num_playlists_to_complete, num songs).
        - k: int, number of song predictions for each playlist

        Returns:
        - 0/1 csr_matrix of shape (num_playlists_to_complete, num_songs) 
        """
        num_playlists = playlists.shape[0]
        dim = self.laten_songs.shape[1]

        pred = []
        for i in range(num_playlists):
            # playlist_songs = playlists.getrow(i).indices

            start = playlists.indptr[i]
            end = playlists.indptr[i+1]
            playlist_songs = playlists.indices[start:end]

            if (len(playlist_songs)==0):
                avg = np.zeros(shape = (dim,1))
            else:
                avg = self.laten_songs[playlist_songs].mean(axis = 0).T

        
            score = (self.laten_songs @ avg).ravel()
        
            
            # Exclude original playlist songs from candidates
            score[playlist_songs] = -np.inf

            knn = np.argpartition(-score, k)[:k]
            pred.append(knn)
            
        return tools.create_sparse_indicator_matrix(pred, self.num_songs)