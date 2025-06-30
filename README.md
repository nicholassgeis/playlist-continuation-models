# playlist-continuation-models

## Team

- Jonathan Bloom  
- Nick Geis  
- Nadia Khoury


## Summary

This project was completed over a 6 week period during Erdos Institute's Data Science Bootcamp in summer 2025.

These repo explores two questions on algorithmic playlist continuation using the [Spotify Million Playlist Dataset](https://engineering.atspotify.com/2018/5/introducing-the-million-playlist-dataset-and-recsys-challenge-2018) and the [AcousticBrainz database](https://acousticbrainz.org). We develop multiple models to predict what tracks are likely to follow in a playlist by exploiting the co-occurrence patterns in the Spotify Million Playlist Dataset. Out models include a naive Random Model (picks songs at random), a KNN (or $k$ Nearest Neighbors) approach via an implicit distance metric, and a SVD (or Singular Value Decomposition) based model that aims to dramatically reduce dimensionality since the dataset references over 2 million tracks. To quantify our models we also include several metrics described below. 

Our work also incorporates tools for genre prediction and music clustering. Namely, we have identified a subset of the tracks from the Spotify Million Playlist Dataset that are included in the AcousticBrainz database. We then extracted additionally sonic and genre features for those tracks. We now project the space of 2.2 million tracks into a smaller dimensional space using the UMAP (or Uniform Manifold Approximation and Projection) technique. Finally, we perform a KNN approach to propagate the newly added genre data to unlabeled tracks.

The repository includes all code along with a Jupyter Notebook illustrating sample evaluation. Although our data was sourced from the aforementioned dataset we transformed it into sparase matrix files (npz) and csv files for our purposes.  Due to storage limits we only upload a sampling of these files.


## Models

### Naive Baseline (Random Sampling)

As a baseline for comparison, we implement a naive model that selects $k$ songs uniformly at random from the global pool of 2+ million tracks, excluding any already present in the input playlist. This model does not incorporate any information about song co-occurrence, playlist structure, or user behavior. While it typically performs poorly on accuracy and vibe-based metrics, it serves as a lower bound for evaluating the effectiveness of more sophisticated approaches.

### Graph-Based Greedy KNN

Songs are treated as nodes in a graph, connected if they co-occur in a playlist. Edges are weighted by a decreasing function of the form $\frac{1}{x^\alpha + 0.5}$. A greedy algorithm then recommends new songs based on how strongly connected they are to songs in the input playlist.

### Spectral KNN via Latent Embeddings

Given the playlist-track sparse matrix $P$ which is a 0-1 matrix encoding which tracks (columns) occur in which playlists (rows), we compute the co-occurrence matrix $C = P^\top P$, where $P$ is a playlist-song incidence matrix. Dimensionality reduction via SVD projects songs into a latent space. Playlists are embedded as averages of their song vectors, and recommendations are made via cosine similarity to this vector.

### Track Clustering and Genre Prediction

Using UMAP on the co-occurrence matrix, we project songs or artists into 2D space and label them with genre metadata from an external source. A KNN classifier predicts genres for unlabeled songs, allowing us to analyze genre clusters and structure in the playlist data.


## Evaluation

We use four complementary metrics to assess playlist continuation:

- **Bulls-eye**: Proportion of exact track matches.
- **Vibe**: Cosine similarity in the embedding space.
- **Artist Match**: Distribution similarity of artist frequencies.
- **Album Match**: Same as above for albums.

These metrics ensure both literal and stylistic accuracy are captured. Experiments show that higher-dimensional embeddings improve all scores. 

To ascertain if our models are just recommending only popular tracks we also include code to look at so-called rank-popularity histograms that depict how the distribution of songs ranked by popularity. Here popularity is measured by the number of playlists a song appears in; tracks appearing in many playlists are considered more popular than those that appear in few playlists. 

Additionally, we apply inverse-frequency weighting to reduce popularity bias, tuning a parameter \( \alpha \) to trade off fairness and accuracy.

![metrics-figure](assets/metrics.png) <!-- Replace with actual visual -->


## Description of Repository

`data/`: Contains sample matrices and lookup tables used throughout the project. Includes playlist-track, album-track, and artist-track sparse matrices, as well as ID-to-name CSV files. Also contains a small training/testing playlist splits in JSON format for experimentation. We strongly recommend downloading the full Spotify Million Playlist Dataset for further exploration.

`spotify_data.py`: Provides functions for loading sparse matrices and dictionary mappings used to structure the dataset for modeling.

`spotify_tools.py`: Includes helper utilities for preprocessing, data transformation, and graph-related operations.

`spotify_models.py`: Implements the core playlist continuation models, including the graph-based greedy KNN and spectral embedding models.

`spotify_metrics.py`: Defines the evaluation metrics used in our experiments, such as vibe score, bulls-eye accuracy, and distribution matching.

`Genre_predict_via_Artist_URI.ipynb` and `Genre_predict_via_track_URI.ipynb`: Notebooks for genre classification via UMAP projection and KNN prediction, using external genre label data.

`spotify_validation.ipynb`: Notebook for running validation routines and visualizing model performance on held-out playlist samples. **NOTE:** this notebook contains examples of the majority of the functions unique to this project. Start here in order to get an understanding of is possible.


### Data

There are two primary datasets:

- [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge): This dataset is the source of all playlist information. You will need to create an AIcrowd account to download the dataset. This dataset is around 35 GB in size.
- [AcousticBrainz Database](https://acousticbrainz.org/download): This is a crowd sourced dataset that contains sonic and genre information for various different recordings. There are many more unique tracks in the AcousticBrainz database than in the Spotify Million Playlist dataset (around 6 million vs 2.2 million). This entire database is around 180 GB in size.

The full dataset is far too large to include in this repository, but we provide a representative sample to support experimentation. The structure of the data folder includes:

    .
    └──data/
        ├── sample_playlist_data.json # Small collection of playlists
        ├── songs_with_genres_1.csv
        ├── songs_with_genres_2.csv # Some tracks from Spotify dataset with new AcousticBrainz features
        ├── table_matrix_data/
            ├── playlist_track_matrix/ # Sparse 0-1 playlist-song matrices
            ├── playlist_track_ordered_matrix/ # Ordered playlist matrices
            ├── track_id_to_name/ # Mapping of track IDs to names
            ├── artist_track.npz # Sparse artist-song matrix
            ├── album_track.npz # Sparse album-song matrix
            ├── *_id_to_name.csv # ID-to-name mappings
        ├── split_data/
            ├── train/ # JSON playlists for training
            └── test/ # JSON playlists for testing

> **Note:** We emphasize again that these files represent a small portion of the original 35GB dataset (1000 playlists instead of 1 million). You can regenerate matrices from the full dataset using the provided utilities.


## Future Work

- **Temporal Modeling**: Extend models to account for track order, enabling next-song predictions that better reflect playlist flow.

- **Bias Mitigation**: Apply inverse-frequency weighting to reduce bias towards recomending only popular songs without compromising quality.

- **Feature Integration**: Incorporate additional data sources such as playlist titles and audio features (e.g., tempo, key, danceability, loudness) to provide greater contextual understanding as a way to improve model metrics.

- **Dimensionality Reduction via Clustering**: By grouping tracks according to the implicit clustering encoded in the Spotify dataset we could dramatically decrease dimensionality while increasing speed/efficiency while preserving quality.
