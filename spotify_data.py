
import sys
import json
import random
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz

# --------------------------------------------------------------------------------------
#        SPARSE MATRIX LOADERS
# --------------------------------------------------------------------------------------

# These matrices can be loaded as needed.

def get_artist_track_matrix():
    matrix = load_npz('data/table_matrix_data/artist_track.npz')
    return matrix

def get_album_track_matrix():
    matrix = load_npz('data/table_matrix_data/album_track.npz')
    return matrix

def get_playlist_track_matrix():
    '''
    Consists of all playlists from the 800,000 training dataset.
    '''
    matrix = load_npz('data/table_matrix_data/playlist_track.npz')
    return matrix

# --------------------------------------------------------------------------------------
#        DICTIONARIES
# --------------------------------------------------------------------------------------

# This sections defines dictionaries where track, artist, and album ids are the keys and the string containing
# track name, artist name and album name are the corresponding values. Since there are many songs of the 
# same name, we cannot make this a two-way dictionary.

def ids_to_names_table():
    '''
    Returns the track_id -> track_name, artist_id -> artist_name, and
    album_id -> album_name dictionaries.
    '''
    data = pd.read_csv('data/table_matrix_data/track_id_to_name.csv')
    d1 = data.set_index('track_uri')['track_name'].to_dict()

    data = pd.read_csv('data/table_matrix_data/artist_id_to_name.csv')
    d2 = data.set_index('artist_uri')['artist_name'].to_dict()

    data = pd.read_csv('data/table_matrix_data/album_id_to_name.csv')
    d3 = data.set_index('album_uri')['album_name'].to_dict()
    
    return d1, d2, d3

# --------------------------------------------------------------------------------------
#        TWO-WAY DICTIONARIES
# --------------------------------------------------------------------------------------

# The unique identifiers for each track, artist and album will be in 1-to-1 correspondence with
# the index set for their respective groups. This is not true for the names corresponding to each 
# track, artist, and album id. This allows us to construct 2-way dictionaries which we couldn't do
# before.

class BiDict:
    '''
    Defines a two-way dictionary or "bidictionary."
    '''
    def __init__(self):
        self.forward = {}
        self.reverse = {}
    
    def __setitem__(self, key, value):
        self.forward[key] = value
        self.reverse[value] = key
    
    def __getitem__(self, key):
        return self.forward[key]
    
    def __len__(self):
        return len(self.forward)
    
    def get_key(self, value):
        return self.reverse[value]
    
    def keys(self):
        return list(self.forward.keys())

def track_index_table():
    '''
    key: track_id
    value: track_index in track space
    
    Use .get_key(track_index) to recover track_id in O(1) time
    '''
    bd = BiDict()
    i = 0
    data = pd.read_csv('data/table_matrix_data/track_id_to_name.csv')
    for id in data['track_uri']:
        bd[id] = i
        i = i+1
    return bd

def artist_index_table():
    '''
    key: artist_id
    value: artist_index in artist space

    Use .get_key(artist_index) to recover artist_id in O(1) time
    '''
    bd = BiDict()
    i = 0
    data = pd.read_csv('data/table_matrix_data/artist_id_to_name.csv')
    for id in data['artist_uri']:
        bd[id] = i
        i = i + 1
    return bd

def album_index_table():
    '''
    key: album_id
    value: album_index in album space

    Use .get_key(album_index) to recover album_id in O(1) time
    '''
    bd = BiDict()
    i = 0
    data = pd.read_csv('data/table_matrix_data/album_id_to_name.csv')
    for id in data['album_uri']:
        bd[id] = i
        i = i + 1
    return bd

# --------------------------------------------------------------------------------------
#        BUILDS USEFUL DICTIONARIES
# --------------------------------------------------------------------------------------

track_index = track_index_table()
artist_index = artist_index_table()
album_index = album_index_table()

get_track_name, get_artist_name, get_album_name = ids_to_names_table()

# --------------------------------------------------------------------------------------
#        JSON FUNCTIONS
# --------------------------------------------------------------------------------------

# List of training data json_files
json_files = [f for f in os.listdir("split_data/train") if f.endswith('.json')]

def playlist_aggregator(files, dtype = 'train'):
    '''
    This functions joins all the playlists across various JSON files into a single
    numpy array object. By default, it gathers all the playlists from the JSON files in
    json_train. If you want to access json_test, then use
                    playlist_aggregator(json_test)
    '''

    aggregate = np.empty(len(files)*100, dtype = object)
    running_count = 0
    for f in files:
        with open('split_data/' + dtype + '/' + f, 'r') as file:
            data = np.array(json.load(file))
            for playlist in data:
                aggregate[running_count] = playlist
                running_count = running_count + 1
    return aggregate
