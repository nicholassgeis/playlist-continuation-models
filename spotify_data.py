
import sys
import json
import random
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack

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
    matrix1 = load_npz('data/table_matrix_data/playlist_track_matrix/playlist_track_1.npz')
    matrix2 = load_npz('data/table_matrix_data/playlist_track_matrix/playlist_track_2.npz')
    return vstack([matrix1, matrix2])

def get_ordered_playlist_track_matrix():
    matrix1 = load_npz('data/table_matrix_data/playlist_track_ordered_matrix/playlist_track_ordered_1.npz')
    matrix2 = load_npz('data/table_matrix_data/playlist_track_ordered_matrix/playlist_track_ordered_2.npz')
    matrix3 = load_npz('data/table_matrix_data/playlist_track_ordered_matrix/playlist_track_ordered_3.npz')
    return vstack([matrix1, matrix2, matrix3])



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
    data = pd.read_csv('data/table_matrix_data/track_id_to_name/track_id_to_name_1.csv')
    d1 = data.set_index('track_uri')['track_name'].to_dict()
    data = pd.read_csv('data/table_matrix_data/track_id_to_name/track_id_to_name_2.csv')
    d1.update(data.set_index('track_uri')['track_name'].to_dict())

    data = pd.read_csv('data/table_matrix_data/artist_id_to_name.csv')
    d2 = data.set_index('artist_uri')['artist_name'].to_dict()

    data = pd.read_csv('data/table_matrix_data/album_id_to_name.csv')
    d3 = data.set_index('album_uri')['album_name'].to_dict()
    
    return d1, d2, d3



# --------------------------------------------------------------------------------------
#        TWO-WAY DICTIONARIES
# --------------------------------------------------------------------------------------

# Each track, artist, and album has a unique identifier that maps one-to-one with an index 
# in its respective group. However, names are not guaranteed to be unique. 
# The IdResolver class allows us to build reliable two-way mappings between IDs and indices, 
# while also storing associated names for reference.

class IdResolver:
    '''
    Defines a two-way dictionary or "bidictionary."
    '''
    def __init__(self):
        self.sid_to_idx_dict = {}
        self.idx_to_sid_dict = {}
        self.sid_to_name_dict = {}
        self.idx_to_name_dict = {}
        
    
    def setitem(self, sid, idx, name):
        self.sid_to_idx_dict[sid] = idx
        self.idx_to_sid_dict[idx] = sid
        self.sid_to_name_dict[sid] = name
        self.idx_to_name_dict[idx] = name
    
    def idx_to_sid(self, value):
        return self.idx_to_sid_dict[value]
    
    def sid_to_idx(self, value):
        return self.sid_to_idx_dict[value]
    
    def lookup_name(self, value):
        if isinstance(value, int):
            return self.idx_to_name_dict[value]

        if isinstance(value, str):
            return self.sid_to_name_dict[value]

def get_track_resolver():
    '''
    Builds an IdResolver for tracks.

    Returns a two-way mapping between:
    - track_id (Spotify URI) and its integer index in track space
    - each ID/index and the corresponding track name

    Enables fast lookup in both directions:
    - Use .get_sid(index) to recover the original track ID
    - Use .get_index(track_id) to get the associated index
    - Use .lookup_name(track_id or index) to get the track name
    '''
    data = pd.read_csv('data/table_matrix_data/track_id_to_name/track_id_to_name_1.csv')
    temp = pd.read_csv('data/table_matrix_data/track_id_to_name/track_id_to_name_2.csv')
    data = pd.concat([data, temp], ignore_index = True)

    trackid_to_name = data.set_index('track_uri')['track_name'].to_dict()
    bd = IdResolver()
    idx = 0

    for sid in data['track_uri']:
        name = trackid_to_name[sid]
        bd.setitem(sid, idx, name)
        idx = idx+1
    return bd

def get_artist_resolver():
    '''
    Builds an IdResolver for artists.

    Returns a two-way mapping between:
    - track_id (Spotify URI) and its integer index in artists space
    - each ID/index and the corresponding artist name

    Enables fast lookup in both directions:
    - Use .get_sid(index) to recover the original artist ID
    - Use .get_index(artist_id) to get the associated index
    - Use .lookup_name(artist_id or index) to get the artist name
    '''
    data = pd.read_csv('data/table_matrix_data/artist_id_to_name.csv')
    artistid_to_name = data.set_index('artist_uri')['artist_name'].to_dict()
    bd = IdResolver()
    idx = 0
    
    for sid in data['artist_uri']:
        name = artistid_to_name[sid]
        bd.setitem(sid, idx, name)
        idx = idx+1
    return bd

def get_album_resolver():
    '''
    Builds an IdResolver for albums.

    Returns a two-way mapping between:
    - album_id (Spotify URI) and its integer index in album space
    - each ID/index and the corresponding album name

    Enables fast lookup in both directions:
    - Use .get_sid(index) to recover the original album ID
    - Use .get_index(album_id) to get the associated index
    - Use .lookup_name(album_id or index) to get the album name
    '''
    data = pd.read_csv('data/table_matrix_data/album_id_to_name.csv')
    albumid_to_name = data.set_index('album_uri')['album_name'].to_dict()
    bd = IdResolver()
    idx = 0

    for sid in data['album_uri']:
        name = albumid_to_name[sid]
        bd.setitem(sid, idx, name)
        idx = idx+1
    return bd



# --------------------------------------------------------------------------------------
#        LANGUAGE INTERPRETER
# --------------------------------------------------------------------------------------

# Preloaded data
artist_resolver = get_artist_resolver()
album_resolver = get_album_resolver()
track_resolver = get_track_resolver()

AT = get_artist_track_matrix()
BT = get_album_track_matrix()

def PrettyPrintPlaylists(m: csr_matrix):
    for i in range(m.shape[0]):
        start = m.indptr[i]
        end = m.indptr[i+1]
        print(f"\n=== Playlist {i} (length: {end - start}) ===")
        track_idx = m.indices[start:end]
        for track_id in track_idx:
            track_name = track_resolver.lookup_name(int(track_id))

            artist_id = AT[:, track_id].nonzero()[0][0]
            artist_name = artist_resolver.lookup_name(int(artist_id))

            album_id = BT[:, track_id].nonzero()[0][0]
            album_name = album_resolver.lookup_name(int(album_id))
            print(f"  • Track: {track_name}\n    Artist: {artist_name}\n    Album: {album_name}")



# --------------------------------------------------------------------------------------
#        JSON FUNCTIONS
# --------------------------------------------------------------------------------------

# List of training data json_files
json_files = [f for f in os.listdir("data/split_data/train") if f.endswith('.json')]

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


'''
# --------------------------------------------------------------------------------------
#        TWO-WAY DICTIONARIES
# --------------------------------------------------------------------------------------

# The unique identifiers for each track, artist and album will be in 1-to-1 correspondence with
# the index set for their respective groups. This is not true for the names corresponding to each 
# track, artist, and album id. This allows us to construct 2-way dictionaries which we couldn't do
# before.

class BiDict:
    # Defines a two-way dictionary or "bidictionary."
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
    
    #key: track_id
    #value: track_index in track space
    
    #Use .get_key(track_index) to recover track_id in O(1) time
    
    bd = BiDict()
    i = 0
    # Load first csv file containing track_ids
    data = pd.read_csv('data/table_matrix_data/track_id_to_name/track_id_to_name_1.csv')
    for id in data['track_uri']:
        bd[id] = i
        i = i+1
    # Load second csv file containing track_ids
    data = pd.read_csv('data/table_matrix_data/track_id_to_name/track_id_to_name_2.csv')
    for id in data['track_uri']:
        bd[id] = i
        i = i+1
    return bd

def artist_index_table():

    # key: artist_id
    # value: artist_index in artist space

    # Use .get_key(artist_index) to recover artist_id in O(1) time
    
    bd = BiDict()
    i = 0
    data = pd.read_csv('data/table_matrix_data/artist_id_to_name.csv')
    for id in data['artist_uri']:
        bd[id] = i
        i = i + 1
    return bd

def album_index_table():

    # key: album_id
    # value: album_index in album space

    # Use .get_key(album_index) to recover album_id in O(1) time

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

'''