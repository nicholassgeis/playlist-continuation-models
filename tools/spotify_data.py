# --------------------------------------------------------------------------------------
#        DICTIONARIES
# --------------------------------------------------------------------------------------

# This sections defines helpful look-up tables to translate between english statements like
# "Humble. by Kendrick Lamar", the unique Spotify track uri for the song, and our internal indexing system.

def sti_to_description_table():
    '''
    Defines a lookup table to retrieve the string "[track name] by [artist name]" given a
    spotify track id in O(1) time.
    '''
    data = pd.read_csv('data/simple_universe.csv')
    return data.set_index('key')['value'].to_dict()



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

def track_index_dict(sti_table):
    '''
    Defines a two-way lookup table to index each track by the spotify track id.
    Being two-way means that we can retrieve the index given a spotify track id and
    retrive the spotify track id given an index in O(1) time.
    '''
    bd = BiDict()
    i = 0
    for id in sti_table.keys():
        bd[id] = i
        i = i+1
    return bd

def artist_index_dict(artist_table):
    '''
    Defines a two-way lookup table to index each artist by the spotify artist id.
    Being two-way means that we can retrieve the index given a spotify artist id and
    retrive the spotify artist id given an index in O(1) time.
    '''
    bd = BiDict()
    i = 0
    for id in artist_table.keys():
        bd[id] = i
        i = i + 1
    return bd

def album_index_dict(album_table):
    '''
    Defines a two-way lookup table to index each album by the spotify album id.
    Being two-way means that we can retrieve the index given a spotify album id and
    retrive the spotify album id given an index in O(1) time.
    '''
    bd = BiDict()
    i = 0
    for id in album_table.keys():
        bd[id] = i
        i = i + 1
    return bd

