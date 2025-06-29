import numpy as np
from scipy.sparse import csr_matrix


def create_sparse_indicator_matrix(col_indices, num_cols):
    """
    Converts a list of index lists into a 0-1 CSR sparse matrix.
    
    Parameters:
        col_indices (List[List[int]]): List of row-wise column indices.
        num_cols (int): Total number of columns in the output matrix.

    Returns:
        csr_matrix: Sparse binary matrix of shape (len(col_indices), num_cols).
    """
    nr = len(col_indices)
    nc = len(col_indices[0])
    n = nr * nc
    data = np.ones(n)
    rows = np.zeros(n, dtype=int)
    cols = np.zeros(n, dtype=int)

    count = 0
    for i in range(nr):
        for j in range(nc):
            rows[count] = i
            cols[count] = col_indices[i][j]
            count+= 1

    return csr_matrix((data, (rows,cols)), shape = (nr, num_cols))


def csr_col_split(m: csr_matrix, k: int, split_type = 'shuffle' ):
    """
    Splits each row of a CSR matrix into two disjoint parts by randomly sampling k non-zero entries.

    Parameters:
    ----------
    m : csr_matrix
        The input sparse matrix.
    k : int
        The number of non-zero entries to randomly select from each row.
        If a row has fewer than k non-zero entries, all are included in one split.
    split_type : str, optional
        'shuffle' = random selection
        'largest' = splits off the k-largest values 
        'smallest' = splits off the k-smallest values 

    Returns:
    -------
    a : csr_matrix
        A sparse matrix with the sampled entries removed from each row.
    b : csr_matrix
        A sparse matrix containing only the sampled entries from each row.

    Notes:
    -----
    - The union of a and b reconstructs the original matrix.
    - Sampling is done independently for each row.
    - Both a and b have the same shape as m.
    """
    a = m.copy()
    b = m.copy()

    num_rows = m.shape[0]

    for i in range(num_rows):
        start = m.indptr[i]
        end = m.indptr[i + 1]
        row_len = end - start

        idx = np.arange(start, end)
        r = min(k, row_len)

        if split_type == 'shuffle':
            sample = np.random.choice(row_len, size=r, replace=False)
        elif split_type == 'largest':
            sample = np.argpartition(m.data[start:end], -r)[-r:]
        elif split_type == 'smallest':
            sample = np.argpartition(m.data[start:end], r)[:r]
        else:
            raise ValueError(f"Invalid split_type: '{split_type}'")

        mask = np.zeros(row_len, dtype=bool)
        mask[sample] = True

        a.data[idx[mask]] = 0
        b.data[idx[~mask]] = 0

    a.eliminate_zeros()
    b.eliminate_zeros()
    return a, b



def split_csr_matrix(csr_matrix, n_splits=3):
    '''
    Splits a csr_matrix by rows into n_splits many different sections. Default set at 3 for a top, middle, and 
    bottom section.

    Parameters:
    ----------
    m : csr_matrix
        The input sparse matrix.
    n_splits : int
        The number of equal horizontal splits desired for the csr_matrix.
        Default is 3.

    Returns:
    -------
    An list of length n_splits with each entry a csr_matrix corresponding to a particular
    horizontal section.
    '''
    n_rows = csr_matrix.shape[0]
    chunk_size = n_rows // n_splits
    remainder = n_rows % n_splits
    
    splits = []
    start = 0
    
    for i in range(n_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunk = csr_matrix[start:end]
        splits.append(chunk)
        start = end
    
    return splits
