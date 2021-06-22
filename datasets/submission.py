import pandas as pd
import numpy as np

"""
arrs_to_csv: convert your solutions (list of numpy arrays) to the submission format
check_submit_phase1: some preliminary check of the submission for phase 1
evaluate: the function for calculating the score between an estimated graph and the ground truth
"""


def arrs_to_csv(arrs, input_path='submit.csv'):
    """
    This can be used to generate the submission file in .csv

    Parameters:
        arrs: list of your solutions for each dataset; each element should be a numpy array of 0 or 1
        input_path: where to save your file; e.g., submit.csv
    -------

    """
    arrs_str = [arr_to_string(arr) for arr in arrs]
    pd.DataFrame(arrs_str).to_csv(input_path, index=False)


def check_submit_phase1(input_path):
    """
    Preliminary check of whether your submission is compatible with the system

    Parameters
        input_path: input_path: where to load your file; e.g., submit.csv

    """
    #  shapes of datasets in phase 1.
    mat_shapes_phase1 = [11, 15, 17, 18, 10, 10, 11, 11, 11, 15, 18, 18, 20, 28, 26, 22, 18, 11, 15, 18]

    arrs = csv_to_arrs(input_path)
    if len(arrs) != len(mat_shapes_phase1):
        raise ValueError('Number of solutions are not correct')

    for i, mat_int in enumerate(arrs):
        if mat_int.shape[0] != mat_shapes_phase1[i]:
            raise ValueError('matrix {} has an incorrect shape'.format(i))

    print('Preliminary check of phase 1: OK')


def evaluate(est_graph_matrix, true_graph_matrix):
    """
    parameters:
        est_graph_matrix: np.ndarray, 0-1 adjacency matrix for the estimated graph
        true_graph_matrix:np.ndarray, 0-1 adjacency matrix for the true graph
    return:
        A score ranges from 0 to 1
    """
    W_p = pd.DataFrame(est_graph_matrix).applymap(lambda elem:1 if elem!=0 else 0)
    W_true = pd.DataFrame(true_graph_matrix).applymap(lambda elem:1 if elem!=0 else 0)
    num_true = W_true.sum(axis=1).sum()
    assert num_true!=0
    # true_positives
    num_tp =  (W_p + W_true).applymap(lambda elem:1 if elem==2 else 0).sum(axis=1).sum()
    # False Positives + Reversed Edges
    num_fn_r = (W_p - W_true).applymap(lambda elem:1 if elem==1 else 0).sum(axis=1).sum()
    score = np.max((num_tp-num_fn_r,0))/num_true
    return score


def arr_to_string(mat):
    """
    Parameters
        mat: numpy array with each entry either 0 or 1

    Returns:
        string of the input array
    """
    mat_int = mat.astype(int)
    mat_flatten = mat_int.flatten().tolist()
    for m in mat_flatten:
        if m not in [0, 1]:
            raise TypeError("Value not in {0, 1}.")
    mat_str = ' '.join(map(str, mat_flatten))
    return mat_str


def csv_to_arrs(input_path):
    """
    read submission csv and transmit it back to list of numpy arrays

    Parameters
        input_path: where to load your file; e.g., submit.csv

    Returns
        a list of numpy arrays
    -------

    """
    arrs = []
    arrs_csv = pd.read_csv(input_path)
    arrs_str = arrs_csv.values.tolist()
    for arr in arrs_str:
        mat_flatten = np.fromstring(arr[0], dtype=int, sep=' ')
        n = int(np.sqrt(len(mat_flatten)))
        mat_int = mat_flatten.reshape(n, n)
        arrs.append(mat_int)
    return arrs