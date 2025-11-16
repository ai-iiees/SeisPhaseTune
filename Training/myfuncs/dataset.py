import numpy as np

def manual_spliter(dataset,
                   mode='PS-Pairs',
                   precentages={'train': 0.9, 'dev': 0.05, 'test': 0.05},
                   random=True,
                   random_state=None):
    """
    Manually splits a SeisBench dataset into train/dev/test subsets based on a filtering condition.

    This function modifies the `dataset.metadata` DataFrame in-place by adding a `split` column
    with values `'train'`, `'dev'`, or `'test'` according to the specified proportions and mode.

    Parameters
    ----------
    dataset : seisbench.data.dataset.WaveformDataset
        The dataset object whose metadata will be used for splitting.
    
    mode : str, optional (default='PS-Pairs')
        Determines which subset of the data to split:
        - 'PS-Pairs': only rows where `PS-pairs` is True.
        - 'not PS-Pairs': only rows where `PS-pairs` is False.
        - 'All': use all rows regardless of `PS-pairs`.

    precentages : dict, optional (default={'train': 0.9, 'dev': 0.05, 'test': 0.05})
        Dictionary defining the proportion of samples to assign to each split.
        The values must sum to 1.0.

    random : bool, optional (default=True)
        If True, shuffle the data before splitting.

    random_state : int or None, optional
        Random seed for reproducible shuffling if `random` is True.

    Raises
    ------
    ValueError
        If the sum of `precentages` values does not equal 1.0,
        or if `mode` is not one of the accepted values.

    Returns
    -------
    None
        The function modifies the dataset in-place by assigning the split labels in `dataset.metadata['split']`.

    Notes
    -----
    This function is particularly useful when you want full control over how a dataset is partitioned
    before training machine learning models.
    """
    if not np.isclose(sum(precentages.values()), 1.0):
        raise ValueError("percentages must sum to 1.0")
    ###
    df = dataset.metadata
    if mode=='PS-Pairs':
        df_subset = df[df['PS-pairs']]
    elif mode == 'not PS-Pairs':
        df_subset = df[~df['PS-pairs']]
    elif mode=='All':
        df_subset = df
    else:
        raise ValueError("mode must be 'PS-Pairs', 'not PS-Pairs', or 'All'")
    ###
    num_elements = len(df_subset)
    print("Number of selected sample:", num_elements)
    n_train = int(num_elements * precentages['train'])
    n_dev = int(num_elements * precentages['dev'])
    n_test = num_elements - n_train - n_dev
    ###
    indexs = df_subset.index.to_numpy()
    if random:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indexs)
    ###
    dataset.metadata['split'] = 'Undefined'
    dataset.metadata.loc[indexs[: n_train], 'split'] = 'train'
    dataset.metadata.loc[indexs[n_train: n_train+n_dev], 'split'] = 'dev'
    dataset.metadata.loc[indexs[n_train+n_dev: ], 'split'] = 'test'
