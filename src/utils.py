def get_train_valid_test_split(option='full_episodes'):
    if option == 'full_episodes':
        train_episodes = gt_dfs['Muppets-02-01-01']
        valid_episodes = gt_dfs['Muppets-02-04-04']
        test_episodes = gt_dfs['Muppets-03-04-03']
    elif option == 'split_episode':
        raise NotImplementedError
    return train_episodes, valid_episodes, test_episodes

