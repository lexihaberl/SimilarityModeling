import sklearn
import pandas as pd
from utils import eval

def train_eval_inner_cv(model, model_name, train_cols, target_col, config, feat_df, gt_df, episode_names):
    inner_cv_folds_ids = [(0, 1), (1, 0)]
    eval_dfs = []
    model_info = {}

    for test_ep in episode_names:
        test_X = feat_df[feat_df.episode == test_ep][train_cols]
        test_y = gt_df[gt_df.episode == test_ep][target_col]

        episode_names_inner = [ep for ep in episode_names if ep != test_ep]

        for inner_cv_fold_id in inner_cv_folds_ids:
            train_ep = episode_names_inner[inner_cv_fold_id[0]]
            val_ep = episode_names_inner[inner_cv_fold_id[1]]

            train_X = feat_df[feat_df.episode == train_ep][train_cols]
            train_y = gt_df[gt_df.episode == train_ep][target_col]

            val_X = feat_df[feat_df.episode == val_ep][train_cols]
            val_y = gt_df[gt_df.episode == val_ep][target_col]

            clf = sklearn.base.clone(model)
            clf.fit(train_X, train_y)

            pred_y = clf.predict(val_X)

            # collect model info & predictions
            model_info[f'train_{train_ep}_val_{val_ep}'] = {}
            model_info[f'train_{train_ep}_val_{val_ep}']['model'] = clf
            model_info[f'train_{train_ep}_val_{val_ep}']['y'] = val_y
            model_info[f'train_{train_ep}_val_{val_ep}']['pred_y'] = pred_y
            
            # add eval info to the df
            eval_df = eval.eval_prediction(val_y, pred_y, model_name=model_name, config=config)

            eval_df['target'] = target_col
            eval_df['type'] = 'validation'
            eval_df['test_fold'] = test_ep
            eval_df['train_fold'] = train_ep
            eval_df['valid_fold'] = val_ep
            eval_df['val_len'] = len(val_y)

            eval_df = pd.DataFrame.from_dict(eval_df, orient='index').T
            eval_dfs.append(eval_df)

    eval_df_all = pd.concat(eval_dfs)

    return eval_df_all, model_info



def train_eval_2_to_1(model, model_name, train_cols, target_col, config, feat_df, gt_df, episode_names):
    inner_cv_folds_ids = [(0, 1), (1, 0)]
    eval_dfs = []
    model_info = {}

    for test_ep in episode_names:
        test_X = feat_df[feat_df.episode == test_ep][train_cols]
        test_y = gt_df[gt_df.episode == test_ep][target_col]

        train_X = feat_df[feat_df.episode != test_ep][train_cols]
        train_y = gt_df[gt_df.episode != test_ep][target_col]

        clf = sklearn.base.clone(model)
        clf.fit(train_X, train_y)

        pred_y = clf.predict(test_X)

        # collect model info & predictions
        model_info[f'test_{test_ep}'] = {}
        model_info[f'test_{test_ep}']['model'] = clf
        model_info[f'test_{test_ep}']['y'] = test_y
        model_info[f'test_{test_ep}']['pred_y'] = pred_y
        
        # add eval info to the df
        eval_df = eval.eval_prediction(test_y, pred_y, model_name=model_name, config=config)

        eval_df['target'] = target_col
        eval_df['type'] = 'testing'
        eval_df['test_fold'] = test_ep
        eval_df['train_fold'] = f"{', '.join([ep for ep in episode_names if ep != test_ep])}"
        eval_df['test_len'] = len(test_y)

        eval_df = pd.DataFrame.from_dict(eval_df, orient='index').T
        eval_dfs.append(eval_df)

    eval_df_all = pd.concat(eval_dfs)

    return eval_df_all, model_info