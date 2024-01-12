import sklearn
import pandas as pd
from utils import eval
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

def train_full_cv(final_df, clf_config, train_cfg, split_ids, train_cols, target_col, split_col, save_results):
    eval_dfs = []
    clf_dict_outer = {cfg['model_name']: [] for cfg in clf_config}
    tprs_dict = {cfg['model_name']: [] for cfg in clf_config}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for test_split_id in split_ids:
        inner_cv_splits = [s for s in split_ids if s != test_split_id]

        test_X = final_df[final_df.episode_split==test_split_id][train_cols]
        test_y = final_df[final_df.episode_split==test_split_id][target_col]

        print(f'Train splits: {inner_cv_splits}, Test split: {test_split_id}')

        for i, cfg in enumerate(clf_config):
            print(f'Training {cfg["model_name"]}...')
            param_grid = cfg['param_grid']
            model_name = cfg['model_name']
            clf = cfg['model']
            clf = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

            # prepare data for inner CV
            inner_df = final_df[final_df.episode_split.isin(inner_cv_splits)]
            train_X = inner_df[train_cols]
            train_y = inner_df[target_col]

            # find best hyperparams on validation set with inner CV
            # train model with best hyperparams on full inner CV set / outer CV train set
            clf_dict = train_inner_cv(train_X, train_y, inner_df, split_col, clf, param_grid, model_name)
            clf_dict_outer[model_name].append(clf_dict)

            # evaluate model with outer CV test set
            predict_y = clf_dict['model'].predict(test_X)
            eval_dict = eval.eval_prediction(test_y, predict_y, model_name=model_name, config=train_cfg)

            eval_dict['target'] = target_col
            eval_dict['test_split'] = test_split_id

            eval_dfs.append(pd.DataFrame.from_dict(eval_dict, orient='index').T)

            # create ROC curve, get
            interp_tpr = eval.create_roc_curve_get_tpr_count(test_X, test_y, clf_dict, test_split_id, axes[i])
            tprs_dict[model_name].append(interp_tpr)

    eval_df = pd.concat(eval_dfs) 

    # add averaged ROC to the ROC plot
    mean_fpr = np.linspace(0, 1, 100)
    for i, cfg in enumerate(clf_config):
        mean_tpr = np.mean(tprs_dict[cfg['model_name']], axis=0)
        mean_tpr[-1] = 1.0
        axes[i].plot(
            mean_fpr,
            mean_tpr,
            color="b",
            label=f"Mean ROC ({model_name})",
            lw=2,
            alpha=0.8,
        )
        axes[i].set(title=f"{cfg['model_name']}")
    
    if save_results:
        eval_df.to_csv(f"../data/eval/DT_{target_col}_video_eval_df.csv", index=False)
        pickle.dump(clf_dict_outer, open(f"../data/models/DT_{target_col}_video_clf_info.pkl", "wb"))
        pickle.dump(tprs_dict, open(f"../data/models/DT_{target_col}_video_tpr_info.pkl", "wb"))
        plt.savefig(f"../data/eval/DT_{target_col}_video_precision_recall.png")

    fig.set_visible(not fig.get_visible())
    plt.draw()

    return eval_df, clf_dict_outer, tprs_dict


def train_inner_cv(train_X, train_y, inner_df, split_col, clf, param_grid, model_name):
    clf_dict = {}
    ps = PredefinedSplit(inner_df[split_col].values)
    grid_search = GridSearchCV(clf, param_grid, cv=ps, n_jobs=-1, scoring=['f1', 'precision', 'recall', 'balanced_accuracy'], refit='f1')
    
    grid_search.fit(train_X, train_y)

    clf_dict['model_name'] = model_name
    clf_dict['model'] = grid_search.best_estimator_
    clf_dict['params'] = grid_search.best_params_
    clf_dict['f1_inner'] = grid_search.best_score_
    clf_dict['cv_results_'] = grid_search.cv_results_

    return clf_dict



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