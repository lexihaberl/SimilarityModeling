from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score
import numpy as np
from sklearn.metrics import roc_curve, RocCurveDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull


def eval_prediction(y_true, y_pred, model_name=None, config=None):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    acc = balanced_accuracy_score(y_true, y_pred)

    tnr = 2*acc - r
    fpr = 1 - tnr
    
    eval_dict = {
        "model": model_name,
        "config": config,
        "precision": p,
        "recall": r,
        "tpr": max(0.0, r),
        "fpr": max(0.0, fpr),
        "f1": f,
        "support": s,
        "acc": acc
    }

    return eval_dict


def create_roc_curve_get_tpr_count(test_X, test_y, clf_dict, test_split_id, ax):
    # plot ROC curve (wiki definition)
    try:
        viz = RocCurveDisplay.from_estimator(
            clf_dict['model'],
            test_X,
            test_y,
            name=f"ROC fold {test_split_id}",
            alpha=0.3,
            lw=1,
            ax=ax
        )
    except:
        pred_y_prob = clf_dict['model'].predict_proba(test_X)
        viz = RocCurveDisplay.from_predictions(
            test_y, 
            pred_y_prob[:, 1],
            name=f"ROC fold {test_split_id}",
            alpha=0.3,
            lw=1,
            ax=ax
        )

    mean_fpr = np.linspace(0, 1, 100)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0

    return interp_tpr


def is_point_below_line(x, y, x1=0, y1=1, x2=1, y2=0, intercept=1):
    slope = (y2 - y1) / float(x2 - x1)
    expected_y = slope * x + intercept
    return y < expected_y


def is_point_below_line2(x, y):
    slope = (1 - 0) / (1 - 0)
    intercept = 0
    expected_y = slope * x + intercept
    return y < x


def plot_precision_recall_curve(precision_dict, recall_dict, model_names, title, ax=None, plot_curve=True, sim_mod='SimMod1'):
    if sim_mod == 'SimMod1':
        colors = {
            'RandomForest': '#1f77b4',
            'DecisionTree': '#ff7f0e',
            'KNN': '#2ca02c',
        }
    else:
        colors = {
            'linSVC': '#1f77b4',
            'NaiveBayes': '#ff7f0e',
            'GM': '#2ca02c',
            'GaussianMixture': '#2ca02c',
        }
    
    if not ax:
        _, ax = plt.subplots(figsize=(6, 6))

    for model_name in model_names:
        precision_points = precision_dict[model_name].copy()
        recall_points = recall_dict[model_name].copy()

        precision_points += [0, 1]
        recall_points += [1, 0]

        sorted_indices = sorted(range(len(recall_points)), key=lambda k: recall_points[k])
        precision_points = [precision_points[i] for i in sorted_indices]
        recall_points = [recall_points[i] for i in sorted_indices]

        if plot_curve:
            hull = ConvexHull(np.array(list(zip(precision_points, recall_points))))
            
            for simplex in hull.simplices:
                x_simplex = [precision_points[j] for j in simplex]
                y_simplex = [recall_points[j] for j in simplex]
                are_points_below_line = sum(is_point_below_line(x, y) for x, y in zip(x_simplex, y_simplex))
                if are_points_below_line < 1:
                    sns.lineplot(x=x_simplex, y=y_simplex, alpha=1, c=colors[model_name], ax=ax)

        sns.scatterplot(x=precision_points[1:-1], y=recall_points[1:-1], label=model_name, ax=ax, color=colors[model_name])
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
    ax.set(xlabel='Precision', ylabel='Recall', title=title)


def plot_roc_curve(fpr_dict, tpr_dict, model_names, title, ax=None, plot_curve=True, sim_mod='SimMod1'):
    if sim_mod == 'SimMod1':
        colors = {
            'RandomForest': '#1f77b4',
            'DecisionTree': '#ff7f0e',
            'KNN': '#2ca02c',
        }
    else:
        colors = {
            'linSVC': '#1f77b4',
            'NaiveBayes': '#ff7f0e',
            'GM': '#2ca02c',
            'GaussianMixture': '#2ca02c'
        }

    if not ax:
        _, ax = plt.subplots(figsize=(6, 6))

    for model_name in model_names:
        fpr_points = fpr_dict[model_name].copy()
        tpr_points = tpr_dict[model_name].copy()

        fpr_points += [0, 1]
        tpr_points += [0, 1]

        sorted_indices = sorted(range(len(fpr_points)), key=lambda k: tpr_points[k])
        fpr_points = [fpr_points[i] for i in sorted_indices]
        tpr_points = [tpr_points[i] for i in sorted_indices]
        if plot_curve:
            hull = ConvexHull(np.array(list(zip(fpr_points, tpr_points))))
            
            for simplex in hull.simplices:
                x_simplex = [fpr_points[j] for j in simplex]
                y_simplex = [tpr_points[j] for j in simplex]
                are_points_below_line = sum(y < x for x, y in zip(x_simplex, y_simplex))
                if are_points_below_line < 1:
                    sns.lineplot(x=x_simplex, y=y_simplex, alpha=0.5, c=colors[model_name], ax=ax)

        sns.scatterplot(x=fpr_points[1:-1], y=tpr_points[1:-1], label=model_name, ax=ax, alpha=0.5, color=colors[model_name])
    
    ax.set(xlabel='FPR', ylabel='TPR', title=title)
    # ax.set_ylim(0, 1)
    # ax.set_xlim(0, 1)
    sns.lineplot(x=[0, 1], y=[0, 1], color='gray', ax=ax, linestyle='--')

def get_eval_info(eval_df, clf_dict_outer):
    n = 5
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    fpr_dict = {}

    for model_name in clf_dict_outer.keys():
        precision_dict[model_name] = []
        recall_dict[model_name] = []
        f1_dict[model_name] = []
        fpr_dict[model_name] = []

        for model in clf_dict_outer[model_name]:
            for i in range(n):
                scores = model['cv_results_']
                scores[f'split{i}_test_recall'] = list([max(0.0, r) for r in scores[f'split{i}_test_recall']])
                precision_dict[model_name] += scores[f'split{i}_test_precision'].tolist()
                recall_dict[model_name] += scores[f'split{i}_test_recall']
                f1_dict[model_name] += scores[f'split{i}_test_f1'].tolist()

                tnr = 2*scores[f'split{i}_test_balanced_accuracy'] - scores[f'split{i}_test_recall']
                fpr_dict[model_name] += list([max(0.0, fpr) for fpr in (1 - tnr).tolist()])

            precision_dict[model_name] += eval_df[eval_df.model==model_name].precision.tolist()
            recall_dict[model_name] += eval_df[eval_df.model==model_name].recall.tolist()
            f1_dict[model_name] += eval_df[eval_df.model==model_name].f1.tolist()

            fpr_dict[model_name] += eval_df[eval_df.model==model_name].fpr.tolist()

    tpr_dict = recall_dict

    return precision_dict, recall_dict, f1_dict, fpr_dict, tpr_dict


def get_feature_importance_rf(train_cols, clf_dict_outer):
    feature_importance_dfs = []

    for i, model_dict in enumerate(clf_dict_outer['RandomForest']):
        forest = model_dict['model']['clf']
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        
        forest_importances = pd.Series(importances, index=train_cols)
        df = pd.DataFrame(forest_importances).reset_index(drop=False)
        df.columns = ['feature', 'importance']
        df['std'] = std
        df['split_id'] = i
        feature_importance_dfs.append(df)

    feature_importance_df = pd.concat(feature_importance_dfs)
    return feature_importance_df