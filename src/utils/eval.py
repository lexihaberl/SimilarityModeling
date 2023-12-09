from sklearn.metrics import precision_recall_fscore_support

def eval_prediction(y_true, y_pred, model_name=None, config=None):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1)
    
    eval_dict = {
        "model": model_name,
        "config": config,
        "precision": p,
        "recall": r,
        "f1": f,
        "support": s
    }

    return eval_dict

def create_roc_curve():
    raise NotImplementedError