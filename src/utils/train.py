from sklearn.model_selection import cross_validate

def train_model(model, model_name, config, X, y, scoring = ['precision', 'recall']):
    scores = cross_validate(model, X, y, scoring=scoring)

    train_dict = {
        "model_name": model_name,
        "config": config,
        "precisions": scores['test_precision'],
        "recalls": scores['test_recall'],
        "precision_mean": scores['test_precision'].mean(),
        "recall_mean": scores['test_recall'].mean()
    }

    model = model.fit(X, y)

    # return model fitted on the entire train X
    return model, train_dict
