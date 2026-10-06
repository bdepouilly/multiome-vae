from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def knn_scores(Z_train, y_train, Z_test, y_test, n_neighbors = 15):
    """Evaluation function that fits a kNN on the training data and returns its accuracy, balanced accuracy and f1 macro score on test data."""
    kNN = KNeighborsClassifier(n_neighbors)
    kNN.fit(Z_train, y_train)
    
    y_pred = kNN.predict(Z_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    return {"accuracy" : accuracy,
            "balanced_accuracy" : balanced_accuracy,
            "f1_score_macro" : f1}