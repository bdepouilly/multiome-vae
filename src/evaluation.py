# Short evaluation function that fit a kNN on the training data and returns its accuracy, balanced accuracy and f1 macro score on test data.

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

def knn_scores(Z_train, y_train, Z_test, y_test, n_neighbors = 15):
    scale_and_classify = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors))
    scale_and_classify.fit(Z_train, y_train)
    
    y_pred = scale_and_classify.predict(Z_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    
    return {"accuracy" : accuracy,
            "balanced_accuracy" : balanced_accuracy,
            "f1_score_macro" : f1}