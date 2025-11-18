from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_model_candidates():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM_RBF": SVC(kernel='rbf', probability=True, C=1.0, gamma='scale', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }
