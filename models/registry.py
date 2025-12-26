from sklearn.linear_model import LogisticRegression

def get_model_candidates():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }
