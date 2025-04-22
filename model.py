# model.py
from sklearn.ensemble import RandomForestRegressor

def build_model(X_train, y_train):
    """
    Builds and trains Random Forest model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model