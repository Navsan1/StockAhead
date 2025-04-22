from sklearn.ensemble import RandomForestRegressor

def build_model(X_train, y_train):
    """
    Builds and trains the Random Forest model.
    :param X_train: The training input data
    :param y_train: The training output data
    :return: Trained model
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model