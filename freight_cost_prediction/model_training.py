from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


def train_models(X_train, y_train):
    """
    Train multiple regression models.
    """

    model1 = LinearRegression()
    model1.fit(X_train, y_train)

    model2 = DecisionTreeRegressor(random_state=42)
    model2.fit(X_train, y_train)

    model3 = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model3.fit(X_train, y_train)

    return model1, model2, model3