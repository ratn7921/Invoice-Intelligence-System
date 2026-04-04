import pandas as pd


def predict_freight(model, quantity, dollars):
    """
    Predict freight cost for new order.
    """

    input_data = pd.DataFrame({
        "Quantity": [quantity],
        "Dollars": [dollars]
    })

    prediction = model.predict(input_data)

    return prediction[0]