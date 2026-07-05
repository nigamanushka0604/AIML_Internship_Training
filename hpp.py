# hpp.py
import joblib
import pandas as pd

def predict_rent(new_house_dict):
    hpp = joblib.load("house_rent_prediction.pkl")
    encoder = hpp['encoder']
    scaler = hpp['scaler']
    model = hpp['model']
    x_train_columns = hpp['x_train_columns']

    new_house = pd.DataFrame(new_house_dict)
    new_house = new_house.reindex(columns=x_train_columns)

    new_encoded = encoder.transform(new_house)
    new_scaled = scaler.transform(new_encoded)
    prediction = model.predict(new_scaled)
    return prediction[0]
