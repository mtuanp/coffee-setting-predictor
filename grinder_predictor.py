import numpy as np
from keras.models import load_model
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import joblib

class CoffeePredictor:
    def __init__(self):
        self.model: Sequential = None
        self.scaler: StandardScaler = None

    def load_model(self, filename='coffee_grinder_model.keras'):
        self.model = load_model(filename)
        self.scaler = joblib.load('coffee_scaler.joblib')

    def predict(self, coffee_bean_type,coffee_bean_origin_continent,coffee_bean_bitterness_taste,desired_grams_of_extracted_coffee,desired_brewing_time,grams_of_coffee,taste_like):
        if self.model is None:
            raise ValueError('Model not loaded')
        if self.scaler is None:
            raise ValueError('Scaler not loaded')

        inputs = np.array([[coffee_bean_type,coffee_bean_origin_continent,coffee_bean_bitterness_taste,desired_grams_of_extracted_coffee,desired_brewing_time,grams_of_coffee,taste_like]])
        
        standardized_input = self.scaler.transform(inputs)
        return self.model.predict(standardized_input)[0][0]

if __name__ == '__main__':
    predictor = CoffeePredictor()
    predictor.load_model('coffee_grinder_model.keras')

    predicted_output = predictor.predict(1,1,4,32,25,16,10)

    print(f'Predicted output: {predicted_output}')