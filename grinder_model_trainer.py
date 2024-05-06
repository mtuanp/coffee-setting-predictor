from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

"""
    Coffee grinder model trainer, it should create the sequential model and feed it with the trainingsdata. 
"""
class CoffeeModelTrainer:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None

    def load_data(self, filename='coffee_training_data.csv'):
        # Load coffee training data from CSV file
        self.data = np.genfromtxt(filename, skip_header=1, delimiter=',')

    def preprocess_data(self):
        # Separate input and output values
        X = self.data[:, :-1]
        y = self.data[:, -1]
        # Standardize the data using a scaler
        self.scaler = StandardScaler()
        standardized_X = self.scaler.fit_transform(X)

        return standardized_X, y

    def define_model(self, dimensionSize):
        self.model = Sequential()
        self.model.add(Input(shape=(dimensionSize,)))  # Input layer with shape derived from input data
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train_model(self, standardized_X, y):
        # Split the data into training and testing sets
        train_X, test_X, train_y, test_y = train_test_split(standardized_X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(train_X, train_y, epochs=500, verbose=0, batch_size=32)

        return test_X, test_y

    def evaluate_model(self, test_X, test_y):
        # Evaluate the trained model
        mse = self.model.evaluate(test_X, test_y)

        return mse

    def save_model(self, filename='coffee_grinder_model.keras', scalaername='coffee_scaler.joblib'):
        # Save the trained model and scaler to a file
        self.model.save(filename)
        joblib.dump(self.scaler, scalaername)

if __name__ == '__main__':
    trainer = CoffeeModelTrainer()
    trainer.load_data()
    standardized_X, y = trainer.preprocess_data()
    trainer.define_model(standardized_X.shape[1])
    test_X, test_y = trainer.train_model(standardized_X, y)
    mse = trainer.evaluate_model(test_X, test_y)
    print(f'Model mean squared error: {mse}')
    trainer.save_model()