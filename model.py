import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



def load_data(csv_file):
    df = pd.read_csv("modified_diabetes.csv")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y

def train_model(X, y):
    # Initialize and fit the scaler
    model = GradientBoostingClassifier()
    model.fit(X, y)
    return model

def predict(model, input_data):
    return model.predict([input_data])

# Example usage
if __name__ == "__main__":
    X, y = load_data('your_preprocessed_data.csv')
    model = train_model(X, y)
    # Test the model with some input data
    # print(predict(model, [1, 85, 66, 29, 0, 26.6, 0.351, 31]))  # Example input
