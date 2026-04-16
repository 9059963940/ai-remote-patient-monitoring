import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('data/patient_data.csv')

X = data[['heart_rate', 'oxygen_level', 'temperature']]
y = data['risk']

model = RandomForestClassifier()
model.fit(X, y)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")