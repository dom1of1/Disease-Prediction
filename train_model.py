import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle

# Load training dataset
train_data = pd.read_csv('data/Training.csv')

# Drop unnamed or extra columns
train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed')]

# Fill NaNs with 0s
train_data.fillna(0, inplace=True)

# Features and target
X_train = train_data.drop('prognosis', axis=1)
y_train = train_data['prognosis']

# Save the symptom list for use in the app
with open('models/symptoms.pkl', 'wb') as f:
    pickle.dump(X_train.columns.tolist(), f)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Save model
with open('models/trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)