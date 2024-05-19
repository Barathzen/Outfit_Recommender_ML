from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Load the dataset from CSV
df = pd.read_csv('ml proj.csv')

# Combine the last three columns into a single column 'Outfit'
df['Outfit'] = df['Outfit1'] + ', ' + df['Outfit2'] + ', ' + df['Outfit3']

# Drop 'Outfit1', 'Outfit2', and 'Outfit3' columns
df.drop(['Outfit1', 'Outfit2', 'Outfit3'], axis=1, inplace=True)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['Weather', 'Color'])

# Split the data into training and testing sets
X = df_encoded.drop('Outfit', axis=1)
y = df_encoded['Outfit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    weather = request.form['weather']
    color = request.form['color']

    # Create a dictionary for the user input
    user_input = {'Weather': weather, 'Color': color}

    # Encode the user input
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_encoded = pd.get_dummies(user_input_df, columns=['Weather', 'Color'])

    # Reindex user_input_encoded to include all possible columns and reorder columns
    user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

    # Make recommendation
    recommended_outfit = knn.predict(user_input_encoded)
    return jsonify({'Recommended outfit': recommended_outfit[0]})

if __name__ == '__main__':
    app.run(debug=True)
