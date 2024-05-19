import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Get user input
weather = input('Enter the weather condition (e.g., Sunny, Foggy, Rainy, etc.): ')
color = input('Enter the color (e.g., Red, Blue, Green, etc.): ')

# Create a dictionary for the user input
user_input = {'Weather': weather, 'Color': color}

# Encode the user input
user_input_encoded = pd.get_dummies(pd.DataFrame(user_input, index=[0]), columns=['Weather', 'Color'])

# Reindex user_input_encoded to include all possible columns and reorder columns
user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

# Make recommendation
recommended_outfit = knn.predict(user_input_encoded)
print(f'Recommended outfit: {recommended_outfit[0]}')
