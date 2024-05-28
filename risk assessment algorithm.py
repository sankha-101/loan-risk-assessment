import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv("C:\\Users\\Sarthak\\Desktop\\dataset.csv")
print(data.head())
print(data.info())

label_encoder = LabelEncoder()
data['age'] = label_encoder.fit_transform(data['age'])
data['employment'] = label_encoder.fit_transform(data['employment'])
data['education'] = label_encoder.fit_transform(data['education'])
data['marital_status'] = label_encoder.fit_transform(data['marital_status'])
data['income'] = label_encoder.fit_transform(data['income'])
data['loan_amount'] = label_encoder.fit_transform(data['loan_amount'])
data['loan_term'] = label_encoder.fit_transform(data['loan_term'])
data['default'] = label_encoder.fit_transform(data['default'])


X = data.drop(columns=['default'])
y = data['default']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Explore the dataset to gain insights
sns.pairplot(data, hue='default')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)


y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Model Accuracy:", accuracy_best)
print("Best Model Classification Report:")
print(classification_report(y_test, y_pred_best))

