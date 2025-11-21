from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

file_path = 'data/congregate_scores.csv'
data = pd.read_csv(file_path)

score_columns = ['Top score', 'Jng score', 'Mid score', 'Bot score', 'Sup score']
target_column = 'Result'

for col in score_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data_cleaned = data.dropna()

features = data_cleaned[score_columns]
target = data_cleaned[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Loss', 'Win'])

feature_importances = pd.Series(
    rf_model.feature_importances_, index=features.columns
).sort_values(ascending=False)

print("Classification Report:\n", report)
print("\nFeature Importances:\n", feature_importances)