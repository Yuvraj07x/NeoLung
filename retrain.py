import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('survey lung cancer.csv')

# Preprocess as per notebook
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 2})
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 2})

# Features and target
x = df.iloc[:, :-1]
y = df['LUNG_CANCER']

# Check class distribution
print("Original class distribution:")
print(y.value_counts(normalize=True))

# Separate classes
no_cancer = df[df['LUNG_CANCER'] == 2]
yes_cancer = df[df['LUNG_CANCER'] == 1]

# Undersample majority (YES cancer) to match minority (NO)
yes_cancer_undersampled = yes_cancer.sample(n=len(no_cancer), random_state=42)
balanced_df = pd.concat([no_cancer, yes_cancer_undersampled])

# New features/target
x_bal = balanced_df.iloc[:, :-1]
y_bal = balanced_df['LUNG_CANCER']

print("Balanced class distribution:")
print(y_bal.value_counts(normalize=True))

# Split
x_train, x_test, y_train, y_test = train_test_split(x_bal, y_bal, test_size=1/3, random_state=42)

# Train RandomForest (same as notebook)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Evaluate
accuracy = rf_model.score(x_test, y_test)
print(f"Balanced model accuracy on test: {accuracy:.4f}")

# Save
joblib.dump(rf_model, 'rf_model.pkl')
print("Unbiased model saved as rf_model.pkl")
