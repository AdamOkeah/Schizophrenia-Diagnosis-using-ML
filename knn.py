import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('schizophrenia_dataset.csv')

# Ensure 'Diagnosis' column exists
if 'Diagnosis' not in df.columns:
    raise KeyError("Column 'Diagnosis' is missing from the dataset.")

# Convert categorical features to numeric using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Convert 'Diagnosis' to numeric if necessary
if df['Diagnosis'].dtype == 'object':
    df['Diagnosis'] = df['Diagnosis'].astype('category').cat.codes

# Compute the correlation matrix (only numeric columns)
correlation_matrix = df.corr()

# Ensure 'Diagnosis' column is still present
if 'Diagnosis' not in correlation_matrix.columns:
    raise KeyError("After processing, 'Diagnosis' is still missing in correlation matrix.")

# Select correlations with the target variable
target_correlation = correlation_matrix['Diagnosis'].sort_values(ascending=False)

# Display correlations
print("\nCorrelation with Diagnosis:")
print(target_correlation)

# Plot correlations with the target variable
plt.figure(figsize=(12, 8))
target_correlation.drop('Diagnosis', errors='ignore').plot(kind='bar')
plt.title('Correlation with Diagnosis', fontsize=16)
plt.ylabel('Correlation Coefficient', fontsize=14)
plt.xlabel('Features', fontsize=14)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Select features with correlation above the threshold
threshold = 0.003  # Adjust as needed
high_corr_features = target_correlation[abs(target_correlation) > threshold].index.tolist()
if "Diagnosis" in high_corr_features:
    high_corr_features.remove("Diagnosis")

print("\nHigh Correlated Features:", high_corr_features)

# Define features and target
X = df[high_corr_features]
y = df["Diagnosis"]


missing_values = df.isnull().sum()
print(missing_values)

# Scale and normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.astype('float32'))

# Scale and normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X.astype('float32'))

# Apply SMOTE to oversample the minority class
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(y_resampled.value_counts())

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize the KNN classifier with a specific number of neighbors
knn = KNeighborsClassifier(n_neighbors=4)

# Fit the classifier to the resampled training data
knn.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.3f}")

# Generate classification report as a dictionary
report = classification_report(y_test, y_pred, output_dict=True)

# Convert the dictionary to a pandas DataFrame for better formatting
report_df = pd.DataFrame(report).transpose()

# Round the metrics to 3 decimal places
report_df = report_df.round(3)

# Display the formatted classification report
print("\nClassification Report:")
print(report_df)

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('KNN Confusion Matrix')
plt.show()
