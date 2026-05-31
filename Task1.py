import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df = pd.read_csv("Iris.csv")

print("First 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nInfo:\n")
print(df.info())

print("\nStatistical Summary:\n", df.describe())

print("\nClass Distribution:\n", df['Species'].value_counts())
# Drop unnecessary column
df = df.drop("Id", axis=1)

# Encode target labels
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

print("\nAfter Encoding:\n", df.head())
sns.pairplot(df, hue='Species')
plt.show()
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()
df.hist()
plt.show()
plt.figure()
sns.boxplot(data=df)
plt.title("Boxplot of Features")
plt.show()
X = df.drop("Species", axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
