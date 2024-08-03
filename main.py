from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()

# Visualize Sepal Dimensions
plt.figure(figsize=(8, 6))
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, cmap='viridis', edgecolor='black', alpha=0.9)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Dimensions')
plt.legend(handles=plt.scatter([], []).legend_elements()[0], labels=iris.target_names, loc="lower right", title="Classes")
plt.show()

# Visualize Petal Dimensions
plt.figure(figsize=(8, 6))
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target, cmap='viridis', edgecolor='black', alpha=0.9)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Dimensions')
plt.legend(handles=plt.scatter([], []).legend_elements()[0], labels=iris.target_names, loc="lower right", title="Classes")
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict the test dataset
y_predicted = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predicted)
print(f"Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_predicted)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_predicted, target_names=iris.target_names))
