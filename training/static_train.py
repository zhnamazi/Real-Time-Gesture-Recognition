import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.svm import SVC

CSV_FILE = "./dataset/hand_gestures_dataset_v6.csv"
MODEL_PATH = "./training/gesture_classifier_mlp.pkl"
# MODEL_PATH = "gesture_classifier_rf.pkl"
# MODEL_PATH = "gesture_classifier_svm.pkl"
SCALER_PATH = "./training/gesture_scaler.pkl"

print("Loading dataset...")
df = pd.read_csv(CSV_FILE)

print(f"Dataset shape: {df.shape}")
print(f"Classes: {df['label'].unique()}")
print(f"Samples per class:\n{df['label'].value_counts()}")

# Prepare features and labels
X = df.iloc[:, 1:].values  # All columns except label
y = df['label'].values

print(f"\nFeature shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build MLP (lightweight for Raspberry Pi)
print("\nTraining MLP classifier...")
classifier = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 layers
    # hidden_layer_sizes=(64),  # 2 layers
    activation='relu',
    max_iter=3,
    learning_rate_init=0.001,
    batch_size=16,
    early_stopping=True,
    validation_fraction=0.1,
    alpha=0.01,
    random_state=42,
    verbose=1
)
classifier.fit(X_train_scaled, y_train)


# classifier = RandomForestClassifier(n_estimators=2, max_depth=3, random_state=42)
# classifier.fit(X_train_scaled, y_train)

# classifier = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
# classifier.fit(X_train_scaled, y_train)


# Evaluate
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classifier.classes_, yticklabels=classifier.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as confusion_matrix.png")

# Save model and scaler
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(classifier, f)
print(f"\nModel saved to {MODEL_PATH}")

with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to {SCALER_PATH}")

# Model info for Raspberry Pi
print("\n--- Model Info ---")
print(f"Model size: ~{len(pickle.dumps(classifier)) / 1024:.1f} KB")