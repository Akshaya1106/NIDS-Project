# ==========================================
# Network Intrusion Detection System (NIDS)
# Using Machine Learning
# ==========================================

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# ---------- STEP 1: LOAD DATASET ----------
train_data = pd.read_csv("kdd_train.csv")
test_data = pd.read_csv("kdd_test.csv")

# ---------- STEP 2: MAP ATTACK TYPES ----------
attack_mapping = {

    # Normal
    "normal": "Normal",

    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",

    # Probe
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",

    # R2L
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",

    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R"
}

train_data["labels"] = train_data["labels"].map(attack_mapping)
test_data["labels"] = test_data["labels"].map(attack_mapping)

# Remove rows whose labels are not mapped
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# ---------- STEP 3: SPLIT FEATURES & TARGET ----------
X_train = train_data.drop("labels", axis=1)
y_train = train_data["labels"]

X_test = test_data.drop("labels", axis=1)
y_test = test_data["labels"]

# ---------- STEP 4: ENCODE CATEGORICAL FEATURES ----------
categorical_columns = ["protocol_type", "service", "flag"]

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()

    combined = pd.concat([X_train[col], X_test[col]], axis=0)

    le.fit(combined)

    X_train[col] = le.transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

    encoders[col] = le

# ---------- STEP 5: ENCODE TARGET ----------
label_encoder = LabelEncoder()

combined_labels = pd.concat([y_train, y_test])

label_encoder.fit(combined_labels)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

# ---------- STEP 6: FEATURE SCALING ----------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------- STEP 7: DEFINE MODELS ----------
models = {

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Naive Bayes":
        GaussianNB()
}

# ---------- STEP 8: TRAIN & TEST ----------
accuracy_results = {}

best_model_object = None

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    accuracy_results[name] = accuracy

    print("\n====================================")
    print("MODEL :", name)
    print("Accuracy :", round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

# ---------- STEP 9: BEST MODEL ----------
best_model = max(accuracy_results, key=accuracy_results.get)

print("\n====================================")
print("BEST MODEL :", best_model)
print("BEST ACCURACY :", round(accuracy_results[best_model] * 100, 2), "%")

best_model_object = models[best_model]

# ---------- STEP 10: SAVE MODEL ----------
joblib.dump(best_model_object, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("\nModel saved successfully.")