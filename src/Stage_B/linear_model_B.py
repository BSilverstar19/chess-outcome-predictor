import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

games = pd.read_csv("projects/Chess-outcome-predictor/data/processed/rapid_games_B_20.csv")


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = games.drop(columns=["result"])
y = games["result"]

# ---------------------------------------------------------
# Create training and test sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)


# ---------------------------------------------------------
# Define the logistic regression pipeline with standard scaler
# ---------------------------------------------------------

model = make_pipeline(StandardScaler(), LogisticRegression())


# ---------------------------------------------------------
# Train the model
# ---------------------------------------------------------

model.fit(X_train, y_train)


# ---------------------------------------------------------
# Find accuracy with cross validation
# ---------------------------------------------------------

cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=["accuracy", "neg_log_loss"])
accuracy = cv_results["test_accuracy"]
log_loss = -cv_results["test_neg_log_loss"]

print(f"Mean accuracy: {accuracy.mean():.3f}")
print(f"Mean log loss: {log_loss.mean():.3f}")

# ---------------------------------------------------------
# Generate example probabilities
# ---------------------------------------------------------

probabilities = model.predict_proba(X_test)

print(model.classes_)
print(probabilities[:5])


# ---------------------------------------------------------
# Compare with the baseline
# ---------------------------------------------------------

baseline = DummyClassifier(strategy="prior")

baseline_results = cross_validate(baseline, X_train, y_train, cv=5, scoring=["accuracy", "neg_log_loss"])

baseline_accuracy = baseline_results["test_accuracy"]
baseline_log_loss = -baseline_results["test_neg_log_loss"]

print(f"\nMean baseline accuracy: {baseline_accuracy.mean():.3f}")
print(f"Mean baseline log loss: {baseline_log_loss.mean():.3f}")


# ---------------------------------------------------------
# Display confusion matrix on the test set
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize="true")
plt.show()