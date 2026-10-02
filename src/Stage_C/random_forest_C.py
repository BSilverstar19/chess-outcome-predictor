import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, accuracy_score, ConfusionMatrixDisplay


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

games = pd.read_csv("projects/Chess-outcome-predictor/data/processed/rapid_games_C_20_full.csv")


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
# Define the random forest
# ---------------------------------------------------------

model = RandomForestClassifier(random_state=7, n_jobs=2)


# ---------------------------------------------------------
# Define hyperparameters to test
# ---------------------------------------------------------

param_grid = {"n_estimators": [100, 250], "max_depth": [10, 20, None], "min_samples_leaf": [5, 25, 100]}


# ---------------------------------------------------------
# Find the best hyperparameters using cross-validation
# ---------------------------------------------------------

grid_search = GridSearchCV(model, param_grid, cv=3, scoring="neg_log_loss", n_jobs=-1)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"Best hyperparameters: {grid_search.best_params_}")


# ---------------------------------------------------------
# Predict probabilities on the test set
# ---------------------------------------------------------

probabilities = best_model.predict_proba(X_test)

predictions = best_model.predict(X_test)


# ---------------------------------------------------------
# Evaluate the model
# ---------------------------------------------------------

print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"Test log loss: {log_loss(y_test, probabilities, labels=best_model.classes_):.3f}")

print("\nClass order:")
print(best_model.classes_)

print("\nFirst five predicted probabilities:")
print(probabilities[:5])


# ---------------------------------------------------------
# Baseline
# ---------------------------------------------------------

baseline = DummyClassifier(strategy="prior")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)
baseline_probabilities = baseline.predict_proba(X_test)

print(f"\nBaseline accuracy: {accuracy_score(y_test, baseline_predictions):.3f}")
print(f"Baseline log loss: {log_loss(y_test, baseline_probabilities, labels=baseline.classes_):.3f}")


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, normalize="true")

plt.show()