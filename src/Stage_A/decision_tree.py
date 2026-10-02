import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, GridSearchCV, cross_validate)
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, log_loss


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

games = pd.read_csv("projects/Chess-outcome-predictor/data/processed/rapid_games_A.csv")


# ---------------------------------------------------------
# Separate features and target
# ---------------------------------------------------------

X = games[["white_elo", "black_elo"]]
y = games["result"]


# ---------------------------------------------------------
# Create training and test sets
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7, stratify=y)


# ---------------------------------------------------------
# Define the decision tree
# ---------------------------------------------------------

model = DecisionTreeClassifier(random_state=7)


# ---------------------------------------------------------
# Define the hyperparameter values to test
# ---------------------------------------------------------

param_grid = {"min_samples_leaf": [10, 25, 50, 100, 250, 500, 1000, 2500, 5000]}


# ---------------------------------------------------------
# Inner cross-validation:
# ---------------------------------------------------------

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="neg_log_loss", n_jobs=-1)


# ---------------------------------------------------------
# Outer cross-validation:
# ---------------------------------------------------------

nested_cv_results = cross_validate(grid_search, X_train, y_train, cv=5, scoring=["accuracy", "neg_log_loss"], n_jobs=-1, return_estimator=True)


# ---------------------------------------------------------
# Calculate nested CV performance
# ---------------------------------------------------------

nested_accuracy = nested_cv_results["test_accuracy"]
nested_log_loss = -nested_cv_results["test_neg_log_loss"]

print(f"Nested CV mean accuracy: " f"{nested_accuracy.mean():.3f}")

print(f"Nested CV mean log loss: " f"{nested_log_loss.mean():.3f}")


# ---------------------------------------------------------
# Show the best hyperparameter found in each outer fold
# ---------------------------------------------------------

print("\nBest min_samples_leaf for each outer fold:")

for i, estimator in enumerate(nested_cv_results["estimator"], start=1):
    print(f"Fold {i}: " f"{estimator.best_params_['min_samples_leaf']}")


# ---------------------------------------------------------
# Fit the hyperparameter search on the entire training set
# ---------------------------------------------------------

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nFinal best min_samples_leaf: " f"{grid_search.best_params_['min_samples_leaf']}")


# ---------------------------------------------------------
# Evaluate probabilities on the held-out test set
# ---------------------------------------------------------

predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)

test_accuracy = (predictions == y_test).mean()
test_log_loss = log_loss(y_test, probabilities, labels=best_model.classes_)

print(f"\nTest accuracy: {test_accuracy:.3f}")
print(f"Test log loss: {test_log_loss:.3f}")

print("\nClass order:")
print(best_model.classes_)

print("\nFirst five predicted probabilities:")
print(probabilities[:5])


# ---------------------------------------------------------
# Compare with the baseline
# ---------------------------------------------------------

baseline = DummyClassifier(strategy="prior")

baseline.fit(X_train, y_train)

baseline_results = cross_validate(baseline, X_train, y_train, cv=5, scoring=["accuracy", "neg_log_loss"])

baseline_accuracy = baseline_results["test_accuracy"]
baseline_log_loss = -baseline_results["test_neg_log_loss"]

print(f"\nMean baseline accuracy: " f"{baseline_accuracy.mean():.3f}")

print(f"Mean baseline log loss: " f"{baseline_log_loss.mean():.3f}")


# ---------------------------------------------------------
# Display confusion matrix on the test set
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, normalize="true")

plt.show()