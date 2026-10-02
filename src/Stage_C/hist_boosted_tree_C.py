import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import ConfusionMatrixDisplay, log_loss


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
# Define the gradient boosting model
# ---------------------------------------------------------

model = HistGradientBoostingClassifier(random_state=7)


# ---------------------------------------------------------
# Define the hyperparameter values to test
# ---------------------------------------------------------

param_grid = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [7, 15, 31, 63],
    "min_samples_leaf": [20, 50, 100]
}


# ---------------------------------------------------------
# Find the best hyperparameter using log loss
# ---------------------------------------------------------

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="neg_log_loss", n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"Best max_leaf_nodes: {grid_search.best_params_['max_leaf_nodes']}")
print(f"Best learning_rate: {grid_search.best_params_['learning_rate']}")
print(f"Best min_samples_leaf: {grid_search.best_params_['min_samples_leaf']}")
print(f"Best cross-validation log loss: {-grid_search.best_score_:.3f}")


# ---------------------------------------------------------
# Evaluate the final model on the held-out test set
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

baseline_results = cross_validate(baseline, X_train, y_train, cv=5, scoring=["accuracy", "neg_log_loss"])

baseline_accuracy = baseline_results["test_accuracy"]
baseline_log_loss = -baseline_results["test_neg_log_loss"]

print(f"\nMean baseline accuracy: {baseline_accuracy.mean():.3f}")
print(f"Mean baseline log loss: {baseline_log_loss.mean():.3f}")


# ---------------------------------------------------------
# Display confusion matrix on the test set
# ---------------------------------------------------------

ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, normalize="true")
plt.show()