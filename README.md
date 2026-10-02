# Chess Outcome Prediction

## Overview
This project investigates the prediction of chess game outcomes utilising supervised machine learning and a large dataset of online chess games. The project progressively evaluates four feature representations: the ratings of each player, material balance, advanced hand-engineered chess features, and a one-hot encoded representation of the position.

Logistic regression and tree-based models are evaluated using classification accuracy and multiclass log loss. The goal of the project is to assess how increasingly detailed representations of a chess position affect predictive performance. The predictions in question are the probabilities of a White win, Black win, or draw.

## Dataset
- All the datasets created in this project contain games from the August 2026 Lichess open database, which can be found here: https://database.lichess.org
- The raw dataset was filtered to only contain rapid time control games. Games with missing data, invalid data or games that ended before the move number being analysed was reached were excluded from the relevant datasets.
- All datasets were passed through an Elo conversion program that converts Lichess ratings to an estimated Chess.com equivalent. This standardises the ratings to the Chess.com rating system, which was chosen due to its larger player base.
- Most datasets contain exactly 1,000,000 games, but some of the datasets (designated as small) only contain 100,000
- Within each dataset, the target was the game result (White win / draw / Black win). All other columns contained the relevant features. 
- The original elo dataset was analysed in the notebook 01_data_exploration to ensure the dataset does not introduce any unexpected errors.

## Methodology

### Stage A — Elo ratings
This stage was the essential starting point for the whole project. Here, we establish how much can be predicted without looking at the chess position itself. Instead, we only look at the playing strength of each player, measured using their 'Elo' rating. 

Features:
- White Elo
- Black Elo

Models:
- Dummy baseline
- Logistic regression
- Decision tree
- Random forest
- Histogram gradient boosting


### Stage B — Material Balance
This stage kept the original elo information and introduced information about the actual state of the game, but only the most basic information. Hence, the new features used were the relative number of each piece and overall material balance.

Python-chess was used to reconstruct the required position from each game, and extract these features. 2 different positions were analysed per game: the position after black's 10th move and the position after black's 20th move.  

Features:
- White Elo
- Black Elo
- Relative piece counts (number of white pieces of this type - number of black pieces of this type)
- Material balance

Models:
- Dummy baseline
- Logistic regression
- Random forest
- Histogram gradient boosting


### Stage C — Advanced hand-engineered features
This stage kept all the previous features and added a suite of more advanced features. These features attempted to capture the many other aspects of the position that may contribute to the game result. A comprehensive list of the added features can be found below.

These features were deliberately hand-engineered using my personal chess domain knowledge. It is important to note that almost all of the features shown below are actually "difference" features. This means that rather than having 2 features, one corresponding to white and one to black, there is just one feature (which is the difference white number - black number).

Again, 2 different positions were analysed per game: the position after black's 10th move and the position after black's 20th move.  

Features:
- White Elo
- Black Elo
- Relative piece counts 
- Material balance
- Number of hanging pieces  -  hanging was defined (and calculated) as there being more attackers on a piece than defenders
- Castling difference  -  1 if white has castled and black hasn't, 0 if both have or neither have, -1 if black has castled and white hasn't
- Pawn cover around king  -  number of pawns adjacent to the king. This is a measure of king safety
- Isolated pawns  -  number of pawns with no pawns on neighbouring files. Isolated pawns are considered a structural weakness 
- Doubled+ pawns  -  number of pawns with another pawn of the same colour on the same file. Also considered a structural weakness
- Average pawn rank  -  a measure of how advanced each side is i.e how much space each side controls
- Controlled squares  -  number of squares each side controls (i.e there are more attackers than defenders). Another measure of controlled space
- Controlled central squares  -  same as above, but only the 4 central squares are considered. Central control is a large strength
- Bishops on long diagonals  -  these bishops have the potential to be very powerful attacking pieces
- Rooks on semi-open/open files  -  these rooks have the potential to be very powerful attacking pieces

Models:
- Dummy baseline
- Logistic regression
- Random forest
- Histogram gradient boosting


### Stage D — Raw board encoding
This stage replaces the features introduced in stages B & C with the actual chess position. Hence, the model is no longer reliant on 'good' features being hand-engineered as it has access to the actual position. To represent the chess board in a tabular format, one-hot encoding was used. 

One-hot encoding represents the board using 768 binary features. Each square is represented using 12 features, with each feature corresponding to a piece type and colour. A 1 means the piece is on that square and a 0 means it isn't. The current castling rights were also added to the encoding of the position.

Only logistic regression was used for this stage because the huge number of features significantly increased the computational cost of training the tree-based models. Logistic regression was much more efficient and still provided a useful comparison with the previous stages.

Again, 2 different positions were analysed per game: the position after black's 10th move and the position after black's 20th move.  

Features:
- White Elo
- Black Elo
- One-hot encoding of the position
- Castling rights in the position (kingside and queenside for each side)


Model:
- Logistic regression


### Model Training and Evaluation
- The dataset was split into training and test sets. For all of the models, 80% of the data was used for training and 20% of the data was used for testing.
- The training data is used for model fitting and hyperparameter selection, while the test data is data used for final evaluation. Cross-validation was used on the training set where appropriate (to select hyperparameters without using information from the test set). 
- Where hyperparameter selection is required, grid search cross validation was used. I decided not to use nested cross-validation due to the much larger computational cost. However, I did use cross-validation to improve the accuracy of the result metrics when hyperparameter tuning was not required.
- The first and easiest metric to evaluate the model is classification accuracy. This measures the proportion of games where the model predicts the final outcome (white win, black win or draw) correctly. The predicted final outcome is the result with the highest probability from the model.
- Multiclass log loss measures the quality of the predicted probabilities assigned to each possible game result. It is calculated by taking the negative logarithm of the probability assigned to the correct outcome for each game (averaged across all games). Hence, lower values actually indicate better predictions.


## Results

| Stage | Model | Move no. | Accuracy | Log Loss |
|---|---|---|---:|---:|
| N/A | Baseline | N/A | 49.5% | 0.850 |
| A | Logistic Regression | N/A | 53.8% | 0.831 | 
| A | Decision Tree | N/A | 53.6% | 0.830 |
| A | Histogram Boosted Tree | N/A | 53.9% | 0.829 |
| A | Random Forest | N/A | 53.6% | 0.830 |
| B | Logistic Regression | 10 | 55.8% | 0.822 |
| B | Histogram Boosted Tree | 10 | 56.0% | 0.818 | 
| B | Random Forest | 10 | 55.5% | 0.823 |
| B | Logistic Regression | 20 | 62.5% | 0.787 |
| B | Histogram Boosted Tree | 20 | 62.7% | 0.779 |
| B | Random Forest | 20 | 62.5% | 0.783 |
| C | Logistic Regression | 10 | 56.5% | 0.817 | 
| C | Histogram Boosted Tree | 10 | 57.1% | 0.810 |
| C | Random Forest | 10 | 56.8% | 0.816 |
| C | Logistic Regression | 20 | 63.6% | 0.776 |
| C | Histogram Boosted Tree | 20 | 64.0% | 0.764 | 
| C | Random Forest | 20 | 63.7% | 0.770 |
| D | Logistic Regression | 10 | 57.0% | 0.815 |
| D | Logistic Regression | 20 | 63.6% | 0.775 |

Note that the log loss quoted in this table can be used to find the mean probability for the correct class via the formula e^(-1*log_loss)
Hence, a log loss of 0.850 corresponds to a geometric mean probability of 42.74%.

## Conclusions
1. More information about the current board state substantially improves the model's prediction.
Stage A (only rating information) reached roughly 53.6–53.9% accuracy and 0.829-0.831 log loss. At the other end of the spectrum, Stage C at move 20  reached 63.6–64.0% accuracy and 0.764-0.776 log loss. Hence, information about the position provides considerable predictive value beyond player ratings.

2. Later-game positions are much more informative than earlier positions.
Across every model and stage, the results at move 20 were much better than at move 10. For example, the Stage C histogram boosted tree improved from 57.1% at move 10 to 64.0% accuracy at move 20. The log loss fell from 0.810 at move 10 to 0.764 at move 20. This makes intuitive sense: by move 20, more of the game's trajectory has been established and hence the prediction should be more accurate.

3. More sophisticated hand-engineered features provide an improvement to predictions.
Moving from Stage B to C improved both the accuracy and the log loss for every model and move number. Taking an example, at move 20:
- Logistic regression: 62.5% → 63.6%, 0.787 → 0.776
- Histogram boosted tree: 62.7% → 64.0%, 0.779 → 0.764
- Random forest: 62.5% → 63.7%, 0.783 → 0.770
This suggests that the additional features used in Stage C contained genuinely useful information not captured by the Stage B features.

4. Model choice matters less than feature representation and move number.
Within a given stage and move number, the three models performed very similarly. Using Stage C at move 20 as an example, the accuracy only ranges from 63.6% to 64.0%. In contrast, changing the input features or moving from move 10 to move 20 produces much larger changes in accuracy.

5. Histogram boosted trees generally give the strongest results (only by a small margin)
The histogram boosted tree has the highest accuracy and lowest log loss for every Stage B/C comparison shown. The strongest result is Stage C, move 20: 64.0% accuracy and 0.764 log loss. However, its advantage over logistic regression and random forest is small, so the results do not suggest that model architecture is the primary driver of performance.

## Limitations
The main limitation of this project was the computational resources and time available. 
With greater compute resources, I would perform more extensive hyperparameter tuning and use nested cross-validation. This applies especially to the random forest and histogram boosted tree models as they were more computationally expensive. 
I would also investigate a wider range of board positions rather than restricting the analysis to moves 10 and 20. This would allow the change in predictive performance throughout a game to be studied and the end goal could be a predictor that can be used for any move of the game. 
Finally, I would use more time to explore more advanced machine-learning models, such as neural networks. They could learn useful positional features directly from the position rather than relying on hand-engineered features, advancing and improving on Stage D. 

## Repository structure
The repository is organised as follows:

​```
Chess-outcome-predictor/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── Stage_A/
│   ├── Stage_B/
│   ├── Stage_C/
│   └── Stage_D/
├── notebooks/
├── README.md
└── requirements.txt
​```

- `data/raw/` — Contains the original Lichess games.
- `data/processed/` — Contains the final datasets produced after preprocessing and feature extraction.
- `src/Stage_A/` — Models using player ratings as features.
- `src/Stage_B/` — Models using simple material balance features.
- `src/Stage_C/` — Models using more advanced, hand-engineered features.
- `src/Stage_D/` — Models using a raw, one-hot encoding of the chess position.
- `notebooks/` — Explores the dataset and visualises key patterns in player ratings and game outcomes.
- `requirements.txt` — Lists the Python dependencies required to run the project.

## Running the project
To generate the required datasets from the large lichess dataset, first download the lichess dataset for August 2026 from https://database.lichess.org 
Then, use the processing scripts from each stage to generate the individual datasets. To run the processing scripts, use the following command:
python [processing_script] [lichess_dataset] [output_dataset]

For example:
python src/Stage_A/process_elo_data.py data/raw/lichess_db_standard_rated_2026-08.pgn.zst data/processed/games.csv

After generating the required datasets, run the model scripts from the root of the repository. For example:

```bash
python src/Stage_A/linear_model.py
```

The scripts for each experiment are organised within their corresponding Stage (A–D).