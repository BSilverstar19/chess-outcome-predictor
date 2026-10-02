# Convert the lichess ratings in a CSV dataset into chesscom ratings
# Ben Silvester-Ratcliffe Sept 2026

import pandas as pd


MAX_CONVERSION_RATING = 2500


def convert_lichess_to_chesscom(lichess_rating):
    """Convert a Lichess rating to an estimated chess.com rating."""
    x = lichess_rating

    return (
        (((((( -1.5643446939104476e-19 * x
            + 1.9181728962123964e-15) * x
            - 9.604852714700698e-12) * x
            + 2.5310705975897389e-8) * x
            - 3.7767766830985903e-5) * x
            + 3.2037977111173628e-2) * x
            - 1.3370078341728121e1) * x
            + 2.2500120508668001e3
    )


def convert_rating(rating):
    """Convert ratings below 2500; leave 2500+ ratings unchanged."""
    if rating >= MAX_CONVERSION_RATING:
        return rating

    return convert_lichess_to_chesscom(rating)


def main():
    # Paths to the input and output CSV files.
    input_file = "projects/Chess-outcome-predictor/data/processed/games.csv"
    output_file = "projects/Chess-outcome-predictor/data/processed/rapid_games_C_20_full.csv"

    # Load the data.
    df = pd.read_csv(input_file)

    # Convert ratings below 2500.
    df["white_elo"] = df["white_elo"].apply(convert_rating)
    df["black_elo"] = df["black_elo"].apply(convert_rating)

    # Round converted ratings to the nearest whole number.
    df["white_elo"] = df["white_elo"].round().astype(int)
    df["black_elo"] = df["black_elo"].round().astype(int)

    # Save the converted data.
    df.to_csv(output_file, index=False)

    print(f"Converted data saved to: {output_file}")


if __name__ == "__main__":
    main()