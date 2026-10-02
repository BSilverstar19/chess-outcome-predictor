# Processes a compressed Lichess PGN database, extracting White Elo, Black Elo,
# and game results into a clean CSV dataset.
# Ben Silvester-Ratcliffe Sept 2026

##### Set Up #####
import chess.pgn
import zstandard as zstd
import csv
import io
from pathlib import Path
import argparse


# Python dictionary for making the game result readable
RESULT_MAPPING = {"1-0": "white", "1/2-1/2": "draw", "0-1": "black"}


##### Data Processing #####
def is_rapid(time_control):
    """Return True if a Lichess time control is classified as rapid."""

    try:
        # Separate the increment from the main time
        initial_time, increment = time_control.split("+")

        initial_time = int(initial_time)
        increment = int(increment)

        # Compute lichess' estimated duration and compare with the rapid time control
        estimated_duration = initial_time + 40 * increment

        return 480 <= estimated_duration < 1500

    except (ValueError, AttributeError):
        return False


def process_pgn(input_path: Path, output_path: Path):
    """
    Extract White Elo, Black Elo, and game result from a compressed
    Lichess PGN database and save the cleaned data as a CSV file.

    Parameters
    ----------
    input_path : Path
        Path to the .pgn.zst Lichess database.
    output_path : Path
        Path to where the processed CSV should be saved.
    """

    games_processed = 0
    games_kept = 0
    games_skipped = 0

    # Open the compressed .zst file in binary read mode
    with open(input_path, "rb") as compressed_file:
        decompressor = zstd.ZstdDecompressor()

        # Create a stream that decompresses the file as it is read
        with decompressor.stream_reader(compressed_file) as decompressed_file:

            # Convert the decompressed binary data into readable UTF-8 text
            text_stream = io.TextIOWrapper(decompressed_file, encoding="utf-8")

            # Open the output CSV file for writing
            with open(output_path, "w", newline="", encoding="utf-8") as csv_file:

                # Create a CSV writer to write rows to the output file
                writer = csv.writer(csv_file) 

                # Write the new CSV file header
                writer.writerow(["white_elo", "black_elo", "result",])

                while True:
                    # Read the header information for the next game
                    header = chess.pgn.read_headers(text_stream)

                    # Stop the loop when there are no more games to read
                    if header is None:
                        break

                    games_processed += 1

                    # Report on progress
                    if games_processed % 100_000 == 0:
                        print(f"Processed {games_processed:,} games ", f"({games_kept:,} kept)")

                    # Retreive the required information from the header
                    white_elo = header.get("WhiteElo")
                    black_elo = header.get("BlackElo")
                    result = header.get("Result")
                    time_control = header.get("TimeControl")

                    # Skip games that are not classified as rapid by Lichess
                    if time_control is None or not is_rapid(time_control):
                        games_skipped += 1
                        continue

                    # Check if the result is valid, skip the game if it isn't
                    if result not in RESULT_MAPPING:
                        games_skipped += 1
                        continue

                    # Check if both Elo ratings exist, skip the game if it they don't
                    if white_elo is None or black_elo is None:
                        games_skipped += 1
                        continue

                    # Convert Elo values to integers, skip the game if this raises an error
                    try:
                        white_elo = int(white_elo)
                        black_elo = int(black_elo)
                    except ValueError:
                        games_skipped += 1
                        continue

                    # Basic sanity check for ratings.
                    if not (0 < white_elo < 4000) or not (0 < black_elo < 4000):
                        games_skipped += 1
                        continue

                    # Update the new csv file with the game entry
                    writer.writerow([white_elo, black_elo, RESULT_MAPPING[result]])

                    games_kept += 1

                    # Stop once we have found 1 million valid games
                    if games_kept >= 1_000_000:
                        break
                    

    print("Processing complete.")
    print(f"Games processed: {games_processed:,}")
    print(f"Games kept:      {games_kept:,}")
    print(f"Games skipped:   {games_skipped:,}")
    print(f"Output file:     {output_path}")


##### Program Interface #####
def main():
    """Parse command-line arguments and process the PGN database."""

    # Create a parser for handling command-line arguments
    parser = argparse.ArgumentParser(description = "Process a compressed Lichess PGN database.")

    # Add the input PGN file path as a required command-line argument
    parser.add_argument("input", type=Path, help="Path to the .pgn.zst file")

    # Add the output CSV file path as a required command-line argument
    parser.add_argument("output", type=Path, help="Path for the output CSV file")

    # Read and store the user's command-line arguments
    args = parser.parse_args()

    # Check that the input file exists.
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create the output directory and any missing parent directories
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Use the previously defined function
    process_pgn(args.input, args.output)


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()