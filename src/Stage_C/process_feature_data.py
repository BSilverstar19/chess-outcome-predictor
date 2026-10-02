# Processes a compressed Lichess PGN database, extracting White Elo, Black Elo,
# game results, and material features after Black's 10th move into a CSV dataset.
# Ben Silvester-Ratcliffe Sept 2026

##### Set Up #####
import chess
import chess.pgn
import zstandard as zstd
import csv
import io
from pathlib import Path
import argparse
from features import get_features


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

        # Compute Lichess' estimated duration and compare with the rapid time control
        estimated_duration = initial_time + 40 * increment
        return 480 <= estimated_duration < 1500

    except (ValueError, AttributeError):
        return False



def process_pgn(input_path: Path, output_path: Path):
    """
    Extract White Elo, Black Elo, game result, and position features after
    Black's 10th move from a compressed Lichess PGN database and save them as CSV.
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
                writer.writerow(["white_elo", "black_elo", "pawn_difference", "knight_difference", "bishop_difference", "rook_difference", "queen_difference", "material_difference", "hanging_material", "castling_difference", "king_pawn_cover", "isolated_pawn_difference", "doubled_pawn_difference", "avg_pawn_rank", "controlled_squares", "controlled_centre", "long_bishops", "open_rooks", "result"])

                while True:
                    # Read the next complete game
                    game = chess.pgn.read_game(text_stream)

                    # Stop the loop when there are no more games to read
                    if game is None:
                        break

                    games_processed += 1

                    # Report on progress
                    if games_processed % 10_000 == 0:
                        print(f"Processed {games_processed:,} games ({games_kept:,} kept)")

                    # Retrieve the required information from the header
                    header = game.headers
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

                    # Check if both Elo ratings exist, skip the game if they don't
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

                    # Basic sanity check for ratings
                    if not (0 < white_elo < 4000) or not (0 < black_elo < 4000):
                        games_skipped += 1
                        continue

                    # Replay the game up to the position after Black's 10th move, checking for castling
                    board = game.board()
                    moves_played = 0
                    white_castled = False
                    black_castled = False

                    for move in game.mainline_moves():
                        if board.is_castling(move):
                            if board.turn == chess.WHITE:
                                white_castled = True
                            else:
                                black_castled = True

                        board.push(move)
                        moves_played += 1

                        if moves_played == 40:
                            break
                    castling_difference = int(white_castled) - int(black_castled)

                    # Skip games that do not reach Black's 20th move
                    if moves_played < 40:
                        games_skipped += 1
                        continue

                    # Calculate piece-count differences and material difference
                    features = get_features(board)

                    # Update the CSV file with the game entry
                    writer.writerow([white_elo, black_elo, features[0], features[1], features[2], features[3], features[4], features[5], features[6], castling_difference, features[7], features[8], features[9], features[10], features[11], features[12], features[13], features[14], RESULT_MAPPING[result]])
                    games_kept += 1

                    # Stop once we have found 1,000,000 valid games
                    if games_kept >= 1_000_000:
                        break

    print("Processing complete.")
    print(f"Games processed: {games_processed:,}")
    print(f"Games kept:      {games_kept:,}")
    print(f"Games skipped:   {games_skipped:,}")
    print(f"Output file:     {output_path}")


##### Program Interface #####
def main():
    """Parse command-line arguments and process the PGN database."""

    # Create a parser for handling command-line arguments
    parser = argparse.ArgumentParser(description="Process a compressed Lichess PGN database.")

    # Add the input PGN file path as a required command-line argument
    parser.add_argument("input", type=Path, help="Path to the .pgn.zst file")

    # Add the output CSV file path as a required command-line argument
    parser.add_argument("output", type=Path, help="Path for the output CSV file")

    # Read and store the user's command-line arguments
    args = parser.parse_args()

    # Check that the input file exists
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    # Create the output directory and any missing parent directories
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Use the previously defined function
    process_pgn(args.input, args.output)


# Run main() only when this file is executed directly
if __name__ == "__main__":
    main()