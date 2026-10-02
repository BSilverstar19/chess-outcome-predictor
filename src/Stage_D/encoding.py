# Processes a PGN position, extracting the one hot encoding of the position
# Ben Silvester-Ratcliffe Sept 2026

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import chess


# ---------------------------------------------------------
# Extract features function
# ---------------------------------------------------------

def get_encoding(board):
    encoding = []

    for square in chess.SQUARES:
        piece = board.piece_at(square)

        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in range(1, 7):
                if piece is not None and piece.color == color and piece.piece_type == piece_type:
                    encoding.append(1)
                else:
                    encoding.append(0)

    encoding.append(int(board.turn))
    encoding.append(int(board.has_kingside_castling_rights(chess.WHITE)))
    encoding.append(int(board.has_queenside_castling_rights(chess.WHITE)))
    encoding.append(int(board.has_kingside_castling_rights(chess.BLACK)))
    encoding.append(int(board.has_queenside_castling_rights(chess.BLACK)))

    return encoding


# Testing
'''
board = chess.Board("3r1rk1/p1p4p/1p3pp1/4nn2/2P1B3/P3P2P/1B3PP1/R2R1K2 w - - 0 21")
print(board)
test = get_encoding(board)
print(test)
'''