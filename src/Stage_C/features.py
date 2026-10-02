# Processes a PGN position, extracting White Elo, Black Elo, the game result
# and many more features to analyse
# Ben Silvester-Ratcliffe Sept 2026

# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

import chess


# ---------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

def get_piece_differences(board):
    differences = [0, 0, 0, 0, 0]
    material_difference = 0

    for i in range (1,6):
        piece_type, value = i, PIECE_VALUES[i]
        difference = len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))
        differences[i-1] = difference
        material_difference += difference * value

    return differences, material_difference



def get_hanging_pieces(board):
    white_value = 0
    black_value = 0

    for square, piece in board.piece_map().items():
        attackers = board.attackers(not piece.color, square)
        defenders = board.attackers(piece.color, square)

        if len(attackers) > len(defenders):
            if piece.color == chess.WHITE:
                white_value += PIECE_VALUES[piece.piece_type]
            else:
                black_value += PIECE_VALUES[piece.piece_type]

    difference = white_value - black_value

    return difference



def get_king_pawn_cover(board):
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    white_adjacent = chess.SquareSet(chess.BB_KING_ATTACKS[white_king])
    black_adjacent = chess.SquareSet(chess.BB_KING_ATTACKS[black_king])

    white_pawns = 0
    for square in white_adjacent:
        if board.piece_at(square) == chess.Piece(chess.PAWN, chess.WHITE):
            white_pawns += 1

    black_pawns = 0
    for square in black_adjacent:
        if board.piece_at(square) == chess.Piece(chess.PAWN, chess.BLACK):
            black_pawns += 1

    difference = white_pawns - black_pawns
    return difference



def get_pawn_difference(board):
    white_files = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)]
    black_files = [chess.square_file(sq) for sq in board.pieces(chess.PAWN, chess.BLACK)]

    def count_isolated(files):
        isolated = 0

        for f in files:
            if (f - 1) not in files and (f + 1) not in files:
                isolated += 1

        return isolated

    def count_doubled(files):
        doubled = 0

        for f in set(files):
            if files.count(f) >= 2:
                doubled += 1

        return doubled


    isolated_difference = count_isolated(white_files) - count_isolated(black_files)
    doubled_difference = count_doubled(white_files) - count_doubled(black_files)
    return isolated_difference, doubled_difference



def get_average_pawn_rank(board):
    white_pawns = board.pieces(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces(chess.PAWN, chess.BLACK)

    white_ranks = []
    black_ranks = []

    for pawn in white_pawns:
        rank = chess.square_rank(pawn) + 1
        white_ranks.append(rank)

    for pawn in black_pawns:
        rank = 8 - chess.square_rank(pawn)
        black_ranks.append(rank)

    try:
        white_average = sum(white_ranks) / len(white_ranks)
    except ZeroDivisionError:
        white_average = 0

    try:
        black_average = sum(black_ranks) / len(black_ranks)
    except ZeroDivisionError:
        black_average = 0

    difference = round((white_average - black_average), 2)
    return difference



def get_square_control(board):
    white_control = 0
    black_control = 0

    for square in chess.SQUARES:
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))

        if white_attackers > black_attackers:
            white_control += 1
        elif black_attackers > white_attackers:
            black_control += 1

    centre = [chess.D4, chess.E4, chess.D5, chess.E5]

    white_centre = 0
    black_centre = 0

    for square in centre:
        white_attackers = len(board.attackers(chess.WHITE, square))
        black_attackers = len(board.attackers(chess.BLACK, square))

        if white_attackers > black_attackers:
            white_centre += 1
        elif black_attackers > white_attackers:
            black_centre += 1

    overall_difference = white_control - black_control
    centre_difference = white_centre - black_centre

    return overall_difference, centre_difference



def get_long_diagonal_bishops(board):
    white_long = 0
    black_long = 0

    for color in [chess.WHITE, chess.BLACK]:
        for square in board.pieces(chess.BISHOP, color):
            attacks = board.attacks(square)

            long_diagonal = False

            for target in attacks:
                bishop_file = chess.square_file(square)
                target_file = chess.square_file(target)
                distance = abs(bishop_file - target_file)

                if distance >= 5:
                    long_diagonal = True
                    break

            if long_diagonal:
                if color == chess.WHITE:
                    white_long += 1
                else:
                    black_long += 1

    difference = white_long - black_long
    return difference



def get_rooks_on_open_files(board):
    white_rooks = 0
    black_rooks = 0

    for color in [chess.WHITE, chess.BLACK]:
        for rook_square in board.pieces(chess.ROOK, color):
            rook_file = chess.square_file(rook_square)
            pawn_on_file = False

            for pawn_square in board.pieces(chess.PAWN, color):
                pawn_file = chess.square_file(pawn_square)

                if rook_file == pawn_file:
                    pawn_on_file = True
                    break

            if not pawn_on_file:
                if color == chess.WHITE:
                    white_rooks += 1
                else:
                    black_rooks += 1

    difference = white_rooks - black_rooks
    return difference



def get_features(board):

    differences, material_difference = get_piece_differences(board)
    hanging_pieces = get_hanging_pieces(board)
    king_pawn_cover = get_king_pawn_cover(board)
    isolated_pawns, doubled_pawns = get_pawn_difference(board)
    average_rank = get_average_pawn_rank(board)
    controlled_squares, central_control = get_square_control(board)
    long_bishops = get_long_diagonal_bishops(board)
    open_rooks = get_rooks_on_open_files(board)

    features = differences + [material_difference] + [hanging_pieces] + [king_pawn_cover] + [isolated_pawns] + [doubled_pawns] + [average_rank] + [controlled_squares] + [central_control] + [long_bishops] + [open_rooks]

    return features

# Testing
'''
board = chess.Board("3r1rk1/p1p4p/1p3pp1/4nn2/2P1B3/P3P2P/1B3PP1/R2R1K2 w - - 0 21")
print(board)
test = get_features(board)
print(test)
'''