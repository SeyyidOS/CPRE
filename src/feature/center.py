# Constants for chessboard and central squares
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
CENTER_SQUARES = {'d4', 'd5', 'e4', 'e5'}
PIECE_WEIGHTS = {'P': 1, 'p': 1, 'N': 3, 'n': 3, 'B': 3, 'b': 3, 'R': 5, 'r': 5, 'Q': 9, 'q': 9}
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
STRAIGHT_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Rook and Queen
DIAGONAL_MOVES = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Bishop and Queen
PAWN_ATTACKS = {'white': [(1, 1), (1, -1)], 'black': [(-1, 1), (-1, -1)]}


# Helper function: Get the square name based on file and row indices
def get_square(file_idx, row_idx):
    if 0 <= file_idx < 8 and 1 <= row_idx <= 8:
        return FILES[file_idx] + str(row_idx)
    return None


# Helper function: Parse FEN and extract the board layout
def parse_fen(fen):
    return fen.split()[0].split('/')


# Helper function: Count pieces in the center squares
def count_pieces_in_center(board_rows):
    white_count = 0
    black_count = 0
    row_index = 8
    for row in board_rows:
        file_index = 0
        for char in row:
            if char.isdigit():  # Empty squares
                file_index += int(char)
            else:
                square = get_square(file_index, row_index)
                if square in CENTER_SQUARES:
                    if char.isupper():  # White piece
                        white_count += 1
                    else:  # Black piece
                        black_count += 1
                file_index += 1
        row_index -= 1
    return white_count, black_count


# Helper function: Calculate threats from knights
def calculate_knight_threats(file_idx, row_idx, color):
    threats = 0
    for move in KNIGHT_MOVES:
        threatened_square = get_square(file_idx + move[1], row_idx + move[0])
        if threatened_square in CENTER_SQUARES:
            threats += PIECE_WEIGHTS['N'] if color == 'white' else PIECE_WEIGHTS['n']
    return threats


# Helper function: Calculate threats from sliding pieces (Rook, Bishop, Queen)
def calculate_sliding_threats(file_idx, row_idx, directions, color, piece_positions):
    threats = 0
    for direction in directions:
        current_file, current_row = file_idx + direction[1], row_idx + direction[0]
        while True:
            square = get_square(current_file, current_row)
            if not square:  # Out of bounds
                break
            if square in piece_positions:  # Stop at the first blocking piece
                break
            if square in CENTER_SQUARES:
                threats += PIECE_WEIGHTS['Q'] if color == 'white' else PIECE_WEIGHTS['q']
            current_file += direction[1]
            current_row += direction[0]
    return threats


# Helper function: Calculate threats from pawns
def calculate_pawn_threats(file_idx, row_idx, color):
    threats = 0
    moves = PAWN_ATTACKS['white'] if color == 'white' else PAWN_ATTACKS['black']
    for move in moves:
        threatened_square = get_square(file_idx + move[1], row_idx + move[0])
        if threatened_square in CENTER_SQUARES:
            threats += PIECE_WEIGHTS['P'] if color == 'white' else PIECE_WEIGHTS['p']
    return threats


# Main function: Calculate proximity and threats
def calculate_proximity_and_threats_to_center(fen):
    """
    Calculate the number of white and black pieces in the central squares (d4, d5, e4, e5)
    and the number of threats to those squares by all pieces.

    Parameters:
        fen (str): A FEN string representing the chessboard.

    Returns:
        tuple: (white_pieces_in_center, black_pieces_in_center, white_threats, black_threats)
    """
    board_rows = parse_fen(fen)
    white_pieces_in_center, black_pieces_in_center = count_pieces_in_center(board_rows)

    white_threats = 0
    black_threats = 0
    piece_positions = set()

    # Collect piece positions for blocking logic
    row_index = 8
    for row in board_rows:
        file_index = 0
        for char in row:
            if char.isdigit():
                file_index += int(char)
            else:
                piece_positions.add(get_square(file_index, row_index))
                file_index += 1
        row_index -= 1

    # Calculate threats for each piece
    row_index = 8
    for row in board_rows:
        file_index = 0
        for char in row:
            if char.isdigit():  # Empty squares
                file_index += int(char)
            else:
                color = 'white' if char.isupper() else 'black'
                if char.upper() == 'N':  # Knight
                    threats = calculate_knight_threats(file_index, row_index, color)
                elif char.upper() == 'P':  # Pawn
                    threats = calculate_pawn_threats(file_index, row_index, color)
                elif char.upper() in {'R', 'Q'}:  # Rook or Queen
                    threats = calculate_sliding_threats(file_index, row_index, STRAIGHT_MOVES, color, piece_positions)
                elif char.upper() in {'B', 'Q'}:  # Bishop or Queen
                    threats = calculate_sliding_threats(file_index, row_index, DIAGONAL_MOVES, color, piece_positions)
                else:
                    threats = 0

                if color == 'white':
                    white_threats += threats
                else:
                    black_threats += threats

                file_index += 1
        row_index -= 1

    return white_pieces_in_center, black_pieces_in_center, white_threats, black_threats
