from chess import Board
import chess

def identify_second_piece(fen, moves):
    board = Board(fen)
    move_list = moves.split()  # Split moves into individual UCI strings

    try:
        # First move
        if len(move_list) >= 1:
            first_move = chess.Move.from_uci(move_list[0])
            board.push(first_move)  # Apply the first move to the board

        # Second move
        if len(move_list) >= 2:
            second_move = chess.Move.from_uci(move_list[1])
            piece_moved = board.piece_at(second_move.from_square)  # Get the piece at the move's origin square
            if piece_moved:
                piece_type = piece_moved.piece_type
                piece_dict = {
                    chess.PAWN: "Pawn",
                    chess.KNIGHT: "Knight",
                    chess.BISHOP: "Bishop",
                    chess.ROOK: "Rook",
                    chess.QUEEN: "Queen",
                    chess.KING: "King",
                }
                return piece_dict.get(piece_type, "Unknown")
        return "Unknown"
    except:
        return "Invalid Move"

# Apply the function to your DataFrame
