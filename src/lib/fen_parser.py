import chess

def parse_fen_without_board(fen):
    components = fen.split()
    
    turn = 1 if components[1].lower() == 'w' else 0
    castling = components[2]
    en_passant = 1 if components[3] != '-' else 0
    halfmove = int(components[4])
    fullmove = int(components[5])

    castling_features = {
        "white_kingside_castle": 1 if 'K' in castling else 0,
        "white_queenside_castle": 1 if 'Q' in castling else 0,
        "black_kingside_castle": 1 if 'k' in castling else 0,
        "black_queenside_castle": 1 if 'q' in castling else 0,
    }

    parsed_features = {
        "white_to_move": turn,
        "en_passant_possible": en_passant,
        "halfmove_clock": halfmove,
        "fullmove_number": fullmove,
        **castling_features,
    }

    return parsed_features

def flatten_board(fen):
    board = fen.split()[0]
    flattened = []

    for char in board:
        if char.isdigit():
            flattened.extend([0] * int(char))
        elif char == '/': 
            continue
        else: 
            flattened.append(ord(char))

    if len(flattened) != 64:
        raise ValueError("Invalid FEN: Incorrect number of squares.")
    
    return flattened


def update_dataframe_with_new_fen(data):
    def apply_move_and_generate_fen(fen, move):
        board = chess.Board(fen)
        chess_move = chess.Move.from_uci(move)
        board.push(chess_move)
        return board.fen()

    updated_data = data.copy()
    updated_fens = []
    updated_moves = []
    
    for index, row in updated_data.iterrows():
        try:
            fen = row['FEN']
            moves = row['Moves'].split()
            
            if moves:
                first_move = moves.pop(0)
                new_fen = apply_move_and_generate_fen(fen, first_move)
                
                updated_fens.append(new_fen)
                updated_moves.append(" ".join(moves))
            else:
                updated_fens.append(fen)
                updated_moves.append("")
        
        except Exception as e:
            updated_fens.append(row['FEN'])
            updated_moves.append(row['Moves'])
    
    updated_data['FEN'] = updated_fens
    updated_data['Moves'] = updated_moves
    return updated_data
