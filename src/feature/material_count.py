def count_pieces(fen_string):
    board_state = fen_string.split()[0]
    
    piece_counts = {
        'WhitePawns': 0, 'WhiteKnights': 0, 'WhiteBishops': 0, 'WhiteRooks': 0, 'WhiteQueens': 0, 'WhiteKings': 0,
        'BlackPawns': 0, 'BlackKnights': 0, 'BlackBishops': 0, 'BlackRooks': 0, 'BlackQueens': 0, 'BlackKings': 0
    }
    
    piece_map = {
        'P': 'WhitePawns', 'N': 'WhiteKnights', 'B': 'WhiteBishops', 'R': 'WhiteRooks', 'Q': 'WhiteQueens', 'K': 'WhiteKings',
        'p': 'BlackPawns', 'n': 'BlackKnights', 'b': 'BlackBishops', 'r': 'BlackRooks', 'q': 'BlackQueens', 'k': 'BlackKings'
    }
    
    for char in board_state:
        if char in piece_map:
            piece_counts[piece_map[char]] += 1
    
    return list(piece_counts.values())


