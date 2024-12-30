def calculate_material_balance(fen):
    piece_values = {'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0}  
    white_pieces = sum([piece_values.get(c.lower(), 0) for c in fen.split()[0] if c.isupper()])
    black_pieces = sum([piece_values.get(c.lower(), 0) for c in fen.split()[0] if c.islower()])
    return white_pieces - black_pieces
