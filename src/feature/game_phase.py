def calculate_game_phase(fen):
    mn = int(fen.split()[5]) 
    if mn <= 10:
        return "Opening"
    elif 10 < mn <= 20:
        return "Early Middlegame"
    elif 20 < mn <= 30:
        return "Middlegame"
    elif 30 < mn <= 40:
        return "Late Middlegame"
    else:
        return "Endgame"
