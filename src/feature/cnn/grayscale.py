import numpy as np
from PIL import Image

piece_map = {
    'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6,
    'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12,
    '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0
}

def fen_to_grid(fen):
    board_part = fen.split()[0]
    rows = board_part.split('/')
    grid = []
    for row in rows:
        grid_row = []
        for char in row:
            if char.isdigit():
                grid_row.extend([0] * int(char))  
            else:
                grid_row.append(piece_map[char]) 
        grid.append(grid_row)
    return np.array(grid)

def fen_to_image(fen, save=False):
    grid = fen_to_grid(fen)
    img = (grid / np.max(grid) * 255).astype(np.uint8)
    
    if save:
        img = Image.fromarray(img, 'L')  # 'L' for grayscale
        output_path = "../results/cnn/grayscale.png"
        img.save(output_path)
        print(f"Chessboard grayscale image saved as {output_path}")
    
    return img

