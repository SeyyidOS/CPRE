import numpy as np
from PIL import Image

piece_colors = {
    'P': (173, 216, 230),  # Light Blue for white pawns
    'R': (144, 238, 144),  # Light Green for white rooks
    'N': (255, 182, 193),  # Pink for white knights
    'B': (238, 130, 238),  # Violet for white bishops
    'Q': (255, 215, 0),    # Gold for white queen
    'K': (255, 255, 255),  # White for white king
    'p': (0, 0, 139),      # Dark Blue for black pawns
    'r': (34, 139, 34),    # Dark Green for black rooks
    'n': (139, 0, 139),    # Dark Violet for black knights
    'b': (85, 26, 139),    # Purple for black bishops
    'q': (184, 134, 11),   # Dark Gold for black queen
    'k': (0, 0, 0),        # Black for black king
}

light_square = (240, 217, 181)  # Beige for light squares
dark_square = (181, 136, 99)   # Brown for dark squares

def fen_to_rgb_image(fen, square_size=60, save=False):
    board_part = fen.split()[0]
    rows = board_part.split('/')
    board = []
    
    for i, row in enumerate(rows):
        board_row = []
        for char in row:
            if char.isdigit():
                board_row.extend([None] * int(char))
            else:
                board_row.append(piece_colors[char])
        board.append(board_row)
    
    image = np.zeros((8 * square_size, 8 * square_size, 3), dtype=np.uint8)
    for i, row in enumerate(board):
        for j, cell in enumerate(row):
            square_color = light_square if (i + j) % 2 == 0 else dark_square
            start_x, start_y = i * square_size, j * square_size
            end_x, end_y = start_x + square_size, start_y + square_size
            
            image[start_x:end_x, start_y:end_y] = square_color
            
            if cell is not None:
                piece_color = np.array(cell, dtype=np.uint8)
                image[start_x:end_x, start_y:end_y] = (
                    0.7 * image[start_x:end_x, start_y:end_y] + 0.3 * piece_color
                ).astype(np.uint8)
    
    if save:
        img = Image.fromarray(image, 'RGB')
        output_path="../results/cnn/rgb.png"
        img.save(output_path)
        print(f"Chessboard RGB image saved as {output_path}")

    return image