from feature.center import calculate_proximity_and_threats_to_center
from feature.material_advantage import calculate_material_balance
from feature.white_to_move import calculate_white_to_move
from feature.first_peace import identify_second_piece
from feature.game_phase import calculate_game_phase
from feature.material_count import count_pieces
import pandas as pd

def apply_feature_engineering(df, load=False):
    path = "/teamspace/studios/this_studio/data/feature/featured_df.csv"
    
    if load:
        print(f"Loading featured engineered dataframe from {path}")
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
        return df
    
    if not "FEN" in df.columns:
        raise ValueError("FEN feature must be presented in the dataframe")
    
    df['MoveLength'] = df['Moves'].apply(lambda x: len(x.split()))
    df['WhiteToMove'] = df['FEN'].apply(calculate_white_to_move)
    df['GamePhase'] = df['FEN'].apply(calculate_game_phase)
    df['MaterialAdvantage'] = df['FEN'].apply(calculate_material_balance)
    df['SecondMovedPiece'] = df.apply(lambda row: identify_second_piece(row['FEN'], row['Moves']), axis=1)
    
    df[['WhiteCenterProximity', 'BlackCenterProximity', 'WhiteThreats', 'BlackThreats']] = df['FEN'].apply(
        calculate_proximity_and_threats_to_center
    ).apply(pd.Series)
    
    piece_columns = ['WhitePawns', 'WhiteKnights', 'WhiteBishops', 'WhiteRooks', 'WhiteQueens', 'WhiteKings',
                 'BlackPawns', 'BlackKnights', 'BlackBishops', 'BlackRooks', 'BlackQueens', 'BlackKings']
    df[piece_columns] = pd.DataFrame(df['FEN'].apply(count_pieces).tolist(), index=df.index)
    
    df.to_csv(path)
    print(f"Feature engineered dataframe saved to {path}")
    
    return df
    
    
    