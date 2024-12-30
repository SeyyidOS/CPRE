import matplotlib.pyplot as plt
import seaborn as sns
import chess

def hist_plot(df, feature):
    plt.figure(figsize=(24, 6))
    sns.histplot(df[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Frequency')
    plt.show()

def box_plot(df, feature):
    plt.figure(figsize=(24, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(f'{feature}')
    plt.show()

def bar_plot(df, feature):
    plt.figure(figsize=(24, 6))
    counts = df[feature].value_counts()
    sns.barplot(x=counts.values, y=counts.index)
    plt.title(f'Frequency of {feature}')
    plt.xlabel('Count')
    plt.ylabel(f'{feature}')
    plt.show()

def pie_chart(df, feature):
    plt.figure(figsize=(24, 6))
    counts = df[feature].value_counts()
    counts.plot.pie(autopct='%1.1f%%', figsize=(8, 8))
    plt.title(f'{feature} Distribution')
    plt.ylabel('')
    plt.show()

def hexbin_plot(df, feature1, feature2):
    plt.figure(figsize=(24, 8))
    plt.hexbin(df[f"{feature1}"], df[f"{feature2}"], gridsize=25, cmap='Reds', mincnt=10, alpha=1)
    plt.colorbar(label='Puzzle Count')
    plt.title(f'Puzzle {feature1} vs {feature2} (Hexbin)', fontsize=15)
    plt.xlabel(f'Puzzle {feature1}', fontsize=12)
    plt.ylabel(f'Puzzle {feature2}', fontsize=12)
    plt.tight_layout()
    plt.show()

def scatter_plot(df, feature1, feature2):
    plt.figure(figsize=(24, 6))
    plt.scatter(df[f"{feature1}"], df[f"{feature2}"], alpha=0.5, color='blue')
    plt.title(f'{feature1} vs {feature2}')
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    plt.show()

def violin_plot(df, feature1, feature2):
    plt.figure(figsize=(24, 12))
    sns.violinplot(x=df[f"{feature1}"], y=df[f"{feature2}"])
    plt.title(f'{feature1} vs {feature2} (Violin Plot)')
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    plt.show()

def pair_plot(df, features):
    sns.pairplot(df[features])
    plt.suptitle('Pairplot for Numerical Variables', y=1.02)
    plt.show()

def is_valid_fen(fen):
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False

def is_valid_move(fen, moves):
    board = chess.Board(fen)
    for move in moves.split():
        if chess.Move.from_uci(move) not in board.legal_moves:
            return False  
        board.push_uci(move)
    return True 

def IQR_outlier_analysis(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    print(f"Lower Bound: {lower_bound}")
    print(f"Upper Bound: {upper_bound}")
    print(f'Number of Outliers: {len(outliers)}')
