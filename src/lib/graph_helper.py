import chess
import networkx as nx
import matplotlib.pyplot as plt

PIECE_VALUES = {
    "p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 1000, None: 0,
}

def build_attack_graph(fen):
    board = chess.Board(fen)
    G = nx.DiGraph()

    for square in chess.SQUARES:
        square_name = chess.square_name(square)
        piece = board.piece_at(square)
        if piece:
            G.add_node(square_name,
                       piece=piece.symbol(),
                       color="white" if piece.color else "black",
                       value=PIECE_VALUES.get(piece.symbol().lower(), 0))
        else:
            G.add_node(square_name, piece=None, color=None, value=0)

    for square in chess.SQUARES:
        for attacker in board.attackers(chess.WHITE, square):
            source = chess.square_name(attacker)
            target = chess.square_name(square)
            piece = board.piece_at(attacker)
            if piece:
                if board.piece_at(square):
                    if board.piece_at(square).symbol().islower():
                        G.add_edge(source, target,
                                piece=piece.symbol(),
                                value=PIECE_VALUES.get(piece.symbol().lower(), 0),
                                color="white")
                else:
                    G.add_edge(source, target,
                        piece=piece.symbol(),
                        value=PIECE_VALUES.get(piece.symbol().lower(), 0),
                        color="white")
        for attacker in board.attackers(chess.BLACK, square):
            source = chess.square_name(attacker)
            target = chess.square_name(square)
            piece = board.piece_at(attacker)
            if piece:
                if board.piece_at(square):
                    if board.piece_at(square).symbol().isupper():
                        G.add_edge(source, target,
                                piece=piece.symbol(),
                                value=PIECE_VALUES.get(piece.symbol().lower(), 0),
                                color="black")
                else:
                    G.add_edge(source, target,
                        piece=piece.symbol(),
                        value=PIECE_VALUES.get(piece.symbol().lower(), 0),
                        color="black")
    return G

def build_legal_moves_graph(fen):
    board = chess.Board(fen)
    G = nx.DiGraph()

    # Add nodes for each square
    for square in chess.SQUARES:
        square_name = chess.square_name(square)
        piece = board.piece_at(square)
        if piece:
            G.add_node(square_name,
                       piece=piece.symbol(),
                       color="white" if piece.color else "black",
                       value=PIECE_VALUES.get(piece.symbol().lower(), 0))
        else:
            G.add_node(square_name, piece=None, color=None, value=0)

    # Add edges for legal moves
    for move in board.legal_moves:
        source = chess.square_name(move.from_square)
        target = chess.square_name(move.to_square)
        piece = board.piece_at(move.from_square)
        if piece:
            G.add_edge(source, target,
                       piece=piece.symbol(),
                       value=PIECE_VALUES.get(piece.symbol().lower(), 0))

    return G

    central_squares = {"d4", "e4", "d5", "e5"}
    central_control = {"white": {sq: 0 for sq in central_squares},
                       "black": {sq: 0 for sq in central_squares}}
    threats = []

    for source, target, data in graph.edges(data=True):
        if target in central_squares:
            if data["color"] == "white":
                central_control["white"][target] += 1
            elif data["color"] == "black":
                central_control["black"][target] += 1

        target_color = graph.nodes[target]["color"]
        if target_color and target_color != data["color"]:
            threats.append((
                source, target, data["piece"], graph.nodes[target]["value"]
            ))

    return central_control, threats

def visualize_graph(graph, name):
    pos = nx.spring_layout(graph)
    node_colors = [
        "lightblue" if graph.nodes[node]["color"] == "white" else
        "lightcoral" if graph.nodes[node]["color"] == "black" else
        "lightgrey"
        for node in graph.nodes
    ]
    node_labels = {
        node: f"{node}\n{graph.nodes[node]['value']}" for node in graph.nodes
    }

    plt.figure(figsize=(32, 32))
    nx.draw(
        graph, pos, with_labels=True, labels=node_labels, node_color=node_colors,
        font_size=8, node_size=700, edge_color="grey", alpha=0.7,
    )
    plt.title("Chess Graph Visualization")
    plt.savefig(f"../results/images/{name}.png")
    plt.close()


