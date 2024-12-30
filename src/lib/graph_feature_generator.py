from .graph_helper import build_attack_graph, build_legal_moves_graph
import networkx as nx
import chess

def calculate_central_control(graph, verbose=0):
    central_squares = {"d4", "e4", "d5", "e5"}
    central_control = {"white": 0, "black": 0}
    for source, target, data in graph.edges(data=True):
        if target in central_squares:
            if data["color"]:
                if verbose:
                    print(f"Source: {source} Target: {target} Data: {data}")
                central_control[data["color"]] += 1
    return central_control

def calculate_threat_features(graph, verbose=0):
    threats = {"white": {"count": 0, "value": 0}, "black": {"count": 0, "value": 0}}
    for source, target, data in graph.edges(data=True):
        target_color = graph.nodes[target]["color"]
        if target_color and target_color != data["color"]:
            if verbose:
                print(f"Source: {source} Target: {target} Data: {data}")
            threats[data["color"]]["count"] += 1
            threats[data["color"]]["value"] += graph.nodes[target]["value"]
    return {
        "white_threat_count": threats["white"]["count"],
        "white_threat_value": threats["white"]["value"],
        "black_threat_count": threats["black"]["count"],
        "black_threat_value": threats["black"]["value"]
    }

def calculate_mobility(graph, verbose=0):
    mobility = {"white": 0, "black": 0}
    for source, data in graph.nodes(data=True):
        if data["color"]:
            if verbose:
                print(f"Source: {source} Data: {data}")
            out_degree = graph.out_degree(source)
            mobility[data["color"]] += out_degree
    return mobility

def calculate_material_advantage(graph, verbose=0):
    material = {"white": -1000, "black": -1000}
    for _, data in graph.nodes(data=True):
        if data["color"]:
            if verbose:
                print(f"Data: {data}")
            material[data["color"]] += data["value"]
    return {
        "white_material": material["white"],
        "black_material": material["black"],
        "material_difference": material["white"] - material["black"]
    }

def calculate_king_safety(fen, color):
    board = chess.Board(fen)
    attack_graph = build_attack_graph(fen)
    
    king_square = None
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.symbol().lower() == 'k' and ((piece.color and color == 'white') or (not piece.color and color == 'black')):
            king_square = chess.square_name(square)
            break
    if not king_square:
        return {"threatened_squares": 0, "safe_squares": 0}
    
    king_adjacent_squares = chess.SquareSet(chess.BB_KING_ATTACKS[chess.parse_square(king_square)])
    
    threatened_squares = 0
    safe_squares = 0
    for square in king_adjacent_squares:
        square_name = chess.square_name(square)
        if attack_graph.has_node(square_name):
            attackers = [
                source for source, target, edge_data in attack_graph.in_edges(square_name, data=True)
                if edge_data["color"] != color
            ]
            if attackers:
                threatened_squares += 1
            else:
                safe_squares += 1

    return {
        "king_threatened_squares": threatened_squares,
        "king_safe_squares": safe_squares
    }

def calculate_connectivity(graph):
    clustering = nx.clustering(graph.to_undirected())
    white_clustering = [
        clustering[node] for node, data in graph.nodes(data=True)
        if data["color"] == "white"
    ]
    black_clustering = [
        clustering[node] for node, data in graph.nodes(data=True)
        if data["color"] == "black"
    ]
    return {
        "white_connectivity": sum(white_clustering) / len(white_clustering) if white_clustering else 0,
        "black_connectivity": sum(black_clustering) / len(black_clustering) if black_clustering else 0,
    }

def generate_features(row):
    fen = row["FEN"]

    attack_graph = build_attack_graph(fen)
    legal_moves_graph = build_legal_moves_graph(fen)

    central_control = calculate_central_control(attack_graph)

    threats = calculate_threat_features(attack_graph)

    mobility = calculate_mobility(legal_moves_graph)

    material = calculate_material_advantage(attack_graph)

    white_king_safety = calculate_king_safety(fen, "white")
    black_king_safety = calculate_king_safety(fen, "white")

    connectivity = calculate_connectivity(attack_graph)

    return {
        "white_central_control": central_control["white"],
        "black_central_control": central_control["black"],
        "white_threat_count": threats["white_threat_count"],
        "white_threat_value": threats["white_threat_value"],
        "black_threat_count": threats["black_threat_count"],
        "black_threat_value": threats["black_threat_value"],
        "white_mobility": mobility["white"],
        "black_mobility": mobility["black"],
        "white_material": material["white_material"],
        "black_material": material["black_material"],
        "material_difference": material["material_difference"],
        "white_king_threatened_squares": white_king_safety["king_threatened_squares"],
        "white_king_safe_squares": white_king_safety["king_safe_squares"],
        "black_king_threatened_squares": white_king_safety["king_threatened_squares"],
        "black_king_safe_squares": white_king_safety["king_safe_squares"],
        "white_connectivity": connectivity["white_connectivity"],
        "black_connectivity": connectivity["black_connectivity"]
    }

