import matplotlib.pyplot as plt
import networkx as nx

def draw_vit_model_diagram():
    G = nx.DiGraph()

    # Add nodes
    G.add_node("Input Image\n(3, 144, 144)", pos=(0, 10))
    G.add_node("Patch Embedding\n(Patch Size: 4x4, Emb Dim: 32)", pos=(0, 8))
    G.add_node("Positional Embedding\n(+ CLS Token)", pos=(0, 6))
    for i in range(1, 7):  # 6 Transformer Layers
        G.add_node(f"Transformer Layer {i}\n(Multi-Head Attention + FFN)", pos=(0, 6 - 2*i))
    G.add_node("Classification Head\n(Layer Norm + Linear)", pos=(0, -8))
    G.add_node("Output\n(37 Classes)", pos=(0, -10))

    # Add edges
    G.add_edge("Input Image\n(3, 144, 144)", "Patch Embedding\n(Patch Size: 4x4, Emb Dim: 32)")
    G.add_edge("Patch Embedding\n(Patch Size: 4x4, Emb Dim: 32)", "Positional Embedding\n(+ CLS Token)")
    G.add_edge("Positional Embedding\n(+ CLS Token)", "Transformer Layer 1\n(Multi-Head Attention + FFN)")
    for i in range(1, 6):
        G.add_edge(f"Transformer Layer {i}\n(Multi-Head Attention + FFN)", f"Transformer Layer {i+1}\n(Multi-Head Attention + FFN)")
    G.add_edge("Transformer Layer 6\n(Multi-Head Attention + FFN)", "Classification Head\n(Layer Norm + Linear)")
    G.add_edge("Classification Head\n(Layer Norm + Linear)", "Output\n(37 Classes)")

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 12))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title("Vision Transformer (ViT) Model Diagram")
    plt.show()

draw_vit_model_diagram()
