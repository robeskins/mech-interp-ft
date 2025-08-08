import os
from pathlib import Path
from eap.graph import Graph
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import math

def node_diff(graph1: Graph, graph2: Graph):
    prev_nodes = graph1.nodes
    curr_nodes = graph2.nodes
    added = sum(
        1 for name, edge in curr_nodes.items()
        if edge.in_graph and not prev_nodes[name].in_graph
    )   

    removed = sum(
        1 for name, edge in prev_nodes.items()
        if edge.in_graph and not curr_nodes[name].in_graph
    )

    print("Nodes added:", added)
    print("Nodes removed:", removed)
    print("Node Total:", added + removed)
    return added, removed, added + removed

def edge_diff(graph1: Graph, graph2: Graph):
    prev_edges = graph1.edges 
    curr_edges = graph2.edges

    added = sum(
        1 for name, edge in curr_edges.items()
        if edge.in_graph and not prev_edges[name].in_graph
    )   

    removed = sum(
        1 for name, edge in prev_edges.items()
        if edge.in_graph and not curr_edges[name].in_graph
    )

    print("Edges added:", added)
    print("Edges removed:", removed)
    print("Edge Total:", added + removed)
    return added, removed, added + removed

def read_accuracy(path):
    with open(path, 'r') as f:
        metrics = json.load(f)
    return metrics.get('percentage_performance_kl')

def find_graphs_and_load(task_results_folder):
    print(task_results_folder)
    task_a_folder = [os.path.join(task_results_folder, f)
                     for f in os.listdir(task_results_folder)
                     if os.path.isdir(os.path.join(task_results_folder, f)) and 'task_a' in f
                     ][0]
    
    task_a_checkpoint_0 = Path(task_a_folder) / 'checkpoint-0'/ 'graph.json'
    task_a_checkpoint_500 = Path(task_a_folder) / 'checkpoint-500'/ 'graph.json'

    graph_a_0 = Graph.from_json(task_a_checkpoint_0)
    graph_a_500 = Graph.from_json(task_a_checkpoint_500)

    total_edges = len(graph_a_0.edges)
    five_percent_edges = int(total_edges * 0.01)
    graph_a_0.apply_topn(five_percent_edges , absolute=True, prune=True)
    graph_a_500.apply_topn(five_percent_edges, absolute=True, prune=True)

    performance_a_0_path = Path(task_a_folder) / 'checkpoint-0'/ 'metrics.json'
    performance_a_500_path = Path(task_a_folder) / 'checkpoint-500'/ 'metrics.json'
    accuracy_0 = read_accuracy(performance_a_0_path)
    accuracy_500 = read_accuracy(performance_a_500_path)
    
    return graph_a_0, graph_a_500, accuracy_0, accuracy_500

def run_graph_changes(folders):
    acc_diffs = []
    edge_changes = []
    node_changes = []

    for folder in tqdm(folders):
        graph_a_0, graph_a_500, accuracy_0, accuracy_500 = find_graphs_and_load(folder)
        edge_changes.append(edge_diff(graph_a_0, graph_a_500)[2])
        node_changes.append(node_diff(graph_a_0, graph_a_500)[2])
        acc_diffs.append(accuracy_0 - accuracy_500)
    return acc_diffs, edge_changes, node_changes

def plot_graph_changes(folders, output_file='graph_changes.png'):
    acc_diffs, edge_changes, node_changes = run_graph_changes(folders)
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.scatter(edge_changes, acc_diffs, color='blue')
    plt.title('Circuit KL vs Edge Changes')
    plt.xlabel('Edge Changes')
    plt.ylabel('Circuit KL')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(node_changes, acc_diffs, color='green')
    plt.title('Circuit KL vs Node Changes')
    plt.xlabel('Node Changes')
    plt.ylabel('Circuit KL')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.show()

if __name__ == '__main__':

    task_results_folder = 'results_1'
    folders = [
        os.path.join(task_results_folder, f)
        for f in os.listdir(task_results_folder)
    ]
    
    plot_graph_changes(folders)

