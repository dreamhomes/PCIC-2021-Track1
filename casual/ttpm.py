#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : ttpm.py
@Time        : 2021-06-22 16:55:26
@Author      : dreamhomes
@Description : TTPM model with topology.
"""
import os
import networkx as nx
import numpy as np
import pandas as pd
from castle.algorithms import TTPM
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from pyvis.network import Network

os.environ["NUMEXPR_MAX_THREADS"] = "12"


alarm_path = "../datasets/with_topology/2/Alarm.csv"
topo_path = "../datasets/with_topology/2/Topology.npy"
dag_path = "../datasets/with_topology/2/DAG.npy"


def main():
    # 历史告警
    alarm_data = pd.read_csv(alarm_path, encoding="utf")
    # 拓扑图
    topo_matrix = np.load(topo_path)
    # 因果图
    dag_matrix = np.load(dag_path)

    X = alarm_data.iloc[:, 0:3]
    X.columns = ["event", "node", "timestamp"]
    X = X.reindex(columns=["event", "timestamp", "node"])

    ttpm = TTPM(topo_matrix, max_iter=20, max_hop=2)

    ttpm.learn(X)  # 迭代时间非常长...

    est_causal_matrix = ttpm.causal_matrix.to_numpy()

    np.save("../output/est_graphs/2.npy", est_causal_matrix)

    GraphDAG(est_causal_matrix, dag_matrix)
    g_score = MetricsDAG(est_causal_matrix, dag_matrix).metrics["gscore"]
    print(f"g-score: {g_score}")

    TP = []
    FP = []
    FN = []
    for i in range(len(est_causal_matrix)):
        for j in range(len(est_causal_matrix)):
            if est_causal_matrix[i][j] == 1 and dag_matrix[i][j] == 1:
                TP.append((i, j))
            if est_causal_matrix[i][j] == 1 and dag_matrix[i][j] == 0:
                FP.append((i, j))
            if est_causal_matrix[i][j] == 0 and dag_matrix[i][j] == 1:
                FN.append((i, j))
    print("TP {}".format(len(TP)))
    print("FP {}".format(len(FP)))
    print("FN {}".format(len(FN)))

    est_net = Network("500px", "900px", notebook=False, directed=True, layout=False)
    est_g = nx.from_numpy_matrix(est_causal_matrix)
    est_net.from_nx(est_g)
    est_net.show("../output/draw_graphs/est_graph.html")

    true_net = Network("500px", "900px", notebook=True, directed=True, layout=False)
    true_g = nx.from_numpy_matrix(dag_matrix)
    true_net.from_nx(true_g)
    true_net.show("../output/draw_graphs/true_graph.html")

    topo_net = Network("500px", "900px", notebook=True, directed=True, layout=False)
    topo_g = nx.from_numpy_matrix(topo_matrix)
    topo_net.from_nx(topo_g)
    topo_net.show("../output/draw_graphs/topo.html")


if __name__ == "__main__":
    main()
