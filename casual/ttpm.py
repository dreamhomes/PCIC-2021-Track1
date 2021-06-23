#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : ttpm.py
@Time        : 2021-06-22 16:55:26
@Author      : dreamhomes
@Description : TTPM model with topology.
"""
import os
import time
import typer

import networkx as nx
import numexpr as ne
import numpy as np
import pandas as pd
from castle.algorithms import TTPM
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from loguru import logger
from pyvis.network import Network

ne.set_vml_num_threads(12)


def preprocessing(alarm_data):
    """alarm data preprocessing.

    Args:
        alarm_data ([type]): [description]
    """

    X = alarm_data.iloc[:, 0:3]
    X.columns = ["event", "node", "timestamp"]
    X = X.reindex(columns=["event", "timestamp", "node"])

    logger.info("Data preprocessing finished.")

    return X


def model_fit(train_data, topo_matrix, iters):
    """model train.

    Args:
        train_data (pd.dataframe): alert data.
        topo_matrix (np.array): device topology.
    """
    model_time_start = time.time()

    ttpm = TTPM(topo_matrix, max_iter=iters, max_hop=2)

    ttpm.learn(train_data)  # 迭代时间非常长...

    est_causal_matrix = ttpm.causal_matrix.to_numpy()

    model_time_end = time.time()

    logger.info(f"Model fitting finished. Elapsed time: {model_time_end-model_time_start}")

    return est_causal_matrix


def evaluate(est_causal_matrix, true_graph_matrix):
    """evaluation.

    Args:
        est_causal_matrix (np.array): estimate casual graph
        true_graph_matrix (np.array): true graph
    """
    g_score = MetricsDAG(est_causal_matrix, true_graph_matrix).metrics["gscore"]
    logger.info(f"g-score: {g_score}")

    TP = []
    FP = []
    FN = []
    for i in range(len(est_causal_matrix)):
        for j in range(len(est_causal_matrix)):
            if est_causal_matrix[i][j] == 1 and true_graph_matrix[i][j] == 1:
                TP.append((i, j))
            if est_causal_matrix[i][j] == 1 and true_graph_matrix[i][j] == 0:
                FP.append((i, j))
            if est_causal_matrix[i][j] == 0 and true_graph_matrix[i][j] == 1:
                FN.append((i, j))
    logger.info("TP {}".format(len(TP)))
    logger.info("FP {}".format(len(FP)))
    logger.info("FN {}".format(len(FN)))

    _g_score = max(0.0, (len(TP) - len(FP))) / (len(TP) + len(FN))
    logger.info(f"g-score(Ref): {_g_score}")  # 源码 False Positives + Reversed Edges

    return g_score


def draw_graph(graph_matrix, name="graph"):
    """draw graph.

    Args:
        graph_matrix ([type]): [description]
    """
    net = Network("500px", "900px", notebook=False, directed=True, layout=False)
    g = nx.from_numpy_matrix(graph_matrix)
    net.from_nx(g)

    os.makedirs("./output/draw_graphs/", exist_ok=True)
    net.show(f"./output/draw_graphs/{name}.html")


def main(
    alarm_path: str = typer.Option(default="", help="alarm csv path."),
    topo_path: str = typer.Option(default="", help="topology path."),
    dag_path: str = typer.Option(default="", help="true graph path."),
    dataset_name: str = typer.Option(default="", help="output file name."),
    draw_graph: bool = typer.Option(default=False, help="draw graph."),
    iters: int = typer.Option(default=10, help="number of iterations."),
):
    logger.info("---start---")

    os.makedirs("./output", exist_ok=True)
    # 历史告警
    alarm_data = pd.read_csv(alarm_path, encoding="utf")
    # 拓扑图
    topo_matrix = np.load(topo_path)
    # 因果图
    dag_matrix = np.load(dag_path)

    alarm_data = preprocessing(alarm_data)

    est_causal_matrix = model_fit(alarm_data, topo_matrix, iters)

    os.makedirs("./output/est_graphs", exist_ok=True)
    np.save(f"./output/est_graphs/{iters}-{dataset_name}.npy", est_causal_matrix)

    evaluate(est_causal_matrix, dag_matrix)

    if draw_graph:
        draw_graph(est_causal_matrix, f"est-{dataset_name}")
        draw_graph(dag_matrix, f"true-{dataset_name}")

    # GraphDAG(est_causal_matrix, dag_matrix)

    logger.info("---finished---")


if __name__ == "__main__":
    typer.run(main)
