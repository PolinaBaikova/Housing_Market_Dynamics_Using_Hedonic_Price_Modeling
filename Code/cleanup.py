#!/usr/bin/env python
"""
slim_road_graph.py – create a smaller road graph for local/VM use
"""
import os
import osmnx as ox
import networkx as nx
ROOT = os.path.dirname(os.path.abspath(__file__))
INFILE  = os.path.join(ROOT, "..", "Data", "orlando_road_network.graphml")
#OUTFILE = os.path.join(ROOT, "..", "Data", "orlando_road_network_light.gpickle")

# ------------------------------------------------------------------- #
# 1.  Load the full graph (requires RAM on a beefy machine)
# ------------------------------------------------------------------- #
print("Loading full GraphML …")
G = ox.load_graphml(INFILE)

# ------------------------------------------------------------------- #
# 2.  Keep only motorway / trunk edges
# ------------------------------------------------------------------- #
KEEP_HWY = {"motorway", "motorway_link", "trunk", "trunk_link"}

edges_to_remove = [
    (u, v, k)
    for u, v, k, d in G.edges(keys=True, data=True)
    if d.get("highway") not in KEEP_HWY
]
G.remove_edges_from(edges_to_remove)
print(f"→ removed {len(edges_to_remove):,} edges not in {KEEP_HWY}")

# Remove any now‑isolated nodes
isolated = list(nx.isolates(G))
G.remove_nodes_from(isolated)
print(f"→ removed {len(isolated):,} isolated nodes")

# ------------------------------------------------------------------- #
# 3.  Drop heavy / unused attributes
# ------------------------------------------------------------------- #
KEEP_EDGE_ATTR = {"geometry", "highway", "name", "ref", "length"}
KEEP_NODE_ATTR = {"highway", "x", "y"}          # x/y always present

for _, _, d in G.edges(data=True):
    for key in list(d):
        if key not in KEEP_EDGE_ATTR:
            d.pop(key)

for _, d in G.nodes(data=True):
    for key in list(d):
        if key not in KEEP_NODE_ATTR:
            d.pop(key)

# ------------------------------------------------------------------- #
# 4.  Save two versions of the graph:
#     a) Standard (modern) with highest protocol (gzip-compressed)
#     b) Legacy-compatible with protocol=4 (no compression)
# ------------------------------------------------------------------- #

import pickle

# (a) Modern compressed version
OUTFILE_NEW = os.path.join(ROOT, "..", "Data", "orlando_road_network_light.pkl.gz")
with open(OUTFILE_NEW, "wb") as fh:
    import gzip
    with gzip.GzipFile(fileobj=fh, mode="wb") as gz:
        pickle.dump(G, gz, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✓ Modern pickle written → {OUTFILE_NEW}")

# (b) Legacy-compatible version
OUTFILE_OLD = os.path.join(ROOT, "..", "Data", "orlando_road_network_light_old.pkl")
with open(OUTFILE_OLD, "wb") as fh:
    pickle.dump(G, fh, protocol=4)
print(f"✓ Legacy pickle written → {OUTFILE_OLD}")

