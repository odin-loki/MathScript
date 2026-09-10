# Graphs, including the negative weights bellman_ford exists for and never saw.
A = [0, 4, 5; 0, 0, 0; 0, -3, 0]
d = graph_bellman_ford(A, 0)
d
G = [0, 1, 0; 1, 0, 1; 0, 1, 0]
graph_degree_centrality(G)
graph_is_connected(G)
graph_diameter(G)
graph_radius(G)
D = [0, 1, 0, 0; 1, 0, 0, 0; 0, 0, 0, 1; 0, 0, 1, 0]
graph_is_connected(D)
