# %% [markdown]
# # **Jazz musician network**

# %%
import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import tree
import pandas as pd
import numpy as np
import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
import math




# %%
jazz = nx.read_gml( 'jazz.gml', label = None)
jazz.add_weighted_edges_from([(u,v,2) for u,v in jazz.edges])

# %%
cmap = {
    0 : 'maroon',
    1 : 'teal',
    2 : 'black', 
    3 : 'orange',
    4 : 'green',
    5 : 'yellow',
    6 : 'red'
}

# %% [markdown]
# **forceatlas2** is a force-directed layout close to other algorithms used for network spatialization
# 
# https://github.com/bhargavchippada/forceatlas2

# %%
#function to plot posicions
forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution=True,  # Dissuade hubs
                        linLogMode=False,  # NOT IMPLEMENTED
                        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence=1.0,

                        # Performance
                        jitterTolerance=1.0,  # Tolerance
                        barnesHutOptimize=True,
                        barnesHutTheta=1.2,
                        multiThreaded=False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio=2.0,
                        strongGravityMode=True,
                        gravity=9.0,

                        # Log
                        verbose=True)

# %%

degree = [v for (u,v) in jazz.degree]
plt.figure(figsize=(20,16))

size_node =  [(v ** 2)/5 for v in degree]
color_node =  [v for v in degree]

positions = forceatlas2.forceatlas2_networkx_layout(jazz, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz, positions, edge_color="black", alpha=0.09)
plt.axis('off')
plt.show()





# %% [markdown]
# **number of nodes:**  198 

# %% [markdown]
# **number of edges:** 2742

# %%

degrees = [v for (u,v) in jazz.degree]

plt.figure(figsize=(20,16))
size_node =  [(v ** 2)/5 for v in degrees]
color_node =  [cmap[0] if v > 10 else cmap[1] for v in degrees]

positions = forceatlas2.forceatlas2_networkx_layout(jazz, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz, positions, edge_color="black", alpha=0.09)
plt.axis('off')
plt.show()


# %% [markdown]
# 

# %% [markdown]
# ## **Community**
# > - girvan_newman
# > 
# > - greedy_modularity_communities
# > 
# > - cliques
# > 
# > - asyn_fluidc

# %% [markdown]
# ### **Comunyties is by *girvan_newman algorithm***
#  
# The Girvan-Newman algorithm for the detection and analysis of community structure relies on the iterative elimination of edges that have the highest number of shortest paths between nodes passing through them. By removing edges from the graph one-by-one, the network breaks down into smaller pieces, so-called communities. The algorithm was introduced by Michelle Girvan and Mark Newman
# 
# **How does it work?**
# 
# The idea was to find which edges in a network occur most frequently between other pairs of nodes by finding edges betweenness centrality. The edges joining communities are then expected to have a high edge betweenness. The underlying community structure of the network will be much more fine-grained once the edges with the highest betweenness are eliminated which means that communities will be much easier to spot.
# 
# The Girvan-Newman algorithm can be divided into four main steps:
# 
# - For every edge in a graph, calculate the edge betweenness centrality.
# - Remove the edge with the highest betweenness centrality.
# - Calculate the betweenness centrality for every remaining edge.
# - Repeat steps 2-4 until there are no more edges left.

# %%
from networkx.algorithms import community

communities_generator = community.girvan_newman(jazz)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
jazz_communities =sorted(map(sorted, next_level_communities))


# %%
#cominities by .girvan_newman
plt.figure(figsize=(20,16))

degree = dict(jazz.degree)

size_node = [ degree[v]**2/5 if v in jazz_communities[0] else 100 if v in jazz_communities[1] else 200 for v in jazz]

color_node =  [cmap[1] if v in jazz_communities[0] else cmap[3] if v in jazz_communities[1] else cmap[4] for v in jazz]

positions = forceatlas2.forceatlas2_networkx_layout(jazz, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz, positions, edge_color="black", alpha=0.09)
plt.axis('off')
plt.show()

# %% [markdown]
# > number of comunities: 3
# > 
# > *number of nodes:*  198 
# > 
# > *number of edges:* 2742
# 
# > - first comunity *teal* nodes: 193
# >
# > - second comunity *orange* nodes: 4
# >
# > - tird comunity *green* nodes: 1

# %% [markdown]
# ### **Comunity based in *greedy_modularity_communities***
# 
# Find communities in G using greedy modularity maximization.
# 
# This function uses Clauset-Newman-Moore greedy modularity maximization to find the community partition with the largest modularity.
# 
# Greedy modularity maximization begins with each node in its own community and repeatedly joins the pair of communities that lead to the largest modularity until no futher increase in modularity is possible (a maximum). Two keyword arguments adjust the stopping condition. cutoff is a lower limit on the number of communities so you can stop the process before reaching a maximum (used to save computation time). best_n is an upper limit on the number of communities so you can make the process continue until at most n communities remain even if the maximum modularity occurs for more. To obtain exactly n communities, set both cutoff and best_n to n.
# 
# This function maximizes the generalized modularity, where resolution is the resolution parameter, often expressed as . See modularity().

# %%
from networkx.algorithms.community import greedy_modularity_communities
greedy_comunities = greedy_modularity_communities(jazz)

# %%
plt.figure(figsize=(20,16))

degree = dict(jazz.degree)

size_node = [ degree[v]**2/5 if v in jazz_communities[0] else 100 if v in jazz_communities[1] else 200 for v in jazz]

color_node =  [cmap[1] if v in greedy_comunities[0] else cmap[3] if v in greedy_comunities[1] else cmap[4] for v in jazz]


positions = forceatlas2.forceatlas2_networkx_layout(jazz, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz, positions, edge_color="black", alpha=0.09)
plt.axis('off')
plt.show()

# %% [markdown]
# > number of comunities: 3
# > 
# > *number of nodes*  198 
# > 
# > *number of edges* 2742
# 
# > - first comunity *teal* nodes 69
# >
# > - second comunity *orange* nodes 66
# >
# > - tird comunity *green* nodes : 63

# %% [markdown]
#                                 after this step jazz graph trasnform to unidrected graph

# %% [markdown]
# ### **CLIQUES**
# 
# Finding the largest clique in a graph is NP-complete problem, so most of these algorithms have an exponential running time

# %% [markdown]
# A clique of a graph G is a set X of vertices of G with the property that every pair of distinct vertices in X are adjacent in G. A maximal clique of a graph G is a clique X of vertices of G, such that there is no clique Y of vertices of G that contains all of X and at least one other vertex.
# 
# Given a graph G, its clique graph K(G) is a graph such that
# 
# - every vertex of K(G) represents a maximal clique of G; and
# - two vertices of K(G) are adjacent when the underlying cliques in G share at least one vertex in common.
# The clique graph K(G) can also be characterized as the intersection graph of the maximal cliques of G

# %%
#trasform to indirect graph
jazz_indirect = jazz.to_undirected()
#cliques = nx.Cliques(jazz_indirect)

# %%

plt.figure(figsize=(20,16))

degree = dict(jazz_indirect.degree)

size_node = [ degree[v]**2/5  for v in jazz_indirect.nodes ]

color_node =  [cmap[4]  for v in jazz_indirect.nodes]

positions = forceatlas2.forceatlas2_networkx_layout(jazz_indirect, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz_indirect, positions, node_size=size_node, node_color='teal', alpha=0.7)
nx.draw_networkx_edges(jazz_indirect, positions, edge_color= color_node, alpha=0.1)
plt.axis('off')
plt.show()

# %% [markdown]
# > plot of undiredcted graph

# %% [markdown]
# ### clique

# %%
#superposition of max cliuqe into graph
#indirect graph
plt.figure(figsize=(20,16))

degree = dict(jazz_indirect.degree)
cliques_jazz = list(nx.find_cliques(jazz_indirect))

size_node = [ 400 if v in cliques_jazz[6] else degree[v]**2/5  for v in jazz_indirect.nodes ]

color_node = [cmap[3] if v in cliques_jazz[6] else cmap[1]  for v in jazz_indirect.nodes]

edge_color =  [cmap[2] if u in cliques_jazz[6] and v in cliques_jazz[6] else cmap[5] for (u,v) in jazz_indirect.edges]

positions = forceatlas2.forceatlas2_networkx_layout(jazz_indirect, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz_indirect, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz_indirect, positions, edge_color=edge_color, alpha=0.2)
plt.axis('off')
plt.show()

# %% [markdown]
# *yellow nodes are all nodes wich are a clique chosing randomly from set of all cliques*
# > number of cliques in jazz network: 746

# %% [markdown]
# ### Bigest clique

# %%
degree = dict(jazz_indirect.degree)
cliques_jazz = list(nx.find_cliques(jazz_indirect))

max_clique =[]
for i in cliques_jazz:
    if len(max_clique) < len(i):
        max_clique = i


# %%
plt.figure(figsize=(20,16))

size_node = [ degree[v]**2/4 if v in max_clique else degree[v]**2/5  for v in jazz_indirect.nodes ]

color_node = [cmap[3] if v in max_clique else cmap[1]  for v in jazz_indirect.nodes]

edge_color =  [cmap[2] if u in max_clique and v in max_clique else cmap[5] for (u,v) in jazz_indirect.edges]

positions = forceatlas2.forceatlas2_networkx_layout(jazz_indirect, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz_indirect, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz_indirect, positions, edge_color=edge_color, alpha=0.2)
plt.axis('off')
plt.show()

# %% [markdown]
# > maximum clique in jazz network
# >> number of elments: 30

# %% [markdown]
# #### **asyn_fluidc**

# %% [markdown]
# Returns communities in G as detected by Fluid Communities algorithm.
# 
# The asynchronous fluid communities algorithm is described in. The algorithm is based on the simple idea of fluids interacting in an environment, expanding and pushing each other. Its initialization is random, so found communities may vary on different executions.

# %%
communities = list(nx.community.asyn_fluidc(jazz_indirect, 4))
plt.figure(figsize=(20,16))

degree = dict(jazz.degree)

size_node = [ degree[v]**2/5  for v in jazz]

color_node =  [cmap[0] if v in communities[0] else cmap[1] if v in communities[1] else cmap[2] if v in communities[2] else cmap[3] if v in communities[3] else cmap[5] for v in jazz]


positions = forceatlas2.forceatlas2_networkx_layout(jazz, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz, positions, edge_color="black", alpha=0.09)
plt.axis('off')
plt.show()

# %% [markdown]
# > number of comunities: 4
# > 
# > *number of nodes:*  198 
# > 
# > *number of edges:* 2742
# 
# > - first comunity *maroon* nodes: 41
# >
# > - second comunity *teal* nodes: 57
# >
# > - tird comunity *black* nodes: 57
# > 
# > - fourth comunity *orange* nodes: 22
# 

# %% [markdown]
# ## *detecting the most important, relevant, influential, or central nodes*

# %% [markdown]
# - Most connected (degree centrality).
# - The closest to the rest of the nodes (closeness centrality ).
# - Through which passes more information (betweenness centrality ).
# - one connected to other important nodes (eigenvector centrality ).
# - Among many others 

# %% [markdown]
# 

# %%
a = dict(jazz.degree)
b = [a[i] for i in range(1,len(a)+ 1)]
c = [i if a[i] == 100 else 0 for i in range(1,len(a)+ 1)]

# %%
degree = dict(jazz.degree)

# %%
#(136,100)(#node, degree)
degree = dict(jazz.degree)
plt.figure(figsize=(20,16))


size_node = [ (deg**2)/5  for deg in b]
color_node =  [cmap[0] if v == 136 else cmap[1] for v in jazz.nodes]
edge_color =  [cmap[2] if u == 136 or v ==  136  else cmap[3] for (u,v) in jazz.edges]

positions = forceatlas2.forceatlas2_networkx_layout(jazz, pos=None, iterations=2000)
nx.draw_networkx_nodes(jazz, positions, node_size=size_node, node_color=color_node, alpha=0.7)
nx.draw_networkx_edges(jazz, positions, edge_color=edge_color, alpha=0.09)
plt.axis('off')
plt.show()

# %% [markdown]
# > degree of the most conected: 100
# >
# >> id of the most node: 136
# >
# > average degree:27.696969696969695

# %% [markdown]
# ### ----------------------------------------------------------------------------------------------------

# %% [markdown]
# 


