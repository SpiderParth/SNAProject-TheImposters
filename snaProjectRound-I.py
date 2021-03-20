# -*- coding: utf-8 -*-
"""snaProjectRound-I.ipynb

**Importing the required Libraries**
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import log

"""**Dataset Download**"""

urlFacebookNetwork = 'https://snap.stanford.edu/data/facebook_combined.txt.gz'
urlEmailNetwork = 'https://snap.stanford.edu/data/email-Eu-core.txt.gz'

dataframeFacebookNetwork = pd.read_csv(urlFacebookNetwork, delimiter=' ', header=None, names=['from', 'to'])
dataframeEmailNetwork = pd.read_csv(urlEmailNetwork, delimiter=' ', header=None, names=['from', 'to'])

facebookNetwork = nx.from_pandas_edgelist(dataframeFacebookNetwork, 'from', 'to')
facebookNetwork.name = 'Facebook Network'
emailNetwork = nx.from_pandas_edgelist(dataframeEmailNetwork, 'from', 'to', create_using=nx.DiGraph())
emailNetwork.name = 'E-mail Network'

"""**Information about the Networks**"""

print(nx.info(facebookNetwork))
print()
print(nx.info(emailNetwork))

"""**Networks**

Facebook Network
"""

plt.figure(figsize=(50, 50))
plt.title('Facebook Network', fontsize=20)
nx.draw_networkx(facebookNetwork, pos=nx.spring_layout(facebookNetwork))

"""Email Network"""

plt.figure(figsize=(50, 50))
plt.title('Email Network', fontsize=20)
nx.draw_networkx(emailNetwork, pos=nx.spring_layout(emailNetwork))

"""**Centrality Measures**

Degree Centrality
"""

degCentralityFacebookNetwork = nx.degree_centrality(facebookNetwork)
pd.DataFrame(sorted(degCentralityFacebookNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'degreeCentrality'])

inDegCentralityEmailNetwork = nx.in_degree_centrality(emailNetwork)
pd.DataFrame(sorted(inDegCentralityEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'inDegreeCentrality'])

outDegCentralityEmailNetwork = nx.out_degree_centrality(emailNetwork)
pd.DataFrame(sorted(outDegCentralityEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'outDegreeCentrality'])

"""EigenVector Centrality"""

evCentralityFaceBookNetwork = nx.eigenvector_centrality(facebookNetwork)
pd.DataFrame(sorted(evCentralityFaceBookNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'eigenvectorCentrality'])

evCentralityEmailNetwork = nx.eigenvector_centrality(emailNetwork)
pd.DataFrame(sorted(evCentralityEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'eigenvectorCentrality'])

"""Katz Centrality"""

katzCentralityFacebookNetwork = nx.katz_centrality_numpy(facebookNetwork)
pd.DataFrame(sorted(katzCentralityFacebookNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'katzCentrality'])

katzCentralityEmailNetwork = nx.katz_centrality_numpy(emailNetwork)
pd.DataFrame(sorted(katzCentralityEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'katzCentrality'])

"""PageRank Centrality"""

pagerankCentralityFacebookNetwork = nx.pagerank(facebookNetwork) 
pd.DataFrame(sorted(pagerankCentralityFacebookNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'pagerankCentrality'])

pagerankCentralityEmailNetwork = nx.pagerank(emailNetwork) 
pd.DataFrame(sorted(pagerankCentralityEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'pagerankCentrality'])

"""Closeness Centrality"""

closenessCentralityFacebookNetwork = nx.closeness_centrality(facebookNetwork)
pd.DataFrame(sorted(closenessCentralityFacebookNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'closenessCentrality'])

closenessCentralityEmailNetwork = nx.closeness_centrality(emailNetwork)
pd.DataFrame(sorted(closenessCentralityEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'closenessCentrality'])

"""Betweenness Centrality"""

betweennessCentralityFacebookNetwork = nx.betweenness_centrality(facebookNetwork)
pd.DataFrame(sorted(betweennessCentralityFacebookNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'betweennessCentrality'])

betweennessCentralityEmailNetwork = nx.betweenness_centrality(emailNetwork)
pd.DataFrame(sorted(betweennessCentralityEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'betweennessCentrality'])

"""**Local Clustering Coefficients**"""

localClusteringCoefficientsFacebookNetwork = nx.clustering(facebookNetwork)
pd.DataFrame(sorted(localClusteringCoefficientsFacebookNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'localClusteringCoefficients'])

localClusteringCoefficientsEmailNetwork = nx.clustering(emailNetwork)
pd.DataFrame(sorted(localClusteringCoefficientsEmailNetwork.items(), key=lambda item: item[1], reverse=True), columns=['Node', 'localClusteringCoefficients'])

"""**Average Clustering Coefficient**"""

averageClusteringCoefficientFacebookNetwork = nx.average_clustering(facebookNetwork)
averageClusteringCoefficientFacebookNetwork

averageClusteringCoefficientEmailNetwork = nx.average_clustering(emailNetwork)
averageClusteringCoefficientEmailNetwork

"""**Transitivity/ Global Clustering Coefficient**"""

transitivityFacebookNetwork = nx.transitivity(facebookNetwork)
transitivityFacebookNetwork

transitivityEmailNetwork = nx.transitivity(emailNetwork)
transitivityEmailNetwork

"""**Reciprocity**"""

reciprocityFacebookNetwork = nx.reciprocity(facebookNetwork)
reciprocityFacebookNetwork

reciprocityEmailNetwork = nx.reciprocity(emailNetwork)
reciprocityEmailNetwork

"""**Giant Components**

Giant Component of Facebook Network
"""

plt.figure(figsize=(50, 50))
plt.title('Giant Component of Facebook Network', fontsize=20)
giantFacebookNetwork = facebookNetwork.subgraph(max(nx.connected_components(facebookNetwork), key=len))
nx.draw_networkx(facebookNetwork, pos=nx.spring_layout(facebookNetwork), edgelist=giantFacebookNetwork.edges())

giantFacebookNetwork.number_of_nodes()

"""Giant Component of Email Network"""

plt.figure(figsize=(50, 50))
plt.title('Giant Component of Email Network', fontsize=20)
giantEmailNetwork = emailNetwork.subgraph(max(nx.strongly_connected_components(emailNetwork), key=len))
nx.draw_networkx(emailNetwork, pos=nx.spring_layout(emailNetwork), edgelist=giantEmailNetwork.edges())

giantEmailNetwork.number_of_nodes()

"""**Evolution of Random Network**"""

plt.figure(figsize=(30,10))
plt.title('Evolution of Random Network', fontsize=20)
nodes = 10000
averageDegree = np.arange(0, 10.1, 0.1)
probabilityK = []
for degree in averageDegree:
  randomNetwork = nx.erdos_renyi_graph(nodes, degree/(nodes-1))
  giantRandomNetwork = randomNetwork.subgraph(max(nx.connected_components(randomNetwork), key=len))
  probabilityK.append(giantRandomNetwork.number_of_nodes()/nodes)
plt.plot(averageDegree, probabilityK, 'm', linewidth=5)
plt.axvline(x=1, color='k', linewidth=2)
plt.axvline(x=log(nodes), color='k', linewidth=2)
plt.xlabel('<k>')
plt.ylabel('N_G/N')
plt.show()