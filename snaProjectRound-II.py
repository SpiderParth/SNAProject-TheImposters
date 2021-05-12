# -*- coding: utf-8 -*-
"""snaProjectRound-II.ipynb

**Importing the required Libraries**
"""

import random
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from math import log
from scipy.cluster import hierarchy
from scipy.spatial import distance
from networkx.algorithms.community.centrality import girvan_newman

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

"""**Community Detection**

Girvan–Newman Algorithm
"""

def girvanNewman(graph, name):
  rand = random.randint(0, 999)
  numberOfSteps = 5
  pos = nx.spring_layout(graph)
  comp = girvan_newman(graph)
  for communities in itertools.islice(comp, numberOfSteps):
    communitiesList = []
    communitiesList = tuple(sorted(c) for c in communities)
    colorMap = []
    for node in graph.nodes:
      for community in communitiesList:
        if node in community:
          index = communitiesList.index(community)
      colorMap.append(index * rand)
    plt.figure(figsize=(50, 50))
    plt.title('Communitites using Givan-Newman Algorithm for' + ' ' + name, fontsize=20)
    plt.text(0.85, 0.85, 'Number of Communitites: ' + str(len(communitiesList)), fontsize=15)
    nx.draw(graph, pos, node_color=colorMap, with_labels=True)
    plt.show()

girvanNewman(facebookNetwork, 'Facebook Network')

girvanNewman(emailNetwork, 'Email Network')

"""Ravasz Algorithm"""

def createDistanceMatrix(graph):
  distanceMatrix = np.zeros((len(graph.nodes),len(graph.nodes)))
  for node in graph.nodes:
    for neighbor in graph.neighbors(node):
      if node != neighbor:
        firstNeighbors = set(graph.neighbors(node))
        secondNeighbors = set(graph.neighbors(neighbor))
        commonNeighbors = list((firstNeighbors & secondNeighbors) - {node, neighbor})
        countCommonNeighbors = len(commonNeighbors)
        minDegree = sorted(list(graph.degree([node, neighbor])), key=lambda item: item[1])[0][1]
        distanceMatrix[node][neighbor] = (countCommonNeighbors + 1)/minDegree
        distanceMatrix[neighbor][node] = (countCommonNeighbors + 1)/minDegree
  return distanceMatrix

def plotDendogram(linkageMatrix, name):
  plt.figure(figsize=(50,20))
  plt.title('Ravasz Algorithm Dendrogram (truncated to 5 clusters) for' + ' ' + name)
  plt.xlabel('Nodes in the Cluster')
  plt.ylabel('Distance')
  hierarchy.dendrogram(linkageMatrix, truncate_mode='lastp', p=5, leaf_rotation=90., leaf_font_size=12.)

def ravasz(graph, name):
  distanceMatrix = createDistanceMatrix(graph)
  upperTriangle = distance.squareform(distanceMatrix)
  linkageMatrix = hierarchy.average(upperTriangle)
  plotDendogram(linkageMatrix, name)

ravasz(facebookNetwork, 'Facebook Network')

ravasz(emailNetwork, 'Email Network')

"""**Independent Cascade Model on Scale-free Network**"""

scalefreeNetwork = nx.scale_free_graph(1000)
scalefreeNetwork.name = 'Scale-free Network'

print(nx.info(scalefreeNetwork))

plt.figure(figsize=(50, 50))
plt.title('Scale-free Network', fontsize=20)
nx.draw(scalefreeNetwork, pos=nx.spring_layout(scalefreeNetwork), with_labels=True)

def assignActivationProbabilities(graph):
  for node in list(graph.nodes):
    neighborList = list(graph.neighbors(node))
    totalNeighbors = 0
    for neighbor in neighborList:
      multiEdges = graph.number_of_edges(node, neighbor)
      totalNeighbors += multiEdges
    activationProbability = np.random.random(totalNeighbors)
    activationProbability /= activationProbability.sum()
    i = 0
    for neighbor in neighborList:
      multiEdges = graph.number_of_edges(node, neighbor)
      for edge in range(0, multiEdges):
        graph[node][neighbor][edge]['weight'] = activationProbability[i]
        i += 1

def ICM(graph, seedNode, nIterations=1):
  totalSteps = 0
  totalSpread = 0
  for iteration in range(0, nIterations):
    np.random.seed(iteration)
    active = seedNode[:]
    newActive = seedNode[:]
    steps = 0
    while newActive:
      activatedNodes = []
      for node in newActive:
        neighbors = graph.neighbors(node)
        for neighbor in list(neighbors):
          maxWeight = -1
          maxWeightIndex = -1
          for index in list(graph[node][neighbor]):
            if dict(graph[node][neighbor])[index]['weight'] > maxWeight:
              maxWeight = dict(graph[node][neighbor])[index]['weight']
              maxWeightIndex = index
          random = np.random.uniform(0, 0.01)
          if random < graph[node][neighbor][maxWeightIndex]['weight']:
            activatedNodes.append(neighbor)
      newActive = list(set(activatedNodes) - set(active))
      active += newActive
      steps += 1
    totalSteps += steps
    totalSpread += len(active)
  return totalSteps/nIterations, totalSpread/nIterations

assignActivationProbabilities(scalefreeNetwork)
icmStepsAndSpread = []
for i in range(0, len(G.nodes)):
  seedNode = [i]
  totalSteps, totalSpread = ICM(scalefreeNetwork, seedNode)
  icmStepsAndSpread.append((i, totalSteps, totalSpread))
pd.DataFrame(sorted(icmStepsAndSpread, key=lambda item: item[2], reverse=True), columns=['Seed Node', 'Steps', 'Spread'])