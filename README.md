# SNAProject-TheImposters
This repository contains the SNA projects made by The Imposters.

## Datasets
You can choose two data sets from the below links (both having several options).
1. http://konect.uni-koblenz.de
2. https://snap.stanford.edu/data/
If the first first link is not working please try this: http://konect.cc/

## Project Goals
ROUND-I

Once you fix the datasets you have to SOLVE TWO PROBLEMS for both the two dataset separately:
1. Find all centrality measures, clustering coefficients (both local and global) and reciprocity and transitivity that we have studied in the class using appropriate algorithms (you may use specific packages for this or write your own algorithm for the same).
2. Try to get an algorithm package in Python to find the maximum connected component (called a giant component in the class) in a given graph G. Let us denote the number of nodes in the giant component of a graph G as N_G. Vary ⟨k⟩ from 0 to 5 with increment of 0.1. For each value of ⟨k⟩ find the ratio N_G/N where N is the number of nodes in the graph. Plot this ratio with respect to ⟨k⟩. Take ⟨k⟩ as x-axis and ratio N_G/N as y-axis.

ROUND-II

1. Find the giant component G in the network (note that giant components it the largest connected subgraph the constricted network/graph). Let N_G denote the number of nodes in G. Find N_G/N where, N is the total number of nodes in the network.
2. Apply Girvan Newman algorithm and Ravasz algorithm to find the communities step by step and illustrate each step the communities got and stop after 5 steps. Show all the communities and give your understanding about the communities that you got through the algorithm. Looking at the output given by the two algorithms compare and contrast the two algorithms.
3. Create a scale-free network using appropriate Python package (find out!). Apply ICM (Independent Cascade Model) to find the maximum number steps required to get to the maximum number of nodes. This you may repeat 5 times by starting from different nodes and see how many steps are required for the above. Activation probabilities for the pair of nodes which is needed for ICM can be assigned randomly. When you are assigning it randomly note this point: from a node say v if there are three edges to different vertices w, x, and y then, it should be p(v,w) + p(v,x) + p(v,y) = 1.

We've chose to work with:-
1. the Social Circles Facebook network: https://snap.stanford.edu/data/ego-Facebook.html
2. the email-Eu-core network: https://snap.stanford.edu/data/email-Eu-core.html
