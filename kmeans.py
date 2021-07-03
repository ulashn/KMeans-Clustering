import math
import random
import numpy as np
import copy
import sys

random.seed(5710414)

def calcEuclidian(p1, p2):
    return np.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]))

def calcManhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

def randColor():
    return int(random.uniform(0, 256)), int(random.uniform(0, 256)), int(random.uniform(0, 256))

def initializeCenters(num):
    centers = []
    for i in range(num):
        temp = randColor()
        centers.append(temp)
    return centers

def assignDataToClusters(data, centers, metric):
    cluster = []
    for k in range(len(centers)):
        temp = []
        cluster.append(temp)

    rows, cols = data.shape[0], data.shape[1]
    i = 0
    minindex = 0

    while(i < cols):
        currentMin = float("inf")
        rgb = data[0][i], data[0][i+1], data[0][i+2]
        for j in range(len(centers)):
            if metric == 'manhattan':
                tempDist = calcManhattan(rgb, centers[j])
                if(tempDist < currentMin):
                    currentMin = tempDist
                    minindex = j
                else:
                    continue
            else:
                tempDist = calcEuclidian(rgb, centers[j])
                if(tempDist < currentMin):
                    currentMin = tempDist
                    minindex = j
                else:
                    continue
        
        cluster[minindex].append(rgb)
        i = i+3

    return cluster

def calcNewCenter(cluster, oldCenters, metric, epsilon):
    newCenters = []
    flag = False
    for i in range(len(cluster)):
        num = len(cluster[i])
        if num == 0:
            rgb = 0,0,0
            newCenters.append(rgb)
            if(metric == 'manhattan'):
                distcents = calcManhattan(newCenters[i], oldCenters[i])
                if(distcents > epsilon):
                    flag = True
            else:
                distcents = calcEuclidian(newCenters[i], oldCenters[i])
                if(distcents > epsilon):
                    flag = True

        else:
            totalr = 0
            totalg = 0
            totalb = 0
            for j in range(num):
                totalr = cluster[i][j][0] + totalr
                totalg = cluster[i][j][1] + totalg
                totalb = cluster[i][j][2] + totalb
            avgr = float(totalr) / num
            avgg = float(totalg) / num
            avgb = float(totalb) / num
            result = avgr,avgg,avgb
            newCenters.append(result)
            if(metric == 'manhattan'):
                distcents = calcManhattan(newCenters[i], oldCenters[i])
                if(distcents > epsilon):
                    flag = True
            else:
                distcents = calcEuclidian(newCenters[i], oldCenters[i])
                if(distcents > epsilon):
                    flag = True

    return newCenters, flag


class KMeans:
    def __init__(self, X, n_clusters, max_iterations=1000, epsilon=0.01, distance_metric="manhattan"):        
        self.X = X
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.clusters = []
        self.cluster_centers = []
        self.epsilon = epsilon
        self.max_iterations = max_iterations

    def fit(self):
        self.cluster_centers = initializeCenters(self.n_clusters)
        iteration = 0
        flag = True
        while iteration < self.max_iterations and flag:
            print("KMeans iteration: ", iteration+1)
            self.clusters = assignDataToClusters(self.X, self.cluster_centers, self.distance_metric)
            newCenters , flag = calcNewCenter(self.clusters, self.cluster_centers, self.distance_metric, self.epsilon)
            self.cluster_centers = newCenters
            iteration += 1
        if(flag == False):
            print("Epsilon boundary reached! Halting...")
        else:
            print("Max iterations reached! Halting...")
        

    def predict(self, instance):
        minIndex = 0
        currentMin = float("inf")
        for i in range(len(self.cluster_centers)):
            if(self.distance_metric == "manhattan"):
                dist = calcManhattan(instance, self.cluster_centers[i])
                if(dist < currentMin):
                    minIndex = i
                    currentMin = dist
                else:
                    continue
            else:
                dist = calcEuclidian(instance, self.cluster_centers[i])
                if(dist < currentMin):
                    minIndex = i
                    currentMin = dist
                else:
                    continue
        
        return minIndex
