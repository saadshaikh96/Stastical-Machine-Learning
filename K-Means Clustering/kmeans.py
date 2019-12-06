import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from random import randint


def Euclidean(x,y):
    return np.linalg.norm(x-y)


def areDifferentCentroids(a,b):
    if(len(a)!=len(b)):
        return 1
    for i in range(0,len(a)):
        if(a[i] != b[i]):
            return 1

    return 0


def findMaxLengthCluster(cluster_list):
    max_index = 0
    for jj in range(len(cluster_list)):
        if(len(cluster_list[jj]) > len(cluster_list[max_index])):
            max_index = jj

    return max_index


data = []
k_list = []
sse_list = []

with open('bc.txt','r') as f:
    for line in f.readlines():
        data.append(list(map(int,line.split(',')[1:-1])))

for ii in range(2,9):
    k_list.append(ii)


for k in k_list:

    print ('k=', k)
    new_centroids = [[] for xx in range(k)]
    old_centroids = []
    sum = 0
    count = 0

    for i in range(k):
        old_centroids.append(data[np.random.randint(0,699)])

    while(areDifferentCentroids(new_centroids,old_centroids)):

        cluster_points = [[] for xx in range(k)]
        if(count>0):
            old_centroids = new_centroids
            new_centroids = [[] for xx in range(k)]

        count += 1

        for data_point in range(len(data)):
            min_dist = 99999999
            min_center = 0
            for iterator,center in enumerate(old_centroids):
                distance = Euclidean(np.array(data[data_point]), np.array(center))

                if(distance < min_dist):
                    min_dist = distance
                    min_center = iterator
            cluster_points[min_center].append(data_point)

        #Shift samples from max cluster to empty cluster
        for ite,cluster in enumerate(cluster_points):
            if(not cluster):
                max_len_cluster_index = findMaxLengthCluster(cluster_points)
                move_samples = np.random.randint(1,len(cluster_points[max_len_cluster_index]))
                for mm in range(move_samples):
                    sample_to_move = cluster_points[max_len_cluster_index][0]
                    cluster_points[ite].append(sample_to_move)
                    cluster_points[max_len_cluster_index].remove(sample_to_move)

        #Calculating new centroids
        for ii in range(len(cluster_points)):
            temp_data = []
            for jj in cluster_points[ii]:
                temp_data.append(data[jj])

            temp_data = np.array(temp_data)
            temp_data = np.average(temp_data, axis=0)
            new_centroids[ii] = temp_data.tolist()

    #Calculating Potential Function
    for index,cluster in enumerate(cluster_points):
        for point in cluster:
            sum += Euclidean(np.array(data[point]), np.array(new_centroids[index])) ** 2

    sse_list.append(sum)


plt.plot(k_list, sse_list, '-', color='#99004C', marker='+', linewidth=2.0)
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("SSE vs K")
plt.show()
