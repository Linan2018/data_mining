#!/usr/bin/env python
# coding: utf-8

# In[113]:


from igraph import Graph
import numpy as np 


# In[263]:


def read_gml(path):
    g = Graph.Read_GML(path)
    adj = g.get_adjacency()
    n = adj.shape[0]
    a = np.array([adj[i] for i in range(n)])
    
    return a


# In[433]:


from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号


def distance(metric):
    def f(x, y):
        _ = np.vstack([x,y])
        return pdist(_, metric)[0]
    return f

def cluster(a, method, metric):
    font1 = {
        'weight': 'normal',
        'size': 30,
    }
    
    z = linkage(a, method=method, metric=distance(metric))
    fig = plt.figure(figsize=(25, 10))
    dn = dendrogram(z)
    plt.title('{} {}'.format(metric, method),font1)
    plt.savefig('clusterTree_{}_{}.svg'.format(metric, method))
    return z


# In[444]:


def computeQ(a, z): 
    n = a.shape[0]
    m = int(np.sum(a) / 2)
    k = [np.sum(a[i]) for i in range(n)]
    c = np.arange(n_v)
    def connect(a, b, it):
        buffer[it + n] = buffer[a].union(buffer[b])
        del buffer[a]
        del buffer[b] 
        temp = list(buffer[it + n].copy())
        tar = np.min(temp)
        for i in range(len(temp)):
            c[temp[i]] = tar

    buffer = {k: {k} for k in range(n)}
    q = list(range(n_v))   # 合并i次的Q值
#     print(z)

    for it, line in enumerate(z):
        connect(int(line[0]), int(line[1]), it)
        q[it + 1] = np.sum([[(a[v][w] - k[v]*k[w]/(2*m)) * (c[v]==c[w])
                         for v in range(n) ]for w in range(n)]) / (2*m) 
#         print(it+1, q[it + 1], line[0], line[1])
        print(c)

    return q


# In[445]:


def main(path, method, metric):
    font = {
        'weight': 'normal',
        'size': 13,
    }
    font1 = {
        'weight': 'normal',
        'size': 18,
    }
    A = read_gml(path)
    Z = cluster(A, method, metric)
    Q = computeQ(A, Z)
    print(max(Q))
    X = list(range(1, A.shape[0] + 1))
#     X.reverse()
#     Q.reverse()
    print(Q)
    plt.figure()
    plt.plot(X, Q)
    plt.title('{} {}'.format(metric, method), font1)
    
    plt.xlabel('合并次数', font) 
    plt.ylabel('modularity Q', font)
    
    plt.savefig('Q_{}_{}_{: 0.5f}_{: 0.5f}.svg'.format(
        metric, method, max(Q), Q[-2]))


# In[442]:


file_path = 'karate.gml'
# euclidean
for metric in ['euclidean', 'cosine', 'jaccard', 'hamming']:
    for method in ["single", "complete", "average", "weighted"]:
        main('karate.gml', method=method, metric=metric)

