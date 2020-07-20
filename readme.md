# [1. 分类技术 二分网络上的链路预测](https://github.com/Linan2018/data_mining/tree/master/hw1)

1. 采用二分网络模型，对`ml-1m`文件夹中的“用户——电影”打分数据进行建模，考虑将用户信息、电影详细信息、以及打分分值作为该网络上的边、点的权重；

2. 根据网络结构特征给出节点相似性度量指标；

3. 基于相似性在二分网络上进行链路预测；

4. 采用交叉验证的方法验证预测结果；

5. 画出 ROC 曲线来度量预测方法的准确性。

# [2. 聚类技术 复杂网络社团检测](https://github.com/Linan2018/data_mining/tree/master/hw2)

1. 导入`karate.gml`中的空手道网络数据；

2. 根据网络结构特征给出节点相似性度量指标；

3. 采用层次聚类过程对网络数据进行聚类；

4. 计算模块性指标 Q 值，当 Q 值最大时输出聚类结果；

5. 可视化聚类结果。

# [3. 关联规则挖掘 美国国会投票记录](https://github.com/Linan2018/data_mining/tree/master/hw3)

1. [数据来源](http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)

2. 使用 Apriori算法，支持度设为 30%，置信度为 90%，挖掘高置信度的规则
