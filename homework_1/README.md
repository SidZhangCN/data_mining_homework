# 数据挖掘课程 实验报告

## 实验1 基于sklearn的聚类测试

##### 张旭东 计算机技术 201934750

### 实验要求
使用python + sklearn在load_digits和fetch_20newsgroups两个数据集上的聚类效果。
使用的聚类算法如下：

| Method name | Parameters | Scalability | Usecase | Geometry (metric used) |
| :- | :-: | :-: | :-: | :-  |
| K-Means | number of clusters | Very large n_samples, medium n_clusters with MiniBatch code | General-purpose, even cluster size, flat geometry, not too many clusters | Distances between points |
| Affinity propagation | damping, sample preference | Not scalable with n_samples | Many clusters, uneven cluster size, non-flat geometry | Graph distance (e.g. nearest-neighbor graph) |
| Mean-shift | 	bandwidth | Not scalable with n_samples | Many clusters, uneven cluster size, non-flat geometry | Distances between points |
| Spectral clustering | number of clusters | Medium n_samples, small n_clusters | Few clusters, even cluster size, non-flat geometry | Graph distance (e.g. nearest-neighbor graph) |
| Ward hierarchical clustering | number of clusters or distance threshold | Large n_samples and n_clusters | Many clusters, possibly connectivity constraints | Distances between points |
| Agglomerative clustering | number of clusters or distance threshold, linkage type, distance | Large n_samples and n_clusters | Many clusters, possibly connectivity constraints, non Euclidean distances | Any pairwise distance |
| DBSCAN | neighborhood size | Very large n_samples, medium n_clusters | Non-flat geometry, uneven cluster sizes | Distances between nearest points |
| Gaussian mixtures | many | Not scalable | Flat geometry, good for density estimation | Mahalanobis distances to centers | 

### 实验环境
硬件环境： Intel I5-3410M， DDR3 1600 8G
软件环境： Ubuntu 18.04， Python 3.6.8， scikit-learn 0.21.3， numpy 1.17.2

### 实验数据
- digital_loads：
    sklearn自带的手写数据集，1797个样本，每个样本包括8*8像素的图像和一个[0, 9]整数的标签
- fetch_20newsgroups：
    用于文本分类、文本挖据和信息检索研究的国际标准数据集之一，数据集收集了大约20,000左右的新闻组文档，均匀分为20个不同主题的新闻组集合。

### 实验测试指标
- Time：
    数据集聚类所用的时间
- Homogeneity：
    同质性，每个群集只包含单个类的成员
- Completeness：
    完整性，给定类的所有成员都分配给同一个群集
- Normalized Mutual Information (NMI) ：
    衡量测试结果（跑出的结果）和标准结果有多大区别

### 实验过程
- digital_loads下的聚类评估：
    1. 实例化数据集；
    2. 数据集标准化；
    3. 使用各种聚类方法后输出评估参数。
- fetch_20newsgroups下的聚类评估：
    1. 事先选定好新闻类别，这里选的是20个分类里的前4个，并实例化；
    2. 计算Tf-idf；
    3. 截断SVD和LSA降维；
    4. 使用各种聚类方法后输出评估参数。

### 实验结果
8种聚类方法在digital_loads数据集下的评估结果如下表：

|init	|						time|	homo|	compl|	NMI|
| ----- | ----                      | ----  | ----  | ---- |
|k-means++|                   	0.25s|	0.602|	0.650|	0.625|	
|Affinity propagation|        	3.76s|	0.964|	0.425|	0.590|	
|Mean-shift|                  	0.17s|	0.009|	0.263|	0.018|	
|Spectral Clustering|         	0.57s|	0.773|	0.878|	0.822|	
|Ward Hierarchical Clustering|	0.16s|	0.827|	0.765|	0.795|	
|Agglomerative clustering|    	0.08s|	0.863|	0.497|	0.630|	
|DBSCAN|                      	0.36s|	0.887|	0.477|	0.621|	
|Gaussian mixture|              0.00s|	0.862|	0.541|	0.665|

8种聚类方法在fetch_20newsgroups数据集下的评估结果如下表：

|init	|						time|	homo|	compl|	NMI|
| ----- | ----                      | ----  | ----  | ---- |
|k-means++|                   	0.05s|	0.363|	0.442|	0.398|	
|Affinity propagation|        	24.24s|	0.642|	0.180|	0.282|	
|Mean-shift|                  	0.19s|	0.000|	1.000|	0.000|	
|Spectral Clustering|         	14.80s|	0.052|	0.277|	0.088|	
|Ward Hierarchical Clustering|	1.28s|	0.482|	0.174|	0.256|	
|Agglomerative clustering|    	0.74s|	0.144|	0.175|	0.158|	
|DBSCAN|                      	0.20s|	0.002|	0.211|	0.005|	
|Gaussian mixture|              0.00s|	0.437|	0.185|	0.260|

### 实验结果分析
- digital_loads下的结果分析：
    从实验结果来看，效果较好的是谱聚类（Spectral Clustering）方法和层次聚类（Hierarchical Clustering）方法。
    耗费时间上来看，因为数据集较简单，所以这几种方法耗费的时间都不多。
- fetch_20newsgroups下的结果分析：
    因为数据集较大，维度较多，多以在降维的情况下，仍然比较耗时，有的聚类方法直接卡死机了。
    从测试指标来看，效果都不怎么好，其中较好的是k-means方法。
