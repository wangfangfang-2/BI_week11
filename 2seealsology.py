# 使用Graph Embedding中的DeepWalk对space_data进行压缩
import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
from ge.models import DeepWalk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
#tsv文件提取区别在 \t
df = pd.read_csv('./seealsology-data.tsv',sep = '\t')
# df.head()
G=nx.from_pandas_edgelist(df,'source','target',edge_attr=True)
# 数据加载，构造图
#G = nx.read_gml('football.gml')

# 初始化Node2Vec模型
model = DeepWalk(G, walk_length=10, num_walks=5, workers=1)
# 模型训练
model.train(window_size=4, iter=20)
# 得到节点的embedding
embeddings = model.get_embeddings()
#print(embeddings)
#print(embeddings.shape)
#print(type(embeddings))
#print(embeddings['lunar escape systems'])

# 在二维空间中绘制所选节点的向量
def plot_nodes(word_list):
    # 每个节点的embedding为100维
    X = []
    for item in word_list:
        X.append(embeddings[item])

    #print(X.shape)
    # 将100维向量减少到2维
    pca = PCA(n_components=2)
    result = pca.fit_transform(X) 
    #print(result)
    # 绘制节点向量
    plt.figure(figsize=(12,9))
    # 创建一个散点图的投影
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(list(word_list)):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))        
    plt.show()

plot_nodes(model.w2v_model.wv.vocab)
#s=model.similar_by_word('critical illness insurance')
#print(model.similar_by_word('critical illness insurance'))


""" 
随机游走
input: 将节点和被遍历的路径的长度作为输入
output: 返回遍历节点的顺序:
"""
def get_randomwalk(node, path_length):
    random_walk = [node]
    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break
        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node        
    return random_walk

#print(get_randomwalk('EastCarolina', 10))
# 从图获取所有节点的列表
all_nodes = list(G.nodes())
# 捕获数据集中所有节点的随机游走序列
random_walks = []
for n in tqdm(all_nodes):
	# 每个节点游走5次，每次最长距离为10
    for i in range(5):
        random_walks.append(get_randomwalk(n,10))
        

# 输出随机游走序列，及序列个数
#print(random_walks)
#print(len(random_walks))

# 使用skip-gram，提取模型学习到的权重
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# 训练skip-gram (word2vec)模型
model = Word2Vec(window = 4, sg = 1, hs = 0,
                 negative = 10, # 负采样
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14)
# 从random_walks中创建词汇表
model.build_vocab(random_walks, progress_per=2)
model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)
print(model)
# 输出和critical illness insurance相似
print(model.similar_by_word('critical illness insurance'))


# # 在二维空间中绘制所选节点的向量
# def plot_nodes(word_list):
# 	# 每个节点的embedding为100维
#     X = model[word_list]
#     #print(type(X))
#     # 将100维向量减少到2维
#     pca = PCA(n_components=2)
#     result = pca.fit_transform(X) 
#     #print(result)
#     # 绘制节点向量
#     plt.figure(figsize=(12,9))
#     # 创建一个散点图的投影
#     plt.scatter(result[:, 0], result[:, 1])
#     for i, word in enumerate(word_list):
#         plt.annotate(word, xy=(result[i, 0], result[i, 1]))
#     plt.show()

# plot_nodes(model.wv.vocab)

