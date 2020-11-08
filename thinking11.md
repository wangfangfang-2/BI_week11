#Thinking1	什么是Graph Embedding，都有哪些算法模型Graph 
Embedding 是深度学习中的“基本操作”，是一种Embedding降维技术，可以有效的挖掘图网络中的节点特征表示，在推荐系统、NLP/CV，搜索排序，计算广告领域中的热点模型。Embedding 指的是用一个低维度向量表示一个实体，可以是一个词（Word2Vec），可以是一个物品（Item2Vec），也可以是网络关系中的节点（Graph Embedding）word2vec等Embedding技术的延伸，作用：帮助我们从图网络中进行特征提取，将图网络中的点用低维的向量表示，并且这些向量要能反应原有网络的特性，比如原网络中两个点的结构类似，那么这两个点表示的向量也应该类似。
主要方法：factorization methods （图因式分解机）；random walk techniques（随机游走）、Deep Walk算法：DeepWalk = Random Walk + Skip-gram；deep learning（深度学习）Node2Vec在工业界很成功：Facebook：广告领域定制化受众；Tencent：微信朋友圈广告(Lookalike)策略；Lookalike，相似人群扩展；DMP是Look-alike技术的核心基础。
#Thinking2	如何使用Graph Embedding在推荐系统，比如NetFlix 电影推荐，请说明简要的思路	
基于人的行为item学出来做embedding，再做后续的推荐，通过session做切分后，构成一张Graph，在做后续的embedding计算，放到推荐系统里完成推荐。		
（1）在深度学习网络中作为Embedding层，完成从高维稀疏特征向量到低维稠密特征向量的转换（比如Wide&Deep、DIN等模型）。推荐场景中大量使用One-hot编码对类别、id型特征进行编码，导致样本特征向量极度稀疏，而深度学习的结构特点使其不利于稀疏特征向量的处理，因此几乎所有的深度学习推荐模型都会由Embedding层负责将高维稀疏特征向量转换成稠密低维特征向量。因此，掌握各类Embedding技术是构建深度学习推荐模型的基础性操作。
（2）作为预训练的Embedding特征向量，与其他特征向量连接后，一同输入深度学习网络进行训练（比如FNN模型）。Embedding本身就是极其重要的特征向量。相比MF等传统方法产生的特征向量，Embedding的表达能力更强，特别是Graph Embedding技术被提出后，Embedding几乎可以引入任何信息进行编码，使其本身就包含大量有价值的信息。在此基础上，Embedding向量往往会与其他推荐系统特征连接后一同输入后续深度学习网络进行训练。
（3）通过计算用户和物品的Embedding相似度，Embedding可以直接作为推荐系统的召回层或者召回策略之一（比如Youtube推荐模型等）。Embedding对物品、用户相似度的计算是常用的推荐系统召回层技术。在局部敏感哈希（Locality-Sensitive Hashing）等快速最近邻搜索技术应用于推荐系统后，Embedding更适用于对海量备选物品进行快速“筛选”，过滤出几百到几千量级的物品交由深度学习网络进行“精排”。
（4）通过计算用户和物品的Embedding，将其作为实时特征输入到推荐或者搜索模型中（比如Airbnb的embedding应用）。	
#Thinking3	数据探索EDA都有哪些常用的方法和工具		
整体情况探索:
df.head()/df.tail();	df.info();df.describe();value_counts()//查看某列的不同数据分布；pandas.profiling/一行代码生成报告；
缺失值处理:df.isnull().any() ;df.isnull().sum();msno.matrix(sample);msno.bar(sample);msno.heatmap(sample);
数据分布：计算偏度和峰度（与正态分布进行比较）;常用数据分布：johnsonsu;norm;lognorm;
