"""config"""
batch_size = 32    # 一次训练所选取的样本数
# lr = 1e-3             # 学习率
# n_epoch = 50          # 训练次数
dropout = 0.5   
d_embedding = 1024 # ProtT5 dim
d_model = 256   # 词向量维度
n_class = 2    # 输出
vocab_size = 21   # 词典大小
nlayers = 4   # transformer encoder layer
nhead = 2    # transformer encoder head
dim_feedforward  =1024 # transformer encoder feedforward
kmers  = [5]
