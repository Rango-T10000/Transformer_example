import torch
from torch import nn

#定义embedding+PE层
class EBD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EBD, self).__init__(*args, **kwargs)

        self.word_ebd = nn.Embedding(28,24)
        self.pos_ebd = nn.Embedding(12,24)
        self.pos_t = torch.arange(0,12).reshape(1,12)

    # X: (batch_size, length),这就是输入，即源
    def forward(self, X:torch.tensor):
        return self.word_ebd(X) + self.pos_ebd(self.pos_t)

#attention: 对某个位置来说其他位置对其的影响的定量表示
#定义一个函数用Q，K，V计算注意力O,这是计算注意力的固定公式
def attention(Q, K, V):
    A = Q @ K.transpose(-1,-2) / (Q.shape[-1] ** 0.5) #Q乘以K的转置，再除以根号d，d是Q形状的最后一个维度，这里就是24
    A = torch.softmax(A, dim = -1)
    O = A @ V
    return O

#定义一个attention模块计算层
class Single_head_Attention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Single_head_Attention_block, self).__init__(*args, **kwargs)
        #下面就是attention计算要经过的层，不用纠结为啥
        self.Wq = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
        self.Wk = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
        self.Wv = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
        self.Wo = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
    
    def forward(self, X):
        #下面就是attention计算要经过的标准流程，不用纠结为啥
        Q, K, V = self.Wq(X), self.Wk(X) ,self.Wv(X)
        O = attention(Q, K, V)
        O = self.Wo(O)
        return O


#multi_head多头注意力
#定义一个调整q,k,v矩阵维度的函数
def transpose_qkv(QKV: torch.tensor):
    QKV = QKV.reshape(QKV.shape[0],QKV.shape[1],4,6) #把Q(2,12,24)变为Q(2,12,4,6)
    QKV = QKV.transpose(-2,-3)                       #把Q(2,12,4,6)变为Q(2,4,12,6)交换维度就是转置
    return QKV

#定义一个调整O矩阵维度的函数
def transpose_o(O):
    O = O.transpose(-2,-3) 
    O = O.reshape(O.shape[0],O.shape[1],24) #同O = O.reshape(O.shape[0],O.shape[1],-1)
    return O

#定义一个多头attention模块计算层
class Multi_head_Attention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Multi_head_Attention_block, self).__init__(*args, **kwargs)
        #下面就是attention计算要经过的层，不用纠结为啥
        self.Wq = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
        self.Wk = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
        self.Wv = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
        self.Wo = nn.Linear(24,24, bias = False) #乘以矩阵就是可以看作是让输入X经过没有偏置的线性层
    
    def forward(self, X):
        #下面就是attention计算要经过的标准流程，不用纠结为啥
        Q, K, V = self.Wq(X), self.Wk(X) ,self.Wv(X)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V)
        O = transpose_o(O)
        O = self.Wo(O)
        return O
    
#加与规范化Add & Norm为了防止过拟合，
class AddNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AddNorm, self).__init__(*args, **kwargs)
        self.add_norm = nn.LayerNorm(24)    #这里的Norm是层归一化，不是batch_size归一化


    def forward(self, X, X1): #这个AddNorm包括：原本的输入X(经过ebd+PE); 多头注意力的输出O
        X = X + X1
        X = self.add_norm(X)
        return X
    
#逐位前馈网络Feed & forward
class Pos_FFN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Pos_FFN, self).__init__(*args, **kwargs)
        self.lin1 = nn.Linear(24,48)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(48,24)
        self.relu2 = nn.ReLU()       

    def forward(self, X):
        X = self.lin1(X)
        X = self.relu1(X)
        X = self.lin2(X)
        X = self.relu2(X)        
        return X
           
#测试一下
if __name__ == "__main__":
    a = torch.ones((2,12)).long() #输入是2个数据，每个长度12
    ebd = EBD()
    b = ebd(a) #经过我们的ebd层，每个数据被编码为长度为24的向量，即隐藏层长度/数量
    print(b.shape)

    atten_en = Multi_head_Attention_block() #经过这个多头注意力计算
    c = atten_en(b)
    print(c.shape)

    addnorm = AddNorm()
    d = addnorm(b, c)
    print(d.shape)

    posfnn = Pos_FFN()
    e = posfnn(d)
    print(e.shape)

    pass