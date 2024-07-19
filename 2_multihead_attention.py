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

#attention: 对某个位置来说其他位置对其的影响的定量表示？
#比方上边输入是2个数据，每个长度12
#对每个数据来说，共含有12个元素，用A来表示每个位置的元素收到其他位置元素以及自己本身的影响，所以A应该是12x12的矩阵
#Query，key, value
#Q, K, V三个矩阵就是用输入乘以矩阵Wq,Wk,Wv算出来的，乘以的这些矩阵不用纠结怎么来的，都是可学习的参数，初始化都是随机定义的
#最终的Attention计算的输出还要乘以矩阵O
#最终的Attention计算的输出就是用上边这些算出来的

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
#因为单头只得到一个A矩阵，信息损失太多
#原本单头的Q(12,24)，K(12,24)，V(12,24)
#拆成Q(12,4,6)，K(12,4,6)，V(12,4,6)，这样最终可以计算出4个A,这就是4头注意力,  注意这里都省略了说明第一个维度batch_size，实际是Q(2,12,4,6)
#为了最终A还是12x12,进行维度转换：Q(4,12,6)，K(4,12,6)，V(4,12,6)
#4是head数，12是每个数据的长度，6是隐藏层长度/数量
#这样计算完O是(4,12,6),注意这里都省略了说明第一个维度batch_size，实际是O(2,4,12,6)
#最后还是要通过交换维度，合并维度等为了让输出满足注意力计算得到的结果的形状应和本来数据的维度"相同",即把O从(4,12,6)->(12,4,6)->(12,24)

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

#定义一个多头attention模块计算层(这里也是self attention, 即多头自注意力MASA,因为3个输入都是来自原本的输入)
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

#测试一下
if __name__ == "__main__":
    a = torch.ones((2,12)).long() #输入是2个数据，每个长度12
    ebd = EBD()
    b = ebd(a) #经过我们的ebd层，每个数据被编码为长度为24的向量，即隐藏层长度/数量
    print(b.shape)

    atten_en1 = Single_head_Attention_block() #经过这个单头注意力计算
    c1 = atten_en1(b)
    print(c1.shape)    #最终经过EBD,单头注意力计算得到的结果的形状应和本来数据的维度"相同"，原来是(2,12),计算完是(2,12,24)

    atten_en2 = Multi_head_Attention_block() #经过这个多头注意力计算
    c2 = atten_en2(b)
    print(c2.shape)

    pass





