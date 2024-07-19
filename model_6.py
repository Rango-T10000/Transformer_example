import torch
from torch import nn

#这个其实就是把每个token做词嵌入生成的向量维度
num_hiddens = 256

#定义embedding+PE层
class EBD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(EBD, self).__init__(*args, **kwargs)

        self.word_ebd = nn.Embedding(29,num_hiddens)
        self.pos_ebd = nn.Embedding(12,num_hiddens)
        self.pos_t = torch.arange(0,12).reshape(1,12)

    # X: (batch_size, length),这就是输入，即源
    def forward(self, X:torch.tensor):
        return self.word_ebd(X) + self.pos_ebd(self.pos_t[:,:X.shape[-1]].to(X.device))

#attention: 对某个位置来说其他位置对其的影响的定量表示
#定义一个函数用Q，K，V计算注意力O,这是计算注意力的固定公式
def attention(Q, K, V, M:torch.Tensor): 
    A = Q @ K.transpose(-1,-2) / (Q.shape[-1] ** 0.5) #Q乘以K的转置，再除以根号d，d是Q形状的最后一个维度，这里就是num_hiddens
    M = M.unsqueeze(1)
    A.masked_fill_(M==0,-torch.tensor(float('inf')))
    A = torch.softmax(A, dim = -1)
    O = A @ V
    return O

#multi_head多头注意力
#定义一个调整q,k,v矩阵维度的函数
def transpose_qkv(QKV: torch.tensor):
    QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], 4, QKV.shape[-1]//4) #把Q(2,12,num_hiddens)变为Q(2,12,4,6)
    QKV = QKV.transpose(-2,-3)                       #把Q(2,12,4,6)变为Q(2,4,12,6)交换维度就是转置
    return QKV

#定义一个调整O矩阵维度的函数
def transpose_o(O):
    O = O.transpose(-2,-3) 
    O = O.reshape(O.shape[0],O.shape[1],num_hiddens) #同O = O.reshape(O.shape[0],O.shape[1],-1)
    return O

#定义一个多头attention模块计算层
class Multi_head_self_Attention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Multi_head_self_Attention_block, self).__init__(*args, **kwargs)
        #下面就是attention计算要经过的层，不用纠结为啥
        self.Wq = nn.Linear(num_hiddens,num_hiddens, bias = False) 
        self.Wk = nn.Linear(num_hiddens,num_hiddens, bias = False) 
        self.Wv = nn.Linear(num_hiddens,num_hiddens, bias = False) 
        self.Wo = nn.Linear(num_hiddens,num_hiddens, bias = False) 
    
    def forward(self, X, M:torch.Tensor): #M可能是I_m或者是O_m
        #下面就是attention计算要经过的标准流程，不用纠结为啥
        Q, K, V = self.Wq(X), self.Wk(X) ,self.Wv(X)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, M)
        O = transpose_o(O)
        O = self.Wo(O)
        return O
    
#加与规范化Add & Norm为了防止过拟合，
class AddNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(AddNorm, self).__init__(*args, **kwargs)
        self.add_norm = nn.LayerNorm(num_hiddens)    #这里的Norm是层归一化，不是batch_size归一化
        self.dropout = nn.Dropout(0.1)      #图里面没画，为了防止过拟合


    def forward(self, X, X1): #这个AddNorm包括：原本的输入X(经过ebd+PE); 多头注意力的输出O
        X1 = self.add_norm(X1)
        X = X + X1
        X = self.dropout(X)
        return X
    
#逐位前馈网络Feed & forward
class Pos_FFN(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Pos_FFN, self).__init__(*args, **kwargs)
        self.lin1 = nn.Linear(num_hiddens, 1024, bias = False)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(1024, num_hiddens, bias = False)
        self.relu2 = nn.ReLU()       

    def forward(self, X):
        X = self.lin1(X)
        X = self.relu1(X)
        X = self.lin2(X)
        X = self.relu2(X)        
        return X

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#定义一个Encoder_block
class Encoder_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder_block, self).__init__(*args, **kwargs)
        self.attention = Multi_head_self_Attention_block()
        self.add_norm_1 = AddNorm()
        self.FFN = Pos_FFN()
        self.add_norm_2 = AddNorm()

    def forward(self, X, I_m):
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X, I_m)
        X = self.add_norm_1(X, X_1)
        X_2 = self.FFN(X)
        X = self.add_norm_2(X, X_2) 
        return X

#Encoder
class Encoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        self.ebd = EBD()
        self.encoder_blks = nn.Sequential()
        self.encoder_blks.append(Encoder_block())
        self.encoder_blks.append(Encoder_block())

        #还有种写法：
        # self.encoder_blk1 = Encoder_block()
        # self.encoder_blk2 = Encoder_block()
    def forward(self, X, I_m):
        X = self.ebd(X)
        for encoder_blk in self.encoder_blks:
            X = encoder_blk(X, I_m)

        return X


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#定义一个多头attention模块计算层,(这里是cross attention, 即多头互注意力MACA,因为K,V输入都是来自Encoder的输出)
class CrossAttention_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(CrossAttention_block, self).__init__(*args, **kwargs)
        self.Wq = nn.Linear(num_hiddens,num_hiddens, bias = False) 
        self.Wk = nn.Linear(num_hiddens,num_hiddens, bias = False) 
        self.Wv = nn.Linear(num_hiddens,num_hiddens, bias = False) 
        self.Wo = nn.Linear(num_hiddens,num_hiddens, bias = False) 
    
    def forward(self, X, X_en, I_m):
        Q, K, V = self.Wq(X), self.Wk(X_en) ,self.Wv(X_en)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, I_m)
        O = transpose_o(O)
        O = self.Wo(O)
        return O
    
#定义一个Decoder_block
class Decoder_block(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder_block, self).__init__(*args, **kwargs)
        self.attention = Multi_head_self_Attention_block()
        self.add_norm_1 = AddNorm()
        self.cross_attention = CrossAttention_block()
        self.add_norm_2 = AddNorm()
        self.FFN = Pos_FFN()
        self.add_norm_3 = AddNorm()
        mask_matrix = torch.ones(12,12)
        self.tril_mask = torch.tril(mask_matrix).unsqueeze(0) #这里创建的就是一个下三角矩阵，用来做mask

    def forward(self, X_t, O_m, X_en, I_m):
        O_m = O_m.unsqueeze(-2)
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X_t, O_m * self.tril_mask[:,:O_m.shape[-1],:O_m.shape[-1]].to(X_t.device))  # *是元素级别相乘，会根据广播机制自动拓展维度
        X_t = self.add_norm_1(X_t, X_1)
        X_2 = self.cross_attention(X_t, X_en, I_m)
        X_t = self.add_norm_2(X_t, X_2)
        X_3 = self.FFN(X_t)
        X_t = self.add_norm_3(X_t, X_3)
        return X_t


#Decoder
class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        self.ebd = EBD()
        self.decoder_blks = nn.Sequential()
        self.decoder_blks.append(Decoder_block())
        self.decoder_blks.append(Decoder_block())
        self.dense = nn.Linear(num_hiddens, 29, bias = False)  #过完所有的Decoder_blk，最终还要经过一个MLP

    def forward(self, X_t, O_m, X_en, I_m):
        X_t = self.ebd(X_t)
        for decoder_blk in self.decoder_blks:
            X_t = decoder_blk(X_t, O_m, X_en, I_m)
        X_t = self.dense(X_t)
        return X_t
    

#构建transformer模型
class Transformer(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Transformer, self).__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder()

    #源的长度是固定的X_s(Encoder的输入)， 目标的长度是可变的X_t(Decoder的输入)
    #I_m表示输入的掩码，O_m表示输出的掩码
    def forward(self, X_s, I_m, X_t, O_m): 
        X_en = self.encoder(X_s, I_m)
        X  = self.decoder(X_t, O_m, X_en, I_m)
        return X

#测试一下
if __name__ == "__main__":
    a = torch.ones((2,12)).long() #输入是2个数据，每个长度12
    b = torch.ones((2,4)).long()
    model = Transformer()
    output = model(a,b)
    # print(output)
    # print(output.shape)
    pass
    

    