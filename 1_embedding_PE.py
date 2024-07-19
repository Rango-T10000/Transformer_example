import torch
from torch import nn

#源（input）
#当然输入是Token序列，是类似['a','b','c',.........]中选一些的组合
#比方我给这个['a','b','c',.........]按顺序标成id,用[0,1,2,........]代替
#则我输入的token序列['a','b','c','d']就可以表示为[0,1,2,3]
a = torch.tensor([[1,2,3,4],[2,3,3,4]]) #表示输入有2个batch，每个序列长度为4

#词嵌入(也是一个层，对输入做计算，就是把输入的每一个编码成一个向量)
ebd1 = nn.Embedding(5,24) #定义词汇表长度是5，每个词汇编码为长度24的向量
b1 = ebd1(a)
print(b1)
print(b1.shape)


#位置编码PE：给输入中的每一个元素加上位置信息
#自己设计一个最简单的就是给每个元素编个号index
pos = torch.tensor([[0,1,2,3]])
ebd2 = nn.Embedding(4,24)
b2 = ebd2(pos)  #对位置编码也做一下词嵌入
print(b2.shape)

# #定义一个自己的模型
# class model(nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)

#         #下面开始写自己的内容，其他的都是固定框架
#         #xxXXXXXXXXX

#     def forward(self):
#         #xxXXXXXXXXX
#         pass

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

#测试一下
if __name__ == "__main__":
    a = torch.ones((2,12)).long() #输入是2个数据，每个长度12
    ebd = EBD()
    b = ebd(a) #经过我们的ebd层，每个数据被编码为长度为24的向量
    print(b.shape)
    pass
    


