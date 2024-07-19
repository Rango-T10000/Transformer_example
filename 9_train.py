from model_6 import Transformer
from torch.utils.data import DataLoader
from data_8 import MyDataset
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import torch


# my_model = Transformer()

source_path = "/home2/wzc/d2l_learn/d2l-zh/learning_code3/dataset/source.txt"
target_path = "/home2/wzc/d2l_learn/d2l-zh/learning_code3/dataset/target.txt"
# dataset = MyDataset(source_path, target_path)

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
# for input_id, input_m, output_id, output_m in dataloader:
#     my_model(input_id, input_m, output_id[:,:-1], output_m[:,:-1])
#     pass

my_model = Transformer().cuda()
dataset = MyDataset(source_path, target_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
loss_func = nn.CrossEntropyLoss(ignore_index=2)  #计算loss的时候不计算“PAD”
trainer = AdamW(params=my_model.parameters(), lr=0.0005) #参数优化器
for epoch in range(200):
    t = tqdm(dataloader) #显示进度条
    for input_id, input_m, output_id, output_m in t:
        
        output = my_model(input_id.cuda(), input_m.cuda(), output_id[:, :-1].cuda(), output_m[:, :-1].cuda())
        target = output_id[:, 1:].cuda() #这个就是label
        loss = loss_func(output.reshape(-1, 29), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), 1) #梯度裁剪
        trainer.step()
        trainer.zero_grad()
        # print(loss.item())
        print("Loss:")
        t.set_description(str(loss.item()))

torch.save(my_model.state_dict(), "/home2/wzc/d2l_learn/d2l-zh/learning_code3/model_pth/model.pth")