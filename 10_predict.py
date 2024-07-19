from model_6 import Transformer
from torch.utils.data import DataLoader
from data_8 import MyDataset
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import torch
from data_8 import process_data

vocab_lst = ["[BOS]", "[EOS]", "[PAD]", 'a', 'b', 'c', 'd','e', 'f', 'g', 'h','i', 'j', 'k', 'l','m', 'n', 
 'o', 'p','q', 'r', 's', 't','u', 'v', 'w', 'x','y', 'z']

def predict(model:Transformer, sentence:str):
    max_len = 12
    device = torch.device("cuda:0")
    source_id, source_m, _, _  = process_data(sentence, "aaaaa") #对输入的字符串提前处理
    target_id_lst = [vocab_lst.index("[BOS]")]  #初始只有"[BOS]"
    target_m_lst = [1]  #现在只有"[BOS]"，就全是有效的，所以就是一个1
    source_id = torch.tensor(source_id).to(device).unsqueeze(0)
    source_m = torch.tensor(source_m).to(device).unsqueeze(0)
    for _ in range(max_len):
        target_id = torch.tensor(target_id_lst).to(device).unsqueeze(0)
        target_m = torch.tensor(target_m_lst).to(device).unsqueeze(0)
        output = model(source_id, source_m, target_id, target_m)
        #第一次output是(1,1,29),每次只看第二个维度的最后一个，这才是新的，29中最大的值对应的id就是预测的词的id
        word_id = torch.argmax(output[0][-1])  
        target_id_lst.append(word_id.item())
        target_m_lst.append(1)
        if word_id == vocab_lst.index("[EOS]"):
            break
    result = ""
    for id in target_id_lst:
        result += vocab_lst[id]
    return result

if __name__ == "__main__":
    my_model = Transformer().cuda()
    my_model.load_state_dict(torch.load("/home2/wzc/d2l_learn/d2l-zh/learning_code3/model_pth/model.pth"))
    print(predict(my_model, "abcdif"))