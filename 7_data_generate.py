import random


#26个小写字母+3 = 共29个elements（词汇表中的元素，这些去组成一个句子/词）
vocab_lst = ["[BOS]","[EOS]","[PAD]",'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
char_lst = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
             'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
bos_token = "[BOS]"
eos_token = "[EOS]"
pas_token = "[PAD]"

#任务：
#对于一个英语字符串，输出加密后的字符串
#加密规则是：对于每一个字符，使其ascii码值循环-5，然后将整个字符串逆序
#例子：abcfg -> vwxab ->baxwv
#让transformer来寻找这个规律

#自己生成数据
source_path = "/home2/wzc/d2l_learn/d2l-zh/learning_code3/dataset/source.txt"
target_path = "/home2/wzc/d2l_learn/d2l-zh/learning_code3/dataset/target.txt"
with open(source_path, "w") as f:
    pass
with open(target_path, "w") as f:
    pass

for _ in range(10000):
    source_str = ""
    target_str = ""
    for idx in range(random.randint(3,10)): #生成的字符串长度为3～13随机的一个值
        i = random.randint(0,25)
        source_str += char_lst[i]
        target_str += char_lst[(i+26 - 5)%26]
    target_str = target_str[::-1] #逆序

    with open(source_path, 'a') as f:
        f.write(source_str + '\n')
    with open(target_path, 'a') as f:
        f.write(target_str + '\n')



