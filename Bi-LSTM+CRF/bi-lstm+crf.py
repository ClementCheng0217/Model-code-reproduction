import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from TorchCRF import CRF
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import torch
import matplotlib.pyplot as plt
from torchtext.vocab import Vectors
import sys
import time

#LSTM_CRF模型
class LSTM_CRF(nn.Module):
    # LSTM_CRF(LSTM_CRF, vocab_size, tag2idx, embedding_size, hidden_size, max_length=train_max_length, vectors=None)
    def __init__(self, vocab_size, tag_to_index, embedding_size, hidden_size, max_length, vectors=None):
        super(LSTM_CRF, self).__init__()
        self.embedding_size = embedding_size #特征向量大小
        self.hidden_size = hidden_size #隐藏层数
        self.vocab_size = vocab_size #单词集长度
        self.tag_to_index = tag_to_index #实体类型集
        self.target_size = len(tag_to_index) #实体类型个数
        if vectors is None:#自己训练
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(vectors)
        self.lstm = nn.LSTM(embedding_size, hidden_size//2, bidirectional=True)#bidirectional:lstm是否为双向：论文设置为false
        self.hidden_to_tag = nn.Linear(hidden_size, self.target_size)#设置网络中的全连接层：用隐藏层全连接
        self.crf = CRF(self.target_size,batch_first=True)#batch_first为True表示batch是第一个参数而不是第二个参数
        self.max_length = max_length

    def get_mask(self, length_list):
        mask = []
        for length in length_list:
            mask.append([1 for i in range(length)] + [0 for j in range(self.max_length - length)])
        return torch.tensor(mask, dtype=torch.bool, device='cuda')

    def LSTM_Layer(self, sentences, length_list):

        embeds = self.embedding(sentences)
        length_list=length_list.cpu()
        packed_sentences = pack_padded_sequence(embeds, lengths=length_list, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_sentences)

        result, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.max_length)

        feature = self.hidden_to_tag(result)
        return feature

    def CRF_layer(self, input, targets, length_list):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        return self.crf(input, targets, self.get_mask(length_list))

    def forward(self, sentences, length_list, targets):
        x = self.LSTM_Layer(sentences, length_list)
        x = self.CRF_layer(x, targets, length_list)

        return x

    def predict(self, sentences, length_list):
        out = self.LSTM_Layer(sentences, length_list)
        mask = self.get_mask(length_list)
        return self.crf.decode(out, mask)


#数据处理相关
def read_data(path):#提取conll2003单词与实体类型，整合成句子
    sentences_list = []         # 每一个元素是一整个句子
    sentences_list_labels = []  # 每个元素是一整个句子的标签
    with open(path, 'r', encoding='UTF-8') as f:
        sentence_labels = []    # 每个元素是这个句子的每个单词的标签
        sentence = []           # 每个元素是这个句子的每个单词

        for line in f:
            line = line.strip()
            if not line:        # 如果遇到了空白行
                if sentence:    # 防止空白行连续多个，导致出现空白的句子
                    sentences_list.append(' '.join(sentence))
                    sentences_list_labels.append(' '.join(sentence_labels))

                    sentence = []
                    sentence_labels = []
                                # 创建新的句子的list，准备读入下一个句子
            else:
                res = line.split()
                assert len(res) == 4#假定一行为一个res，且res有四个值
                if res[0] == '-DOCSTART-':
                    continue
                sentence.append(res[0])#单词
                sentence_labels.append(res[3])#短语类型

        if sentence:            # 防止最后一行没有空白行，导致最后一句话录入不到
            sentences_list.append(sentence)
            sentences_list_labels.append(sentence_labels)
    return sentences_list, sentences_list_labels

def build_vocab(sentences_list):#合并训练集和测试集，分别把单词和实体类型变成词典
    ret = []
    for sentences in sentences_list:
        ret += [word for word in sentences.split()]
    return list(set(ret))

class mydataset(Dataset):
    def __init__(self, x : torch.Tensor, y : torch.Tensor, length_list):
        self.x = x
        self.y = y
        self.length_list = length_list
    def __getitem__(self, index):
        #存每一个句子对应的单词下标集合，实体类型下标集合和长度
        data = self.x[index]
        labels = self.y[index]
        length = self.length_list[index]
        return data, labels, length
    def __len__(self):
        return len(self.x)

def get_idx(word, d):#查找单词在单词表中的位置下标
    if d[word] is not None:
        return d[word]
    else:
        return d['<unknown>']

def sentence2vector(sentence, d):#把具体句子中的单词转换成word2idx里面的位置下标
    #word是句子中一个个单词
    return [get_idx(word, d) for word in sentence.split()]

def padding(x, max_length, d):#填充句子，让所有句子相同长度
    length = 0
    for i in range(max_length - len(x)):
        x.append(d['<pad>'])
    return x

def get_dataloader(x, y, batch_size):#生成dataloader
    #此时word2idx和tag2idx是train的单词与实体类型数组
    #x=x_train,y=y_train
    word2idx, tag2idx, vocab_size = pre_processing()
    #s是一句一句的
    inputs = [sentence2vector(s, word2idx) for s in x]
    targets = [sentence2vector(s, tag2idx) for s in y]

    length_list = [len(sentence) for sentence in inputs]#统计各句子中单词个数

    # 在Conll2000和2003数据集中，由于训练集最大句子长度为113，测试集最大句子长度为124，所以直接设置max_length=124
    max_length = 124

    # max_length = 0
    # max_length = max(max(length_list), max_length)
    # print(max_length)
    # print(max(length_list))
    # sys.exit()
    # max_length = max(max(length_list), max_length)


    inputs = torch.tensor([padding(sentence, max_length, word2idx) for sentence in inputs])
    targets = torch.tensor([padding(sentence, max_length, tag2idx) for sentence in targets], dtype=torch.long)

    dataset = mydataset(inputs, targets, length_list)
    #shuffle为false：词法分析单词之间的位置不可以打乱
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)

    return dataloader, max_length

def pre_processing():#生成单词与实体类型数组（不含重复项）
    x_train, y_train = read_data("C:/Users/成子健/Desktop/CRF++-0.58/example/conll2003_ner/train.txt")
    x_test, y_test = read_data("C:/Users/成子健/Desktop/CRF++-0.58/example/conll2003_ner/test.txt")
    d_x = build_vocab(x_train+x_test)#单词集
    d_y = build_vocab(y_train+y_test)#实体类型集
    word2idx = {d_x[i]: i for i in range(len(d_x))}#给单词编号
    tag2idx = {d_y[i]: i for i in range(len(d_y))}#给实体类型编号
    #penn treebank
    # tag2idx["<START>"] = 39
    # tag2idx["<STOP>"] = 40
    # conll2000
    # tag2idx["<START>"] = 23
    # tag2idx["<STOP>"] = 24
    #conll2003
    tag2idx["<START>"] = 9
    tag2idx["<STOP>"] = 10
    pad_idx = len(word2idx)
    word2idx['<pad>'] = pad_idx
    tag2idx['<pad>'] = len(tag2idx)
    # tag2idx['<unknown>'] = len(tag2idx)
    vocab_size = len(word2idx)
    idx2tag = {value: key for key, value in tag2idx.items()}
    print(tag2idx)
    return word2idx, tag2idx, vocab_size

def compute_f1(pred, targets, length_list):#计算f1值
    tp, fn, fp, tn = [], [], [], []

    # #penn treebank
    # x = 45
    # y=41

    #conll2000
    # x=30
    # y=25

    # conll2003
    x = 15
    y = 9#为实体类型数组长度

    for i in range(x):
        tp.append(0)
        fn.append(0)
        fp.append(0)
        tn.append(0)
    for i, length in enumerate(length_list):
        for j in range(length):
            a, b = pred[i][j], targets[i][j]
            if (a == b):
                tp[a] += 1
                tn[a] += 1
            else:
                fp[a] += 1
                fn[b] += 1
    tps = 0
    fps = 0
    fns = 0
    tns = 0
    for i in range(y):
        tps += tp[i]
        fps += fp[i]
        fns += fn[i]
        tns += tn[i]
    precision = tps / (tps + fps)
    recall = tps / (tps + fns)
    accurary = (tps + tns) / (tps + tns + fns + fps)
    f1=2 * precision * recall / (precision + recall)
    return f1


#main:

batch_size = 100 #一次训练所抓取的数据样本数量：论文里设置为100
embedding_size = 50 #特征向量的大小:论文里设置为50
hidden_size = 300 #隐藏层：论文里设置为300
epochs = 10 #循环次数：论文里设置为10
totalnorm = []
clipcoef = []


def train(model, vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=None):
    model = model(vocab_size, tag2idx, embedding_size, hidden_size, max_length, vectors=vectors)#初始化模型
    if torch.cuda.is_available():
        model = model.cuda()  # model 在 GPU 上进行训练

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)#lr为学习率，论文里设置为0.1
    start_time = time.time()
    loss_history = []
    print("dataloader length: ", len(train_dataloader))#表示训练集按照batchsize分成了多少组

    #训练模式
    model.train()
    f1_history = []
    idx2tag = {value: key for key, value in tag2idx.items()}
    for epoch in range(epochs):#循环
        total_loss = 0.
        f1 = 0
        for idx, (inputs, targets, length_list) in enumerate(train_dataloader):#其中每次循环有batch_size个句子
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                length_list = length_list.cuda()

            model.zero_grad()#梯度初始化为0
            loss = (-1) * model(inputs, length_list, targets)
            total_loss += loss.item()
            pred = model.predict(inputs, length_list)#预测训练集的实体类型
            # print(inputs,targets,length_list)
            # print(loss)
            # print(pred)
            # sys.exit()
            f=compute_f1(pred, targets, length_list)
            f1 += f#计算f1
            loss.backward()#反向传播（计算梯度）
            max_norm = 0.5
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)#裁剪梯度：解决梯度爆炸问题
            clip_coef = max_norm/total_norm#用于max_norm调参
            totalnorm.append(total_norm)
            clipcoef.append(clip_coef)
            optimizer.step()#更新网络参数
            # if (idx + 1) % 10 == 0 and idx:#10组含有batch_size条句子为一组
            cur_loss = total_loss
            loss_history.append(cur_loss / (idx + 1))
            f1_history.append(f1 / (idx + 1))
            total_loss = 0
            print("epochs : {}, batch : {}, loss : {}, f1 : {}".format(epoch + 1, (idx+1) * batch_size,
                                                                       cur_loss / ((idx + 1) * batch_size),
                                                                       f1 / (idx + 1)))
    #绘制损失函数图像
    plt.plot(np.arange(len(loss_history)), np.array(loss_history))
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    plt.title('Bi-LSTM+CRF model')
    plt.show()

    #绘制f1图像
    plt.plot(np.arange(len(f1_history)), np.array(f1_history))
    plt.title('train f1 scores')
    plt.show()

    #保存模型
    # torch.save(model.state_dict(), "model.pth")

    # 恢复为预保存好特定参数的网络模型
    # model.load_state_dict(torch.load("model.pth"))
    # print(model)

    #测试模式
    model.eval()
    f1 = 0
    accuracy = 0
    precision = 0
    recall = 0
    f1_history = []
    s = 0
    with torch.no_grad():#不会更新已经训练好的模型的网络参数
        for idx, (inputs, targets, length_list) in enumerate(test_dataloader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
                length_list = length_list.cuda()
            loss = (-1) * model(inputs, length_list, targets)
            total_loss += loss.item()
            pred = model.predict(inputs, length_list)
            f= compute_f1(pred, targets, length_list)
            f1 += f
            f1_history.append(f1 / (idx + 1))
            s = idx
    print("f1 score : {},test size = {}".format(f1 / (s + 1), (s + 1)))
    # 绘制f1图像
    plt.plot(np.arange(len(f1_history)), np.array(f1_history))
    plt.title('test f1 scores')
    plt.show()
    print(totalnorm)
    print(clipcoef)


if __name__ == '__main__':
    x_train, y_train = read_data("C:/Users/成子健/Desktop/CRF++-0.58/example/conll2003_ner/train.txt")
    x_test, y_test = read_data("C:/Users/成子健/Desktop/CRF++-0.58/example/conll2003_ner/test.txt")
    word2idx, tag2idx, vocab_size = pre_processing()
    train_dataloader, train_max_length = get_dataloader(x_train, y_train, batch_size)
    test_dataloader, test_max_length = get_dataloader(x_test, y_test, 32)
    train(LSTM_CRF, vocab_size, tag2idx, embedding_size, hidden_size, max_length=train_max_length, vectors=None)



