import numpy as np
# from elmoformanylangs import Embedder
import gensim
import torch.nn.functional as F
import torch

max_len = 500
# Embedding
model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec/sgns.merge.word')


def preprocess(filename):
    textlist = []
    scorelist = []
    with open(filename, 'r') as fin:
        for line in fin.readlines():
            seg = line.strip().split('\t')
            textlist.append(seg[2])
            score = [int(x.split(':')[1]) for x in seg[1].split(' ')]
            ten = torch.tensor(score[1:]).type('torch.FloatTensor')
            scorelist.append(np.array(F.softmax(ten)))
    embed_list = np.zeros((len(textlist), max_len, 300))
    num = -1
    oov = 0
    inc = 0
    for text in textlist:
        num += 1
        tem = np.zeros((max_len, 300))
        count = -1
        for word in text:
            count += 1
            if count >= max_len:
                break
            if word in model:
                inc += 1
                tem[count] = np.array(model.wv[word])
            else:
                oov += 1
                tem[count] = np.random.random(300)*2 - 1.0
        embed_list[num] = tem
    print('oov = ', oov)
    print('inc = ', inc)
    print('embed:', np.shape(embed_list))
    np.save('./dataset/' + filename.split('.')[2] + '.embed', embed_list)
    np.save('./dataset/' + filename.split('.')[2] + '.score', scorelist)
    return


if __name__ == '__main__':
    preprocess('./dataset/sinanews.test')
    preprocess('./dataset/sinanews.train')
