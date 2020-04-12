#!/usr/bin/env python3
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import gensim
import multiprocessing
import sys
import os.path
import logging
import warnings
import jieba
import numpy as np

source_file = '../data/en-zh/UNv1.0.en-zh.zh'
file_with_white_space = 'tmp_UNv1.0.en-zh.zh_with_white_space'

# f_write = open(file_with_white_space, 'w')
# with open(source_file, 'r') as f_read:
#     for line in f_read:
#         line = [char for char in line]
#         line = ' '.join(line)
#         f_write.write(line)
# f_write.close()

# 忽略警告
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

if __name__ == '__main__':

    # inp为输入语料, outp1为输出模型, outp2为vector格式的模型
    input_file = file_with_white_space
    out_model = 'out.model'
    out_vector = 'out.vector'

    # 训练skip-gram模型
    model = Word2Vec(LineSentence(input_file), size=10, window=5, min_count=5,
                     workers=multiprocessing.cpu_count())

    # 保存模型
    model.save(out_model)
    # 保存词向量
    model.wv.save_word2vec_format(out_vector, binary=False)
