# %%
# coding: utf-8
# Neural Machine Translation using word level language model and embeddings in Keras
from nltk.translate.bleu_score import sentence_bleu
import math
import time
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from zhon.hanzi import punctuation
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
import pandas as pd
import numpy as np
import string
from string import digits
# import matplotlib.pyplot as plt
import re
# from sklearn.cross_validation import train_test_split

UNIT_OUTPUT = 256
SAMPLES = 10000
EPOCH = 100
BATCH_SIZE = 32


def createCmn():
    lines = pd.read_table(
        'data/cmn.txt', names=['eng', 'zh', '_'],
        # nrows=SAMPLES,  # used for debug
    )
    lines.zh = lines.zh.astype(str)
    lines.eng = lines.eng.astype(str)
    lines = lines.sample(frac=1)  # shuffle
    return lines


def createWikititles():
    lines = pd.read_table('data/wikititles-v2.zh-en.tsv',
                          names=['zh', 'eng'],
                          nrows=SAMPLES,  # used for debug
                          )
    lines.zh = lines.zh.astype(str)
    lines.eng = lines.eng.astype(str)
    lines = lines.sample(frac=1)  # shuffle
    return lines


def createNews():
    lines = pd.read_table('data/news-commentary-v15.en-zh.tsv',
                          names=['eng', 'zh'],
                          nrows=SAMPLES,  # used for debug
                          )
    lines.zh = lines.zh.astype(str)
    lines.eng = lines.eng.astype(str)
    lines = lines.sample(frac=1)  # shuffle
    return lines


# #### Data Cleanup
def pre_process(lines):
    lines.eng = lines.eng.apply(lambda x: x.lower())
    lines.zh = lines.zh.apply(lambda x: x.lower())

    # Take the length as 50
    lines.eng = lines.eng.apply(lambda x: re.sub("'", '', x)).apply(
        # lambda x: re.sub(",", ' COMMA', x))
        lambda x: re.sub(",", '', x))
    lines.zh = lines.zh.apply(lambda x: re.sub("'", '', x)).apply(
        # lambda x: re.sub(",", ' COMMA', x))
        lambda x: re.sub(",", '', x))

    exclude = set(string.punctuation)
    lines.eng = lines.eng.apply(lambda x: ''.join(
        ch for ch in x if ch not in exclude))
    exclude_zh = set(punctuation)
    lines.zh = lines.zh.apply(lambda x: ''.join(
        ch for ch in x if ch not in exclude_zh))

    remove_digits = str.maketrans('', '', digits)
    lines.eng = lines.eng.apply(lambda x: x.translate(remove_digits))
    lines.zh = lines.zh.apply(lambda x: x.translate(remove_digits))

    # #### Generate synthetic data

    lines.zh = lines.zh.apply(lambda x: ' '.join(x))
    lines.zh = lines.zh.apply(lambda x: 'START_ ' + x + ' _END')
    return lines

# 生成器


def generate_batch(X, y, batch_size=32):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = np.zeros(
                (batch_size, max_len_en), dtype='float32')
            decoder_input_data = np.zeros(
                (batch_size, max_len_zh), dtype='float32')
            decoder_target_data = np.zeros(
                (batch_size, max_len_zh, num_decoder_tokens), dtype='float32')
            for i, (input_text, target_text) in enumerate(zip(X[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    # encoder input seq
                    encoder_input_data[i, t] = input_token_index[word]
                for t, word in enumerate(target_text.split()):
                    if t < len(target_text.split())-1:
                        # decoder input seq
                        decoder_input_data[i, t] = target_token_index[word]
                    if t > 0:
                        # decoder target sequence (one hot encoded)
                        # does not include the START_ token
                        # Offset by one timestep
                        decoder_target_data[i, t - 1,
                                            target_token_index[word]] = 1.
            yield([encoder_input_data, decoder_input_data], decoder_target_data)


def createEmbedding():
    # 添加英文预训练的embedding
    embeddings_index_en = dict()
    f = open('word2vec/glove.6B/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index_en[word] = coefs
    f.close()

    cnt = 0
    unadded_en = []
    print('>>>>>>>>>>正在加载英文无监督训练的词向量...')
    embedding_dim = 300
    embedding_matrix_en = np.random.randn(num_encoder_tokens, embedding_dim)
    for word, index in input_token_index.items():
        embedding_vector = embeddings_index_en.get(word)
        if embedding_vector is not None:
            embedding_matrix_en[index] = embedding_vector
            cnt += 1
        else:
            unadded_en.append(word)
    print(
        '>>>>>>>>>>英文总共添加了{:.1f}%的词向量'.format(cnt/num_encoder_tokens*100))

    # 添加中文预训练embedding
    print('>>>>>>>>>>正在加载中文无监督训练的词向量...')
    embeddings_index_zh = dict()
    f = open(
        'word2vec/glove.6B/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5')
    jump_first_line = True
    for line in f:
        if jump_first_line:
            jump_first_line = False
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index_zh[word] = coefs
    f.close()

    cnt = 0
    unadded_zh = []
    embedding_matrix_zh = np.random.randn(num_decoder_tokens, 300)
    for word, index in target_token_index.items():
        embedding_vector = embeddings_index_zh.get(word)
        if embedding_vector is not None:
            embedding_matrix_zh[index] = embedding_vector
            cnt += 1
        else:
            unadded_zh.append(word)
    print(
        '>>>>>>>>>>中文总共添加了{:.1f}%的词向量'.format(cnt/num_decoder_tokens*100))
    time.sleep(3)
    print('英文空的词向量：', unadded_en)
    print('中文空的词向量：', unadded_zh)
    time.sleep(3)
    return embedding_matrix_en, embedding_matrix_zh


def createModel():
    encoder_inputs = Input(shape=(None,))
    # 300为预训练embedding的特征维度
    # embedding_matrix_en, embedding_matrix_zh = createEmbedding()
    en_x = Embedding(num_encoder_tokens, 300,
                     #  weights=[embedding_matrix_en],
                     trainable=False)(encoder_inputs)
    encoder = LSTM(UNIT_OUTPUT, return_state=True)
    encoder_outputs, state_h, state_c = encoder(en_x)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]
    # Decoder model
    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    embedding = Embedding(num_decoder_tokens, 300,
                          #   weights=[embedding_matrix_zh],
                          trainable=False)
    final_dex = embedding(decoder_inputs)
    decoder_lstm = LSTM(UNIT_OUTPUT, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(final_dex,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model_train = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model_train.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy')
    model_train.summary()
    encoder_model = Model(encoder_inputs, encoder_states)

    # #### Create sampling model
    decoder_state_input_h = Input(shape=(None,))
    decoder_state_input_c = Input(shape=(None,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    final_dex2 = embedding(decoder_inputs)

    decoder_outputs2, state_h2, state_c2 = decoder_lstm(
        final_dex2, initial_state=decoder_states_inputs)
    decoder_states2 = [state_h2, state_c2]
    decoder_outputs2 = decoder_dense(decoder_outputs2)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs2] + decoder_states2)
    return model_train, encoder_model, decoder_model


def scheduler(epoch, lr):
    # 自己调整学习率的策略
    # if epoch < 40:
    #     return 0.001
    # elif 40 <= epoch < 50:
    #     # return float(0.001 * math.exp(0.1 * (40 - epoch)))
    #     return 0.001*0.1
    return 0.001


def decode_sequence(input_seq):
    # 创建字典
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = ''
    while True:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or len(decoded_sentence) > max_len_zh):
            break

        decoded_sentence += sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


if __name__ == "__main__":
    lines = createCmn()
    # lines = createWikititles()
    # lines = createNews()
    lines = pre_process(lines)

    all_eng_words = set()
    for eng in lines.eng:
        for word in eng.split():
            if word not in all_eng_words:
                all_eng_words.add(word)

    all_zh_words = set()
    for zh in lines.zh:
        for word in zh.split():
            if word not in all_zh_words:
                all_zh_words.add(word)

    lenght_list = []
    for l in lines.zh:
        lenght_list.append(len(l.split(' ')))
    max_len_zh = np.max(lenght_list)

    lenght_list = []
    for l in lines.eng:
        lenght_list.append(len(l.split(' ')))
    max_len_en = np.max(lenght_list)

    input_words = sorted(list(all_eng_words))
    target_words = sorted(list(all_zh_words))
    num_encoder_tokens = len(all_eng_words)
    num_decoder_tokens = len(all_zh_words)

    input_token_index = dict(
        [(word, i) for i, word in enumerate(input_words)])
    target_token_index = dict(
        [(word, i) for i, word in enumerate(target_words)])

    model, encoder_model, decoder_model = createModel()
    plot_model(to_file='img/model.png', model=model)
    plot_model(to_file='img/encoder.png', model=encoder_model)
    plot_model(to_file='img/decoder.png', model=decoder_model)

    checkpoint = ModelCheckpoint("checkpoint/epoch_{epoch:02d}.hdf5", monitor='val_loss', verbose=0,
                                 save_best_only=False, mode='auto', period=20)
    LearningRate = LearningRateScheduler(scheduler, verbose=0)

    print('learning rate: ', K.eval(model.optimizer.lr))  # 学习率0.001
    # 数据量多的时候用fit_generator
    model.fit_generator(generate_batch(lines.eng, lines.zh, batch_size=BATCH_SIZE),
                        steps_per_epoch=lines.shape[0] // BATCH_SIZE,
                        epochs=EPOCH,
                        callbacks=[checkpoint, LearningRate])
    # 加载模型的检查点
    # model.load_weights('checkpoint/epoch_500.hdf5')  # slc

    # 模型预测方法
    encoder_input_data = np.zeros((1, max_len_en), dtype='float32')
    cnt = 0
# %%
    sum_score = 0
    break_cnt = 100
    for i, input_text in enumerate(lines.eng):
        for t, word in enumerate(input_text.split()):
            encoder_input_data[0, t] = input_token_index[word]
        decoded_sentence = decode_sequence(encoder_input_data)
        # compute BLEU score
        reference = lines.zh[i].split()
        reference = [reference[1:-1]]
        candidate = [e for e in decoded_sentence]
        score = sentence_bleu(reference, candidate,
                              weights=(0.25, 0.25, 0.25, 0.25))
        sum_score += score
        print('-')
        print('Input sentence:', input_text)
        print('Decoded sentence:', decoded_sentence)
        print('BLEU score:', score)
        cnt += 1
        if cnt == break_cnt:
            break
    average_score = sum_score / break_cnt
    print('-')
    print('Average BLEU is {:.2f}'.format(average_score))


# %%
