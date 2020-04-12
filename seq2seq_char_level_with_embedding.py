
# coding: utf-8

# ### Neural Machine Translation using word level language model and embeddings in Keras
import math
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from zhon.hanzi import punctuation
import ipdb
# from IPython.core.display import display, HTML
# from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense
import pandas as pd
import numpy as np
import string
from string import digits
# import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import re
from sklearn.cross_validation import train_test_split
# Building a english to french translator

SAMPLES = 20000
EPOCH = 500
BATCH_SIZE = 64


# display(HTML("<style>.container { width:100% !important; }</style>"))


lines = pd.read_table('data/cmn.txt', names=['eng', 'fr', '_'])


lines = lines[0:SAMPLES]


lines.shape


lines.sample(10)


# #### Data Cleanup

lines.eng = lines.eng.apply(lambda x: x.lower())
lines.fr = lines.fr.apply(lambda x: x.lower())

# char-level的分词
lines.eng = lines.eng.apply(lambda x: ' '.join(x))

# Take the length as 50
lines.eng = lines.eng.apply(lambda x: re.sub("'", '', x)).apply(
    # lambda x: re.sub(",", ' COMMA', x))
    lambda x: re.sub(",", '', x))
lines.fr = lines.fr.apply(lambda x: re.sub("'", '', x)).apply(
    # lambda x: re.sub(",", ' COMMA', x))
    lambda x: re.sub(",", '', x))


exclude = set(string.punctuation)
lines.eng = lines.eng.apply(lambda x: ''.join(
    ch for ch in x if ch not in exclude))
exclude_zh = set(punctuation)
lines.fr = lines.fr.apply(lambda x: ''.join(
    ch for ch in x if ch not in exclude_zh))


remove_digits = str.maketrans('', '', digits)
lines.eng = lines.eng.apply(lambda x: x.translate(remove_digits))
lines.fr = lines.fr.apply(lambda x: x.translate(remove_digits))


lines.head()


# #### Generate synthetic data

lines.fr = lines.fr.apply(lambda x: ' '.join(x))

lines.fr = lines.fr.apply(lambda x: 'START_ ' + x + ' _END')


all_eng_words = set()
for eng in lines.eng:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)

all_french_words = set()
for fr in lines.fr:
    for word in fr.split():
        if word not in all_french_words:
            all_french_words.add(word)


len(all_eng_words), len(all_french_words)


lenght_list = []
for l in lines.fr:
    lenght_list.append(len(l.split(' ')))
max_len_fr = np.max(lenght_list)


lenght_list = []
for l in lines.eng:
    lenght_list.append(len(l.split(' ')))
max_len_en = np.max(lenght_list)


input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_french_words))
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_french_words)
# del all_eng_words, all_french_words


input_token_index = dict(
    [(word, i) for i, word in enumerate(input_words)])
target_token_index = dict(
    [(word, i) for i, word in enumerate(target_words)])


# len(lines.fr)*16*num_decoder_tokens


encoder_input_data = np.zeros(
    (len(lines.eng), max_len_en),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(lines.fr), max_len_fr),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(lines.fr), max_len_fr, num_decoder_tokens),
    dtype='float32')


for i, (input_text, target_text) in enumerate(zip(lines.eng, lines.fr)):
    for t, word in enumerate(input_text.split()):
        encoder_input_data[i, t] = input_token_index[word]
    for t, word in enumerate(target_text.split()):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = target_token_index[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[word]] = 1.


# ### Build keras encoder-decoder model

# #### Encoder model

# 添加英文预训练的embedding
embeddings_index_en = dict()
# f = open('glove.6B/glove.6B.50d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index_en[word] = coefs
# f.close()

cnt = 0
unadded = []
embedding_matrix_en = np.random.randn(num_encoder_tokens, 50)
# for word, index in input_token_index.items():
#     embedding_vector = embeddings_index_en.get(word)
#     if embedding_vector is not None:
#         embedding_matrix_en[index] = embedding_vector
#         cnt += 1
#     else:
#         unadded.append(word)
# print('------>>>>>英文总共添加了{:.1f}%的embedding'.format(cnt/num_encoder_tokens*100))

# 添加中文预训练embedding
embeddings_index_zh = dict()
# f = open('glove.6B/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5')
# jump_first_line = True
# for line in f:
#     if jump_first_line:
#         jump_first_line = False
#         continue
#     values = line.split()
#     word = values[0]
#     coefs = np.asarray(values[1:], dtype='float32')
#     embeddings_index_zh[word] = coefs
# f.close()

cnt = 0
embedding_matrix_zh = np.random.randn(num_decoder_tokens, 300)
# for word, index in target_token_index.items():
#     embedding_vector = embeddings_index_zh.get(word)
#     if embedding_vector is not None:
#         embedding_matrix_zh[index] = embedding_vector
#         cnt += 1
#     else:
#         unadded.append(word)
print(
    '------->>>>>>中文总共添加了{:.1f}%的embedding'.format(cnt/num_decoder_tokens*100))
print('空的词向量：', unadded)
# ipdb.set_trace()

encoder_inputs = Input(shape=(None,))
en_x = Embedding(num_encoder_tokens, num_encoder_tokens,
                 trainable=True)(encoder_inputs)
encoder = LSTM(50, return_state=True)
encoder_outputs, state_h, state_c = encoder(en_x)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
# #### Decoder model

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))

dex = Embedding(num_decoder_tokens, num_decoder_tokens,  trainable=True)

final_dex = dex(decoder_inputs)


decoder_lstm = LSTM(50, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(final_dex,
                                     initial_state=encoder_states)

decoder_dense = Dense(num_decoder_tokens, activation='softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy')
model.summary()


# #### Fit the model
checkpoint = ModelCheckpoint("checkpoint/char_rmsprop_epoch_{epoch:02d}_valloss_{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0,
                             save_best_only=False, mode='auto', period=5)


def scheduler(epoch, lr):
    if epoch < 40:
        return 0.001
    # elif 40 <= epoch < 50:
    #     # return float(0.001 * math.exp(0.1 * (40 - epoch)))
    #     return 0.001*0.1
    else:
        return 0.001


LearningRate = LearningRateScheduler(scheduler, verbose=0)


# model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
#           batch_size=BATCH_SIZE,
#           epochs=EPOCH,
#           validation_split=0.1,
#           callbacks=[checkpoint, LearningRate])

checkpointPath = 'checkpoint/char_rmsprop_epoch_450_valloss_2.98.hdf5'
model.load_weights(checkpointPath)

# print('learning rate: ', K.eval(model.optimizer.lr)) # 学习率0.001
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()

# #### Create sampling model
decoder_state_input_h = Input(shape=(50,))
decoder_state_input_c = Input(shape=(50,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

final_dex2 = dex(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    final_dex2, initial_state=decoder_states_inputs)
decoder_states2 = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states2)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = target_token_index['START_']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '_END' or
                len(decoded_sentence) > max_len_fr):
            stop_condition = True
            break

        decoded_sentence += sampled_char

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence


# #### Look at the some translations


for seq_index in range(SAMPLES-100, SAMPLES):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:',
          lines.eng[seq_index: seq_index + 1].values.tolist()[0])
    print('Decoded sentence:', decoded_sentence)
