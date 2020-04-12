from langconv import *


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


if __name__ == "__main__":
    f_simplified = open('wikititles-v2.zh-en_simplified.tsv', 'w')
    f_traditional = open('wikititles-v2.zh-en_traditional.tsv', 'r')
    for traditional_sentence in f_traditional:
        simplified_sentence = Traditional2Simplified(traditional_sentence)
        f_simplified.write(simplified_sentence)
    f_traditional.close()
    f_simplified.close()
