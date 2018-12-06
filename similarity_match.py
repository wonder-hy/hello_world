import numpy as np
import jieba
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

# 相关语料及字典集路径
userdict_path = './jiebauserdict.txt'
jieba.load_userdict(userdict_path)
test_class_path = './test_class.txt'
test_class_path2 = './test_class2.txt'
standard_cor_path1 = './key.txt'
standard_cor_path2 = './key2.txt'


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim 余弦相似度
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denom != 0:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
    else:
        sim = 0
    return sim


def build_sim_vec(standard_sample_path, input_sample):
    """
    建立余弦相似度矩阵
    :param standard_sample_path: 参考词集路径
    :param input_sample: 测试词条
    :return: sim_vec 返回相似度矩阵
    """
    c = open(standard_sample_path, 'r', encoding='utf-8-sig')
    cor = c.readlines()
    corpus = []
    for citiao in cor:
        corpus.append(citiao)
    vec = vectorizer.fit_transform(corpus)
    standard_vec = vec.toarray()
    key_word_corpus = vectorizer.get_feature_names()
    # 建立分词文本
    cut = jieba.cut_for_search(input_sample)
    test_words = []
    for i in cut:
        test_words.append(i)
    # 建立测试文本词频率向量
    test_vec = []
    frequency = defaultdict(int)
    for word in key_word_corpus:
        for i in test_words:
            if word == i:
                frequency[word] += 1
    # one-hot 编码
    for word in key_word_corpus:
        test_vec.append(frequency[word])
    frequency.clear()
    sim_vec = []
    for vector in standard_vec:
        sim_vec.append(cos_sim(vector, test_vec))

    c.close()

    return sim_vec


# 依据相关性得出分类结论
def match(input_sample):
    """
    得出相关性结论
    :param input_sample: 输入的待分类语料条目
    :return: class_str 字符串型分类结果
    """
    t_class = open(test_class_path, 'r', encoding='utf-8-sig')
    test_class = t_class.readlines()
    t_class2 = open(test_class_path2, 'r', encoding='utf-8-sig')
    test_class2 = t_class2.readlines()
    sim_vec1 = build_sim_vec(standard_cor_path1, input_sample)
    if max(sim_vec1) > 0.7:
        if np.argmax(sim_vec1) > 3:
            class_str = str(test_class[int(np.argmax(sim_vec1))]).replace('\n', '')
        else:
            sim_vec2 = build_sim_vec(standard_cor_path2, input_sample)
            if max(sim_vec2) > 0.2:
                class_str = str(test_class2[int(np.argmax(sim_vec2))]).replace('\n', '')
            else:
                class_str = '其他'
    else:
        class_str = '其他'

    t_class.close()
    t_class2.close()

    return class_str




