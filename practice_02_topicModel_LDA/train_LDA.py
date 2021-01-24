import numpy as np
# 准备数据集，建一个class来加载数据集，对数据进行预处理
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora
import logging
import logging.config


class DataSet:

    def __init__(self, txt_file_path_1, txt_file_path_2):
        self.__txt_file_1 = txt_file_path_1
        self.__txt_file_2 = txt_file_path_2
        logging.config.fileConfig('./logging.conf')
        self.trainLogger = logging.getLogger('trainLogger')

    def __load_txt(self, path):  # 从txt文档中加载文本内容，逐行读入
        with open(path, 'r') as file:
            content = file.readlines()  # 一次性将所有的行都读入
        return [line[:-1] for line in content]  # 去掉每一行末尾的\n

    def __tokenize(self, lines_list):  # 预处理之一：对每一行文本进行分词
        tokenizer = RegexpTokenizer('\w+')
        # 此处用正则表达式分词器而不用word_tokenize的原因是：排除带有标点的单词
        return [tokenizer.tokenize(line.lower()) for line in lines_list]

    def __remove_stops(self, lines_list):  # 预处理之二：对每一行取出停用词
        # 我们要删除一些停用词，避免这些词的噪声干扰，故而需要一个停用词表
        stop_words_list = stopwords.words('english')  # 获取英文停用词表
        return [[token for token in line if token not in stop_words_list]
                for line in lines_list]
        # 这儿有点难以理解，lines_list含有的元素也是list，这一个list就是一行文本，
        # 而一行文本内部有N个分词组成，故而lines_list可以看出二维数组，需要用两层generator

    def __word_stemm(self, lines_list):  # 预处理之三：对每个分词进行词干提取
        stemmer = SnowballStemmer('english')
        return [[stemmer.stem(word) for word in line] for line in lines_list]

    def prepare(self):
        '''供外部调用的函数，用于准备数据集'''
        # 先从txt文件中加载文本内容，再进行分词，再去除停用词，再进行词干提取
        stemmed_words_1 = self.__word_stemm(self.__remove_stops(self.__tokenize(self.__load_txt(self.__txt_file_1))))
        stemmed_words_2 = self.__word_stemm(self.__remove_stops(self.__tokenize(self.__load_txt(self.__txt_file_2))))
        self.trainLogger.info(stemmed_words_2)
        # 后面的建模需要用到基于dict的词矩阵，故而先用corpora构建dict在建立词矩阵
        dict_words = corpora.Dictionary(stemmed_words_2)
        self.trainLogger.info(dict_words)

        # word_list = list()
        # with open('./data/generative_vocab.txt') as f:
        #     for line in f.readlines():
        #         word_list.append(line.rstrip('\n'))
        #         print(line.rstrip('\n'))
        # dict_words = corpora.Dictionary([word_list])
        # print(dict_words.values())

        matrix_words = [dict_words.doc2bow(text) for text in stemmed_words_1]
        return dict_words, matrix_words

    # 以下函数主要用于测试上面的几个函数是否运行正常

    def get_content(self):
        return self.__load_txt()

    def get_tokenize(self):
        return self.__tokenize(self.__load_txt())

    def get_remove_stops(self):
        return self.__remove_stops(self.__tokenize(self.__load_txt()))

    def get_word_stemm(self):
        return self.__word_stemm(self.__remove_stops(self.__tokenize(self.__load_txt())))


if __name__ == '__main__':

    logging.config.fileConfig('./logging.conf')
    trainLogger = logging.getLogger('trainLogger')
    np.random.seed(37)

    # 获取数据集
    dataset = DataSet("./data/data_topic_modeling1.txt", './data/generative_vocab.txt')
    dict_words, matrix_words = dataset.prepare()

    # 使用LDAModel建模
    lda_model = models.ldamodel.LdaModel(matrix_words, num_topics=30,
                                         id2word=dict_words, passes=20)
    # 此处假设原始文档有两个主题

    # 查看模型中最重要的N个单词
    print('Most important words to topics: ')
    for item in lda_model.print_topics(num_topics=30, num_words=5):
        # 此处只打印最重要的5个单词
        print('Topic: {}, words: {}'.format(item[0], item[1]))

    para_matrix = lda_model.get_topics()
    print(para_matrix)
    np.save('matrix_info', para_matrix)
    np.savetxt('LDA_matrix.txt', para_matrix)
