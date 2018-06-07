# -*- coding:utf8 -*-
# ==============================================================================
# Copyright 2017 Baidu.com, Inc. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
This module implements data process strategies.
"""

import os
import json
import logging
import numpy as np
from collections import Counter


class BRCDataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """
    def __init__(self, max_p_num, max_p_len, max_q_len,word_size,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_p_num = max_p_num
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len

        self.word_size=word_size

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.train_set += self._load_dataset(train_file, train=True)    #每次选取每个问题的最优解
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.dev_set += self._load_dataset(dev_file) #
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)   #
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []   #用来存放所有的sample
            char_data_set=[]
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:   #根据answer span来剔除数据，实际上对于给定max_p_len时，一部分数据会浪费，目前想到的方法是将max_p_len设置足够大，已知平均长度是394
                    if len(sample['answer_spans']) == 0:    #剔除未给出answer span的数据
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:  #剔除answer span的下界超出文章设定最大长度的数据
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']
                    
                sample['question_tokens'] = sample['segmented_question']
                sample['question_char']=sample['question']

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):   
                    if train:   #对所有相关的给出的问题中，将其分词后的最优回答及该问题的'is_selected'项存入sample['passages']列表中
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['segmented_paragraphs'][most_related_para],
                             'is_selected': doc['is_selected'],
                             'passage_char':doc['paragraphs'][most_related_para]}
                        )
                    else:   #对所有的问题中，非训练集时，将每个参考问题的得分最高的回答返回（无法考虑问题是否被选中），得分标准就是问题在答案中的召回率
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)  #使用Counter类统计每段文章中词语频率和问题中词语频率交集！！
                            correct_preds = sum(common_with_question.values())  #统计上一集合中所有数的出现频率之和
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)   #计算问题召回率=该交集总频率/问题长度
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))    #按照-问题召回率和问题长度排序
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0] #选择排第一的答案作为最优解
                        sample['passages'].append({'passage_tokens': fake_passage_tokens,
                                                   'passage_char':''.join(fake_passage_tokens)})
                data_set.append(sample)
        return data_set

    def new_load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        with open(data_path) as fin:
            data_set = []   #用来存放所有的sample
            for lidx, line in enumerate(fin):
                sample = json.loads(line.strip())
                if train:   #根据answer span来剔除数据，实际上对于给定max_p_len时，一部分数据会浪费，目前想到的方法是将max_p_len设置足够大，已知平均长度是394
                    if len(sample['answer_spans']) == 0:    #剔除未给出answer span的数据
                        continue
                    if sample['answer_spans'][0][1] >= self.max_p_len:  #剔除answer span的下界超出文章设定最大长度的数据
                        continue

                if 'answer_docs' in sample:
                    sample['answer_passages'] = sample['answer_docs']

                sample['question_tokens'] = sample['question'].strip()

                sample['passages'] = []
                for d_idx, doc in enumerate(sample['documents']):   
                    if train:   #对所有相关的给出的问题中，将其分词后的最优回答及该问题的'is_selected'项存入sample['passages']列表中
                        most_related_para = doc['most_related_para']
                        sample['passages'].append(
                            {'passage_tokens': doc['paragraphs'][most_related_para].strip(),
                             'is_selected': doc['is_selected']}
                        )
                    else:   #对所有的问题中，非训练集时，将每个参考问题的得分最高的回答返回（无法考虑问题是否被选中），得分标准就是问题在答案中的召回率
                        para_infos = []
                        for para_tokens in doc['segmented_paragraphs']:
                            question_tokens = sample['segmented_question']
                            common_with_question = Counter(para_tokens) & Counter(question_tokens)  #使用Counter类统计每段文章中词语频率和问题中词语频率交集！！
                            correct_preds = sum(common_with_question.values())  #统计上一集合中所有数的出现频率之和
                            if correct_preds == 0:
                                recall_wrt_question = 0
                            else:
                                recall_wrt_question = float(correct_preds) / len(question_tokens)   #计算问题召回率=该交集总频率/问题长度
                            para_infos.append((para_tokens, recall_wrt_question, len(para_tokens)))
                        para_infos.sort(key=lambda x: (-x[1], x[2]))    #按照-问题召回率和问题长度排序
                        fake_passage_tokens = []
                        for para_info in para_infos[:1]:
                            fake_passage_tokens += para_info[0] #选择排第一的答案作为最优解
                        sample['passages'].append({'passage_tokens': fake_passage_tokens})
                data_set.append(sample)
        return data_set

    def generate_word2vec_trainset(self):
        #为训练词向量提供材料，并lower化
        trainset=[]
        for data in self.train_set:
            trainset.append(data['segmented_question'])
            for document in data['documents']:
                trainset.append(document['segmented_title'])
                trainset.extend(document['segmented_paragraphs'])
        new_train_set=[[token.lower() for token in data] for data in trainset]
        return new_train_set

    def generate_char2vec_trainset(self):
        trainset=[]
        for data in self.train_set:
            trainset.append(data['question_char'])
            for document in data['documents']:
                trainset.extend(document['paragraphs'])
        new_train_set=[data.lower() for data in trainset]
        return new_train_set

    def generate_word2vec_testset(self):
        trainset=[]
        for data in self.train_set:
            trainset.append(data['question'])
            for document in data['documents']:
                trainset.append(document['title'])
                trainset.extend(document['paragraphs'])
        new_train_set=[data.strip().lower() for data in trainset]
        return new_train_set

    def _one_mini_batch(self, data, indices, pad_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {'raw_data': [data[i] for i in indices],   #将切片中的数据读进来，每个元素都是一条数据sample
                      'question_token_ids': [], #每个元素都是每个问题的token
                      'question_length': [],    #每个元素都是每个问题长度
                      'passage_token_ids': [],  #每个元素都是一个文章的token
                      'passage_length': [], #每个元素都是文章的长度
                      'start_id': [],
                      'end_id': [],
                      'question_token_char_ids':[],
                      'passage_token_char_ids':[]}
        max_passage_num = max([len(sample['passages']) for sample in batch_data['raw_data']])   #关联文章最多数量
        max_passage_num = min(self.max_p_num, max_passage_num)
        for sidx, sample in enumerate(batch_data['raw_data']):  #对batch_data中的所有问题，全部将问题、回答、关联文章等有关部分读取进来
            for pidx in range(max_passage_num):
                if pidx < len(sample['passages']):  #这样针对每一篇关联文章，就有同样的问题与之对应，即一个问题对应了多个可能答案？？
                    batch_data['question_token_ids'].append(sample['question_token_ids'])
                    batch_data['question_length'].append(len(sample['question_token_ids']))
                    passage_token_ids = sample['passages'][pidx]['passage_token_ids']
                    batch_data['passage_token_ids'].append(passage_token_ids)
                    batch_data['passage_length'].append(min(len(passage_token_ids), self.max_p_len))
                    batch_data['passage_token_char_ids'].append(sample['passages'][pidx]['passage_token_char_ids'])
                    batch_data['question_token_char_ids'].append(sample['question_token_char_ids'])
                else:   #文章数量不足部分补偿，过长就截断
                    batch_data['question_token_ids'].append([])
                    batch_data['question_length'].append(0)
                    batch_data['passage_token_ids'].append([])
                    batch_data['passage_length'].append(0)
                    batch_data['passage_token_char_ids'].append([])
                    batch_data['question_token_char_ids'].append([])
        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id)
        for sample in batch_data['raw_data']:
            if 'answer_passages' in sample and len(sample['answer_passages']):
                gold_passage_offset = padded_p_len * sample['answer_passages'][0]   #这里得到的是答案在五篇para中的偏移量，后面将五篇para拼接到一起形成一段！
                batch_data['start_id'].append(gold_passage_offset + sample['answer_spans'][0][0])   #计算每个问题的最佳答案的偏移量
                batch_data['end_id'].append(gold_passage_offset + sample['answer_spans'][0][1])
            else:
                # fake span for some samples, only valid for testing
                batch_data['start_id'].append(0)
                batch_data['end_id'].append(0)
            sample=[]
        #batch_data.pop('raw_data')  #删了看看能不能快一点
        return batch_data   #返回batch_data字典中每个元素都是一个列表，start_id和end_id是其他长度的1/5

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_p_len = min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = min(self.max_q_len, max(batch_data['question_length']))
        batch_data['passage_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]   #空格填充过长部分，并再截断一次
                                           for ids in batch_data['passage_token_ids']]
        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]

        
        new_token=[]
        for sentence in batch_data['passage_token_char_ids']:
            word_list=[]
            for word in sentence:
                if len(word)>=self.word_size:
                    word_list.append(word[:self.word_size])
                else:
                    word_list.append(word+(self.word_size-len(word))*[pad_id])
            if len(word_list)<pad_p_len:
                word_list+=(pad_p_len-len(word_list))*[self.word_size*[pad_id]]
            else:
                word_list=word_list[:pad_p_len]
            new_token.append(word_list)
        batch_data['passage_token_char_ids']=new_token

        new_token=[]
        for sentence in batch_data['question_token_char_ids']:
            word_list=[]
            for word in sentence:
                if len(word)>=self.word_size:
                    word_list.append(word[:self.word_size])
                else:
                    word_list.append(word+(self.word_size-len(word))*[pad_id])
            if len(word_list)<pad_q_len:
                word_list+=(pad_q_len-len(word_list))*[self.word_size*[pad_id]]
            else:
                word_list=word_list[:pad_q_len]
            new_token.append(word_list)
        batch_data['question_token_char_ids']=new_token

        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']: #将问题的分词return
                    yield token
                for passage in sample['passages']:  #将所有被选中段落分词返回
                    for token in passage['passage_tokens']:
                        yield token

    def char_iter(self,set_name=None):
        if set_name is None:
            data_set=self.train_set+self.dev_set+self.test_set
        elif set_name=='train':
            data_set=self.train_set
        elif set_name=='dev':
            data_set=self.dev_set
        elif set_name=='test':
            data_set=self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_char']:
                    yield token
                for passage in sample['passages']:
                    for token in passage['passage_char']:
                        yield token

    def convert_to_ids(self, vocab):        #变word成index
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_to_ids(sample['question_tokens'])
                for passage in sample['passages']:
                    passage['passage_token_ids'] = vocab.convert_to_ids(passage['passage_tokens'])

    def convert_to_char_ids(self,vocab):
        for data_set in [self.train_set,self.dev_set,self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_char_ids']=[vocab.convert_to_ids(word) for word in sample['question_tokens']]
                for passage in sample['passages']:
                    passage['passage_token_char_ids']=[vocab.convert_to_ids(word) for word in passage['passage_tokens']]

    def gen_mini_batches(self, set_name, batch_size, pad_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)  #返回0-data_size的数组
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id)
