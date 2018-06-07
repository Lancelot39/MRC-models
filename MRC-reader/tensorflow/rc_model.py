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
This module implements the reading comprehension models based on:
1. the BiDAF algorithm described in https://arxiv.org/abs/1611.01603
2. the Match-LSTM algorithm described in https://openreview.net/pdf?id=B1-q5Pqxl
Note that we use Pointer Network for the decoding stage of both models.
"""

import os
import time
import logging
import json
import numpy as np
import tensorflow as tf
from utils import compute_bleu_rouge
from utils import normalize
from layers.basic_rnn import rnn,highway_network
from layers.match_layer import MatchLSTMLayer
from layers.match_layer import AttentionFlowMatchLayer
from layers.match_layer import RnetLSTMLayer
from layers.match_layer import MRCLayer
from layers.pointer_net import PointerNetDecoder
from layers.pointer_net import MRCDecoder

class RCModel(object):
    """
    Implements the main reading comprehension model.
    """

    def __init__(self, vocab,char_vocab, args):

        # logging
        self.logger = logging.getLogger("brc")

        # basic config
        self.algo = args.algo
        self.hidden_size = args.hidden_size
        self.optim_type = args.optim
        #self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.use_dropout = args.dropout_keep_prob < 1

        # length limit
        self.max_p_num = args.max_p_num
        self.max_p_len = args.max_p_len
        self.max_q_len = args.max_q_len
        self.max_a_len = args.max_a_len

        # the vocab
        self.vocab = vocab
        self.char_vocab=char_vocab

        self.use_rnn_char=True
        self.use_highway_network=True

        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self._build_graph()

        # save info
        self.saver = tf.train.Saver()

        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        start_t = time.time()
        self._setup_placeholders()
        self._embed()
        self._encode()
        #self._selfAttention()   #
        self._match()#match_layer的输入是两个LSTM拼接起来的结果，((2*sequence_size)*hidden_size)一维，之后需要做reshape到二维
        self._fuse()
        #self._selfAttention_fuse()
        self._decode()
        self._compute_loss()
        self._create_train_op()
        self.logger.info('Time to build graph: {} s'.format(time.time() - start_t))
        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None]) #句子长度*句子个数
        self.q = tf.placeholder(tf.int32, [None, None]) #句子长度*句子个数
        self.p_length = tf.placeholder(tf.int32, [None])    #句子长度
        self.q_length = tf.placeholder(tf.int32, [None])    #句子长度
        self.start_label = tf.placeholder(tf.int32, [None]) #案例数量
        self.end_label = tf.placeholder(tf.int32, [None])   #案例数量
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.learning_rate=tf.placeholder(tf.float32)   #LR
        self.p_char=tf.placeholder(tf.int32,[None,None,None])	#batch_size*p_length*word_size
        self.q_char=tf.placeholder(tf.int32,[None,None,None])

    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings), #等价于直接加载词向量，可以预训练的
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)   #三维：句子长度*句子个数*embedding维度
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)   #三维：句子长度*句子个数*embedding维度
            
            self.char_embeddings=tf.get_variable('char_embeddings',shape=(self.char_vocab.size(),self.char_vocab.embed_dim),
                                                       initializer=tf.constant_initializer(self.char_vocab.embeddings),trainable=True)

            #self.p_char=tf.squeeze(tf.reshape(self.p_char,[-1,1,tf.shape(self.p_char)[-1]]))
            #self.q_char=tf.squeeze(tf.reshape(self.q_char,[-1,1,tf.shape(self.q_char)[-1]]))
            self.p_char_emb=tf.nn.embedding_lookup(self.char_embeddings,self.p_char)
            self.q_char_emb=tf.nn.embedding_lookup(self.char_embeddings,self.q_char)

        with tf.variable_scope('char_embedding_rnn'):
            word_length=tf.shape(self.p_char)[-1]
            batch_size=tf.shape(self.p_char)[0]
            p_length=tf.shape(self.p_char)[1]
            q_length=tf.shape(self.q_char)[1]
            p_embedding=tf.reshape(self.p_char_emb,[batch_size*p_length,word_length,self.char_vocab.embed_dim])
            q_embedding=tf.reshape(self.q_char_emb,[batch_size*q_length,word_length,self.char_vocab.embed_dim])

        if self.use_rnn_char:        
            with tf.variable_scope('p_char_lstm'):
                _,p_state=rnn('bi-lstm',p_embedding,tf.tile([word_length],[batch_size*p_length]),self.char_vocab.embed_dim)#p_state 1*(batch_size*p_length)*2hidden_size
                self.p_state_f=tf.reshape(p_state,[batch_size,p_length,2*self.char_vocab.embed_dim])	#遇到出现shape丢失的情况，使用reshape在丢失源头进行重塑
                self.new_p_emb=tf.concat([self.p_emb,self.p_state_f],-1)
            #self.new_p_emb.set_shape([batch_size,p_length,self.vocab.embed_dim+2*self.char_vocab.embed_dim])
            #self.final_p_emb=tf.reshape(self.new_p_emb,[batch_size,p_length,tf.shape(self.new_p_emb)[-1]])
            with tf.variable_scope('q_char_lstm'):
                _,q_state=rnn('bi-lstm',q_embedding,tf.tile([word_length],[batch_size*q_length]),self.char_vocab.embed_dim)
                self.q_state_f=tf.reshape(q_state,[batch_size,q_length,2*self.char_vocab.embed_dim])
                self.new_q_emb=tf.concat([self.q_emb,self.q_state_f],-1)
            #self.new_q_emb.set_shape([batch_size,q_length,self.vocab.embed_dim+2*self.char_vocab.embed_dim])
            #self.final_q_emb=tf.reshape(self.new_q_emb,[batch_size,q_length,tf.shape(self.new_q_emb)[-1]])
        
        else:
            with tf.variable_scope('p_char_cnn'):
                self.cnn_width=3
                p_conv_before=tf.expand_dims(p_embedding,1)	#batch_size*1*word_length*embed
                self.p_kernel=tf.Variable(tf.random_normal(shape=[1,self.cnn_width,self.char_vocab.embed_dim,self.char_vocab.embed_dim]))
                self.p_conv_after=tf.nn.conv2d(p_conv_before,filter=self.p_kernel,strides=[1,1,1,1],padding='VALID')	#strides:batch_size*p_length*word_size*embed
                self.p_maxpooling=tf.reduce_max(tf.squeeze(self.p_conv_after,1),1)	#batch_size*1*embed

                self.p_final_char=tf.reshape(self.p_maxpooling,[batch_size,p_length,self.char_vocab.embed_dim])
                self.new_p_emb=tf.concat([self.p_emb,self.p_final_char],-1)

            with tf.variable_scope('q_char_cnn'):
                self.cnn_width=3
                q_conv_before=tf.expand_dims(q_embedding,1)     #batch_size*1*word_length*embed
                self.q_kernel=tf.Variable(tf.random_normal(shape=[1,self.cnn_width,self.char_vocab.embed_dim,self.char_vocab.embed_dim]))
                self.q_conv_after=tf.nn.conv2d(q_conv_before,filter=self.q_kernel,strides=[1,1,1,1],padding='VALID')    #strides:batch_size*p_length*word_size*embed
                self.q_maxpooling=tf.reduce_max(tf.squeeze(self.q_conv_after,1),1)        #batch_size*1*embed
                self.q_final_char=tf.reshape(self.q_maxpooling,[batch_size,q_length,self.char_vocab.embed_dim])
                self.new_q_emb=tf.concat([self.q_emb,self.q_final_char],-1)
        if self.use_highway_network:
            with tf.variable_scope('highway_p'):
                self.new_p_emb=highway_network(self.new_p_emb)
            with tf.variable_scope('highway_q'):
                self.new_q_emb=highway_network(self.new_q_emb)


    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-lstm', self.new_p_emb, self.p_length, self.hidden_size) #三维：句子长度*句子个数*2倍hidden_size
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-lstm', self.new_q_emb, self.q_length, self.hidden_size) #三维：句子长度*句子个数*2倍hidden_size
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)

    def _match(self):
        """
        The core of RC model, get the question-aware passage encoding with either BIDAF or MLSTM
        """
        if self.algo == 'MLSTM':
            match_layer = MatchLSTMLayer(self.hidden_size)
        elif self.algo == 'BIDAF':
            #match_layer=MRCLayer(self.hidden_size)
            #match_layer=RnetLSTMLayer(self.hidden_size)
            match_layer = AttentionFlowMatchLayer(self.hidden_size)
        else:
            raise NotImplementedError('The algorithm {} is not implemented.'.format(self.algo))
        self.match_p_encodes, _ = match_layer.match(self.sep_p_encodes, self.sep_q_encodes,
                                                    self.p_length, self.q_length)   #BIDAF输出是三维：para句子长度*batch_size*8倍hidden_size
        if self.use_dropout:
            self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _fuse(self):
        """
        Employs Bi-LSTM again to fuse the context information after match layer
        """
        with tf.variable_scope('fusion'):
            self.match_p_encodes, _ = rnn('bi-lstm', self.match_p_encodes, self.p_length,
                                         self.hidden_size, layer_num=1) #返回三维：para句子长度*batch_size*两倍hidden_size
            if self.use_dropout:
                self.match_p_encodes = tf.nn.dropout(self.match_p_encodes, self.dropout_keep_prob)

    def _decode(self):
        """
        Employs Pointer Network to get the the probs of each position
        to be the start or end of the predicted answer.
        Note that we concat the fuse_p_encodes for the passages in the same document.
        And since the encodes of queries in the same document is same, we select the first one.
        """
        with tf.variable_scope('same_question_concat'):
            batch_size = tf.shape(self.start_label)[0]
            concat_passage_encodes = tf.reshape(
                self.match_p_encodes,
                [batch_size, -1, 2 * self.hidden_size]  #这一步实现5个句子信息汇聚到一起，三维：batch_size*5倍para句子长度*两倍hidden_size
            )
            no_dup_question_encodes = tf.reshape(
                self.sep_q_encodes,
                [batch_size, -1, tf.shape(self.sep_q_encodes)[1], 2 * self.hidden_size] #三维：batch_size*question句子长度*两倍hidden_size
            )[0:, 0, 0:, 0:]    #这一步实现label对应句子1:1
        decoder = PointerNetDecoder(self.hidden_size)
        #decoder=MRCDecoder(self.hidden_size)
        self.start_probs, self.end_probs = decoder.decode(concat_passage_encodes,
                                                          no_dup_question_encodes)

    def _compute_loss(self):
        """
        The loss function
        """

        def sparse_nll_loss(probs, labels, epsilon=1e-9, scope=None):#labels是batch_size个start_label，probs是batch_size*para_length的概率矩阵
            """
            negative log likelyhood loss
            """
            with tf.name_scope(scope, "log_loss"):
                labels = tf.one_hot(labels, tf.shape(probs)[1], axis=1)
                losses = - tf.reduce_sum(labels * tf.log(probs + epsilon), 1)
            return losses

        self.start_loss = sparse_nll_loss(probs=self.start_probs, labels=self.start_label)
        self.end_loss = sparse_nll_loss(probs=self.end_probs, labels=self.end_label)
        self.all_params = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.add(self.start_loss, self.end_loss))
        if self.weight_decay > 0:
            with tf.variable_scope('l2_loss'):
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.all_params])
            self.loss += self.weight_decay * l2_loss

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)

    def _train_epoch(self, train_batches, dropout_keep_prob,learning_rate):
        """
        Trains the model for a single epoch.
        Args:
            train_batches: iterable batch data for training
            dropout_keep_prob: float value indicating dropout keep probability
        """
        total_num, total_loss = 0, 0
        log_every_n_batch, n_batch_loss = 50, 0
        for bitx, batch in enumerate(train_batches, 1):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: dropout_keep_prob,
                         self.learning_rate: learning_rate,
                         self.p_char:batch['passage_token_char_ids'],
                         self.q_char:batch['question_token_char_ids']}
            #a,b=self.sess.run([self.shape,self.new_shape],feed_dict)
           
            _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])
            n_batch_loss += loss
            if log_every_n_batch > 0 and bitx % log_every_n_batch == 0:
                self.logger.info('Average loss from batch {} to {} is {}'.format(
                    bitx - log_every_n_batch + 1, bitx, n_batch_loss / log_every_n_batch))
                n_batch_loss = 0
        return 1.0 * total_loss / total_num

    def train(self, data, epochs, batch_size, save_dir, save_prefix,
              dropout_keep_prob=1.0, learning_rate=0.004, evaluate=True,fetch_pro=10):    #fetch_pro表示n个问题中抽一个
        """
        Train the model with data
        Args:
            data: the BRCDataset class implemented in dataset.py
            epochs: number of training epochs
            batch_size:
            save_dir: the directory to save the model
            save_prefix: the prefix indicating the model type
            dropout_keep_prob: float value indicating dropout keep probability
            evaluate: whether to evaluate the model on test set after each epoch
        """
        pad_id = self.vocab.get_id(self.vocab.pad_token)
        max_bleu_4 = 0
        last_rouge = 0
        for epoch in range(1, epochs + 1):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            self.logger.info('learning rate is {}'.format(learning_rate))
            train_batches = data.gen_mini_batches('train', batch_size, pad_id, shuffle=True)
            train_batch=[data1 for data1 in train_batches]
            train_batch_select=[train_batch[i] for i in range(len(train_batch)-1) if i%fetch_pro==0]
            train_loss = self._train_epoch(train_batch_select, dropout_keep_prob,learning_rate)
            self.logger.info('Average train loss for epoch {} is {}'.format(epoch, train_loss))

            if evaluate:
                self.logger.info('Evaluating the model after epoch {}'.format(epoch))
                if data.dev_set is not None:
                    eval_batches = data.gen_mini_batches('dev', batch_size, pad_id, shuffle=False)
                    eval_batch=[data1 for data1 in eval_batches]
                    eval_loss, bleu_rouge = self.evaluate(eval_batch[:-1])
                    self.logger.info('Dev eval loss {}'.format(eval_loss))
                    self.logger.info('Dev eval result: {}'.format(bleu_rouge))
                    if last_rouge>bleu_rouge['Rouge-L']:
                        learning_rate/=2
                        self.restore(save_dir,save_prefix)
                    else:
                        last_rouge=bleu_rouge['Rouge-L']
                    if bleu_rouge['Rouge-L'] > max_bleu_4:   #只有在效果上升时才会保存模型
                        self.save(save_dir, save_prefix)
                        max_bleu_4 = bleu_rouge['Rouge-L']
                else:
                    self.logger.warning('No dev set is loaded for evaluation in the dataset!')
            else:
                self.save(save_dir, save_prefix + '_' + str(epoch))

    def evaluate(self, eval_batches, result_dir=None, result_prefix=None, save_full_info=False):
        """
        Evaluates the model performance on eval_batches and results are saved if specified
        Args:
            eval_batches: iterable batch data
            result_dir: directory to save predicted answers, answers will not be saved if None
            result_prefix: prefix of the file for saving predicted answers,
                           answers will not be saved if None
            save_full_info: if True, the pred_answers will be added to raw sample and saved
        """
        pred_answers, ref_answers = [], []
        total_loss, total_num = 0, 0
        for b_itx, batch in enumerate(eval_batches):
            feed_dict = {self.p: batch['passage_token_ids'],
                         self.q: batch['question_token_ids'],
                         self.p_length: batch['passage_length'],
                         self.q_length: batch['question_length'],
                         self.start_label: batch['start_id'],
                         self.end_label: batch['end_id'],
                         self.dropout_keep_prob: 1.0,
                         self.p_char:batch['passage_token_char_ids'],
                         self.q_char:batch['question_token_char_ids']}
            start_probs, end_probs, loss = self.sess.run([self.start_probs,
                                                          self.end_probs, self.loss], feed_dict)

            total_loss += loss * len(batch['raw_data'])
            total_num += len(batch['raw_data'])

            padded_p_len = len(batch['passage_token_ids'][0])
            for sample, start_prob, end_prob in zip(batch['raw_data'], start_probs, end_probs):

                best_answer = self.find_best_answer(sample, start_prob, end_prob, padded_p_len)#返回的是一句一句的答案
                if save_full_info:
                    sample['pred_answers'] = [best_answer]
                    pred_answers.append(sample)
                else:
                    if sample['question_type']=='Yes_No':
                        #yes_no_result=classify_model(best_answer,sample['question_token_ids'])
                        pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': ["Yes"]}) #对于是非类问题进行回答
                    else:
                        pred_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': [best_answer],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})
                if 'answers' in sample:
                    ref_answers.append({'question_id': sample['question_id'],
                                         'question_type': sample['question_type'],
                                         'answers': sample['answers'],
                                         'entity_answers': [[]],
                                         'yesno_answers': []})

        if result_dir is not None and result_prefix is not None:
            result_file = os.path.join(result_dir, result_prefix + '.json')
            with open(result_file, 'w') as fout:
                for pred_answer in pred_answers:
                    fout.write(json.dumps(pred_answer, ensure_ascii=False) + '\n')

            self.logger.info('Saving {} results to {}'.format(result_prefix, result_file))

        # this average loss is invalid on test set, since we don't have true start_id and end_id
        ave_loss = 1.0 * total_loss / total_num
        # compute the bleu and rouge scores if reference answers is provided
        if len(ref_answers) > 0:
            pred_dict, ref_dict = {}, {}
            for pred, ref in zip(pred_answers, ref_answers):
                question_id = ref['question_id']
                if len(ref['answers']) > 0:
                    pred_dict[question_id] = normalize(pred['answers'])
                    ref_dict[question_id] = normalize(ref['answers'])
            bleu_rouge = compute_bleu_rouge(pred_dict, ref_dict)#pred_dict和ref_dict是对应每个问题ID的答案字典
        else:
            bleu_rouge = None
        return ave_loss, bleu_rouge

    def find_best_answer(self, sample, start_prob, end_prob, padded_p_len):
        """
        Finds the best answer for a sample given start_prob and end_prob for each position.
        This will call find_best_answer_for_passage because there are multiple passages in a sample
        """
        best_p_idx, best_span, best_score = None, None, 0
        for p_idx, passage in enumerate(sample['passages']):
            if p_idx >= self.max_p_num:
                continue
            passage_len = min(self.max_p_len, len(passage['passage_tokens']))
            answer_span, score = self.find_best_answer_for_passage(
                start_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                end_prob[p_idx * padded_p_len: (p_idx + 1) * padded_p_len],
                passage_len)
            if score > best_score:
                best_score = score
                best_p_idx = p_idx
                best_span = answer_span
        if best_p_idx is None or best_span is None:
            best_answer = ''
        else:
            best_answer = ''.join(
                sample['passages'][best_p_idx]['passage_tokens'][best_span[0]: best_span[1] + 1])
        return best_answer

    def find_best_answer_for_passage(self, start_probs, end_probs, passage_len=None):
        """
        Finds the best answer with the maximum start_prob * end_prob from a single passage
        """
        if passage_len is None:
            passage_len = len(start_probs)
        else:
            passage_len = min(len(start_probs), passage_len)
        best_start, best_end, max_prob = -1, -1, 0
        for start_idx in range(passage_len):
            for ans_len in range(self.max_a_len):
                end_idx = start_idx + ans_len
                if end_idx >= passage_len:
                    continue
                prob = start_probs[start_idx] * end_probs[end_idx]
                if prob > max_prob:
                    best_start = start_idx
                    best_end = end_idx
                    max_prob = prob
        return (best_start, best_end), max_prob

    def save(self, model_dir, model_prefix):
        """
        Saves the model into model_dir with model_prefix as the model indicator
        """
        self.saver.save(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model saved in {}, with prefix {}.'.format(model_dir, model_prefix))

    def restore(self, model_dir, model_prefix):
        """
        Restores the model into model_dir from model_prefix as the model indicator
        """
        self.saver.restore(self.sess, os.path.join(model_dir, model_prefix))
        self.logger.info('Model restored from {}, with prefix {}'.format(model_dir, model_prefix))
