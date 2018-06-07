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
This module implements the core layer of Match-LSTM and BiDAF
"""

import tensorflow as tf
import tensorflow.contrib as tc
from .basic_rnn import rnn

class MatchLSTMAttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        super(MatchLSTMAttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,
                                                    num_outputs=self._num_units,
                                                    activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)
            G = tf.tanh(self.fc_context
                        + tf.expand_dims(tc.layers.fully_connected(ref_vector,
                                                                   num_outputs=self._num_units,
                                                                   activation_fn=None), 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context,
                                    inputs - attended_context, inputs * attended_context],
                                   -1)
            return super(MatchLSTMAttnCell, self).__call__(new_inputs, state, scope)


class MatchLSTMLayer(object):
    """
    Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Match-LSTM algorithm
        """
        with tf.variable_scope('match_lstm'):
            cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                             inputs=passage_encodes,
                                                             sequence_length=p_length,
                                                             dtype=tf.float32)
            match_outputs = tf.concat(outputs, 2)
            state_fw, state_bw = state
            c_fw, h_fw = state_fw
            c_bw, h_bw = state_bw
            match_state = tf.concat([h_fw, h_bw], 1)
        return match_outputs, match_state

class RnetLSTMAttnCell(tf.contrib.rnn.LSTMCell):
    def __init__(self,num_units,context):
        super(RnetLSTMAttnCell,self).__init__(num_units,state_is_tuple=True)
        self.context=context
        self.num_units=num_units
        self.fc_context=tf.contrib.layers.fully_connected(self.context,num_outputs=num_units,activation_fn=None)

    def __call__(self,input,state,scope=None):
        (c,h)=state
        with tf.variable_scope('match-cell-rnet'):
            fusion=tf.concat([input,h],-1)
            self.up=tf.contrib.layers.fully_connected(fusion,num_outputs=self.num_units,activation_fn=None)
            st=tf.contrib.layers.fully_connected(tf.tanh(self.fc_context+tf.expand_dims(self.up,1)),num_outputs=1,activation_fn=None)
            ct=tf.matmul(tf.transpose(tf.nn.softmax(st,1),[0,2,1]),self.context)	#batch_size*1*embed
            fusion2=tf.concat([input,tf.squeeze(ct,1)],-1)
            gt=tf.contrib.layers.fully_connected(fusion2,num_outputs=fusion2.get_shape().as_list()[-1],activation_fn=tf.sigmoid)
        return super(RnetLSTMAttnCell,self).__call__(fusion2*gt,state,scope)

class RnetLSTMselfAttnCell(tf.contrib.rnn.LSTMCell):
    def __init__(self,num_units,context):
        super(RnetLSTMselfAttnCell,self).__init__(num_units,state_is_tuple=True)
        self.context=context
        self.num_units=num_units
        self.fc_context=tf.contrib.layers.fully_connected(self.context,num_outputs=num_units,activation_fn=None)

    def __call__(self,input,state,scope=None):
        (c,h)=state
        with tf.variable_scope('self-cell-rnet'):
            fusion=tf.concat([input,h],-1)
            self.up=tf.contrib.layers.fully_connected(fusion,num_outputs=self.num_units,activation_fn=None)
            st=tf.contrib.layers.fully_connected(tf.tanh(self.fc_context+tf.expand_dims(self.up,1)),num_outputs=1,activation_fn=None)
            ct=tf.matmul(tf.transpose(tf.nn.softmax(st,1),[0,2,1]),self.context)    #batch_size*1*embed
            fusion2=tf.concat([input,tf.squeeze(ct,1)],-1)
            gt=tf.contrib.layers.fully_connected(fusion2,num_outputs=fusion2.get_shape().as_list()[-1],activation_fn=tf.sigmoid)
        return super(RnetLSTMselfAttnCell,self).__call__(gt*fusion2,state,scope)

class RnetLSTMLayer():
    def __init__(self,num_units):
        self.num_units=num_units

    def match(self,p,q,p_length,q_length):
        with tf.variable_scope('rnet-match-atte'):
            match_cell=RnetLSTMAttnCell(self.num_units,q)
            output,state=tf.nn.dynamic_rnn(match_cell,inputs=p,sequence_length=p_length,dtype=tf.float32)
        with tf.variable_scope('rnet-self-atte'):
            fw_cell=RnetLSTMselfAttnCell(self.num_units,output)
            bw_cell=RnetLSTMselfAttnCell(self.num_units,output)
            outputs,states=tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,inputs=output,sequence_length=p_length,dtype=tf.float32)
            output_final=tf.concat(outputs,-1)
            #states=tf.concat(states,-1)
        return output_final,states

class AttentionFlowMatchLayer(object):
    """
    Implements the Attention Flow layer,
    which computes Context-to-question Attention and question-to-context Attention
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.use_complex_s=True

    def match(self, passage_encodes, question_encodes, p_length, q_length):
        """
        Match the passage_encodes with question_encodes using Attention Flow Match algorithm
        """
        with tf.variable_scope('bidaf'):
            if self.use_complex_s:
                p_length=tf.shape(passage_encodes)[1]
                q_length=tf.shape(question_encodes)[1]
                hp=tf.tile(tf.expand_dims(passage_encodes,2),[1,1,q_length,1])
                hq=tf.tile(tf.expand_dims(question_encodes,1),[1,p_length,1,1])
                fusion=tf.concat([hp,hq,hp*hq],-1)
                sim_matrix=tf.squeeze(tf.contrib.layers.fully_connected(fusion,num_outputs=1,activation_fn=None),-1)
            else:
                sim_matrix = tf.matmul(passage_encodes, question_encodes, transpose_b=True)
            context2question_attn = tf.matmul(tf.nn.softmax(sim_matrix, -1), question_encodes)
            b = tf.nn.softmax(tf.expand_dims(tf.reduce_max(sim_matrix, 2), 1), -1)
            question2context_attn = tf.tile(tf.matmul(b, passage_encodes),
                                         [1, tf.shape(passage_encodes)[1], 1])
            concat_outputs = tf.concat([passage_encodes, context2question_attn,
                                        passage_encodes * context2question_attn,
                                        passage_encodes * question2context_attn], -1)
            return concat_outputs, None

class MRCLayer():
    def __init__(self,hidden_size):
        self.hidden_size=hidden_size
        self.t=3
        
    def match(self,p,q,p_length,q_length):
        p_align=[]
        #p_size=tf.shape(p).as_list()[1]
        for i in range(self.t): 
            with tf.variable_scope('match-align'+str(i)):
                hp=tf.contrib.layers.fully_connected(p,num_outputs=self.hidden_size,activation_fn=tf.nn.relu)
                hq=tf.contrib.layers.fully_connected(q,num_outputs=self.hidden_size,activation_fn=tf.nn.relu)
                s=tf.matmul(hp,tf.transpose(hq,[0,2,1]))
                v=tf.matmul(tf.nn.softmax(s,-1),q)
                fusion=tf.concat([p,v,p*v,p-v],-1)
                x_hat=tf.contrib.layers.fully_connected(fusion,num_outputs=2*self.hidden_size,activation_fn=tf.nn.relu)
                g=tf.contrib.layers.fully_connected(fusion,num_outputs=2*self.hidden_size,activation_fn=tf.sigmoid)
                o=g*x_hat+(1-g)*p
            with tf.variable_scope('self-atten'+str(i)):
                hp_o=tf.contrib.layers.fully_connected(o,num_outputs=self.hidden_size,activation_fn=tf.nn.relu)
                hq_o=tf.contrib.layers.fully_connected(o,num_outputs=self.hidden_size,activation_fn=tf.nn.relu)
                s_o=tf.matmul(hp_o,tf.transpose(hq_o,[0,2,1]))
                v_o=tf.matmul(tf.nn.softmax(s_o,-1),o)
                fusion_o=tf.concat([o,v_o,o*v_o,o-v_o],-1)
                x_hat_o=tf.contrib.layers.fully_connected(fusion_o,num_outputs=2*self.hidden_size,activation_fn=tf.nn.relu)
                g_o=tf.contrib.layers.fully_connected(fusion_o,num_outputs=2*self.hidden_size,activation_fn=tf.sigmoid)
                oo=g_o*x_hat_o+(1-g_o)*o
                p_align.append(oo)
            with tf.variable_scope('mrc-lstm'+str(i)):
                if i==2:
                    oo=tf.concat(p_align,-1)
                p,state=rnn('bi-lstm',oo,p_length,self.hidden_size)
        return p,state

