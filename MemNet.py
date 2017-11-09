import tensorflow as tf
import numpy as np
# http://arxiv.org/abs/1503.08895v4
# q8
class MemNet:
    def __init__(self, memory_size, sentence_size, vocab_size, embedding_size, hop):
        self._memory_size = memory_size
        self._sentence_size = sentence_size
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._hop = hop

        self.C_embeddings = []
        self.A_embeddings = []
        # special matrix for temporal encoding
        self.TA_matrix = []
        self.TC_matrix = []
        self.PE = self._position_encoding()

        self._embedding()
        self._build_inputs()
        self.model()

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._lr = tf.placeholder(tf.float32, [], name="learning_rate")

    def _embedding(self):
        for i in range(self._hop):
            #  index starts from 1
            self.C_embeddings.append(tf.get_variable('c_embedding{}'.format(i + 1), [self._vocab_size, self._embedding_size],
                            initializer=tf.contrib.layers.xavier_initializer()))
            self.A_embeddings.append(tf.get_variable('a_embedding{}'.format(i + 1), [self._vocab_size, self._embedding_size],
                            initializer=tf.contrib.layers.xavier_initializer()))
            self.B_embedding = tf.get_variable("b_embedding", [self._vocab_size, self._embedding_size],
                            initializer=tf.contrib.layers.xavier_initializer())

            self.TA_matrix.append(tf.get_variable('ta_matrix{}'.format(i + 1), [self._memory_size, self._embedding_size],
                            initializer=tf.contrib.layers.xavier_initializer()))
            self.TC_matrix.append(tf.get_variable('tc_matrix{}'.format(i + 1), [self._memory_size, self._embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer()))

    # described in the paper section 4.1
    def _position_encoding(self):
        '''
        :return: shape: [sentence_size, embedding_size]
        '''
        J = self._sentence_size
        d = self._embedding_size
        # initializing like this shape is for convenient traversal
        encode = np.ones([self._embedding_size, self._sentence_size], dtype=np.float32)
        for k in range(d):
            for j in range(J):
                encode[k][j] = (1 - 1.0*j/J) - (1.0*k/d)*(1-2.0*j/J)
        return tf.convert_to_tensor(np.transpose(encode))

    # described in the paper section 4.1
    def _sentence_representation(self, sentence_matrix, special_matrix):
        '''
        process sentence representation for single hop.
        :param sentence_matrix: shape: [batch_size, memory_size, sentence_size, embedding_size]
        :return: sentence_matrix(processed), shape: [batch_size, sentence_size, embedding_size]
        '''

        # use position encoding
        sentence_matrix *= self.PE
        # Temporal Encoding
        sentence_matrix = tf.reduce_sum(sentence_matrix, axis=2)
        sentence_matrix = tf.add(sentence_matrix, special_matrix)
        return sentence_matrix

    def model(self):
        # the variable names are the same as that described in the paper.
        # the shape after embedding_lookup is extended one dimension, e.g.(1, 3, 6)=>(1, 3, 6, 10), where 10 is embedding size.
        u = tf.nn.embedding_lookup(self.B_embedding, self._queries)
        # The shape of self.PE is [sentence_size, embedding_size], so the last two dimensions of query_input
        # should be [sentence_size, embedding_size] as well.
        u *= self.PE
        # convert word embedding to sentence embedding by simply adding.
        u = tf.reduce_sum(u, axis=1)
        for i_hop in range(self._hop):
            sentences = tf.nn.embedding_lookup(self.A_embeddings[i_hop], self._stories)
            m = self._sentence_representation(sentences, self.TA_matrix[i_hop])
            # inner product and softmax, formula (1)
            P = tf.nn.softmax(tf.reduce_sum(m * u, axis=2))
            c = self._sentence_representation(sentences, self.TC_matrix[i_hop])
            # formula (2)
            c = tf.transpose(c, [0, 2, 1])
            o = tf.reduce_sum(P * c, axis=2)

            print (o)

# (?, 3)
if __name__ == "__main__":
    net = MemNet(3,6,8,10,1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # print(sess.run(net.u, feed_dict={net._queries: [[0,1,2,3,4,5],[0,1,2,3,4,5]]}).shape)
        # print(sess.run(net.sentences, feed_dict={net._stories: [[[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]]}).shape)