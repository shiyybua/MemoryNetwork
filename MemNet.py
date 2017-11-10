import tensorflow as tf
import numpy as np
# http://arxiv.org/abs/1503.08895v4
# q8

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.op_scope([t, stddev], name, "add_gradient_noise") as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class MemNet:
    def __init__(self, memory_size, sentence_size, vocab_size, embedding_size, hop, sess):
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
        self.PE = tf.constant(self._position_encoding(), name="encoding")

        self._embedding()
        self._build_inputs()
        self.model()
        self._sess = sess
        self._sess.run(tf.global_variables_initializer())

    def _build_inputs(self):
        self._stories = tf.placeholder(tf.int32, [None, self._memory_size, self._sentence_size], name="stories")
        self._queries = tf.placeholder(tf.int32, [None, self._sentence_size], name="queries")
        self._answers = tf.placeholder(tf.int32, [None, self._vocab_size], name="answers")
        self._lr = 0.1

    def _embedding(self):
        self.B_embedding = tf.get_variable("b_embedding", [self._vocab_size, self._embedding_size],
                                           initializer=tf.random_normal_initializer(stddev=0.1))
        # self.final_weight_matrix = tf.get_variable("projection", [self._embedding_size, self._vocab_size],
        #                                            initializer=tf.random_normal_initializer(stddev=0.1))
        self.A_embeddings.append(
            tf.get_variable('a_embedding', [self._vocab_size, self._embedding_size],
                            initializer=tf.random_normal_initializer(stddev=0.1)))
        for i in range(self._hop):
            #  index starts from 1
            self.C_embeddings.append(tf.get_variable('c_embedding{}'.format(i + 1), [self._vocab_size, self._embedding_size],
                                                     initializer=tf.random_normal_initializer(stddev=0.1)))

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
        return np.transpose(encode)

    # described in the paper section 4.1
    def _sentence_representation(self, sentence_matrix,special_matrix):
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
        u = [u]

        for i_hop in range(self._hop):
            if i_hop == 0:
                sentences_a = tf.nn.embedding_lookup(self.A_embeddings[i_hop], self._stories)
                m = self._sentence_representation(sentences_a, self.TA_matrix[i_hop])
            else:
                sentences_a = tf.nn.embedding_lookup(self.C_embeddings[i_hop-1], self._stories)
                m = self._sentence_representation(sentences_a, self.TC_matrix[i_hop])

            print u
            # hack to get around no reduce_dot
            u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
            print u_temp
            dotted = tf.reduce_sum(m * u_temp, 2)

            # Calculate probabilities
            probs = tf.nn.softmax(dotted)

            probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            with tf.variable_scope('hop_{}'.format(i_hop)):
                m_emb_C = tf.nn.embedding_lookup(self.C_embeddings[i_hop], self._stories)
            m_C = tf.reduce_sum(m_emb_C * self.PE, 2)

            c_temp = tf.transpose(m_C, [0, 2, 1])
            o_k = tf.reduce_sum(c_temp * probs_temp, 2)

            # Dont use projection layer for adj weight sharing
            # u_k = tf.matmul(u[-1], self.H) + o_k

            u_k = u[-1] + o_k


            u.append(u_k)

            # m = tf.transpose(m, [1, 0, 2])
            # # inner product and softmax, formula (1)
            # P = tf.nn.softmax(tf.reduce_sum(m * u, axis=2))
            # P = tf.transpose(P, [1,0])  # [batch_size, memory_size]
            # sentences_c = tf.nn.embedding_lookup(self.C_embeddings[i_hop], self._stories)
            # c = self._sentence_representation(sentences_c)
            # # formula (2)
            # c = tf.transpose(c, [2, 0, 1])
            # o = tf.reduce_sum(P * c, axis=2)
            # o = tf.transpose(o, [1, 0])
            # # the current hop output is the input of next hop.
            # u = o + u

        # formula (3) without softmax
        projection_layer = tf.matmul(u_k, tf.transpose(self.C_embeddings[-1], [1,0]))

        self.predict = tf.argmax(projection_layer, 1, name="predict_op")
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=projection_layer, labels=tf.cast(self._answers, tf.float32))
        self.loss = tf.reduce_sum(losses)

        # self._opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        # grads_and_vars = self._opt.compute_gradients(self.loss)
        # grads_and_vars = [(tf.clip_by_norm(g, 40.0), v) for g, v in grads_and_vars]
        # grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
        # # print grads_and_vars, '111' * 10
        # nil_grads_and_vars = []
        # for g, v in grads_and_vars:
        #     nil_grads_and_vars.append((g, v))
        # train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")
        #
        # self.optimizer = train_op
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def train(self, stories, queries, answers):
        feed_dict = {self._stories: stories, self._queries: queries, self._answers: answers}
        loss, _ = self._sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
        return loss

    def prediction(self, stories, queries):
        feed_dict = {self._stories: stories, self._queries: queries}
        return self._sess.run(self.predict, feed_dict=feed_dict)


if __name__ == "__main__":
    with tf.Session() as sess:
        net = MemNet(3, 6, 8, 10, 1,sess)
        sess.run(tf.global_variables_initializer())
        # print(sess.run(net.loss, feed_dict={net._queries: [[0,1,2,3,4,5],[0,1,2,3,4,5]]}).shape)
        # print(sess.run(net.sentences, feed_dict={net._stories: [[[0,1,2,3,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5]]]}).shape)