import tensorflow as tf
import numpy as np

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

    # in the paper section 4.1
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



    def _sentence_representation(self, sentence_matrix):
        '''
        :param sentence_matrix: shape: [batch_size, memory_size, sentence_size, embedding_size]
        :return:
        '''


        pass


    def model(self):
        self.query_input = tf.nn.embedding_lookup(self.B_embedding, self._queries)



if __name__ == "__main__":
    net = MemNet(6,6,6,6,1)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(net.query_input, feed_dict={net._queries: [[0,1,2,3,4,5]]}))