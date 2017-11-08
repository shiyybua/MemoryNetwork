import tensorflow as tf
# q8
class MemNet:
    def __init__(self,memory_size, sentence_size, vocab_size, embedding_size, hop):
        self._memory_size = memory_size
        self._sentence_size = sentence_size
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._hop = hop

        self.C_embeddings = []
        self.A_embeddings = []

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
