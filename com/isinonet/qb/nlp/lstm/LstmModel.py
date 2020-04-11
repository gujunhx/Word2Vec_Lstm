import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class BiLSTM(object):
    def __init__(self, config, wordEmbedding):

        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        l2Loss = tf.constant(0.0)

        with tf.name_scope("embedding"):
            self.w = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="w")
            self.embeddedWords = tf.nn.embedding_lookup(self.w, self.inputX)

        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSize):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                               output_keep_prob=self.dropoutKeepProb)
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                                                               output_keep_prob=self.dropoutKeepProb)
                    outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, self.embeddedWords, dtype=tf.float32,
                                                          scope="bi-lstm" + str(idx))
                    self.embeddedWords = tf.concat(outputs, 2)

        finalOutput = self.embeddedWords[:, 0, :]
        outputSize = config.model.hiddenSize[-1] * 2
        output = tf.reshape(finalOutput, [-1, outputSize])
        with tf.name_scope("output"):
            outputW = tf.get_variable("outputW",
                                      shape=[outputSize, config.numClasses])
                                      # initializer=tf.contrib.layers.xavier_initializer())
            outputB = tf.Variable(tf.constant(0.1,
                                              shape=[config.numClasses]),
                                  name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0),
                                           tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        with tf.name_scope("loss"):
            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss