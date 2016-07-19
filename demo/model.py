#import skipthoughts as skip
import tensorflow as tf
import numpy as np
from IPython import embed


'''
Modification of Memery Network for multiple choice QA
- Data consists of story, question and answers
- Each sentence is encoded into a vector using Skip-Thought Vectors
  (https://github.com/ryankiros/skip-thoughts)
- Encoded vectors are linearly projected by matrix T
'''

class MemN2N(object):
  def __init__(self, config, sess):
    self.init_std = config.init_std
    self.batch_size = config.batch_size
    self.nhop = config.nhop
    self.idim = config.idim     # input vector dimension (=skip-though vector)
    self.edim = config.edim     # encoding dimension
    self.nstory = config.nstory
    self.nanswer = config.nanswer
    self.embedding_method = 'word2vec'
    self.gamma = config.gamma

    self.story = tf.placeholder(tf.float32,
                                [None, self.nstory, self.idim],
                                name="story")
    self.query = tf.placeholder(tf.float32,
                                [None, 1, self.idim],
                                name="query")
    self.answer = tf.placeholder(tf.float32,
                                 [None, self.nanswer, self.idim],
                                 name="answer")
    self.target = tf.placeholder(tf.int64,
                                 [None],
                                 name="target")
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    self.lr = tf.Variable(config.init_lr)
    self.sess = sess

  def build_memory(self):
    self.global_step = tf.Variable(0, name="global_step")

    # Linear Projection Layeri
    self.T = tf.Variable(tf.random_uniform([self.idim, self.edim],
                                          minval=-0.1, maxval=0.1,
                                          name="projection")) # [idim, edim]

    self.params = [self.T]

    reshape = tf.reshape(self.story, [-1, self.idim]) # [batch_size * nstory, idim ]
    m = tf.matmul(reshape, self.T)   # [batch_size * nstory, edim]
    m = tf.reshape(m, [-1, self.nstory, self.edim]) # [batch_size, nstory, edim]
    self.m = m
    c = tf.matmul(reshape, self.T) # [batch_size * nstory , edim]
    c = tf.reshape(c, [-1, self.nstory, self.edim]) # [batch_size, nstory, edim]

    self.c = c
    reshape = tf.reshape(self.query, [-1, self.idim])
    u = tf.matmul(reshape, self.T)   # [batch_size * 1, edim]
    u = tf.reshape(u, [-1, 1, self.edim]) # [batch_size, 1, edim]

    reshape = tf.reshape(self.answer, [-1, self.idim])
    g = tf.matmul(reshape, self.T)  # [batch_size * nanswer, edim]
    g = tf.reshape(g, [-1, self.nanswer, self.edim]) # [batch_size, nanswer, edim]

    for h in xrange(self.nhop):
      p = tf.batch_matmul(m, u, adj_y=True)  # [batch_size, nstory. 1]
      p = tf.reshape(p, [-1, self.nstory]) # [batch_size, nstory]
      p = tf.nn.softmax(p, name='p')  # [batch_size, nstory]
      p_reshape = tf.reshape(p, [-1, self.nstory, 1]) # [batch_size, nstory, 1]
      o = tf.reduce_sum(p_reshape * c, 1) # [batch_size, edim]
      o = tf.reshape(o, [-1, 1, self.edim])
      u = tf.add(o, u) # [batch_size, 1, edim]

    logits = tf.batch_matmul(g, u, adj_y=True)  # [batch_size, nanswer, 1]
    logits = tf.reshape(logits, [-1, self.nanswer])
    self.logits = logits
    self.probs = tf.nn.softmax(logits)
    self.output = tf.argmax(self.probs, 1)

  def build_model(self, mode, embedding_method):
    self.build_memory()
    # self.skip_model = skip.load_model()
    self.skip_model = None
    self.reg_loss = tf.mul(tf.nn.l2_loss(self.T), self.gamma, name='regularization_loss')
    self.data_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.probs, self.target, name='data_loss')
    self.loss = tf.add(self.reg_loss, self.data_loss, name = 'total_loss')
    self.average_loss = tf.reduce_mean(self.loss)
    self.opt = tf.train.GradientDescentOptimizer(self.lr)
    self.correct_prediction = tf.equal(self.target, tf.argmax(self.probs,1))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    grads_and_vars = self.opt.compute_gradients(self.loss, self.params)
    cliped_grads_and_vars = [(tf.clip_by_norm(gv[0], 40), gv[1]) for gv in grads_and_vars]
    inc_op = self.global_step.assign_add(1)
    with tf.control_dependencies([inc_op]):
      self.apply_grad_op = self.opt.apply_gradients(cliped_grads_and_vars)

    self.saver = tf.train.Saver()

    # At Inference mode
    if mode == 'inference':
        if embedding_method == 'word2vec':
            self.saver.restore(self.sess, './MN_shortcut/model.ckpt')
        elif embedding_method == 'skip':
            print('Restoring model from ./MN_shortcut/skip_plot_40.ckpt')
            self.saver.restore(self.sess, './MN_shortcut/skip_plot_40.ckpt')
    else:
        tf.initialize_all_variables().run()

  def encode(self, inputs):
    story = skip.encode(self.skip_model, inputs.story)
    story = np.asarray(story, dtype=np.float32).reshape([self.batch_size, self.nstory, -1])
    query = skip.encode(self.skip_model, inputs.query)
    query = np.asarray(query, dtype=np.float32).reshape([self.batch_size, 1, -1])
    answer = skip.encode(self.skip_model, inputs.answer)
    answer = np.asarray(answer, dtype=np.float32).reshape([self.batch_size, self.nanswer, -1])
    target = np.asarray(inputs.target, dtype=np.int64).reshape([self.batch_size])
    return story, query, answer, target

  def train(self, inputs, save_flag=False, step = 0):
    story, query, answer, target = inputs
    _, loss = self.sess.run([self.apply_grad_op, self.average_loss],
                            feed_dict={
                              self.story: story,
                              self.query: query,
                              self.answer: answer,
                              self.target: target,
                              self.keep_prob: 0.5
                            })

    if save_flag == True:
        self.saver = tf.train.Saver()
        self.saver.save(self.sess,
                        './MN_shortcut/' +
                        str(self.embedding_method) +
                        '_plot_' + str(self.nstory) +
                        '_' + str(self.gamma) + '_' + str(step) + '.ckpt')
        print( "model shortcut saved...")
    return loss

  def inference(self, inputs):
    story, query, answer = inputs
    answer_index = self.sess.run([self.output], feed_dict={self.story: story, self.query: query, self.answer: answer, self.keep_prob: 1.0})
    return answer_index[0]

  def test(self, inputs):
    story, query, answer, target = inputs
    print( '========================================================')
    loss, accuracy, reg_loss, data_loss = self.sess.run([self.average_loss, self.accuracy, self.reg_loss, self.data_loss],
                                feed_dict={
                                  self.story: story,
                                  self.query: query,
                                  self.answer: answer,
                                  self.target: target,
                                  self.keep_prob: 1.0
                                  })
    print( '| reg loss >> ', reg_loss)
    print( '| data loss >> ', data_loss)
    print( '| edim >> ', self.edim)
    print( '| gamma >> ', self.gamma)
    print( '| Validation loss: ', loss)
    print( '[*] Accuracy >> ', accuracy)
    return accuracy
