from layers import *
from metrics import *
import tensorflow.contrib.slim as slim
from flip_gradient import flip_gradient

flags = tf.app.flags
FLAGS = flags.FLAGS


def label_classifier(inputs, label_num):
    with tf.variable_scope("label_classifier", reuse=True):
        weights = glorot([FLAGS.hash_bit, label_num], name='lc_weights')
        bias = zeros([label_num], name='lc_bias')
        predict = tf.matmul(inputs, weights) + bias
    return predict


def siamese_net(inputs, reuse=False):
    with tf.variable_scope("siamese_net", reuse=reuse):
        weights = glorot([FLAGS.siamese_bit, FLAGS.hash_bit], name='sia_weights')
        bias = zeros([FLAGS.hash_bit], name='sia_bias')
        representation = tf.nn.tanh(tf.matmul(inputs, weights) + bias)
        hash_represetation = tf.sign(representation)
    return representation, hash_represetation


def domain_adversarial(inputs, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = slim.fully_connected(inputs, FLAGS.hash_bit/2, scope='dc_fc_0')
        net = slim.fully_connected(net, FLAGS.hash_bit/4, scope='dc_fc_1')
        net = slim.fully_connected(net, 2, scope='dc_fc_2')
    return net


def domain_classifier(E, l, is_training=True, reuse=False):
    """
        this function is from ACMR (ACM MM 2017)
    """
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        E = flip_gradient(E, l)
        net = slim.fully_connected(E, FLAGS.hash_bit/2, scope='domain_fc_0')
        net = slim.fully_connected(net, FLAGS.hash_bit/4, scope='domain_fc_1')
        net = slim.fully_connected(net, 2, scope='domain_fc_2')
    return net


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.regular_loss = 0

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            # invoke __call__()
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = [var for var in variables]

        # Build metrics
        self._regular_loss()

    def _regular_loss(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class ImgModel(Model):
    """
        Img model
    """
    def __init__(self, placeholders, img_input_dim, img_output_dim, **kwargs):
        super(ImgModel, self).__init__(**kwargs)

        self.inputs = placeholders['img_feature']
        self.img_support = placeholders['img_support']
        self.input_dim = img_input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = img_output_dim
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)

        self.build()

    def _regular_loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.regular_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    def _build(self):

        # self.layers.append(Dense(input_dim=self.input_dim,
        #                          output_dim=FLAGS.img_fc1,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.tanh,
        #                          dropout=False,
        #                          logging=self.logging))

        # self.layers.append(Dense(input_dim=FLAGS.img_fc1,
        #                          output_dim=self.output_dim,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.tanh,
        #                          dropout=False,
        #                          logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.img_gc1,
                                            placeholders=self.placeholders,
                                            support=self.img_support,
                                            act=tf.nn.tanh,
                                            dropout=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.img_gc1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.img_support,
                                            act=tf.nn.tanh,
                                            dropout=False,
                                            logging=self.logging))


class TxtModel(Model):
    """
        Txt model
    """
    def __init__(self, placeholders, txt_input_dim, txt_output_dim, **kwargs):
        super(TxtModel, self).__init__(**kwargs)

        self.inputs = placeholders['txt_feature']
        self.input_dim = txt_input_dim
        self.txt_support = placeholders['txt_support']
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = txt_output_dim
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_txt)

        self.build()

    def interp_block(self, text_input, level, dimTxt):
        shape = [1, 1, 5 * level, 1]
        stride = [1, 1, 5 * level, 1]
        batch_size = text_input.get_shape()[0]
        dim = text_input.get_shape()[1]
        height = 1
        channel = 1
        reshaped_text = tf.reshape(text_input, [batch_size, height, dim, channel])
        prev_layer = tf.nn.avg_pool(reshaped_text, ksize=shape, strides=stride, padding='VALID')
        W_fc1 = tf.random_normal([1, 1, 1, 1], stddev=1.0) * 0.01
        fc1W = tf.Variable(W_fc1)
        prev_layer = tf.nn.conv2d(prev_layer, fc1W, strides=[1, 1, 1, 1], padding='VALID')
        prev_layer = tf.squeeze(prev_layer)
        return prev_layer

    def MultiScaleTxt(self, text_input, dimTxt):
        interp_block1 = self.interp_block(text_input, 10, dimTxt)
        interp_block2 = self.interp_block(text_input, 6, dimTxt)
        interp_block3 = self.interp_block(text_input, 3, dimTxt)
        interp_block6 = self.interp_block(text_input, 2, dimTxt)
        interp_block10 = self.interp_block(text_input, 1, dimTxt)
        output = tf.concat([text_input,
                            interp_block10,
                            interp_block6,
                            interp_block3,
                            interp_block2,
                            interp_block1], axis=1)
        return output

    def _regular_loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.regular_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            # invoke __call__()
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = [var for var in variables]

        # Build metrics
        self._regular_loss()

    def _build(self):

        # self.layers.append(Dense(input_dim=self.input_dim,
        #                          output_dim=FLAGS.txt_fc1,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.tanh,
        #                          dropout=False,
        #                          logging=self.logging))

        # self.layers.append(Dense(input_dim=FLAGS.txt_fc1,
        #                          output_dim=self.output_dim,
        #                          placeholders=self.placeholders,
        #                          act=tf.nn.tanh,
        #                          dropout=False,
        #                          logging=self.logging))
        ##self.txtFusion = self.MultiScaleTxt(self.inputs, self.input_dim)
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.txt_gc1,
                                            placeholders=self.placeholders,
                                            support=self.txt_support,
                                            act=tf.nn.tanh,
                                            dropout=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.txt_gc1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            support=self.txt_support,
                                            act=tf.nn.tanh,
                                            dropout=False,
                                            logging=self.logging))


