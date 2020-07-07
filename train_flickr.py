from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from metrics import calculate_map, calc_map, calc_precision_recall
from models import ImgModel, TxtModel, label_classifier, siamese_net, domain_adversarial
import os
from datetime import datetime
import matplotlib.pyplot as plt
import random

import numpy as np

# Set random seed
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
# random.seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('density', 20, 'density for construct graph')
flags.DEFINE_string('dataset', 'Flickr-25k', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
#img-model setting
flags.DEFINE_integer('img_gc1', 1000, 'Initial learning rate.')
# flags.DEFINE_integer('img_gc2', 100, 'Initial learning rate.')
#txt-model setting
flags.DEFINE_integer('txt_gc1', 300, 'Initial learning rate.')
# flags.DEFINE_integer('txt_gc2', 100, 'Initial learning rate.')
#out-put dim
flags.DEFINE_integer('hash_bit', 16, 'Initial learning rate.')
#learning rate
flags.DEFINE_float('lr_img', 0.001, 'Initial learning rate.')
flags.DEFINE_float('lr_txt', 0.001, 'Initial learning rate.')
flags.DEFINE_float('lr_domain', 0.001, 'Initial learning rate.')
#other setting
flags.DEFINE_integer('epochs', 50, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# all img_feats, labels, txt_vecs, adj are csr_matrix
load_time = time.time()
train_img_feats, train_txt_vecs, train_labels, test_img_feats, test_txt_vecs, test_labels = load_Flickr()
img_dim = train_img_feats.shape[1]
label_dim = train_labels.shape[1]
txt_dim = train_txt_vecs.shape[1]

# get img/txt adjacent matrix
img_adj = compute_img_adj(FLAGS.dataset, train_img_feats, test_img_feats, train_labels, FLAGS.density)
txt_adj = compute_txt_adj(FLAGS.dataset, train_txt_vecs, test_txt_vecs, train_labels, FLAGS.density)
img_adj = img_adj.toarray()
txt_adj = txt_adj.toarray()
print("load adjacent matrix finished in {}s!".format(time.time()-load_time))

# construct mask
preprocess_time = time.time()
train_num = train_txt_vecs.shape[0]
test_num = test_txt_vecs.shape[0]
total_num = train_num + test_num
shape = (total_num, total_num)
train_mask = np.zeros(shape)
train_mask[:train_num, :train_num] += 1
test_mask = np.ones(shape)
test_mask[:train_num, :train_num] -= 1
label_mask = np.zeros(total_num)
label_mask[:train_num] += 1

# concat training feature and test feature
img_feature = np.vstack((train_img_feats, test_img_feats))
#img_feature_tuple = sparse_to_tuple(img_feature)
#img_feature_array = img_feature.toarray()
print('img_feature:\n{}'.format(img_feature))

txt_feature = np.vstack((train_txt_vecs, test_txt_vecs))
# txt_feature_tuple = sparse_to_tuple(txt_feature)
# txt_feature_array = txt_feature.toarray()
print('txt_feature:\n{}'.format(txt_feature))

labels = np.vstack((train_labels, test_labels))
print('labels:\n{}'.format(labels))

# construct Laplacian matrix
img_support = preprocess_adj_dense(img_adj)
print('img_support:\n{}'.format(img_support))
txt_support = preprocess_adj_dense(txt_adj)
print('txt_support:\n{}'.format(txt_support))
print('preprocess finished in {}s!'.format(time.time()-preprocess_time))

# Define placeholders
placeholders = {
    # 'img_feature': tf.sparse_placeholder(tf.float32, shape=tf.constant(img_feature_tuple[2], dtype=tf.int64)),
    # 'txt_feature': tf.sparse_placeholder(tf.float32, shape=tf.constant(txt_feature_tuple[2], dtype=tf.int64)),
    # 'img_support': tf.sparse_placeholder(tf.float32),
    # 'txt_support': tf.sparse_placeholder(tf.float32),
    'img_feature': tf.placeholder(tf.float32, shape=img_feature.shape),
    'txt_feature': tf.placeholder(tf.float32, shape=txt_feature.shape),
    'img_support': tf.placeholder(tf.float32, shape=shape),
    'txt_support': tf.placeholder(tf.float32, shape=shape),
    'labels': tf.placeholder(tf.float32, shape=labels.shape),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}
# Create model
img_model = ImgModel(placeholders, img_input_dim=img_dim, img_output_dim=FLAGS.hash_bit,  logging=True)
txt_model = TxtModel(placeholders, txt_input_dim=txt_dim, txt_output_dim=FLAGS.hash_bit,  logging=True)
emb_v = img_model.outputs
emb_w = txt_model.outputs
hash_emb_v = tf.sign(emb_v)
hash_emb_w = tf.sign(emb_w)

#siamese network
# emb_v, hash_emb_v = siamese_net(emb_v)
# emb_w, hash_emb_w = siamese_net(emb_w)


# label predict loss
predict_v = label_classifier(emb_v, label_dim)
predict_w = label_classifier(emb_w, label_dim)
label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict_v) + \
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict_w)
label_loss = label_loss * label_mask
label_loss = tf.reduce_sum(label_loss) * 300

# # domain adversarial
# # flip_gradient_rate = tf.placeholder(tf.float32)
# emb_v_class = domain_adversarial(emb_v)
# emb_w_class = domain_adversarial(emb_w, reuse=True)
# all_emb_v = tf.concat([tf.ones([total_num, 1]), tf.zeros([total_num, 1])], 1)
# all_emb_w = tf.concat([tf.zeros([total_num, 1]), tf.ones([total_num, 1])], 1)
# # discriminator loss
# domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=emb_v_class, labels=all_emb_v) + \
#             tf.nn.softmax_cross_entropy_with_logits(logits=emb_w_class, labels=all_emb_w)
# domain_class_loss = tf.reduce_mean(domain_class_loss)
# # generator loss
# domain_class_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=emb_v_class, labels=all_emb_w) + \
#             tf.nn.softmax_cross_entropy_with_logits(logits=emb_w_class, labels=all_emb_v)
# domain_class_loss_fake = tf.reduce_mean(domain_class_loss_fake)
# alpha = 5000000
# domain_class_loss = domain_class_loss * alpha
# domain_class_loss_fake = domain_class_loss_fake * alpha
#
# # discriminator accuracy
# domain_img_class_acc = tf.equal(tf.greater(emb_v_class, 0.5), tf.greater(all_emb_v, 0.5))
# domain_txt_class_acc = tf.equal(tf.greater(0.5, emb_w_class), tf.greater(0.5, all_emb_w))
# domain_class_acc = tf.reduce_mean(tf.cast(tf.concat([domain_img_class_acc, domain_txt_class_acc], axis=0), tf.float32))

# img loss
theta_v = 1.0 / 2 * tf.matmul(emb_v, tf.transpose(emb_w))
likely_v = tf.multiply(tf.cast(img_adj, dtype=tf.float32), theta_v) - tf.log(1.0 + tf.exp(theta_v))
neg_likely_v = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask, dtype=tf.float32), likely_v))

#loss using hash code
hash_theta_v = 1.0 / 2 * tf.matmul(hash_emb_v, tf.transpose(hash_emb_w))
hash_likely_v = tf.multiply(tf.cast(img_adj, dtype=tf.float32), hash_theta_v) - tf.log(1.0 + tf.exp(hash_theta_v))
hash_neg_likely_v = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask, dtype=tf.float32), hash_likely_v))

# quantization loss
quantization_img = 2 * tf.nn.l2_loss(emb_v - hash_emb_v)
img_loss = neg_likely_v
#
# # txt loss
theta_w = 1.0 / 2 * tf.matmul(emb_w, tf.transpose(emb_v))
likely_w = tf.multiply(tf.cast(txt_adj, dtype=tf.float32), theta_w) - tf.log(1.0 + tf.exp(theta_w))
neg_likely_w = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask, dtype=tf.float32), likely_w))

# loss using hash code
hash_theta_w = 1.0 / 2 * tf.matmul(hash_emb_w, tf.transpose(hash_emb_v))
hash_likely_w = tf.multiply(tf.cast(txt_adj, dtype=tf.float32), hash_theta_w) - tf.log(1.0 + tf.exp(hash_theta_w))
hash_neg_likely_w = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask, dtype=tf.float32), hash_likely_w))

# quantization loss
quantization_txt = 2 * tf.nn.l2_loss(emb_w - hash_emb_w)
txt_loss = neg_likely_w

# total loss
# total_loss = neg_likely_v + neg_likely_w + quantization_img + 50 * quantization_txt
total_loss = neg_likely_w + neg_likely_v

lc_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="label_classifier")
classifier_vars = [var for var in lc_variables]
siamese_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="siamese_net")
siamese_vars = [var for var in siamese_variables]
t_vars = tf.trainable_variables()
dc_vars = [v for v in t_vars if 'dc_' in v.name]

# img optimizer
img_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)
img_vars = img_model.vars
img_op = img_optimizer.minimize(img_loss, var_list=img_vars)

# txt optimizer
txt_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_txt)
txt_vars = txt_model.vars
txt_op = txt_optimizer.minimize(txt_loss, var_list=txt_vars)

# label classifier optimizer
lc_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)
lc_op = lc_optimizer.minimize(label_loss, var_list=classifier_vars)

# domain optimizer
# dc_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_domain)
# dc_op = dc_optimizer.minimize(domain_class_loss, var_list=dc_vars)

# combine optimize
total_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)
total_op = total_optimizer.minimize(total_loss, var_list=img_vars+txt_vars)


print('optimization variables:')
print(img_vars)
print(txt_vars)
print(siamese_vars)

# Initialize session
sess = tf.Session()

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# plt loss curve
sim_loss_img = []
hash_sim_img = []
q_loss_img = []
sim_loss_txt = []
q_loss_txt = []
hash_sim_txt = []


# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()

    # Training step
    feed_dict = construct_feed_dict(img_feature, txt_feature, img_support, txt_support, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    feed_dict.update({placeholders['num_features_nonzero']: img_feature.shape})
    # op_img, loss_img, likely_loss_v, hash_likely_loss_v, q_img, temp_emb_v = sess.run([img_op, img_loss, neg_likely_v, hash_neg_likely_v, quantization_img, emb_v], feed_dict=feed_dict)
    # feed_dict.update({placeholders['num_features_nonzero']: txt_feature.shape})
    # op_txt, loss_txt, likely_loss_w, hash_likely_loss_w, q_txt, temp_emb_w = sess.run([txt_op, txt_loss, neg_likely_w, hash_neg_likely_w, quantization_txt, emb_w], feed_dict=feed_dict)

    # combine optimization
    op_ = sess.run(total_op, feed_dict=feed_dict)
    likely_loss_v, hash_likely_loss_v, q_img, temp_emb_v = sess.run([neg_likely_v, hash_neg_likely_v, quantization_img, emb_v], feed_dict=feed_dict)
    likely_loss_w, hash_likely_loss_w, q_txt, temp_emb_w = sess.run([neg_likely_w, hash_neg_likely_w, quantization_txt, emb_w], feed_dict=feed_dict)
    # loss_label = sess.run(label_loss, feed_dict=feed_dict)

    sim_loss_img.append(likely_loss_v)
    hash_sim_img.append(hash_likely_loss_v)
    q_loss_img.append(q_img)
    sim_loss_txt.append(likely_loss_w)
    q_loss_txt.append(q_txt)
    hash_sim_txt.append(hash_likely_loss_w)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "neg_v:", "{:.5f}".format(likely_loss_v), "quantization_img:", "{:.5f}".format(q_img),
          "neg_w:", "{:.5f}".format(likely_loss_w), "quantization_txt:", "{:.5f}".format(q_txt), "domain_loss:",
          "time=", "{:.5f}".format(time.time() - t))
    if epoch == FLAGS.epochs-1:
        print("img_emb:{}\n".format(temp_emb_v))
        print("txt_emb:{}\n".format(temp_emb_w))
print("Optimization Finished!")

# # Testing
feed_dict = construct_feed_dict(img_feature, txt_feature, img_support, txt_support, placeholders)
feed_dict.update({placeholders['dropout']: 0.})
feed_dict.update({placeholders['num_features_nonzero']: img_feature.shape})
img_feature_trans = sess.run(emb_v, feed_dict=feed_dict)
feed_dict.update({placeholders['num_features_nonzero']: txt_feature.shape})
txt_feature_trans = sess.run(emb_w, feed_dict=feed_dict)


test_img_feats_trans = img_feature_trans[train_num:]
test_txt_vecs_trans = txt_feature_trans[train_num:]
train_img_feats_trans = img_feature_trans[:train_num]
train_txt_vecs_trans = txt_feature_trans[:train_num]
# binary code
test_img_feats_trans_binary = np.sign(test_img_feats_trans)
test_txt_vecs_trans_binary = np.sign(test_txt_vecs_trans)
train_img_feats_trans_binary = np.sign(train_img_feats_trans)
train_txt_vecs_trans_binary = np.sign(train_txt_vecs_trans)

print("--------------binary map:---------------")
calculate_map(test_img_feats_trans_binary, test_txt_vecs_trans_binary, test_labels)
mapi2t = calc_map(test_img_feats_trans_binary, train_txt_vecs_trans_binary, test_labels, train_labels)
mapt2i = calc_map(test_txt_vecs_trans_binary, train_img_feats_trans_binary, test_labels, train_labels)
print ('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))
print("--------------continuous map------------")
calculate_map(test_img_feats_trans, test_txt_vecs_trans, test_labels)
c_mapi2t = calc_map(test_img_feats_trans, train_txt_vecs_trans, test_labels, train_labels)
c_mapt2i = calc_map(test_txt_vecs_trans, train_img_feats_trans, test_labels, train_labels)
print ('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (c_mapi2t, c_mapt2i))

precision_i2t, recall_i2t = calc_precision_recall(test_img_feats_trans_binary, train_txt_vecs_trans_binary, test_labels, train_labels)
precision_t2i, recall_t2i = calc_precision_recall(test_txt_vecs_trans_binary, train_img_feats_trans_binary, test_labels, train_labels)
print('precision_t2i')
print(precision_t2i)
print('recall_t2i')
print(recall_t2i)
print('precision_i2t')
print(precision_i2t)
print('recall_i2t')
print(recall_i2t)

# result_save_path = './data/'+FLAGS.dataset+'/'+'result/'+datetime.now().strftime("%d-%h-%m-%s")+'_'+str(FLAGS.hash_bit)+'_bit_result.pkl'
result_save_path = './data/'+FLAGS.dataset+'/'+'result/'+'epoch'+str(FLAGS.epochs)+'_simple'+'_'+str(FLAGS.hash_bit)+'_bit_result.pkl'
result = dict()
result['desc'] = 'no quantization'
result['bit'] = FLAGS.hash_bit
result['epoch'] = FLAGS.epochs
result['lr'] = FLAGS.lr_img
result['mapi2t'] = mapi2t
result['mapt2i'] = mapt2i
result['c_mapi2t'] = c_mapi2t
result['c_mapt2i'] = c_mapt2i
result['precision_i2t'] = precision_i2t
result['precision_t2i'] = precision_t2i
result['recall_i2t'] = recall_i2t
result['recall_t2i'] = recall_t2i
with open(result_save_path, 'w') as f:
    cPickle.dump(result, f)


# plot loss curve
img_plot_save_path = './figure/img_'+str(FLAGS.epochs)+'_'+str(FLAGS.lr_img)+'_'+str(FLAGS.hash_bit)+'.png'
txt_plot_save_path = './figure/txt_'+str(FLAGS.epochs)+'_'+str(FLAGS.lr_txt)+'_'+str(FLAGS.hash_bit)+'.png'
x_index = np.linspace(1, FLAGS.epochs, FLAGS.epochs)
plt.figure(num='img')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(x_index, sim_loss_img, 'r-s', label='sim_loss')
plt.plot(x_index, q_loss_img, 'g-s', label='quantization')
plt.plot(x_index, hash_sim_img, 'b-s', label='hash_sim_loss')
plt.legend(loc='upper right')
plt.savefig(img_plot_save_path)
plt.show()

plt.figure(num='txt')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(x_index, sim_loss_txt, 'r-s', label='sim_loss')
plt.plot(x_index, q_loss_txt, 'g-s', label='quantization')
plt.plot(x_index, hash_sim_txt, 'b-s', label='hash_sim_loss')
plt.legend(loc='upper right')
plt.savefig(txt_plot_save_path)
plt.show()
