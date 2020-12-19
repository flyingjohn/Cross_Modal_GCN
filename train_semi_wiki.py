from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from metrics import calculate_map, calc_map, calc_precision_recall
from models import ImgModel, TxtModel, label_classifier, siamese_net_single, domain_adversarial
from comparisonExp import PR_curve
import os
from datetime import datetime
import matplotlib.pyplot as plt
import random

import numpy as np

# Set random seed
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('threshold', 20, 'density for construct graph')
flags.DEFINE_float('percentage', 1.0, 'density for construct graph')
flags.DEFINE_string('dataset', 'wikipedia_dataset', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
#img-model setting
flags.DEFINE_integer('img_gc1', 500, 'Initial learning rate.')
#txt-model setting
flags.DEFINE_integer('txt_gc1', 500, 'Initial learning rate.')
#out-put dim
flags.DEFINE_integer('hash_bit', 64, 'Initial learning rate.')
flags.DEFINE_integer('siamese_bit', 128, 'Initial learning rate.')
flags.DEFINE_integer('siamese_bit_2', 64, 'Initial learning rate.')
#learning rate
flags.DEFINE_float('lr_img', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('lr_txt', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('lr_domain', 0.0001, 'Initial learning rate.') # defualt 0.0001
#other setting
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

'''
    Load data : all img_feats, labels, txt_vecs, adj are csr_matrix
'''
load_time = time.time()
train_img_feats, train_txt_vecs, train_labels, test_img_feats, test_txt_vecs, test_labels = load_wiki()
img_dim = train_img_feats.shape[1]
label_dim = train_labels.shape[1]
txt_dim = train_txt_vecs.shape[1]

'''
    get img/txt adjacent matrix
'''
img_adj = compute_img_adj_semi(FLAGS.dataset, train_img_feats, test_img_feats, train_labels, FLAGS.percentage, FLAGS.threshold)
txt_adj = compute_txt_adj_semi(FLAGS.dataset, train_txt_vecs, test_txt_vecs, train_labels, FLAGS.percentage, FLAGS.threshold)
img_adj = img_adj.toarray()
txt_adj = txt_adj.toarray()
common_adj = (img_adj + txt_adj) / 2
img_adj = common_adj
txt_adj = common_adj
print("load adjacent matrix finished in {}s!".format(time.time()-load_time))

'''
    construct mask
'''
preprocess_time = time.time()
train_num = train_txt_vecs.shape[0]
test_num = test_txt_vecs.shape[0]
labeled_num = int(train_num * FLAGS.percentage)
total_num = train_num + test_num
shape = (total_num, total_num)
train_mask = np.zeros(shape)
train_mask[:train_num, :train_num] += 1
train_mask_labeled = np.zeros(shape)
train_mask_labeled[:labeled_num, :labeled_num] += 1
test_mask = np.ones(shape)
test_mask[:train_num, :train_num] -= 1
label_mask = np.zeros(total_num)
label_mask[:train_num] += 1

'''
    concat training feature and test feature
'''
img_feature = np.vstack((train_img_feats, test_img_feats))
print('img_feature:\n{}'.format(img_feature))
txt_feature = np.vstack((train_txt_vecs, test_txt_vecs))
print('txt_feature:\n{}'.format(txt_feature))
labels = np.vstack((train_labels, test_labels))
print('labels:\n{}'.format(labels))

'''
    construct Laplacian matrix
'''
img_support = preprocess_adj_dense(img_adj)
print('img_support:\n{}'.format(img_support))
txt_support = preprocess_adj_dense(txt_adj)
print('txt_support:\n{}'.format(txt_support))
print('preprocess finished in {}s!'.format(time.time()-preprocess_time))

# Define placeholders
placeholders = {
    'img_feature': tf.placeholder(tf.float32, shape=img_feature.shape),
    'txt_feature': tf.placeholder(tf.float32, shape=txt_feature.shape),
    'img_support': tf.placeholder(tf.float32, shape=shape),
    'txt_support': tf.placeholder(tf.float32, shape=shape),
    'labels': tf.placeholder(tf.float32, shape=labels.shape),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

'''
    model construct
'''
# Create model
img_model = ImgModel(placeholders, img_input_dim=img_dim, img_output_dim=FLAGS.siamese_bit,  logging=True)
txt_model = TxtModel(placeholders, txt_input_dim=txt_dim, txt_output_dim=FLAGS.siamese_bit,  logging=True)
emb_v = img_model.outputs
emb_w = txt_model.outputs
# hash_emb_v = tf.sign(emb_v)
# hash_emb_w = tf.sign(emb_w)

#siamese network
emb_v, hash_emb_v = siamese_net_single(emb_v)
emb_w, hash_emb_w = siamese_net_single(emb_w, reuse=True)

'''
    --------------------loss------------------
'''

'''
    label predict loss
'''
# predict_v = label_classifier(emb_v, label_dim)
# predict_w = label_classifier(emb_w, label_dim)
# label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict_v) + \
#             tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predict_w)
# label_loss = label_loss * label_mask
# label_loss = tf.reduce_sum(label_loss) * 300


'''
    domain adversarial
'''
# flip_gradient_rate = tf.placeholder(tf.float32)
emb_v_class = domain_adversarial(emb_v)
emb_w_class = domain_adversarial(emb_w, reuse=True)
all_emb_v = tf.concat([tf.ones([total_num, 1]), tf.zeros([total_num, 1])], 1)
all_emb_w = tf.concat([tf.zeros([total_num, 1]), tf.ones([total_num, 1])], 1)
# discriminator loss
domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=emb_v_class, labels=all_emb_v) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=emb_w_class, labels=all_emb_w)
domain_class_loss = tf.reduce_mean(domain_class_loss)
# # generator loss
domain_class_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=emb_v_class, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=emb_w_class, labels=all_emb_v)
domain_class_loss_fake = tf.reduce_mean(domain_class_loss_fake)
# alpha = 1000000
alpha = 1000000 # default 1000000
# domain_class_loss = domain_class_loss * alpha # do not matter
domain_class_loss_fake = domain_class_loss_fake * alpha

# discriminator accuracy
domain_img_class_acc = tf.equal(tf.greater(emb_v_class, 0.5), tf.greater(all_emb_v, 0.5))
domain_txt_class_acc = tf.equal(tf.greater(0.5, emb_w_class), tf.greater(0.5, all_emb_w))
domain_class_acc = tf.reduce_mean(tf.cast(tf.concat([domain_img_class_acc, domain_txt_class_acc], axis=0), tf.float32))

'''
    similarity preserving loss
'''
# img loss
theta_v = 1.0 / 2 * tf.matmul(emb_v, tf.transpose(emb_w))
likely_v = tf.multiply(tf.cast(img_adj, dtype=tf.float32), theta_v) - tf.log(1.0 + tf.exp(theta_v))
neg_likely_v = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask_labeled, dtype=tf.float32), likely_v))

#loss using hash code
hash_theta_v = 1.0 / 2 * tf.matmul(hash_emb_v, tf.transpose(hash_emb_w))
hash_likely_v = tf.multiply(tf.cast(img_adj, dtype=tf.float32), hash_theta_v) - tf.log(1.0 + tf.exp(hash_theta_v))
hash_neg_likely_v = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask, dtype=tf.float32), hash_likely_v))

# # txt loss
theta_w = 1.0 / 2 * tf.matmul(emb_w, tf.transpose(emb_v))
likely_w = tf.multiply(tf.cast(txt_adj, dtype=tf.float32), theta_w) - tf.log(1.0 + tf.exp(theta_w))
neg_likely_w = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask_labeled, dtype=tf.float32), likely_w))

# loss using hash code
hash_theta_w = 1.0 / 2 * tf.matmul(hash_emb_w, tf.transpose(hash_emb_v))
hash_likely_w = tf.multiply(tf.cast(txt_adj, dtype=tf.float32), hash_theta_w) - tf.log(1.0 + tf.exp(hash_theta_w))
hash_neg_likely_w = -1 * tf.reduce_sum(tf.multiply(tf.cast(train_mask, dtype=tf.float32), hash_likely_w))

'''
    quantization loss
'''
quantization_img = 2 * tf.nn.l2_loss(emb_v - hash_emb_v)
quantization_txt = 2 * tf.nn.l2_loss(emb_w - hash_emb_w)

'''
    balance loss
'''
temp_ones = np.zeros(shape=(total_num, 1))
temp_ones[:train_num, 0] += 1
ones = tf.constant(temp_ones, dtype=tf.float32)
img_balance = tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(emb_w), ones), 2))
txt_balance = tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(emb_v), ones), 2))
balance_loss = 0.1 * img_balance + 0.1 * txt_balance

'''
    equality loss
'''
equality_loss = tf.nn.l2_loss(emb_v - emb_w)
equality_loss = 500 * equality_loss

'''
    total loss
'''
img_loss = neg_likely_v
txt_loss = neg_likely_w
# total_loss = neg_likely_v + neg_likely_w + quantization_img + 50 * quantization_txt
total_loss = txt_loss + img_loss + balance_loss

'''
    optimizer
'''
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
#
# # txt optimizer
txt_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_txt)
txt_vars = txt_model.vars
txt_op = txt_optimizer.minimize(txt_loss, var_list=txt_vars)

# # label classifier optimizer
# lc_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)
# lc_op = lc_optimizer.minimize(label_loss, var_list=classifier_vars)
#
# # # domain optimizer
dc_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_domain)
dc_op = dc_optimizer.minimize(domain_class_loss, var_list=dc_vars)

# combine optimize
total_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)
total_op = total_optimizer.minimize(total_loss, var_list=img_vars+txt_vars+siamese_vars)


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
    '''
        separately optimization
    '''
    # op_img, loss_img, likely_loss_v, hash_likely_loss_v, q_img, temp_emb_v = sess.run([img_op, img_loss, neg_likely_v, hash_neg_likely_v, quantization_img, emb_v], feed_dict=feed_dict)
    # feed_dict.update({placeholders['num_features_nonzero']: txt_feature.shape})
    # op_txt, loss_txt, likely_loss_w, hash_likely_loss_w, q_txt, temp_emb_w = sess.run([txt_op, txt_loss, neg_likely_w, hash_neg_likely_w, quantization_txt, emb_w], feed_dict=feed_dict)

    '''
        combine optimization with domain classifier
    '''
    domain_op_ = sess.run(dc_op, feed_dict=feed_dict)
    op_ = sess.run(total_op, feed_dict=feed_dict)
    likely_loss_v, hash_likely_loss_v, q_img, temp_emb_v, temp_balance_img = sess.run([neg_likely_v, hash_neg_likely_v, quantization_img, emb_v, img_balance], feed_dict=feed_dict)
    likely_loss_w, hash_likely_loss_w, q_txt, temp_emb_w, temp_balance_txt = sess.run([neg_likely_w, hash_neg_likely_w, quantization_txt, emb_w, txt_balance], feed_dict=feed_dict)
    domain_loss_fake_, domain_loss_ = sess.run([domain_class_loss_fake, domain_class_loss], feed_dict=feed_dict)
    domain_acc_ = sess.run(domain_class_acc, feed_dict=feed_dict)

    '''
        combine optimization without domain classifier
    '''
    # op_ = sess.run(total_op, feed_dict=feed_dict)
    # likely_loss_v, hash_likely_loss_v, q_img, temp_emb_v, temp_balance_img = sess.run([neg_likely_v, hash_neg_likely_v, quantization_img, emb_v, img_balance], feed_dict=feed_dict)
    # likely_loss_w, hash_likely_loss_w, q_txt, temp_emb_w, temp_balance_txt = sess.run([neg_likely_w, hash_neg_likely_w, quantization_txt, emb_w, txt_balance], feed_dict=feed_dict)

    sim_loss_img.append(likely_loss_v)
    hash_sim_img.append(hash_likely_loss_v)
    q_loss_img.append(q_img)
    sim_loss_txt.append(likely_loss_w)
    q_loss_txt.append(q_txt)
    hash_sim_txt.append(hash_likely_loss_w)

    '''
    evaluation per epoch
    '''
    if epoch % 50 == 0:
        test_img_feats_trans = temp_emb_v[train_num:]
        test_txt_vecs_trans = temp_emb_w[train_num:]
        train_img_feats_trans = temp_emb_v[:train_num]
        train_txt_vecs_trans = temp_emb_w[:train_num]
    #     # binary code
        test_img_feats_trans_binary = np.sign(test_img_feats_trans)
        test_txt_vecs_trans_binary = np.sign(test_txt_vecs_trans)
        train_img_feats_trans_binary = np.sign(train_img_feats_trans)
        train_txt_vecs_trans_binary = np.sign(train_txt_vecs_trans)
        calculate_map(test_img_feats_trans_binary, test_txt_vecs_trans_binary, test_labels)
        mapi2t = calc_map(test_img_feats_trans_binary, train_txt_vecs_trans_binary, test_labels, train_labels)
        mapt2i = calc_map(test_txt_vecs_trans_binary, train_img_feats_trans_binary, test_labels, train_labels)
        print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))
        calculate_map(test_img_feats_trans, test_txt_vecs_trans, test_labels)
        c_mapi2t = calc_map(test_img_feats_trans, train_txt_vecs_trans, test_labels, train_labels)
        c_mapt2i = calc_map(test_txt_vecs_trans, train_img_feats_trans, test_labels, train_labels)
        print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (c_mapi2t, c_mapt2i))

    '''
        print results with domain classifier
    '''
    print("Epoch:", '%04d' % (epoch + 1),
          "neg_v:", "{:.5f}".format(likely_loss_v),
          "quantization_img:", "{:.5f}".format(q_img),
          "balance_img:", "{:.5f}".format(temp_balance_img),
          "neg_w:", "{:.5f}".format(likely_loss_w),
          "quantization_txt:", "{:.5f}".format(q_txt),
          "balance_txt:", "{:.5f}".format(temp_balance_txt),
          "domain_loss:", '{:.5f}'.format(domain_loss_),
          "domain_loss_fake:", '{:.5f}'.format(domain_loss_fake_),
          "domain_acc:", '{:.5f}'.format(domain_acc_),
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

'''
    save embedding
'''
with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_img_binary.pkl', 'w') as f:
    cPickle.dump(train_img_feats_trans_binary, f)
with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_txt_binary.pkl', 'w') as f:
    cPickle.dump(train_txt_vecs_trans_binary, f)
with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_img_float.pkl', 'w') as f:
    cPickle.dump(train_img_feats_trans, f)
with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_txt_float.pkl', 'w') as f:
    cPickle.dump(train_txt_vecs_trans, f)

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

'''
    plot PR-curve
'''
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('text2img@' + str(FLAGS.hash_bit) + '-bit')
plt.xlabel('recall')
plt.ylabel('precision')
plt.axis([0, 1, 0.2, 1])
plt.grid(True)
PR_curve.plotRPcurve(precision_t2i, recall_t2i, 'r-s', 'GCN')
plt.subplot(122)
plt.title('img2text@' + str(FLAGS.hash_bit) + '-bit')
plt.xlabel('recall')
plt.ylabel('precision')
plt.axis([0, 1, 0.1, 1])
plt.grid(True)
PR_curve.plotRPcurve(precision_i2t, recall_i2t, 'r-s', 'GCN')
plt.show()

# result_save_path = './data/'+FLAGS.dataset+'/'+'result/'+datetime.now().strftime("%d-%h-%m-%s")+'_'+str(FLAGS.hash_bit)+'_bit_result.pkl'
result_save_path = './data/'+FLAGS.dataset+'/'+'result/'+'test2'+'_'+str(FLAGS.hash_bit)+'_bit_result.pkl'
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
