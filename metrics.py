import tensorflow as tf
import numpy as np
import time

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


def calc_precision_recall(qB, rB, query_L, retrieval_L, eps=2.2204e-16):
    """
    calculate precision recall
    Input:
        query_L: 0-1 label matrix (numQuery * numLabel) for query set.
        retrieval_L: 0-1 label matrix (numQuery * numLabel) for retrieval set.
        qB: compressed binary code for query set.
        rB: compressed binary code for retrieval set.
    Output:
        Pre: maxR-dims vector. Precision within different hamming radius.
        Rec: maxR-dims vector. Recall within different hamming radius.
    """
    Wtrue = (np.dot(query_L, np.transpose(retrieval_L)) > 0).astype(int)
    Dhamm = calc_hammingDist(qB, rB)

    maxHamm = int(np.max(Dhamm))
    totalGoodPairs = np.sum(Wtrue)

    precision = np.zeros((maxHamm+1, 1))
    recall = np.zeros((maxHamm+1, 1))
    for i in range(maxHamm+1):
        j = (Dhamm <= (i + 0.001)).astype(int)
        retrievalPairs = np.sum(j)
        retrievalGoodPairs = np.sum(np.multiply(Wtrue, j))
        print(retrievalGoodPairs, retrievalPairs)
        precision[i] = retrievalGoodPairs * 1.0 / (retrievalPairs + eps)
        recall[i] = retrievalGoodPairs * 1.0 / totalGoodPairs

    return precision, recall


def calc_map(qB, rB, query_L, retrieval_L):
    """from deep cross modal hashing"""
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    for iter in xrange(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map


def calculate_map(test_img_feats_trans, test_txt_vecs_trans, test_labels):
    """Calculate top-50 mAP"""
    start = time.time()
    avg_precs = []
    all_precs = []
    all_k = [50]
    for k in all_k:
        for i in range(len(test_txt_vecs_trans)):
            query_label = test_labels[i]

            # distances and sort by distances
            wv = test_txt_vecs_trans[i]
            #dists = calc_l2_norm(wv, test_img_feats_trans)
            dists = calc_hammingDist(wv, test_img_feats_trans)
            sorted_idx = np.argsort(dists)

            # for each k do top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0: topk]
                # if query_label != test_labels[top_k[-1]]:
                #     continue
                if np.any(query_label != test_labels[top_k[-1]]):
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if np.all(retrieved_label == query_label):
                        hits += 1
                precs.append(float(hits) / float(topk))
            if len(precs) == 0:
                precs.append(0)
            avg_precs.append(np.average(precs))
        mean_avg_prec = np.mean(avg_precs)
        all_precs.append(mean_avg_prec)
    print('[Eval - txt2img] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))

    avg_precs = []
    all_precs = []
    all_k = [50]
    for k in all_k:
        for i in range(len(test_img_feats_trans)):
            query_img_feat = test_img_feats_trans[i]
            ground_truth_label = test_labels[i]
            # calculate distance and sort
            #dists = calc_l2_norm(query_img_feat, test_txt_vecs_trans)
            dists = calc_hammingDist(query_img_feat, test_txt_vecs_trans)
            sorted_idx = np.argsort(dists)

            # for each k in top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0: topk]
                if np.any(ground_truth_label != test_labels[top_k[-1]]):
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if np.all(ground_truth_label == retrieved_label):
                        hits += 1
                precs.append(float(hits) / float(topk))
            if len(precs) == 0:
                precs.append(0)
            avg_precs.append(np.average(precs))
        mean_avg_prec = np.mean(avg_precs)
        all_precs.append(mean_avg_prec)
    print('[Eval - img2txt] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))


def calc_hammingDist(request, retrieval_all):
    K = retrieval_all.shape[1]
    distH = 0.5 * (K - np.dot(request, retrieval_all.transpose()))
    return distH

def calc_l2_norm(request, retrieval_all):
    diffs = retrieval_all - request
    dists = np.linalg.norm(diffs, axis=1)
    return dists