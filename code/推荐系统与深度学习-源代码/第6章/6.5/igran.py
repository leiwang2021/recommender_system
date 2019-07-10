# refer to https://github.com/geek-ai/irgan/blob/master/item_recommendation/cf_gan.py
# reference:

# @inproceedings{wang2017irgan,
#   title={Irgan: A minimax game for unifying generative and discriminative information retrieval models},
#   author={Wang, Jun and Yu, Lantao and Zhang, Weinan and Gong, Yu and Xu, Yinghui and Wang, Benyou and Zhang, Peng and Zhang, Dell},
#   booktitle={Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval},
#   pages={515--524},
#   year={2017},
#   organization={ACM}
# }

import tensorflow as tf
import numpy as np
import random
# import utils as ut
import time
import multiprocessing


EMB_DIM = 5
USER_NUM = 943
ITEM_NUM = 1683
BATCH_SIZE = 16
INIT_DELTA = 0.05

all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'
#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

all_users = list(user_pos_train.keys())
all_users.sort()



def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


def simple_test(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    batch_size = 128
    test_users = list(user_pos_test.keys())
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        if index + batch_size <= test_user_num:
            user_batch = test_users[index:index + batch_size]
        else:
            user_batch = test_users[index:]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_for_d(sess, model, filename):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]

        rating = sess.run(model.all_rating, {model.u: [u]})
        rating = np.array(rating[0]) / 0.2  # Temperature!!!!!!!!!!!!!!!!!
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg = np.random.choice(np.arange(ITEM_NUM), size=len(pos), p=prob)
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


# Get batch data from training set
def get_batch_data(file,  size):  # 1,5->1,2,3,4,5
    user,item,label = [],[],[]
    with open(DIS_TRAIN_FILE) as f:
        for line in f:
            tokens= line.strip().split()
            user.append(int(tokens[0]))
            user.append(int(tokens[0]))
            item.append(int(tokens[1]))
            item.append(int(tokens[2]))
            label.append(1.)
            label.append(0.)
            if len(user) == size:
                yield user, item, label
                user,item,label = [],[],[]



# model definination                
class MF(object):
    def __init__(self, itemNum, userNum, emb_dim, lamda, initdelta=0.05, learning_rate=0.05):
        with tf.variable_scope('generator'):
            self.user_embeddings = tf.Variable(
                tf.random_uniform([userNum, emb_dim], minval=-initdelta, maxval=initdelta,
                                  dtype=tf.float32))
            self.item_embeddings = tf.Variable(
                tf.random_uniform([itemNum, emb_dim], minval=-initdelta, maxval=initdelta,
                                  dtype=tf.float32))
            self.item_bias = tf.Variable(tf.zeros([itemNum]))

        self.params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.reward = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
class DIS(MF):
    def __init__(self, itemNum, userNum, emb_dim, lamda, initdelta=0.05, learning_rate=0.05):
        super(DIS, self).__init__(itemNum, userNum, emb_dim, lamda, initdelta=0.05, learning_rate=0.05)

        self.label = tf.placeholder(tf.float32)

        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                logits=self.pre_logits) + lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )

        self.update = self.optimizer.minimize(self.loss, var_list=self.params)


        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           1) + self.i_bias
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

class GEN(MF):
    def __init__(self, itemNum, userNum, emb_dim, lamda,  initdelta=0.05, learning_rate=0.05):
        super(GEN, self).__init__(itemNum, userNum, emb_dim, lamda,  initdelta=0.05, learning_rate=0.05)


        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.i_prob = tf.gather(
            tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]),
            self.i)
        self.loss = -tf.reduce_mean(tf.log(self.i_prob) * self.reward) + lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias))
        self.update=self.optimizer.minimize(self.loss, var_list=self.params)


        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias
  
                                        
if __name__ == '__main__':

    g1 = tf.Graph()
    g2 = tf.Graph()
    sess1 = tf.InteractiveSession(graph=g1)        
    sess2 = tf.InteractiveSession(graph=g2)
    with g1.as_default():
        generator=GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.0 / BATCH_SIZE, initdelta=INIT_DELTA,
                    learning_rate=0.01)
        sess1.run(tf.global_variables_initializer())
    with g2.as_default():
        discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1 / BATCH_SIZE, initdelta=INIT_DELTA,
                        learning_rate=0.01)    
        sess2.run(tf.global_variables_initializer())


    for epoch in range(100):
        if epoch >= 0:
            for d_epoch in range(10):
                if d_epoch%5==0:
                    generate_for_d(sess1, generator, DIS_TRAIN_FILE)
                for input_user, input_item, input_label in    get_batch_data(DIS_TRAIN_FILE,BATCH_SIZE):
                    _ = sess2.run(discriminator.update,
                             feed_dict={discriminator.u: input_user, discriminator.i: input_item,
                                        discriminator.label: input_label})

                print("dis: "+str(simple_test(sess2, discriminator)))
        for g_epoch in range(5):  # 50
            for u in user_pos_train:
                sample_lambda = 0.2
                pos = user_pos_train[u]
                rating = sess1.run(generator.all_logits, {generator.u: u})
                exp_rating = np.exp(rating)
                prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta
                pn = (1 - sample_lambda) * prob
                pn[pos] += sample_lambda * 1.0 / len(pos)
                # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta
                sample = np.random.choice(np.arange(ITEM_NUM), 2 * len(pos), p=pn)
                # Get reward and adapt it with importance sampling
                reward = sess2.run(discriminator.reward, {discriminator.u: u, discriminator.i: sample})
                reward = reward * prob[sample] / pn[sample]
                _ = sess1.run(generator.update,{generator.u: u, generator.i: sample, generator.reward: reward})

            print("gen: "+str(simple_test(sess1, generator)))