import tensorflow as tf
import numpy as np
from dataset import load_dataset
from utils import chunker, revert_expected_value, expand, outer
from math import sqrt
import sys


class RBM(object):

    def __init__(self, num_hidden):

        self.num_hidden = num_hidden
        self.predict = None

    def init_model(self, num_visble, num_hidden):

        with tf.variable_scope("model_dim"):
            self.dim = (num_visble, num_hidden)
            self.num_visble = num_visble
            self.num_hidden = num_hidden

        with tf.variable_scope("model_input"):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.num_visble],
                                        name="input")

            self.mask = tf.placeholder(dtype=tf.float32, shape=[None, self.num_visble],
                                       name="mask")

        with tf.variable_scope("model"):
            normal_initializer = tf.random_normal_initializer(mean=0.0,
                                                              stddev=sqrt(2.0/(self.num_hidden + self.num_visble)),
                                                              seed=0, dtype=tf.float32)

            self.weights = tf.get_variable(name="weights", shape=self.dim, initializer=normal_initializer,
                                           dtype=tf.float32)
            self.hid_bias = tf.get_variable(name="hid_bias", shape=[self.num_hidden], initializer=normal_initializer,
                                            dtype=tf.float32)
            self.vis_bias = tf.get_variable(name="vis_bias", shape=[self.num_visble], initializer=normal_initializer,
                                            dtype=tf.float32)
            self.prev_gw = tf.get_variable(name="prev_grad_weights", shape=self.dim,
                                           dtype=tf.float32)
            self.prev_gbh = tf.get_variable(name="prev_grad_hid_bias", shape=[self.num_hidden],
                                            dtype=tf.float32)
            self.prev_gbv = tf.get_variable(name="prev_grad_vis_bias", shape=[self.num_visble],
                                            dtype=tf.float32)

    def sample_hidden(self, vis):

        hid_probs = tf.nn.sigmoid(tf.matmul(vis, self.weights) + self.hid_bias)
        hid_sample = tf.nn.relu(tf.sign(hid_probs - tf.random_uniform(tf.shape(hid_probs))))

        return hid_sample, hid_probs

    def sample_visible(self, hid, k=5):

        act = tf.nn.sigmoid(tf.matmul(hid, tf.transpose(self.weights)) + self.vis_bias)
        partition = tf.expand_dims(tf.reduce_sum(tf.reshape(act, (-1, self.num_visble // k, k)), 2), -1) * tf.ones((1, k))

        vis_probs = act / tf.reshape(partition, tf.shape(act))
        vis_sample = tf.nn.relu(tf.sign(vis_probs - tf.random_uniform(tf.shape(vis_probs))))

        return vis_sample, vis_probs

    def contrastive_divergence(self, vis_0, k):

        hid_0, hid_probs_0 = self.sample_hidden(vis_0)

        vis_k = vis_0
        hid_k = hid_0

        vis_probs_k = None
        hid_probs_k = None

        for i in range(k):
            vis_k, vis_probs_k = self.sample_visible(hid_k)
            hid_k, hid_probs_k = self.sample_hidden(vis_k)

        self.predict = vis_probs_k

        return [hid_probs_0, vis_k, hid_probs_k]

    @staticmethod
    def gradient(vis_0, hid_probs_0, vis_k, hid_probs_k, masks):

        v1h1_mask = outer(masks, hid_probs_0)

        gw = tf.reduce_mean(outer(vis_0, hid_probs_0) * v1h1_mask - outer(vis_k, hid_probs_k) * v1h1_mask, axis=0)
        gbv = tf.reduce_mean((vis_0 * masks) - (vis_k * masks), axis=0)
        gbh = tf.reduce_mean(hid_probs_0 - hid_probs_k, axis=0)

        return [gw, gbv, gbh]

    def update_weight(self, w_lr, v_lr, h_lr, w_reg, momentum, cdk):

        hid_probs_0, vis_k, hid_probs_k = self.contrastive_divergence(self.input, cdk)
        gw, gbv, gbh = self.gradient(self.input, hid_probs_0, vis_k, hid_probs_k, self.mask)

        update_w = tf.assign_add(self.weights, w_lr * (momentum * self.prev_gw + gw) - w_reg * self.weights)
        update_bh = tf.assign_add(self.hid_bias, h_lr * (momentum * self.prev_gbh + gbh))
        update_bv = tf.assign_add(self.vis_bias, v_lr * (momentum * self.prev_gbv + gbv))

        update_prev_gw = tf.assign(self.prev_gw, momentum * self.prev_gw + gw)
        update_prev_gbh = tf.assign(self.prev_gbh, momentum * self.prev_gbh + gbh)
        update_prev_gbv = tf.assign(self.prev_gbv, momentum * self.prev_gbv + gbv)

        optimizer = (update_w, update_bh, update_bv, update_prev_gw, update_prev_gbh, update_prev_gbv)
        return optimizer

    def fit(self, data_path, sep="\t", user_based=True, epochs=10,
            batch_size=10, w_lr=0.01, v_lr=0.01, h_lr=0.01, w_reg=0.001, momentum=0.9, cdk=1):

        all_users, all_movies, train_data, test_data = load_dataset(data_path, sep, user_based=user_based)

        self.init_model(len(all_movies) * 5, self.num_hidden)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        print("model created")

        for e in range(epochs):

            for batch_i, batch in enumerate(chunker(list(train_data.keys()), batch_size)):

                size = min(len(batch), batch_size)

                # create needed binary vectors
                user_one_hot_ratings = {}
                masks = {}
                for user_id in batch:

                    user_ratings = np.array([0.] * len(all_movies))
                    mask = [0] * (len(all_movies) * 5)

                    for item_id, rat in train_data[user_id]:
                        user_ratings[all_movies.index(item_id)] = rat
                        for _i in range(5):
                            mask[5 * all_movies.index(item_id) + _i] = 1

                    one_hot_ratings = expand(np.array([user_ratings])).astype('float32')
                    user_one_hot_ratings[user_id] = one_hot_ratings
                    masks[user_id] = mask

                ratings_batch = [user_one_hot_ratings[el] for el in batch]
                masks_batch = [masks[iid] for iid in batch]
                train_batch = np.array(ratings_batch).reshape(size, len(all_movies) * 5)

                optimizer = self.update_weight(w_lr, v_lr, h_lr, w_reg, momentum, cdk)

                _ = sess.run([optimizer], feed_dict={self.input: train_batch, self.mask: masks_batch})
                sys.stdout.write('.')
                sys.stdout.flush()

            # test step
            ratings = []
            predictions = []
            for batch in chunker(list(test_data.keys()), batch_size):
                size = min(len(batch), batch_size)

                # create needed binary vectors
                user_one_hot_ratings = {}
                masks = {}
                for user_id in batch:
                    user_ratings = [0.] * len(all_movies)
                    mask = [0] * (len(all_movies) * 5)
                    for item_id, rat in train_data[user_id]:
                        user_ratings[all_movies.index(item_id)] = rat
                        for _i in range(5):
                            mask[5 * all_movies.index(item_id) + _i] = 1
                    one_hot_ratings = expand(np.array([user_ratings])).astype('float32')
                    user_one_hot_ratings[user_id] = one_hot_ratings
                    masks[user_id] = mask

                positions = {profile_id: pos for pos, profile_id
                             in enumerate(batch)}
                ratings_batch = [user_one_hot_ratings[el] for el in batch]
                test_batch = np.array(ratings_batch).reshape(size, len(all_movies) * 5)
                predict = sess.run(self.predict, feed_dict={self.input: test_batch})
                user_preds = revert_expected_value(predict)

                for profile_id in batch:
                    test_movies = test_data[profile_id]
                    try:
                        for movie, rating in test_movies:
                            current_profile = user_preds[positions[profile_id]]
                            predicted = current_profile[all_movies.index(movie)]
                            rating = float(rating)
                            ratings.append(rating)
                            predictions.append(predicted)
                    except Exception:
                        pass

            vabs = np.vectorize(abs)
            distances = np.array(ratings) - np.array(predictions)

            mae = vabs(distances).mean()
            rmse = sqrt((distances ** 2).mean())
            print("\nepoch: {}, mae/rmse: {}/{}".format(e, mae, rmse))
