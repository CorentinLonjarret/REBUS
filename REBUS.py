# Implementation of the REBUS model
import pandas as pd
import scipy.sparse as sp
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import dataset
import sys
import time
import math
import os
import commons
import json


class REBUS:
    def __init__(self, dataset, args):
        print('In class REBUS')
        self.dataset = dataset
        self.args = args

        if not os.path.exists(os.path.join("tmp", "REBUS"+"_"+str(self.args.user_min)+"_"+str(self.args.item_min), self.dataset.data_name)):
            os.makedirs(os.path.join("tmp", "REBUS"+"_"+str(self.args.user_min)+"_"+str(self.args.item_min), self.dataset.data_name))
        random_id = str(random.randint(0, 1000000))
        self.path_saver_parameters = os.path.join("tmp", "REBUS"+"_"+str(self.args.user_min)+"_"+str(self.args.item_min), self.dataset.data_name, random_id+".ckpt")
        args_dict = vars(args)
        with open(os.path.join("tmp", "REBUS"+"_"+str(self.args.user_min)+"_"+str(self.args.item_min), self.dataset.data_name, random_id+"-args.json"), 'w') as fp:
            json.dump(args_dict, fp)

        # Use a training batch to figure out feature dimensionality
        _, _, _, prev_items, _, _, _, _ = self.dataset.generate_train_shuffled_batch_sp_with_prev_items()
        self.feature_dim = prev_items.shape[1]
        print('Feature dimension = ' + str(self.feature_dim))

    def initialize_parameters(self):
        var_emb_items = tf.get_variable('emb_items', [self.dataset.nb_items, self.args.num_dims],
                                        # initializer=tf.random_uniform_initializer( -self.args.init_mean, self.args.init_mean))
                                        initializer=tf.contrib.layers.xavier_initializer(seed=self.args.seed))
        # Add a null vector for embedding_lookup, the null vector is equal tu nb_items
        var_emb_items = tf.concat((var_emb_items[0:self.dataset.nb_items, :], tf.zeros(shape=[1, self.args.num_dims])), 0)

        var_bias_items = tf.get_variable('bias_items', [self.dataset.nb_items, 1],
                                         # initializer=tf.random_uniform_initializer( -self.args.init_mean, self.args.init_mean))
                                         initializer=tf.zeros_initializer())
        # Add a null vector for embedding_lookup, the null vector is equal tu nb_items
        var_bias_items = tf.concat((var_bias_items[0:self.dataset.nb_items, :], tf.zeros(shape=[1, 1])), 0)

        parameters = {
            "var_emb_items": var_emb_items,
            "var_bias_items": var_bias_items
        }

        return parameters

    def create_constant(self):
        self.alpha = tf.constant(self.args.alpha, dtype=tf.float32)
        self.gamma_long = tf.constant(self.args.gamma, dtype=tf.float32)
        self.gamma_short = tf.constant(1-self.args.gamma, dtype=tf.float32)
        self.ones_items = tf.constant(np.repeat([1], self.feature_dim), shape=[self.feature_dim, 1], dtype=tf.float32)

    def create_placeholder(self):
        pl_user_list = tf.placeholder(tf.int32, shape=[None], name='user_list')  # List of all users
        pl_prev_items = tf.placeholder(tf.int32, shape=[None, 1], name='prev_items')
        pl_list_fsub_items_id = tf.placeholder(tf.int32, shape=[None, self.args.L], name='list_fsub_items_id')
        pl_list_fsub_items_values = tf.placeholder(tf.float32, shape=[None, self.args.L], name='list_fsub_items_values')
        pl_list_prev_items_pos = tf.placeholder(tf.int32, shape=[None, self.args.max_lens], name='list_prev_item_pos')
        pl_list_prev_items_neg = tf.placeholder(tf.int32, shape=[None, self.args.max_lens], name='list_prev_item_neg')
        pl_pos = tf.placeholder(tf.int32, shape=[None], name='pos')
        pl_neg = tf.placeholder(tf.int32, shape=[None], name='neg')  # Shape for the neg sparseTensor

        placeholders = {
            'pl_user_list': pl_user_list,
            'pl_prev_items': pl_prev_items,
            'pl_list_fsub_items_id': pl_list_fsub_items_id,
            'pl_list_fsub_items_values': pl_list_fsub_items_values,
            'pl_list_prev_items_pos': pl_list_prev_items_pos,
            'pl_list_prev_items_neg': pl_list_prev_items_neg,
            'pl_pos': pl_pos,
            'pl_neg': pl_neg,
        }

        return placeholders

    def create_feed_dict(self, placeholders, users, prev_items, list_fsub_items_id, list_fsub_items_values, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items):
        feed_dict = {
            placeholders['pl_user_list']: users,
            placeholders['pl_prev_items']: prev_items,
            placeholders['pl_list_fsub_items_id']: list_fsub_items_id,
            placeholders['pl_list_fsub_items_values']: list_fsub_items_values,
            placeholders['pl_list_prev_items_pos']: list_prev_items_pos,
            placeholders['pl_list_prev_items_neg']: list_prev_items_neg,
            placeholders['pl_pos']: pos_items,
            placeholders['pl_neg']: neg_items,
        }

        return feed_dict

    def get_preds(self, placeholders, parameters):

        # Get back variables
        var_emb_items = parameters["var_emb_items"]
        var_bias_items = parameters["var_bias_items"]

        # Get item and bias for pos and neg examples
        item_pos = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_pos"])
        bias_pos = tf.nn.embedding_lookup(var_bias_items, placeholders["pl_pos"])
        item_neg = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_neg"])
        bias_neg = tf.nn.embedding_lookup(var_bias_items, placeholders["pl_neg"])

        # Lookup items for long and short term
        emb_list_prev_items_pos = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_list_prev_items_pos"])
        emb_list_prev_items_neg = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_list_prev_items_neg"])
        emb_list_fsub_items_id = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_list_fsub_items_id"])

        # long term
        # Count the number of list_prev_item_pos withtout null item (null item is equal to nb_items)
        wu_pos = tf.pow(tf.add(self.args.max_lens - tf.reduce_sum(tf.cast(tf.equal(placeholders["pl_list_prev_items_pos"], self.dataset.nb_items), tf.float32), axis=1, keepdims=True), 1e-10), self.alpha)
        sum_long_pos = tf.multiply(tf.reduce_sum(emb_list_prev_items_pos, axis=1), wu_pos)

        wu_neg = tf.pow(tf.add(self.args.max_lens - tf.reduce_sum(tf.cast(tf.equal(placeholders["pl_list_prev_items_neg"], self.dataset.nb_items), tf.float32), axis=1, keepdims=True), 1e-10), self.alpha)
        sum_long_neg = tf.multiply(tf.reduce_sum(emb_list_prev_items_neg, axis=1), wu_neg)
        # wu_neg = tf.pow(tf.add(self.args.max_lens - tf.reduce_sum(tf.cast(tf.equal(placeholders["pl_list_prev_items_pos"], self.dataset.nb_items), tf.float32), axis=1, keepdims=True), 1e-10), self.alpha)
        # sum_long_neg = tf.multiply(tf.reduce_sum(emb_list_prev_items_pos, axis=1), wu_neg)

        # Short term
        pl_list_fsub_items_values_reshape = tf.reshape(placeholders["pl_list_fsub_items_values"], [tf.shape(placeholders["pl_list_fsub_items_values"])[0], tf.shape(placeholders["pl_list_fsub_items_values"])[1], 1])
        sum_short = tf.reduce_sum(tf.multiply(emb_list_fsub_items_id, pl_list_fsub_items_values_reshape), axis=1)

        # Prediction
        if self.args.gamma == -1.0:
            dist_pos = tf.subtract(tf.add(sum_long_pos, sum_short), item_pos)
            dist_squared_pos = tf.multiply(dist_pos, dist_pos)
            preds_pos = tf.add(bias_pos, tf.reduce_sum(dist_squared_pos, axis=1, keepdims=True))

            dist_neg = tf.subtract(tf.add(sum_long_neg, sum_short), item_neg)
            dist_squared_neg = tf.multiply(dist_neg, dist_neg)
            preds_neg = tf.add(bias_neg, tf.reduce_sum(dist_squared_neg, axis=1, keepdims=True))
        else:
            dist_pos = tf.subtract(tf.add(tf.multiply(sum_long_pos, self.gamma_long), tf.multiply(sum_short, self.gamma_short)), item_pos)
            dist_squared_pos = tf.multiply(dist_pos, dist_pos)
            preds_pos = tf.add(bias_pos, tf.reduce_sum(dist_squared_pos, axis=1, keepdims=True))

            dist_neg = tf.subtract(tf.add(tf.multiply(sum_long_neg, self.gamma_long), tf.multiply(sum_short, self.gamma_short)), item_neg)
            dist_squared_neg = tf.multiply(dist_neg, dist_neg)
            preds_neg = tf.add(bias_neg, tf.reduce_sum(dist_squared_neg, axis=1, keepdims=True))

        return -preds_pos, -preds_neg

    def get_preds_for_evaluate(self, placeholders, parameters):

        # Get back variables
        var_emb_items = parameters["var_emb_items"]
        var_bias_items = parameters["var_bias_items"]

        # Get item and bias for pos and neg examples
        item_pos = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_pos"])
        bias_pos = tf.nn.embedding_lookup(var_bias_items, placeholders["pl_pos"])

        # Lookup items for long and short term
        emb_list_prev_items_pos = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_list_prev_items_pos"])
        emb_list_fsub_items_id = tf.nn.embedding_lookup(var_emb_items, placeholders["pl_list_fsub_items_id"])

        # long term
        # Count the number of list_prev_item_pos withtout null item (null item is equal to nb_items)
        wu_pos = tf.pow(tf.add(self.args.max_lens - tf.reduce_sum(tf.cast(tf.equal(placeholders["pl_list_prev_items_pos"], self.dataset.nb_items), tf.float32), axis=1, keepdims=True), 1e-10), self.alpha)
        sum_long_pos = tf.multiply(tf.reduce_sum(emb_list_prev_items_pos, axis=1), wu_pos)

        # Short term
        pl_list_fsub_items_values_reshape = tf.reshape(placeholders["pl_list_fsub_items_values"], [tf.shape(placeholders["pl_list_fsub_items_values"])[0], tf.shape(placeholders["pl_list_fsub_items_values"])[1], 1])
        sum_short = tf.reduce_sum(tf.multiply(emb_list_fsub_items_id, pl_list_fsub_items_values_reshape), axis=1)

        # Prediction
        if self.args.gamma == -1.0:
            dist_pos = tf.subtract(tf.add(sum_long_pos, sum_short), item_pos)
            dist_squared_pos = tf.multiply(dist_pos, dist_pos)
            preds_pos = tf.add(bias_pos, tf.reduce_sum(dist_squared_pos, axis=1, keepdims=True))
        else:
            dist_pos = tf.subtract(tf.add(tf.multiply(sum_long_pos, self.gamma_long), tf.multiply(sum_short, self.gamma_short)), item_pos)
            dist_squared_pos = tf.multiply(dist_pos, dist_pos)
            preds_pos = tf.add(bias_pos, tf.reduce_sum(dist_squared_pos, axis=1, keepdims=True))

        return -preds_pos

    def BPR_loss(self, preds_pos, preds_pneg, parameters):
        # Get back variables
        var_emb_items = parameters["var_emb_items"]
        var_bias_items = parameters["var_bias_items"]

        l2_reg_emb_items = self.args.emb_reg * tf.reduce_sum(tf.square(var_emb_items))
        l2_reg_bias_items = self.args.bias_reg * tf.reduce_sum(tf.square(var_bias_items))

        # BPR training op (add 1e-10 to help numerical stability)
        bprloss_op = tf.reduce_sum(tf.log(1e-10 + tf.sigmoid(preds_pos - preds_pneg))) - l2_reg_emb_items - l2_reg_bias_items

        return -bprloss_op  # We take the opposite beacuse we minimize

    def train(self):
        start_train_time = time.time()
        config = tf.ConfigProto()

        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

        # to keep consistent results
        tf.set_random_seed(1)
        np.random.seed(3)

        self.create_constant()
        self.placeholders = self.create_placeholder()
        self.parameters = self.initialize_parameters()
        self.preds_pos, self.preds_neg = self.get_preds(self.placeholders, self.parameters)
        self.preds_eval = self.get_preds_for_evaluate(self.placeholders, self.parameters)
        self.bprloss_op = self.BPR_loss(self.preds_pos, self.preds_neg, self.parameters)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.bprloss_op, global_step=self.global_step)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(self.init)

            best_epoch = 0
            best_val_auc = -1
            best_test_auc = -1
            best_save_path = self.saver.save(sess, self.path_saver_parameters)

            for epoch in range(self.args.max_iters):
                start_time = time.time()
                mini_batch_idx, batch_size, users, prev_items, list_fsub_items_id, list_fsub_items_values, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items = self.dataset.generate_train_shuffled_batch_sp_with_prev_items_fsub()
                num_complete_minibatches = math.floor(batch_size/self.args.mini_batch_size)  # number of mini batches of size self.args.mini_batch_size in your partitionning
                for k in range(0, num_complete_minibatches):
                    mini_batch_users = users[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size]
                    mini_batch_prev_items = prev_items[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size, :]
                    mini_batch_list_fsub_items_id = list_fsub_items_id[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size, :]
                    mini_batch_list_fsub_items_values = list_fsub_items_values[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size, :]
                    mini_batch_list_prev_items_pos = list_prev_items_pos[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size, :]
                    mini_batch_list_prev_items_neg = list_prev_items_neg[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size, :]
                    mini_batch_pos_items = pos_items[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size]
                    mini_batch_neg_items = neg_items[k * self.args.mini_batch_size: k * self.args.mini_batch_size + self.args.mini_batch_size]

                    feed_dict = self.create_feed_dict(self.placeholders, mini_batch_users, mini_batch_prev_items, mini_batch_list_fsub_items_id, mini_batch_list_fsub_items_values, mini_batch_list_prev_items_pos, mini_batch_list_prev_items_neg, mini_batch_pos_items, mini_batch_neg_items)
                    _, bprloss = sess.run([self.optimizer, self.bprloss_op], feed_dict=feed_dict)

                if batch_size % self.args.mini_batch_size != 0:
                    mini_batch_users = users[num_complete_minibatches * self.args.mini_batch_size: batch_size]
                    mini_batch_prev_items = prev_items[num_complete_minibatches * self.args.mini_batch_size: batch_size, :]
                    mini_batch_list_fsub_items_id = list_fsub_items_id[num_complete_minibatches * self.args.mini_batch_size: batch_size, :]
                    mini_batch_list_fsub_items_values = list_fsub_items_values[num_complete_minibatches * self.args.mini_batch_size: batch_size, :]
                    mini_batch_list_prev_items_pos = list_prev_items_pos[num_complete_minibatches * self.args.mini_batch_size: batch_size, :]
                    mini_batch_list_prev_items_neg = list_prev_items_neg[num_complete_minibatches * self.args.mini_batch_size: batch_size, :]
                    mini_batch_pos_items = pos_items[num_complete_minibatches * self.args.mini_batch_size: batch_size]
                    mini_batch_neg_items = neg_items[num_complete_minibatches * self.args.mini_batch_size: batch_size]

                    feed_dict = self.create_feed_dict(self.placeholders, mini_batch_users, mini_batch_prev_items, mini_batch_list_fsub_items_id, mini_batch_list_fsub_items_values, mini_batch_list_prev_items_pos, mini_batch_list_prev_items_neg, mini_batch_pos_items, mini_batch_neg_items)
                    _, bprloss = sess.run([self.optimizer, self.bprloss_op], feed_dict=feed_dict)

                print('\tEpoch: {} BPR-Loss = {} (time : {})'.format(epoch, bprloss, time.time() - start_time))

                if epoch % self.args.eval_freq == 0:
                    start_eval_time = time.time()
                    val_auc = commons.sample_evaluate_valid_faster(self, self.dataset, sess)['AUC']
                    test_auc = 0.0

                    print('\tEpoch: {} Val AUC = {}, tTest AUC = {} (time : {})'.format(epoch, val_auc, test_auc, time.time() - start_eval_time))

                    if val_auc > best_val_auc:
                        best_epoch = epoch
                        best_val_auc = val_auc
                        best_test_auc = test_auc
                        best_save_path = self.saver.save(sess, self.path_saver_parameters)
                    else:
                        if epoch >= (best_epoch + self.args.quit_delta):
                            print('Overfitted, exiting...')
                            break

                    print('\tCurrent max = {} at epoch {}'.format(best_val_auc, best_epoch))

            time_to_train = time.time() - start_train_time

            print("Restore best parameters")
            self.saver.restore(sess, self.path_saver_parameters)
            self.parameters = sess.run(self.parameters)
            print(self.parameters["var_emb_items"][0])
            print(sess.run('emb_items:0'))

            start_eval_time = time.time()
            valid_metrics = commons.evaluate_valid(self, self.dataset, sess)
            test_metrics = commons.evaluate_test(self, self.dataset, sess)
            time_to_eval = time.time() - start_eval_time

            print('\n\t time_to_train = {}'.format(time_to_train))
            print('\t time_to_eval = {}'.format(time_to_eval))
            print('\t Best Epoch = {}'.format(best_epoch))
            print('\t Sample Validation AUC = {} (estimation with {} random items) '.format(best_val_auc, self.dataset.args.item_per_user))
            print('\t Validation AUC = {} '.format(valid_metrics['AUC']))
            print('\t Sample Test AUC = {} (estimation with {} random items)'.format(best_test_auc, self.dataset.args.item_per_user))
            print('\t Test AUC = {} '.format(test_metrics['AUC']))

            if self.dataset.args.cold_start_user:
                cold_metrics = commons.evaluate_cold_start(self, self.dataset, sess)
                print('\t cold_metrics AUC = {} '.format(cold_metrics['AUC']))
            else:
                cold_metrics = {}

            return {
                'time_to_train': time_to_train,
                'time_to_eval': time_to_eval,
                'best_epoch': best_epoch,
                'best_val_auc': best_val_auc,
                'best_test_auc': best_test_auc,
                'valid_metrics': valid_metrics,
                'test_metrics': test_metrics,
                'cold_metrics': cold_metrics
            }

    def load(self):
        start_train_time = time.time()
        config = tf.ConfigProto()

        ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables

        # to keep consistent results
        tf.set_random_seed(1)
        np.random.seed(3)

        self.create_constant()
        self.placeholders = self.create_placeholder()
        self.parameters = self.initialize_parameters()
        self.preds_pos, self.preds_neg = self.get_preds(self.placeholders, self.parameters)
        self.preds_eval = self.get_preds_for_evaluate(self.placeholders, self.parameters)
        self.bprloss_op = self.BPR_loss(self.preds_pos, self.preds_neg, self.parameters)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate).minimize(self.bprloss_op, global_step=self.global_step)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def predict(self, sess, users_tmp, prev_items_tmp, list_fsub_items_id_tmp, list_fsub_items_values_tmp, pos_items_tmp, list_prev_items_pos_tmp):
        feed_dict = {
            self.placeholders['pl_user_list']: users_tmp,
            self.placeholders['pl_prev_items']: prev_items_tmp,
            self.placeholders['pl_list_fsub_items_id']: list_fsub_items_id_tmp,
            self.placeholders['pl_list_fsub_items_values']: list_fsub_items_values_tmp,
            self.placeholders['pl_list_prev_items_pos']: list_prev_items_pos_tmp,
            self.placeholders['pl_pos']: pos_items_tmp,
        }
        return sess.run([self.preds_eval], feed_dict=feed_dict)
