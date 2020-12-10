import pandas as pd
import scipy.sparse as sp
import numpy as np
from collections import defaultdict
import copy
import os
import ntpath
import commons
import time
import gc

# ---- Class to represent a dataset ---- #


class Dataset:
    def __init__(self, args, grid_search=False):
        self.args = args
        self.grid_search = grid_search
        self.path = self.args.path
        self.user_min = self.args.user_min
        self.item_min = self.args.item_min

        self.data_name = ntpath.basename(args.path).split(".")[0]
        self.ITEMONLYMODEL = ["REBUS", "REBUS_ST", "REBUS_simple", "REBUS_ST_simple", "REBUS_LT"]
        self.COLDSTARTUSERONLYMODEL = ["REBUS", "REBUS_ST", "REBUS_simple", "REBUS_ST_simple", "REBUS_LT"]
        self.FSUBMODEL = ["REBUS", "REBUS_ST"]
        self.REBUSMODEL = ["REBUS", "REBUS_ST", "REBUS_simple", "REBUS_ST_simple"]

        # -- Load Specific FSUB relevante data if Model is REBUS -- #
        if self.args.model in self.FSUBMODEL:
            self.path_fsub = os.path.join("96-FSUB",
                                          ("userMin_"+str(self.args.user_min)+"_itemMin_"+str(self.args.item_min)),
                                          (self.data_name+"_root_fsub_minCount_"+str(self.args.min_count)+"_L_"+str(self.args.L)+".txt"))
            with open(self.path_fsub, "r") as f:
                fsub_set = f.readlines()
            fsub_set = set([x.strip() for x in fsub_set])

            if self.args.production:
                self.path_fsub_prod = os.path.join("96-FSUB",
                                                   ("userMin_"+str(self.args.user_min)+"_itemMin_"+str(self.args.item_min)),
                                                   (self.data_name+"_root_fsub_minCount_"+str(self.args.min_count)+"_L_"+str(self.args.L)+"_prod.txt"))

                with open(self.path_fsub_prod, "r") as f:
                    fsub_set_prod = f.readlines()
                fsub_set_prod = set([x.strip() for x in fsub_set_prod])
                self.fsub_set_prod = fsub_set_prod

            if self.args.damping_fsub == "linear_softmax":
                fsub_damping = [commons.softmax([commons.damping_linear(x, L) for x in np.arange(L)]) for L in np.arange(1, 25+1)]
            elif self.args.damping_fsub == "linear":
                fsub_damping = [[commons.damping_linear(x, L) for x in np.arange(L)] for L in np.arange(1, 25+1)]

            self.fsub_set = fsub_set
            self.fsub_damping = fsub_damping
        else:
            if self.args.damping_fsub == "linear_softmax":
                fsub_damping = [commons.softmax([commons.damping_linear(x, L) for x in np.arange(L)]) for L in np.arange(1, 25+1)]
            elif self.args.damping_fsub == "linear":
                fsub_damping = [[commons.damping_linear(x, L) for x in np.arange(L)] for L in np.arange(1, 25+1)]

            self.fsub_damping = fsub_damping

        # -- Load Data -- #
        if self.path.split('.')[len(self.path.split('.'))-1] == 'csv':
            self.df_data = pd.read_csv(self.path, sep=",", header=None, names=['userID', 'itemID', 'rating', 'time'])
        elif self.path.split('.')[len(self.path.split('.'))-1] == 'txt':
            self.df_data = pd.read_csv(self.path, sep="\s+", header=None, names=['userID', 'itemID', 'rating', 'time'])
        else:
            raise('Type de donnees non pris en charge')

        self.df_data["userID"] = self.df_data["userID"].fillna("user_null")
        self.df_data["itemID"] = self.df_data["itemID"].fillna("item_null")

        print(self.df_data.head(5))

        # -- FIRST PASS : Count the number of unique user and item (no filter) -- #
        print('--- First pass ---')
        user_counts = self.df_data.groupby(['userID']).size().to_dict()
        self.nb_users_init = len(self.df_data['userID'].unique())
        print('Collected user counts...')
        item_counts = self.df_data.groupby(['itemID']).size().to_dict()
        self.nb_items_init = len(self.df_data['itemID'].unique())
        print('Collected item counts...')
        print('\t self.nb_users = ' + str(len(self.df_data['userID'].unique())))
        print('\t self.nb_items = ' + str(len(self.df_data['itemID'].unique())))
        print('\t df_shape  = ' + str(self.df_data.shape))

        # -- SECOND PASS : Filter based on user and item counts and create usefull dict -- #
        print('--- Second pass ---')
        self.nb_events_init = 0
        self.nb_events = 0
        self.nb_items = 0
        self.nb_users = 0
        self.dict_user_name_to_id = {}
        self.dict_item_name_to_id = {}
        self.dict_user_id_to_name = {}
        self.dict_item_id_to_name = {}
        self.train_data = []
        self.train_data_with_valid_item = []
        self.train_data_with_valid_test_item = []
        for row in self.df_data.itertuples():
            # print(row)
            user_name = row[1]
            item_name = row[2]
            value = row[3]
            timespam = row[4]
            self.nb_events_init += 1

            if user_counts[user_name] < self.args.user_min:
                continue

            if item_counts[item_name] < self.args.item_min:
                continue

            self.nb_events += 1

            if item_name not in self.dict_item_name_to_id:  # New item
                self.dict_item_id_to_name[self.nb_items] = item_name
                self.dict_item_name_to_id[item_name] = self.nb_items
                self.nb_items += 1

            if user_name not in self.dict_user_name_to_id:  # New user
                self.dict_user_id_to_name[self.nb_users] = user_name
                self.dict_user_name_to_id[user_name] = self.nb_users
                self.nb_users += 1
                self.train_data.append([])  # add a new vec
                self.train_data_with_valid_item.append([])  # add a new vec
                self.train_data_with_valid_test_item.append([])  # add a new vec

            self.train_data[self.dict_user_name_to_id[user_name]].append([self.dict_item_name_to_id[item_name], timespam])
            self.train_data_with_valid_item[self.dict_user_name_to_id[user_name]].append([self.dict_item_name_to_id[item_name], timespam])
            self.train_data_with_valid_test_item[self.dict_user_name_to_id[user_name]].append([self.dict_item_name_to_id[item_name], timespam])

        # Sort item with timespam
        for u in range(0, self.nb_users):
            self.train_data[u] = sorted(self.train_data[u], key=lambda user: user[1], reverse=False)
            self.train_data_with_valid_item[u] = sorted(self.train_data_with_valid_item[u], key=lambda user: user[1], reverse=False)
            self.train_data_with_valid_test_item[u] = sorted(self.train_data_with_valid_test_item[u], key=lambda user: user[1], reverse=False)

        print('\tself.nb_users = ' + str(self.nb_users))
        print('\tself.nb_items = ' + str(self.nb_items))
        print('\tself.nb_events  = ' + str(self.nb_events))

        # -- Optional Pass to get all filtered user -- #
        if self.args.production:
            self.generate_production_data(user_counts, item_counts)
        else:
            self.nb_users_filtered = -1
            self.nb_users_invalid = -1

        # -- Optional Pass to get all filtered user -- #
        if self.args.cold_start_user and self.args.model in self.COLDSTARTUSERONLYMODEL:
            self.generate_cold_start_user_data(user_counts, item_counts)

        # -- Split dataset into train, validation and test -- #
        self.train_data_set = []
        self.train_data_item = []
        self.valid_data = []
        self.valid_data_set = []
        self.valid_data_set_np = []
        self.valid_times = []
        self.test_data = []
        self.test_data_set = []
        self.test_data_set_np = []
        self.test_times = []
        self.nb_train_events = 0
        self.np_list_items = np.array(range(self.nb_items))
        for u in range(0, self.nb_users):  # Leave out the last two items for each user
            # When need at least 3 actions per user to split the data
            if len(self.train_data[u]) < 3:
                # print("  Warning: user ", u, " has only ", len(self.train_data[u]), " actions")
                self.test_data.append([-1, -1])
                self.test_times.append([-1, -1])
                self.valid_data.append([-1, -1])
                self.valid_times.append([-1, -1])
            else:
                # Get the truth item for the test data
                pop_test = self.train_data[u].pop()
                self.train_data_with_valid_item[u].pop()
                item_test = pop_test[0]
                time_test = pop_test[1]

                # Get the truth item for the val data and the test item for the test data
                pop_val = self.train_data[u].pop()
                item_val = pop_val[0]
                time_val = pop_val[1]

                # Get the val item for the val data
                last_item = self.train_data[u][-1][0]
                last_time = self.train_data[u][-1][1]

                self.test_data.append([item_test, item_val])
                self.test_times.append([time_test, time_val])

                self.valid_data.append([item_val, last_item])
                self.valid_times.append([time_val, last_time])

            self.train_data_set.append(set(map(lambda x: x[0], self.train_data[u])))

            self.valid_data_set.append(set(map(lambda x: x[0], self.train_data[u])))
            self.valid_data_set[u].add(self.valid_data[u][0])
            self.valid_data_set_np.append(np.array(list(self.valid_data_set[u])))
            # self.valid_data_set_eval_np.append(np.setdiff1d(self.np_list_items, self.valid_data_set_np[u]))

            self.test_data_set.append(set(map(lambda x: x[0], self.train_data[u])))
            self.test_data_set[u].add(self.valid_data[u][0])
            self.test_data_set[u].add(self.test_data[u][0])
            self.test_data_set_np.append(np.array(list(self.test_data_set[u])))
            # self.test_data_set_eval_np.append(np.setdiff1d(self.np_list_items, self.test_data_set_np[u]))

            self.train_data_item.append(list(map(lambda x: x[0], self.train_data[u])))
            self.nb_train_events += len(self.train_data[u])

            # if u % 1000 == 0:
            #     print("User {}, time {}".format(u, time.time()-start_time_test))
            #     gc.collect()

        print("self.nb_train_events = ", self.nb_train_events)

        # -- Create all variables for all possible pos item (i.e. The first item of all user is remove) -- #
        self.train_user_id = []
        self.train_pos_item_id = []
        self.train_prev_item_id = []
        self.train_list_prev_items_id = []
        self.train_fsub_items_id = []
        self.train_fsub_items_value = []
        self.nb_train_events_possible = 0
        for user in range(0, self.nb_users):
            list_prev_items_id = []
            for i in range(1, len(self.train_data[user])):
                item_pos = self.train_data[user][i][0]
                item_prev = self.train_data[user][i-1][0]
                self.train_user_id.append(user)
                self.train_pos_item_id.append(item_pos)
                self.train_prev_item_id.append(item_prev)
                list_prev_items_id.append(item_prev)
                len_list_prev_items_id = len(list_prev_items_id)
                self.train_list_prev_items_id.append([])
                if len_list_prev_items_id > self.args.max_lens:
                    for y in list_prev_items_id[len_list_prev_items_id-self.args.max_lens:]:
                        # if y != item_pos:
                        self.train_list_prev_items_id[self.nb_train_events_possible].append(y)
                else:
                    for y in list_prev_items_id:
                        # if y != item_pos:
                        # if not y in self.train_list_prev_items_id[self.nb_train_events_possible] and y != item_pos:
                        self.train_list_prev_items_id[self.nb_train_events_possible].append(y)
                    self.train_list_prev_items_id[self.nb_train_events_possible] += [self.nb_items] * (self.args.max_lens - len_list_prev_items_id)
                    # self.train_list_prev_items_id[self.nb_train_events_possible] += [self.nb_items] * (self.args.max_lens - len(self.train_list_prev_items_id[self.nb_train_events_possible]))

                if self.args.model in self.FSUBMODEL:
                    self.train_fsub_items_id.append([])
                    self.train_fsub_items_value.append([])
                    fsub_items = commons.find_path_stars(fsub_set, list_prev_items_id.copy())
                    pos = 0
                    for y in fsub_items:
                        self.train_fsub_items_id[self.nb_train_events_possible].append(y)
                        self.train_fsub_items_value[self.nb_train_events_possible].append(fsub_damping[len(fsub_items)-1][pos])
                        pos += 1
                    self.train_fsub_items_id[self.nb_train_events_possible] += [self.nb_items] * (self.args.L - len(self.train_fsub_items_id[self.nb_train_events_possible]))
                    self.train_fsub_items_value[self.nb_train_events_possible] += [0] * (self.args.L - len(self.train_fsub_items_value[self.nb_train_events_possible]))
                else:
                    self.train_fsub_items_id.append([])
                    self.train_fsub_items_value.append([])
                    fsub_items = commons.find_markov_chains_k(list_prev_items_id.copy(), self.args.L)
                    pos = 0
                    for y in fsub_items:
                        self.train_fsub_items_id[self.nb_train_events_possible].append(y)
                        self.train_fsub_items_value[self.nb_train_events_possible].append(fsub_damping[len(fsub_items)-1][pos])
                        pos += 1
                    self.train_fsub_items_id[self.nb_train_events_possible] += [self.nb_items] * (self.args.L - len(self.train_fsub_items_id[self.nb_train_events_possible]))
                    self.train_fsub_items_value[self.nb_train_events_possible] += [0] * (self.args.L - len(self.train_fsub_items_value[self.nb_train_events_possible]))

                self.nb_train_events_possible += 1

        self.train_user_id = np.array(self.train_user_id, dtype=np.int32)
        self.train_pos_item_id = np.array(self.train_pos_item_id, dtype=np.int32)
        self.train_prev_item_id = np.array(self.train_prev_item_id, dtype=np.int32).reshape((len(self.train_prev_item_id), 1))
        self.train_list_prev_items_id = np.array(self.train_list_prev_items_id, dtype=np.int32).reshape((len(self.train_list_prev_items_id), self.args.max_lens))
        self.train_fsub_items_id = np.array(self.train_fsub_items_id, dtype=np.int32).reshape((len(self.train_fsub_items_id), self.args.L))
        self.train_fsub_items_value = np.array(self.train_fsub_items_value, dtype=np.float32).reshape((len(self.train_fsub_items_value), self.args.L))
        print("Shape of self.train_user_id = ", self.train_user_id.shape)
        print("Shape of self.train_pos_item_id = ", self.train_pos_item_id.shape)
        print("Shape of self.train_prev_item_id = ", self.train_prev_item_id.shape)
        print("Shape of self.train_list_prev_items_id = ", self.train_list_prev_items_id.shape)
        print("Shape of self.train_fsub_items_id = ", self.train_fsub_items_id.shape)
        print("Shape of self.train_fsub_items_value = ", self.train_fsub_items_value.shape)
        print("self.nb_train_events_possible = ", self.nb_train_events_possible)
        self.train_list_prev_items_id_pos = self.train_list_prev_items_id.copy()
        self.train_list_prev_items_id_pos[self.train_list_prev_items_id_pos == self.train_pos_item_id.reshape((self.train_pos_item_id.shape[0], 1))] = self.nb_items

        # -- Create all variables for all possible pos item (i.e. The first item of all user is remove) -- #
        self.valid_user_id = []
        self.valid_pos_item_id = []
        self.valid_prev_item_id = []
        self.valid_list_prev_items_id = []
        self.valid_fsub_items_id = []
        self.valid_fsub_items_value = []
        self.nb_valid_events_possible = 0
        for user in range(0, self.nb_users):
            item_pos = self.valid_data[user][0]
            item_prev = self.valid_data[user][1]
            if item_pos == -1 or item_prev == -1:
                continue
            self.valid_user_id.append(user)
            self.valid_pos_item_id.append(item_pos)
            self.valid_prev_item_id.append(item_prev)
            self.valid_list_prev_items_id.append([])
            len_train_data = len(self.train_data[user])
            if len_train_data > self.args.max_lens:
                for i in range(len_train_data-self.args.max_lens, len_train_data):
                    # if self.train_data[user][i][0] != item_pos:
                    self.valid_list_prev_items_id[self.nb_valid_events_possible].append(self.train_data[user][i][0])
            else:
                for i in range(0, len_train_data):
                    # if self.train_data[user][i][0] != item_pos:
                    self.valid_list_prev_items_id[self.nb_valid_events_possible].append(self.train_data[user][i][0])
                self.valid_list_prev_items_id[self.nb_valid_events_possible] += [self.nb_items] * (self.args.max_lens - len_train_data)

            if self.args.model in self.FSUBMODEL:
                self.valid_fsub_items_id.append([])
                self.valid_fsub_items_value.append([])
                fsub_items = commons.find_path_stars(fsub_set, self.train_data_item[user].copy())
                pos = 0
                for y in fsub_items:
                    self.valid_fsub_items_id[self.nb_valid_events_possible].append(y)
                    self.valid_fsub_items_value[self.nb_valid_events_possible].append(fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.valid_fsub_items_id[self.nb_valid_events_possible] += [self.nb_items] * (self.args.L - len(self.valid_fsub_items_id[self.nb_valid_events_possible]))
                self.valid_fsub_items_value[self.nb_valid_events_possible] += [0] * (self.args.L - len(self.valid_fsub_items_value[self.nb_valid_events_possible]))
            else:
                self.valid_fsub_items_id.append([])
                self.valid_fsub_items_value.append([])
                fsub_items = commons.find_markov_chains_k(self.train_data_item[user].copy(), self.args.L)
                pos = 0
                for y in fsub_items:
                    self.valid_fsub_items_id[self.nb_valid_events_possible].append(y)
                    self.valid_fsub_items_value[self.nb_valid_events_possible].append(fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.valid_fsub_items_id[self.nb_valid_events_possible] += [self.nb_items] * (self.args.L - len(self.valid_fsub_items_id[self.nb_valid_events_possible]))
                self.valid_fsub_items_value[self.nb_valid_events_possible] += [0] * (self.args.L - len(self.valid_fsub_items_value[self.nb_valid_events_possible]))

            self.nb_valid_events_possible += 1

        self.valid_user_id = np.array(self.valid_user_id, dtype=np.int32)
        self.valid_pos_item_id = np.array(self.valid_pos_item_id, dtype=np.int32)
        self.valid_prev_item_id = np.array(self.valid_prev_item_id, dtype=np.int32).reshape((len(self.valid_prev_item_id), 1))
        self.valid_list_prev_items_id = np.array(self.valid_list_prev_items_id, dtype=np.int32).reshape((len(self.valid_list_prev_items_id), self.args.max_lens))
        self.valid_fsub_items_id = np.array(self.valid_fsub_items_id, dtype=np.int32).reshape((len(self.valid_fsub_items_id), self.args.L))
        self.valid_fsub_items_value = np.array(self.valid_fsub_items_value, dtype=np.float32).reshape((len(self.valid_fsub_items_value), self.args.L))
        print("Shape of self.valid_user_id = ", self.valid_user_id.shape)
        print("Shape of self.valid_pos_item_id = ", self.valid_pos_item_id.shape)
        print("Shape of self.valid_prev_item_id = ", self.valid_prev_item_id.shape)
        print("Shape of self.valid_list_prev_items_id = ", self.valid_list_prev_items_id.shape)
        print("Shape of self.valid_fsub_items_id = ", self.valid_fsub_items_id.shape)
        print("Shape of self.valid_fsub_items_value = ", self.valid_fsub_items_value.shape)
        print("self.nb_valid_events_possible = ", self.nb_valid_events_possible)
        self.valid_list_prev_items_id_pos = self.valid_list_prev_items_id.copy()
        self.valid_list_prev_items_id_pos[self.valid_list_prev_items_id_pos == self.valid_pos_item_id.reshape((self.valid_pos_item_id.shape[0], 1))] = self.nb_items

        # -- Create all variables for all possible pos item (i.e. The first item of all user is remove) -- #
        self.test_user_id = []
        self.test_pos_item_id = []
        self.test_prev_item_id = []
        self.test_list_prev_items_id = []
        self.test_fsub_items_id = []
        self.test_fsub_items_value = []
        self.nb_test_events_possible = 0
        for user in range(0, self.nb_users):
            item_pos = self.test_data[user][0]
            item_prev = self.test_data[user][1]
            if item_pos == -1 or item_prev == -1:
                continue
            self.test_user_id.append(user)
            self.test_pos_item_id.append(item_pos)
            self.test_prev_item_id.append(item_prev)
            self.test_list_prev_items_id.append([])
            len_train_data = len(self.train_data[user])+1  # We add +1 beacause we need to append item_prev (val_pos_item)
            if len_train_data > self.args.max_lens:  # We conserve the most recent action
                for i in range(len_train_data-self.args.max_lens, len(self.train_data[user])):
                    # if self.train_data[user][i][0] != item_pos:
                    self.test_list_prev_items_id[self.nb_test_events_possible].append(self.train_data[user][i][0])
                # if item_prev != item_pos:
                self.test_list_prev_items_id[self.nb_test_events_possible].append(item_prev)
            else:
                for i in range(0, len(self.train_data[user])):
                    # if self.train_data[user][i][0] != item_pos:
                    self.test_list_prev_items_id[self.nb_test_events_possible].append(self.train_data[user][i][0])
                # if item_prev != item_pos:
                self.test_list_prev_items_id[self.nb_test_events_possible].append(item_prev)
                self.test_list_prev_items_id[self.nb_test_events_possible] += [self.nb_items] * (self.args.max_lens - len_train_data)

            if self.args.model in self.FSUBMODEL:
                self.test_fsub_items_id.append([])
                self.test_fsub_items_value.append([])
                tmp_set = self.train_data_item[user].copy()
                tmp_set.append(item_prev)
                fsub_items = commons.find_path_stars(fsub_set, tmp_set)
                pos = 0
                for y in fsub_items:
                    self.test_fsub_items_id[self.nb_test_events_possible].append(y)
                    self.test_fsub_items_value[self.nb_test_events_possible].append(fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.test_fsub_items_id[self.nb_test_events_possible] += [self.nb_items] * (self.args.L - len(self.test_fsub_items_id[self.nb_test_events_possible]))
                self.test_fsub_items_value[self.nb_test_events_possible] += [0] * (self.args.L - len(self.test_fsub_items_value[self.nb_test_events_possible]))
            else:
                self.test_fsub_items_id.append([])
                self.test_fsub_items_value.append([])
                tmp_set = self.train_data_item[user].copy()
                tmp_set.append(item_prev)
                fsub_items = commons.find_markov_chains_k(tmp_set, self.args.L)
                pos = 0
                for y in fsub_items:
                    self.test_fsub_items_id[self.nb_test_events_possible].append(y)
                    self.test_fsub_items_value[self.nb_test_events_possible].append(fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.test_fsub_items_id[self.nb_test_events_possible] += [self.nb_items] * (self.args.L - len(self.test_fsub_items_id[self.nb_test_events_possible]))
                self.test_fsub_items_value[self.nb_test_events_possible] += [0] * (self.args.L - len(self.test_fsub_items_value[self.nb_test_events_possible]))

            self.nb_test_events_possible += 1

        self.test_user_id = np.array(self.test_user_id, dtype=np.int32)
        self.test_pos_item_id = np.array(self.test_pos_item_id, dtype=np.int32)
        self.test_prev_item_id = np.array(self.test_prev_item_id, dtype=np.int32).reshape((len(self.test_prev_item_id), 1))
        self.test_list_prev_items_id = np.array(self.test_list_prev_items_id, dtype=np.int32).reshape((len(self.test_list_prev_items_id), self.args.max_lens))
        self.test_fsub_items_id = np.array(self.test_fsub_items_id, dtype=np.int32).reshape((len(self.test_fsub_items_id), self.args.L))
        self.test_fsub_items_value = np.array(self.test_fsub_items_value, dtype=np.float32).reshape((len(self.test_fsub_items_value), self.args.L))
        print("Shape of self.test_user_id = ", self.test_user_id.shape)
        print("Shape of self.test_pos_item_id = ", self.test_pos_item_id.shape)
        print("Shape of self.test_prev_item_id = ", self.test_prev_item_id.shape)
        print("Shape of self.test_list_prev_items_id = ", self.test_list_prev_items_id.shape)
        print("Shape of self.test_fsub_items_id = ", self.test_fsub_items_id.shape)
        print("Shape of self.test_fsub_items_value = ", self.test_fsub_items_value.shape)
        print("self.nb_test_events_possible = ", self.nb_test_events_possible)
        self.test_list_prev_items_id_pos = self.test_list_prev_items_id.copy()
        self.test_list_prev_items_id_pos[self.test_list_prev_items_id_pos == self.test_pos_item_id.reshape((self.test_pos_item_id.shape[0], 1))] = self.nb_items

        gc.collect()
        print("End of init dataset")

    def random_neq(self, nb_items, train_set_item, pos_items):
        t = np.random.randint(0, nb_items)
        while t in train_set_item or t == pos_items:
            t = np.random.randint(0, nb_items)
        return t

    def generate_train_shuffled_batch_sp_with_prev_items(self):
        # Rand all neg item
        neg_items = np.random.randint(0, self.nb_items, size=(len(self.train_user_id)), dtype=np.int32)
        # neg_items = np.fromiter((self.random_neq(self.nb_items, self.train_data_set[u], self.train_pos_item_id[u]) for u in self.train_user_id), dtype=np.int32)
        train_list_prev_items_id_neg = self.train_list_prev_items_id.copy()
        train_list_prev_items_id_neg[train_list_prev_items_id_neg == neg_items.reshape((neg_items.shape[0], 1))] = self.nb_items

        # Shuffle
        batch_size = self.train_user_id.shape[0]
        mini_batch_idx = np.random.choice(batch_size, size=batch_size, replace=False)
        users = self.train_user_id[mini_batch_idx]
        prev_items = self.train_prev_item_id[mini_batch_idx]
        list_prev_items_pos = self.train_list_prev_items_id_pos[mini_batch_idx, ]
        list_prev_items_neg = train_list_prev_items_id_neg[mini_batch_idx, ]
        pos_items = self.train_pos_item_id[mini_batch_idx]
        neg_items = neg_items[mini_batch_idx]

        # # no shuffle for test when change (comment previous line)
        # batch_size = users.shape[0]
        # mini_batch_idx = None
        # list_prev_items = self.sp_train_items_prev

        return (mini_batch_idx, batch_size, users, prev_items, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items)

    def generate_train_shuffled_batch_sp_with_prev_items_fsub(self):
        # Rand all neg item
        neg_items = np.random.randint(0, self.nb_items, size=(len(self.train_user_id)), dtype=np.int32)
        # neg_items = np.fromiter((self.random_neq(self.nb_items, self.train_data_set[u], self.train_pos_item_id[u]) for u in self.train_user_id), dtype=np.int32)
        train_list_prev_items_id_neg = self.train_list_prev_items_id.copy()
        train_list_prev_items_id_neg[train_list_prev_items_id_neg == neg_items.reshape((neg_items.shape[0], 1))] = self.nb_items

        # Shuffle
        batch_size = self.train_user_id.shape[0]
        mini_batch_idx = np.random.choice(batch_size, size=batch_size, replace=False)
        users = self.train_user_id[mini_batch_idx]
        prev_items = self.train_prev_item_id[mini_batch_idx]
        list_prev_items_pos = self.train_list_prev_items_id_pos[mini_batch_idx, ]
        list_prev_items_neg = train_list_prev_items_id_neg[mini_batch_idx, ]
        list_fsub_items_id = self.train_fsub_items_id[mini_batch_idx, ]
        list_fsub_items_values = self.train_fsub_items_value[mini_batch_idx, ]
        pos_items = self.train_pos_item_id[mini_batch_idx]
        neg_items = neg_items[mini_batch_idx]

        # # no shuffle for test when change (comment previous line)
        # batch_size = users.shape[0]
        # mini_batch_idx = None
        # list_prev_items = self.sp_train_items_prev

        return (mini_batch_idx, batch_size, users, prev_items, list_fsub_items_id, list_fsub_items_values, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items)

    def generate_val_batch_sp_with_prev_items(self, items_per_user=100):
        users = np.repeat(self.valid_user_id, items_per_user)
        prev_items = np.repeat(self.valid_prev_item_id, items_per_user, axis=0)
        list_prev_items_pos = np.repeat(self.valid_list_prev_items_id_pos, items_per_user, axis=0)
        pos_items = np.repeat(self.valid_pos_item_id, items_per_user)
        neg_items = np.random.randint(0, self.nb_items, size=len(self.valid_user_id)*items_per_user, dtype=np.int32)
        list_prev_items_neg = np.repeat(self.valid_list_prev_items_id, items_per_user, axis=0)
        list_prev_items_neg[list_prev_items_neg == neg_items.reshape((neg_items.shape[0], 1))] = self.nb_items

        return (users, prev_items, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items)

    def generate_val_batch_sp_with_prev_items_fsub(self, items_per_user=100):
        users = np.repeat(self.valid_user_id, items_per_user)
        prev_items = np.repeat(self.valid_prev_item_id, items_per_user, axis=0)
        list_prev_items_pos = np.repeat(self.valid_list_prev_items_id_pos, items_per_user, axis=0)
        list_fsub_items_id = np.repeat(self.valid_fsub_items_id, items_per_user, axis=0)
        list_fsub_items_values = np.repeat(self.valid_fsub_items_value, items_per_user, axis=0)
        pos_items = np.repeat(self.valid_pos_item_id, items_per_user)
        neg_items = np.random.randint(0, self.nb_items, size=len(self.valid_user_id)*items_per_user, dtype=np.int32)
        list_prev_items_neg = np.repeat(self.valid_list_prev_items_id, items_per_user, axis=0)
        list_prev_items_neg[list_prev_items_neg == neg_items.reshape((neg_items.shape[0], 1))] = self.nb_items

        return (users, prev_items, list_fsub_items_id, list_fsub_items_values, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items)

    def generate_test_batch_sp_with_prev_items(self, items_per_user=100):
        users = np.repeat(self.test_user_id, items_per_user)
        prev_items = np.repeat(self.test_prev_item_id, items_per_user, axis=0)
        list_prev_items_pos = np.repeat(self.test_list_prev_items_id_pos, items_per_user, axis=0)
        pos_items = np.repeat(self.test_pos_item_id, items_per_user)
        neg_items = np.random.randint(0, self.nb_items, size=len(self.test_user_id)*items_per_user, dtype=np.int32)
        list_prev_items_neg = np.repeat(self.test_list_prev_items_id, items_per_user, axis=0)
        list_prev_items_neg[list_prev_items_neg == neg_items.reshape((neg_items.shape[0], 1))] = self.nb_items

        return (users, prev_items, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items)

    def generate_test_batch_sp_with_prev_items_fsub(self, items_per_user=100):
        users = np.repeat(self.test_user_id, items_per_user)
        prev_items = np.repeat(self.test_prev_item_id, items_per_user, axis=0)
        list_prev_items_pos = np.repeat(self.test_list_prev_items_id_pos, items_per_user, axis=0)
        list_fsub_items_id = np.repeat(self.test_fsub_items_id, items_per_user, axis=0)
        list_fsub_items_values = np.repeat(self.test_fsub_items_value, items_per_user, axis=0)
        pos_items = np.repeat(self.test_pos_item_id, items_per_user)
        neg_items = np.random.randint(0, self.nb_items, size=len(self.test_user_id)*items_per_user, dtype=np.int32)
        list_prev_items_neg = np.repeat(self.test_list_prev_items_id, items_per_user, axis=0)
        list_prev_items_neg[list_prev_items_neg == neg_items.reshape((neg_items.shape[0], 1))] = self.nb_items

        return (users, prev_items, list_fsub_items_id, list_fsub_items_values, list_prev_items_pos, list_prev_items_neg, pos_items, neg_items)

    def generate_production_data(self, user_counts, item_counts):
        # -- Prodution pass : We take all user with at least 1 action on an items with at least "args.item_min" actions (do not use for experimental test) -- #
        print('--- Production pass ---')
        if self.args.model in self.ITEMONLYMODEL:
            self.nb_events_prod = 0
            self.nb_users_prod = 0
            self.dict_user_name_to_id_prod = {}
            self.dict_user_id_to_name_prod = {}
            self.prod_data = []
            for row in self.df_data.itertuples():
                # print(row)
                user_name = row[1]
                item_name = row[2]
                value = row[3]
                timespam = row[4]

                if item_name not in self.dict_item_name_to_id:  # New item
                    continue

                self.nb_events_prod += 1

                if user_name not in self.dict_user_name_to_id_prod:  # New user
                    self.dict_user_id_to_name_prod[self.nb_users_prod] = user_name
                    self.dict_user_name_to_id_prod[user_name] = self.nb_users_prod
                    self.nb_users_prod += 1
                    self.prod_data.append([])  # add a new vec

                self.prod_data[self.dict_user_name_to_id_prod[user_name]].append([self.dict_item_name_to_id[item_name], timespam])

            # Sort item with timespam
            for u in range(0, self.nb_users_prod):
                self.prod_data[u] = sorted(self.prod_data[u], key=lambda user: user[1], reverse=False)
        else:
            self.nb_events_prod = self.nb_events
            self.nb_users_prod = self.nb_users
            self.dict_user_name_to_id_prod = self.dict_user_name_to_id.copy()
            self.dict_user_id_to_name_prod = self.dict_user_id_to_name.copy()
            self.prod_data = self.train_data.copy()

        self.nb_users_filtered = self.nb_users_prod - self.nb_users
        self.nb_users_invalid = self.nb_users_init - self.nb_users_prod

        print('\t nb_users_prod  = ' + str(self.nb_users_prod))
        print('\t nb_users_filtered  = ' + str(self.nb_users_filtered))
        print('\t nb_users_invalid  = ' + str(self.nb_users_invalid))
        print('\t nb_items = ' + str(self.nb_items))
        print('\t nb_events_prod  = ' + str(self.nb_events_prod))

        self.prod_data_set = []
        self.prod_data_set_np = []
        self.prod_data_item = []
        for u in range(0, self.nb_users_prod):  # Leave out the last two items for each user

            self.prod_data_set.append(set(map(lambda x: x[0], self.prod_data[u])))
            self.prod_data_set_np.append(np.array(list(self.prod_data_set[u])))
            self.prod_data_item.append(list(map(lambda x: x[0], self.prod_data[u])))

        self.prod_user_id = []
        self.prod_prev_item_id = []
        self.prod_list_prev_items_id = []
        self.prod_fsub_items_id = []
        self.prod_fsub_items_value = []
        self.nb_prod_events_possible = 0
        for user in range(0, self.nb_users_prod):
            item_prev = self.prod_data_item[user][-1]  # Get the last item
            self.prod_user_id.append(user)
            self.prod_prev_item_id.append(item_prev)
            self.prod_list_prev_items_id.append([])
            len_prod_data = len(self.prod_data[user])
            if len_prod_data > self.args.max_lens:  # We conserve the most recent action
                for i in range(len_prod_data-self.args.max_lens, len(self.prod_data[user])):
                    self.prod_list_prev_items_id[self.nb_prod_events_possible].append(self.prod_data[user][i][0])
            else:
                for i in range(0, len(self.prod_data[user])):
                    self.prod_list_prev_items_id[self.nb_prod_events_possible].append(self.prod_data[user][i][0])
                self.prod_list_prev_items_id[self.nb_prod_events_possible] += [self.nb_items] * (self.args.max_lens - len_prod_data)

            if self.args.model in self.FSUBMODEL:
                self.prod_fsub_items_id.append([])
                self.prod_fsub_items_value.append([])
                fsub_items = commons.find_path_stars(self.fsub_set_prod, self.prod_data_item[user].copy())
                pos = 0
                for y in fsub_items:
                    self.prod_fsub_items_id[self.nb_prod_events_possible].append(y)
                    self.prod_fsub_items_value[self.nb_prod_events_possible].append(self.fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.prod_fsub_items_id[self.nb_prod_events_possible] += [self.nb_items] * (self.args.L - len(self.prod_fsub_items_id[self.nb_prod_events_possible]))
                self.prod_fsub_items_value[self.nb_prod_events_possible] += [0] * (self.args.L - len(self.prod_fsub_items_value[self.nb_prod_events_possible]))
            else:
                self.prod_fsub_items_id.append([])
                self.prod_fsub_items_value.append([])
                fsub_items = commons.find_markov_chains_k(self.prod_data_item[user].copy(), self.args.L)
                pos = 0
                for y in fsub_items:
                    self.prod_fsub_items_id[self.nb_prod_events_possible].append(y)
                    self.prod_fsub_items_value[self.nb_prod_events_possible].append(self.fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.prod_fsub_items_id[self.nb_prod_events_possible] += [self.nb_items] * (self.args.L - len(self.prod_fsub_items_id[self.nb_prod_events_possible]))
                self.prod_fsub_items_value[self.nb_prod_events_possible] += [0] * (self.args.L - len(self.prod_fsub_items_value[self.nb_prod_events_possible]))

            self.nb_prod_events_possible += 1

        self.prod_user_id = np.array(self.prod_user_id, dtype=np.int32)
        self.prod_prev_item_id = np.array(self.prod_prev_item_id, dtype=np.int32).reshape((len(self.prod_prev_item_id), 1))
        self.prod_list_prev_items_id = np.array(self.prod_list_prev_items_id, dtype=np.int32).reshape((len(self.prod_list_prev_items_id), self.args.max_lens))
        self.prod_fsub_items_id = np.array(self.prod_fsub_items_id, dtype=np.int32).reshape((len(self.prod_fsub_items_id), self.args.L))
        self.prod_fsub_items_value = np.array(self.prod_fsub_items_value, dtype=np.float32).reshape((len(self.prod_fsub_items_value), self.args.L))
        print("Shape of prod_user_id = ", self.prod_user_id.shape)
        print("Shape of prod_prev_item_id = ", self.prod_prev_item_id.shape)
        print("Shape of prod_list_prev_items_id = ", self.prod_list_prev_items_id.shape)
        print("Shape of prod_fsub_items_id = ", self.prod_fsub_items_id.shape)
        print("nb_prod_events_possible = ", self.nb_prod_events_possible)

    # Data for cold experiment
    # A Cold user is a user that do not apper in train data (2 < user history < user_min, in this case we take only item that our modelhave a embedding)
    def generate_cold_start_user_data(self, user_counts, item_counts):
        print('--- Cold Start User pass ---')
        self.nb_events_cold_start_user = 0
        self.nb_users_cold_start_user = 0
        self.dict_user_name_to_id_cold_start_user = {}
        self.dict_user_id_to_name_cold_start_user = {}
        self.cold_start_user_data = []
        for row in self.df_data.itertuples():
            # print(row)
            user_name = row[1]
            item_name = row[2]
            value = row[3]
            timespam = row[4]

            if item_name not in self.dict_item_name_to_id or user_name in self.dict_user_name_to_id:  # Remove all none cold user
                continue

            self.nb_events_cold_start_user += 1

            if user_name not in self.dict_user_name_to_id_cold_start_user:  # New user
                self.dict_user_id_to_name_cold_start_user[self.nb_users_cold_start_user] = user_name
                self.dict_user_name_to_id_cold_start_user[user_name] = self.nb_users_cold_start_user
                self.nb_users_cold_start_user += 1
                self.cold_start_user_data.append([])  # add a new vec

            self.cold_start_user_data[self.dict_user_name_to_id_cold_start_user[user_name]].append([self.dict_item_name_to_id[item_name], timespam])

        # Sort item with timespam
        for u in range(0, self.nb_users_cold_start_user):
            self.cold_start_user_data[u] = sorted(self.cold_start_user_data[u], key=lambda user: user[1], reverse=False)

        self.nb_users_filtered = self.nb_users_cold_start_user - self.nb_users
        self.nb_users_invalid = self.nb_users_init - self.nb_users_cold_start_user

        print('\t nb_users_cold_start_user = ' + str(self.nb_users_cold_start_user))
        print('\t nb_items = ' + str(self.nb_items))
        print('\t nb_events_cold_start_user = ' + str(self.nb_events_cold_start_user))

        self.cold_start_user_data_set = []
        self.cold_start_user_data_item = []
        self.cold_start_user_test_data = []
        self.cold_start_user_test_data_set = []
        self.cold_start_user_test_data_set_np = []
        self.nb_cold_start_user_events = 0
        self.nb_cold_start_user_events_filter = 0
        self.nb_users_cold_start_user_filter = 0
        for u in range(0, self.nb_users_cold_start_user):  # Leave out the last two items for each user
            # When need at least 2 actions per user to split the data
            if len(self.cold_start_user_data[u]) < 2:
                # print("  Warning: user ", u, " has only ", len(self.cold_start_user_data[u]), " actions")
                self.cold_start_user_test_data.append([-1, -1])
            else:
                self.nb_users_cold_start_user_filter += 1
                self.nb_cold_start_user_events_filter += len(self.cold_start_user_data[u])
                # Get the truth item for the test data
                pop_test = self.cold_start_user_data[u].pop()
                item_test = pop_test[0]

                # Get the val item for the val data
                # last_item = self.cold_start_user_data[u][-1][0]
                last_item = self.cold_start_user_data[u].pop()
                last_item = last_item[0]

                self.cold_start_user_test_data.append([item_test, last_item])

            self.cold_start_user_data_set.append(set(map(lambda x: x[0], self.cold_start_user_data[u])))

            self.cold_start_user_test_data_set.append(set(map(lambda x: x[0], self.cold_start_user_data[u])))
            self.cold_start_user_test_data_set[u].add(self.cold_start_user_test_data[u][0])
            self.cold_start_user_test_data_set[u].add(self.cold_start_user_test_data[u][1])
            self.cold_start_user_test_data_set_np.append(np.array(list(self.cold_start_user_test_data_set[u])))

            self.cold_start_user_data_item.append(list(map(lambda x: x[0], self.cold_start_user_data[u])))
            self.nb_cold_start_user_events += len(self.cold_start_user_data[u])

            # if u % 1000 == 0:
            #     print("User {}, time {}".format(u, time.time()-start_time_test))
            #     gc.collect()

        print("self.nb_users_cold_start_user_filter = ", self.nb_users_cold_start_user_filter)
        print("self.nb_cold_start_user_events = ", self.nb_cold_start_user_events)

        self.cold_start_user_test_user_id = []
        self.cold_start_user_test_pos_item_id = []
        self.cold_start_user_test_prev_item_id = []
        self.cold_start_user_test_list_prev_items_id = []
        self.cold_start_user_test_fsub_items_id = []
        self.cold_start_user_test_fsub_items_value = []
        self.nb_cold_start_user_test_events_possible = 0
        for user in range(0, self.nb_users_cold_start_user):
            item_pos = self.cold_start_user_test_data[user][0]
            item_prev = self.cold_start_user_test_data[user][1]
            if item_pos == -1 or item_prev == -1:
                continue
            self.cold_start_user_test_user_id.append(user)
            self.cold_start_user_test_pos_item_id.append(item_pos)
            self.cold_start_user_test_prev_item_id.append(item_prev)
            self.cold_start_user_test_list_prev_items_id.append([])
            len_cold_start_user_data = len(self.cold_start_user_data[user])+1  # We add +1 beacause we need to append item_prev (val_pos_item)
            if len_cold_start_user_data > self.args.max_lens:  # We conserve the most recent action
                for i in range(len_cold_start_user_data-self.args.max_lens, len(self.cold_start_user_data[user])):
                    # if self.cold_start_user_data[user][i][0] != item_pos:
                    self.cold_start_user_test_list_prev_items_id[self.nb_cold_start_user_test_events_possible].append(self.cold_start_user_data[user][i][0])
                # if item_prev != item_pos:
                self.cold_start_user_test_list_prev_items_id[self.nb_cold_start_user_test_events_possible].append(item_prev)
            else:
                for i in range(0, len(self.cold_start_user_data[user])):
                    # if self.cold_start_user_data[user][i][0] != item_pos:
                    self.cold_start_user_test_list_prev_items_id[self.nb_cold_start_user_test_events_possible].append(self.cold_start_user_data[user][i][0])
                # if item_prev != item_pos:
                self.cold_start_user_test_list_prev_items_id[self.nb_cold_start_user_test_events_possible].append(item_prev)
                self.cold_start_user_test_list_prev_items_id[self.nb_cold_start_user_test_events_possible] += [self.nb_items] * (self.args.max_lens - len_cold_start_user_data)

            if self.args.model in self.FSUBMODEL:
                self.cold_start_user_test_fsub_items_id.append([])
                self.cold_start_user_test_fsub_items_value.append([])
                tmp_set = self.cold_start_user_data_item[user].copy()
                tmp_set.append(item_prev)
                fsub_items = commons.find_path_stars(self.fsub_set, tmp_set)
                pos = 0
                for y in fsub_items:
                    self.cold_start_user_test_fsub_items_id[self.nb_cold_start_user_test_events_possible].append(y)
                    self.cold_start_user_test_fsub_items_value[self.nb_cold_start_user_test_events_possible].append(self.fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.cold_start_user_test_fsub_items_id[self.nb_cold_start_user_test_events_possible] += [self.nb_items] * (self.args.L - len(self.cold_start_user_test_fsub_items_id[self.nb_cold_start_user_test_events_possible]))
                self.cold_start_user_test_fsub_items_value[self.nb_cold_start_user_test_events_possible] += [0] * (self.args.L - len(self.cold_start_user_test_fsub_items_value[self.nb_cold_start_user_test_events_possible]))
            else:
                self.cold_start_user_test_fsub_items_id.append([])
                self.cold_start_user_test_fsub_items_value.append([])
                tmp_set = self.cold_start_user_data_item[user].copy()
                tmp_set.append(item_prev)
                fsub_items = commons.find_markov_chains_k(tmp_set, self.args.L)
                pos = 0
                for y in fsub_items:
                    self.cold_start_user_test_fsub_items_id[self.nb_cold_start_user_test_events_possible].append(y)
                    self.cold_start_user_test_fsub_items_value[self.nb_cold_start_user_test_events_possible].append(self.fsub_damping[len(fsub_items)-1][pos])
                    pos += 1
                self.cold_start_user_test_fsub_items_id[self.nb_cold_start_user_test_events_possible] += [self.nb_items] * (self.args.L - len(self.cold_start_user_test_fsub_items_id[self.nb_cold_start_user_test_events_possible]))
                self.cold_start_user_test_fsub_items_value[self.nb_cold_start_user_test_events_possible] += [0] * (self.args.L - len(self.cold_start_user_test_fsub_items_value[self.nb_cold_start_user_test_events_possible]))

            self.nb_cold_start_user_test_events_possible += 1

        self.cold_start_user_test_user_id = np.array(self.cold_start_user_test_user_id, dtype=np.int32)
        self.cold_start_user_test_pos_item_id = np.array(self.cold_start_user_test_pos_item_id, dtype=np.int32)
        self.cold_start_user_test_prev_item_id = np.array(self.cold_start_user_test_prev_item_id, dtype=np.int32).reshape((len(self.cold_start_user_test_prev_item_id), 1))
        self.cold_start_user_test_list_prev_items_id = np.array(self.cold_start_user_test_list_prev_items_id, dtype=np.int32).reshape((len(self.cold_start_user_test_list_prev_items_id), self.args.max_lens))
        self.cold_start_user_test_fsub_items_id = np.array(self.cold_start_user_test_fsub_items_id, dtype=np.int32).reshape((len(self.cold_start_user_test_fsub_items_id), self.args.L))
        self.cold_start_user_test_fsub_items_value = np.array(self.cold_start_user_test_fsub_items_value, dtype=np.float32).reshape((len(self.cold_start_user_test_fsub_items_value), self.args.L))
        print("Shape of self.cold_start_user_test_user_id = ", self.cold_start_user_test_user_id.shape)
        print("Shape of self.cold_start_user_test_pos_item_id = ", self.cold_start_user_test_pos_item_id.shape)
        print("Shape of self.cold_start_user_test_prev_item_id = ", self.cold_start_user_test_prev_item_id.shape)
        print("Shape of self.cold_start_user_test_list_prev_items_id = ", self.cold_start_user_test_list_prev_items_id.shape)
        print("Shape of self.cold_start_user_test_fsub_items_id = ", self.cold_start_user_test_fsub_items_id.shape)
        print("Shape of self.cold_start_user_test_fsub_items_value = ", self.cold_start_user_test_fsub_items_value.shape)
        print("self.nb_cold_start_user_test_events_possible = ", self.nb_cold_start_user_test_events_possible)
        self.cold_start_user_test_list_prev_items_id_pos = self.cold_start_user_test_list_prev_items_id.copy()
        self.cold_start_user_test_list_prev_items_id_pos[self.cold_start_user_test_list_prev_items_id_pos == self.cold_start_user_test_pos_item_id.reshape((self.cold_start_user_test_pos_item_id.shape[0], 1))] = self.nb_items
