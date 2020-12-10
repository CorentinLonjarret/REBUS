# -*- coding: utf-8 -*-
"""
Corpus class for the data preparation

Based on "Fusing Similarity Models with Markov Chains for Sparse Sequential Recommendation",
   Ruining He, Julian McAuley
    link : http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm16a.pdf
"""
import time
import pandas as pd
import os
import yaml
import datetime
from pytz import timezone


class Corpus(object):
    """
    Corpus constructor
        pos_per_user : Vector of Vector that store per user (#u) a pair <Item (#i), timespam>
        nUsers : Number of users
        nItems : Number of items
        nClicks : Number of clicks/Actions
        userIds : Maps a user's string-valued ID to an integer
        itemIds : Maps a item's string-valued ID to an integer
        rUserIds : Maps an integer to a user's string-valued ID
        rItemIds : Maps an integer to a item's string-valued ID

        clicked_per_user : Vector of Set of action (store #item) for a user
        val_per_user : Vector of pair that store item for validation (item_val, item_prev)
        test_per_user :Vector of pair that store item for test (item_test, item_prev_val)
        teststamp_per_user : Vector of pair that store item's timespam for test (t_test, t_val)

        num_pos_events : Number of events in train data
        userMin : Minimal number of action for a user
        itemMin : Minimal number of action for a item
    """

    def __init__(self):
        print("Construtor of the Corpus's class : \n")
        self.data = pd.DataFrame({'A': []})
        self.pos_per_user = []
        self.pos_per_user_test = []
        self.pos_per_user_all = []
        self.nUsers = 0
        self.nItems = 0
        self.nClicks = 0
        self.userIds = {}
        self.itemIds = {}
        self.rUserIds = {}
        self.rItemIds = {}
        # Ancienement dans model.def foo():
        self.clicked_per_user = []
        self.val_per_user = []
        self.test_per_user = []
        self.teststamp_per_user = []
        self.splited = False

        self.num_pos_events = 0
        self.userMin = -1
        self.itemMin = -1

    """
    Load the data
        pathFile : path of the data file to load
        userMin : thresholds for the number of users
        itemMin : thresholds for the number of items
    """

    def loadData(self, pathFile, userMin, itemMin, suffix_file_data, name_file_data, savePrep=False, myCADservice_DL_by_days=25, savePath=None):
        print("--- Loading data with clicks from ", pathFile,
              ", userMin = ", userMin, ", itemMin = ", itemMin, "\n")

        self.__init__()
        self.userMin = userMin
        self.itemMin = itemMin

        if pathFile.split('.')[len(pathFile.split('.'))-1] == 'csv':
            self.data = pd.read_csv(pathFile, sep=",", header=None, names=[
                                    'userID', 'itemID', 'rating', 'time'])
        elif pathFile.split('.')[len(pathFile.split('.'))-1] == 'txt':
            self.data = pd.read_csv(pathFile, sep="\s+", header=None,
                                    names=['userID', 'itemID', 'rating', 'time'])
        else:
            raise('Type de données non pris en charge')

        self.data["userID"] = self.data["userID"].fillna("user_null")
        self.data["itemID"] = self.data["itemID"].fillna("item_null")

        print('-- No filter data : #user = ', len(self.data.groupby(['userID']).size().to_dict()), ' & #item = ', len(
            self.data.groupby(['itemID']).size().to_dict()), '& #line = ', len(self.data))

        print('-- Filter data : #user = ', len(self.data.groupby(['userID']).size().to_dict()), ' & #item = ', len(
            self.data.groupby(['itemID']).size().to_dict()), '& #line = ', len(self.data))

        uCounts = self.data.groupby(['userID']).size().to_dict()
        iCounts = self.data.groupby(['itemID']).size().to_dict()

        print('--- First pass: #users = ', len(uCounts), ', #items = ',
              len(iCounts), ', #clicks = ', len(self.data))

        # # Filter all user with less than userMin actions
        nRead = 0
        for row in self.data.itertuples():
            # print(row)
            uName = row[1]
            iName = row[2]
            value = row[3]
            voteTime = row[4]
            nRead += 1

            if uCounts[uName] < self.userMin:
                continue

            if iCounts[iName] < self.itemMin:
                continue

            self.nClicks += 1

            if iName not in self.itemIds:  # New item
                self.rItemIds[self.nItems] = iName
                self.itemIds[iName] = self.nItems
                self.nItems += 1

            if uName not in self.userIds:  # New user
                self.rUserIds[self.nUsers] = uName
                self.userIds[uName] = self.nUsers
                self.nUsers += 1
                self.pos_per_user.append([])  # add a new vec
                self.pos_per_user_test.append([])  # add a new vec
                self.pos_per_user_all.append([])  # add a new vec

            self.pos_per_user[self.userIds[uName]].append([self.itemIds[iName], voteTime])
            self.pos_per_user_test[self.userIds[uName]].append([self.itemIds[iName], voteTime])
            self.pos_per_user_all[self.userIds[uName]].append([self.itemIds[iName], voteTime])

        for u in range(0, self.nUsers):
            self.pos_per_user[u] = sorted(
                self.pos_per_user[u], key=lambda user: user[1], reverse=False)
            self.pos_per_user_test[u] = sorted(
                self.pos_per_user_test[u], key=lambda user: user[1], reverse=False)
            self.pos_per_user_all[u] = sorted(
                self.pos_per_user_all[u], key=lambda user: user[1], reverse=False)

        print('--- Final pass: nUsers = ', self.nUsers, ', nItems = ',
              self.nItems, ', nClicks = ', self.nClicks)

    def splitDataTrainValTest(self):
        print("--- Split Data Train Val Test \n")

        if not self.splited:
            for u in range(0, self.nUsers):  # Leave out the last two items for each user
                # When need at least 3 actions per user to split the data
                if len(self.pos_per_user[u]) < 3:
                    # print("  Warning: user ", u, " has only ", len(self.pos_per_user[u]), " actions")
                    self.test_per_user.append([-1, -1])
                    self.teststamp_per_user.append([-1, -1])
                    self.val_per_user.append([-1, -1])
                else:
                    # Get the truth item for the test data
                    pop_test = self.pos_per_user[u].pop()
                    self.pos_per_user_test[u].pop()
                    item_test = pop_test[0]
                    test_stamp = pop_test[1]

                    # Get the truth item for the val data and the test item for the test data
                    pop_val = self.pos_per_user[u].pop()
                    item_val = pop_val[0]
                    val_stamp = pop_val[1]

                    # Get the val item for the val data
                    item_prev = self.pos_per_user[u][-1][0]

                    self.test_per_user.append([item_test, item_val])
                    self.teststamp_per_user.append([test_stamp, val_stamp])
                    self.val_per_user.append([item_val, item_prev])

                self.clicked_per_user.append(set(map(lambda x: x[0], self.pos_per_user[u])))
                self.num_pos_events += len(self.pos_per_user[u])

            print("num_pos_events = ", self.num_pos_events)
            self.splited = True
        else:
            print("--- Data already splited !")

    def get_all_substrings(self, input_string):
        length = len(input_string)
        return [str("-").join(str(x) for x in input_string[i:j+1]) for i in range(length) for j in range(i, length)]

    def create_fsub(self, dataset, tabMinCount, tabL):
        l_dlStart = time.clock()

        if not os.path.exists(os.path.join('96-FSUB', ('userMin_' + str(self.userMin) + "_itemMin_" + str(self.itemMin)))):
            os.makedirs(os.path.join('96-FSUB', ('userMin_' +
                                                 str(self.userMin) + "_itemMin_" + str(self.itemMin))))

        list_histo_users = [[y[0] for y in x] for x in self.pos_per_user]
        count_sub_sequences_user = []
        dict_sub_sequences = {}

        for user_id in range(self.nUsers):
            if user_id % 1000 == 0:
                print(user_id)
            sub_sequences = self.get_all_substrings(list_histo_users[user_id])
            count_sub_sequences_user.append({x: sub_sequences.count(x) for x in sub_sequences})
            for sub in count_sub_sequences_user[user_id]:
                if sub not in dict_sub_sequences:
                    dict_sub_sequences[sub] = [
                        sub.count('-') + 1, count_sub_sequences_user[user_id][sub]]
                else:
                    dict_sub_sequences[sub][1] = dict_sub_sequences[sub][1] + \
                        count_sub_sequences_user[user_id][sub]

            del sub_sequences

        dict_sub_sequences['Root'] = [1, 200]
        len(dict_sub_sequences)
        end = time.clock() - l_dlStart
        print("Fsub : ", end)

        file_to_save = dataset + '_root_fsub_minCount_' + str(1) + '_L_' + str(1) + '.txt'

        path_to_save = os.path.join(
            '96-FSUB', ('userMin_' + str(self.userMin) + "_itemMin_" + str(self.itemMin)), file_to_save)
        pd.DataFrame([x for x in range(self.nItems)]).to_csv(
            path_to_save, encoding='utf-8', index=False, header=False)

        for minCount in tabMinCount:
            for L in tabL:
                print('minCount : ', minCount, ' L : ', L)

                file_to_save = dataset + '_root_fsub_minCount_' + \
                    str(minCount) + '_L_' + str(L) + '.txt'

                path_to_save = os.path.join(
                    '96-FSUB', ('userMin_' + str(self.userMin) + "_itemMin_" + str(self.itemMin)), file_to_save)

                if minCount == 0:
                    pd.DataFrame([x for x in range(self.nItems)]).to_csv(
                        path_to_save, encoding='utf-8', index=False, header=False)
                else:
                    df_dataset_fsub = pd.DataFrame.from_dict(dict_sub_sequences, orient='index')
                    df_dataset_fsub.reset_index(level=0, inplace=True)
                    df_dataset_fsub.columns = ['sub_string', 'size', 'freq_sub']
                    df_dataset_fsub = df_dataset_fsub.sort_values(
                        by=['freq_sub', 'size'], ascending=[0, 1])
                    df_dataset_fsub = df_dataset_fsub.loc[(
                        df_dataset_fsub['freq_sub'] >= minCount) & (df_dataset_fsub['size'] <= L)]

                    df_dataset_fsub['sub_string'].to_csv(
                        path_to_save, encoding='utf-8', index=False, header=False)
