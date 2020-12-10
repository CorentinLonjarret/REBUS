import numpy as np
import pandas as pd
import random
import os
import argparse
import sys
import time
from datetime import date


###################################################
###################### ARGS #######################
###################################################
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        help='Filename of the input dataset.',
                        required=True)
    parser.add_argument('--model',
                        help='Model to run.',
                        choices=['REBUS', 'REBUS_reg', 'REBUS_simple', 'REBUS_ST', 'REBUS_ST_simple', 'REBUS_LT'],
                        required=True)
    parser.add_argument('--damping_fsub',
                        help='Type of damping factor use for REBUS',
                        choices=['linear_softmax', 'linear'],
                        default="linear_softmax")
    parser.add_argument('--max_iters',
                        help='Max number of iterations to run',
                        default=10000,
                        type=int)
    parser.add_argument('--quit_delta',
                        help='Number of iterations at which to quit if no improvement.',
                        default=250,
                        type=int)
    parser.add_argument('--eval_freq',
                        help='Frequency at which to evaluate model.',
                        default=25,
                        type=int)
    parser.add_argument('--item_per_user',
                        help='Number of items test during validation .',
                        default=100,
                        type=int)
    parser.add_argument('--learning_rate',
                        help='Initial learning rate.',
                        default=0.001,
                        type=float)
    parser.add_argument('--num_dims',
                        help='Model dimensionality.',
                        default=10,
                        type=int)
    parser.add_argument('--mini_batch_size',
                        help='Size of the mini Batch.',
                        default=128,
                        type=int)
    parser.add_argument('--max_lens',
                        help='maximun lenght for long term history',
                        default=100,
                        type=int)
    parser.add_argument('--user_min',
                        help='Number of minimal actions for a user',
                        default=5,
                        type=int)
    parser.add_argument('--item_min',
                        help='Number of minimal interaction for a item',
                        default=5,
                        type=int)
    parser.add_argument('--min_count',
                        help='Minimun times that a sequence appears in all users history',
                        default=1,
                        type=int)
    parser.add_argument('--L',
                        help='Maximun size of a sequence',
                        default=1,
                        type=int)
    parser.add_argument('--emb_reg',
                        help='L2 regularization: embbeding regularization.',
                        default=0.001,
                        type=float)
    parser.add_argument('--bias_reg',
                        help='L2 regularization: Bias regularization.',
                        default=0.001,
                        type=float)
    parser.add_argument('--alpha',
                        help='Alpha for long term.',
                        default=-1.0,
                        type=float)
    parser.add_argument('--gamma',
                        help='Gamma to unified the short term and the long term. 0 equal to have only short term, 1 equal to have only long term',
                        default=0.5,
                        type=float)
    parser.add_argument('--seed',
                        help='Seed',
                        default=1,
                        type=int)
    parser.add_argument('--prediction_name_file',
                        help='Prediction\'s name file',
                        default=None)
    parser.add_argument('--prediction_TopN',
                        help='TopN recommandation to keep per users',
                        default=25,
                        type=int)
    parser.add_argument('--evaluation_name_file',
                        help='Evaluation\'s name file',
                        default=None)
    parser.add_argument('--search_hyperparameters_name_file',
                        help='Name of the search hyperparameters file (i.e. All evaluation of a grid serach)',
                        default=None)
    parser.add_argument('--production', type=str2bool, nargs='?', const=True,
                        help='Args to make prediction in all data available (Mode for production use cases not experimental study) ',
                        default=False)
    parser.add_argument('--cold_start_user', type=str2bool, nargs='?', const=True,
                        help='Args to make prediction in cold start data ',
                        default=False)
    args = parser.parse_args()
    print(args)
    print('')
    return(args)


###################################################
###### EVALUATIONS & PREDICTIONS FUNCTIONS ########
###################################################
HEADER_EVALUATION = ['model', 'dataset', 'date',
                     'nUsers_init', 'nUsers', 'nUsers_invalid', 'nItems_init', 'nItems', 'nClicks_init', 'nClicks', 'num_pos_events',
                     'damping_fsub', 'max_iters', 'quit_delta', 'eval_freq', 'item_per_user',
                     'learning_rate', 'num_dims', 'mini_batch_size', 'max_lens', 'user_min', 'item_min',
                     'min_count', 'L', 'emb_reg', 'bias_reg', 'alpha', 'gamma',
                     'timeToTrain', 'timeToEval', 'bestIte',
                     'best_AUC_val', 'best_AUC_test', 'AUC_val', 'AUC_test',
                     'HIT5_val', 'HIT5_test', 'HIT10_val', 'HIT10_test', 'HIT25_val', 'HIT25_test', 'HIT50_val', 'HIT50_test',
                     'NDCG5_val', 'NDCG5_test', 'NDCG10_val', 'NDCG10_test', 'NDCG25_val', 'NDCG25_test', 'NDCG50_val', 'NDCG50_test',
                     'MRR_val', 'MRR_test']

HEADER_PREDICTION = ['model', 'dataset', 'date',
                     'user', 'userID', 'item', 'itemID',
                     'ranking', 'prediction']


def save_results(model, results_model):
    if model.args.evaluation_name_file is None:
        results_name_file = (model.args.model + "_" + model.dataset.data_name +
                             "_fsubT_nextItem1_batch_" + str(model.args.mini_batch_size) +
                             "_userMin_" + str(model.args.user_min) + "_itemMin_" + str(model.args.item_min) +
                             "_dims_" + str(model.args.num_dims) + "_max_lens_" + str(model.args.max_lens) +
                             "_" + model.args.damping_fsub + ".csv")
    else:
        results_name_file = model.args.evaluation_name_file

    results_path = os.path.join("02-Resultats", "Evaluations", results_name_file)

    print("Creation of \"" + results_path + "\"")
    df_results_model = pd.DataFrame(columns=HEADER_EVALUATION)
    df_results_model.loc[0] = [model.args.model, model.dataset.data_name, date.today().strftime("%d/%m/%Y"),
                               model.dataset.nb_users_init, model.dataset.nb_users, model.dataset.nb_users_invalid, model.dataset.nb_items_init, model.dataset.nb_items, model.dataset.nb_events_init, model.dataset.nb_events, model.dataset.nb_train_events,
                               model.args.damping_fsub, model.args.max_iters, model.args.quit_delta, model.args.eval_freq, model.args.item_per_user,
                               model.args.learning_rate, model.args.num_dims, model.args.mini_batch_size, model.args.max_lens, model.args.user_min, model.args.item_min,
                               model.args.min_count, model.args.L, model.args.emb_reg, model.args.bias_reg, model.args.alpha, model.args.gamma,
                               results_model["time_to_train"], results_model["time_to_eval"], results_model["best_epoch"],
                               results_model["best_val_auc"], results_model["best_test_auc"], results_model["valid_metrics"]['AUC'], results_model["test_metrics"]['AUC'],
                               results_model["valid_metrics"]['HIT_5'], results_model["test_metrics"]['HIT_5'], results_model["valid_metrics"]['HIT_10'], results_model["test_metrics"]['HIT_10'],
                               results_model["valid_metrics"]['HIT_25'], results_model["test_metrics"]['HIT_25'], results_model["valid_metrics"]['HIT_50'], results_model["test_metrics"]['HIT_50'],
                               results_model["valid_metrics"]['NDCG_5'], results_model["test_metrics"]['NDCG_5'], results_model["valid_metrics"]['NDCG_10'], results_model["test_metrics"]['NDCG_10'],
                               results_model["valid_metrics"]['NDCG_25'], results_model["test_metrics"]['NDCG_25'], results_model["valid_metrics"]['NDCG_50'], results_model["test_metrics"]['NDCG_50'],
                               results_model["valid_metrics"]['MRR'], results_model["test_metrics"]['MRR']]

    df_results_model.to_csv(results_path, encoding='utf-8', index=False)

    if model.args.prediction_name_file is None:
        prediction_name_file = (model.args.model + "_" + model.dataset.data_name +
                                "_fsubT_nextItem1_batch_" + str(model.args.mini_batch_size) +
                                "_userMin_" + str(model.args.user_min) + "_itemMin_" + str(model.args.item_min) +
                                "_dims_" + str(model.args.num_dims) + "_max_lens_" + str(model.args.max_lens) +
                                "_" + model.args.damping_fsub + ".csv")
    else:
        prediction_name_file = model.args.prediction_name_file

    prediction_path = os.path.join("02-Resultats", "Predictions", prediction_name_file)
    print("Creation of \"" + prediction_path + "\"")

    if len(results_model["test_metrics"]['topK_predictions_users']) != len(results_model["test_metrics"]['topK_predictions_score']):
        print("Error : In lenght for prediction")
        exit()

    if len(results_model["test_metrics"]['topK_predictions_users']) != len(results_model["test_metrics"]['topK_predictions_items']):
        print("Error : In lenght for prediction")
        exit()

    list_prediction_model = []
    for u in range(len(results_model["test_metrics"]['topK_predictions_users'])):
        for k in range(model.dataset.args.prediction_TopN):
            if model.dataset.args.production:  # Take dict prodution else take dick train
                list_prediction_model .append([model.dataset.args.model, model.dataset.data_name, date.today().strftime("%d/%m/%Y"),
                                               results_model["test_metrics"]['topK_predictions_users'][u], model.dataset.dict_user_id_to_name_prod[results_model["test_metrics"]['topK_predictions_users'][u]],
                                               results_model["test_metrics"]['topK_predictions_items'][u][k], model.dataset.dict_item_id_to_name[results_model["test_metrics"]['topK_predictions_items'][u][k]],
                                               k, results_model["test_metrics"]['topK_predictions_score'][u][k]
                                               ])
            else:
                list_prediction_model .append([model.dataset.args.model, model.dataset.data_name, date.today().strftime("%d/%m/%Y"),
                                               results_model["test_metrics"]['topK_predictions_users'][u], model.dataset.dict_user_id_to_name[results_model["test_metrics"]['topK_predictions_users'][u]],
                                               results_model["test_metrics"]['topK_predictions_items'][u][k], model.dataset.dict_item_id_to_name[results_model["test_metrics"]['topK_predictions_items'][u][k]],
                                               k, results_model["test_metrics"]['topK_predictions_score'][u][k]
                                               ])

    df_prediction_model = pd.DataFrame(list_prediction_model)
    df_prediction_model.columns = HEADER_PREDICTION
    df_prediction_model.head(5)
    df_prediction_model.to_csv(prediction_path, encoding='utf-8', index=False)


def save_results_cold_start(model, results_model):
    if model.args.evaluation_name_file is None:
        results_name_file = (model.args.model + "_" + model.dataset.data_name +
                             "_fsubT_nextItem1_batch_" + str(model.args.mini_batch_size) +
                             "_userMin_" + str(model.args.user_min) + "_itemMin_" + str(model.args.item_min) +
                             "_dims_" + str(model.args.num_dims) + "_max_lens_" + str(model.args.max_lens) +
                             "_" + model.args.damping_fsub + ".csv")
    else:
        results_name_file = model.args.evaluation_name_file

    results_path = os.path.join("02-Resultats", "Evaluations", results_name_file)

    print("Creation of \"" + results_path + "\"")
    df_results_model = pd.DataFrame(columns=HEADER_EVALUATION)
    df_results_model.loc[0] = [model.args.model, model.dataset.data_name, date.today().strftime("%d/%m/%Y"),
                               model.dataset.nb_users_init, model.dataset.nb_users, model.dataset.nb_users_invalid, model.dataset.nb_items_init, model.dataset.nb_items, model.dataset.nb_events_init, model.dataset.nb_events, model.dataset.nb_train_events,
                               model.args.damping_fsub, model.args.max_iters, model.args.quit_delta, model.args.eval_freq, model.args.item_per_user,
                               model.args.learning_rate, model.args.num_dims, model.args.mini_batch_size, model.args.max_lens, model.args.user_min, model.args.item_min,
                               model.args.min_count, model.args.L, model.args.emb_reg, model.args.bias_reg, model.args.alpha, model.args.gamma,
                               results_model["time_to_train"], results_model["time_to_eval"], results_model["best_epoch"],
                               results_model["best_val_auc"], results_model["best_test_auc"], results_model["valid_metrics"]['AUC'], results_model["test_metrics"]['AUC'],
                               results_model["valid_metrics"]['HIT_5'], results_model["test_metrics"]['HIT_5'], results_model["valid_metrics"]['HIT_10'], results_model["test_metrics"]['HIT_10'],
                               results_model["valid_metrics"]['HIT_25'], results_model["test_metrics"]['HIT_25'], results_model["valid_metrics"]['HIT_50'], results_model["test_metrics"]['HIT_50'],
                               results_model["valid_metrics"]['NDCG_5'], results_model["test_metrics"]['NDCG_5'], results_model["valid_metrics"]['NDCG_10'], results_model["test_metrics"]['NDCG_10'],
                               results_model["valid_metrics"]['NDCG_25'], results_model["test_metrics"]['NDCG_25'], results_model["valid_metrics"]['NDCG_50'], results_model["test_metrics"]['NDCG_50'],
                               results_model["valid_metrics"]['MRR'], results_model["test_metrics"]['MRR']]

    df_results_model.to_csv(results_path, encoding='utf-8', index=False)

    if model.args.prediction_name_file is None:
        prediction_name_file = (model.args.model + "_" + model.dataset.data_name +
                                "_fsubT_nextItem1_batch_" + str(model.args.mini_batch_size) +
                                "_userMin_" + str(model.args.user_min) + "_itemMin_" + str(model.args.item_min) +
                                "_dims_" + str(model.args.num_dims) + "_max_lens_" + str(model.args.max_lens) +
                                "_" + model.args.damping_fsub + ".csv")
    else:
        prediction_name_file = model.args.prediction_name_file

    prediction_path = os.path.join("02-Resultats", "Predictions", prediction_name_file)
    print("Creation of \"" + prediction_path + "\"")

    if len(results_model["valid_metrics"]['topK_predictions_users']) != len(results_model["valid_metrics"]['topK_predictions_score']):
        print("Error : In lenght for prediction")
        exit()

    if len(results_model["valid_metrics"]['topK_predictions_users']) != len(results_model["valid_metrics"]['topK_predictions_items']):
        print("Error : In lenght for prediction")
        exit()

    model.dataset.dict_user_id_to_name_cold_start_user[0]
    model.dataset.dict_user_id_to_name[0]
    list_prediction_model = []
    for u in range(len(results_model["valid_metrics"]['topK_predictions_users'])):
        for k in range(model.dataset.args.prediction_TopN):
            list_prediction_model .append([model.dataset.args.model, model.dataset.data_name, date.today().strftime("%d/%m/%Y"),
                                           results_model["valid_metrics"]['topK_predictions_users'][u], model.dataset.dict_user_id_to_name_cold_start_user[results_model["valid_metrics"]['topK_predictions_users'][u]],
                                           results_model["valid_metrics"]['topK_predictions_items'][u][k], model.dataset.dict_item_id_to_name[results_model["valid_metrics"]['topK_predictions_items'][u][k]],
                                           k, results_model["valid_metrics"]['topK_predictions_score'][u][k]
                                           ])

    df_prediction_model = pd.DataFrame(list_prediction_model)
    df_prediction_model.columns = HEADER_PREDICTION
    df_prediction_model.head(5)
    df_prediction_model.to_csv(prediction_path, encoding='utf-8', index=False)


###################################################
################ EVALUATION FUNCTIONS #############
###################################################
def sample_evaluate_valid(model, d, sess):
    item_to_eval = d.args.item_per_user + 1

    AUC = 0.0
    MRR = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_25 = 0.0
    NDCG_50 = 0.0
    HIT_5 = 0.0
    HIT_10 = 0.0
    HIT_25 = 0.0
    HIT_50 = 0.0
    valid_user = 0.0
    u = -1
    for u_temp in range(d.nb_users):
        # u = 0
        if d.valid_data[u_temp][0] == -1 or d.valid_data[u_temp][1] == -1:
            # print(u)
            continue
        else:
            u += 1
            valid_user += 1.0

        users_tmp = np.repeat(d.valid_user_id[u], item_to_eval)
        prev_items_tmp = np.repeat(d.valid_prev_item_id[u].reshape((1, 1)), item_to_eval, axis=0)
        if d.args.model in d.REBUSMODEL:
            list_fsub_items_id_tmp = np.repeat(d.valid_fsub_items_id[u].reshape((1, d.args.L)), item_to_eval, axis=0)
            list_fsub_items_values_tmp = np.repeat(d.valid_fsub_items_value[u].reshape((1, d.args.L)), item_to_eval, axis=0)
        else:
            list_fsub_items_id_tmp = None
            list_fsub_items_values_tmp = None
        pos_items_tmp = np.append(np.random.randint(0, d.nb_items, size=item_to_eval-1, dtype=np.int32), d.valid_pos_item_id[u])
        list_prev_items_pos_tmp = np.append(np.repeat(d.valid_list_prev_items_id[u].reshape((1, d.args.max_lens)), item_to_eval-1, axis=0), d.valid_list_prev_items_id_pos[u].reshape((1, d.args.max_lens)), axis=0)
        list_prev_items_pos_tmp[list_prev_items_pos_tmp == pos_items_tmp.reshape((pos_items_tmp.shape[0], 1))] = d.nb_items
        predictions = model.predict(sess, users_tmp, prev_items_tmp, list_fsub_items_id_tmp, list_fsub_items_values_tmp, pos_items_tmp, list_prev_items_pos_tmp)

        predictions = predictions[0].reshape(item_to_eval)
        count_auc_test = np.float32(predictions[item_to_eval-1] > predictions[0:item_to_eval-1]).sum()
        rank = item_to_eval - (count_auc_test + 1)  # rank strat from 0 to item_to_eval-1

        AUC += count_auc_test / (item_to_eval - 1)  # We take off one item that corresponding to pos_item
        MRR += 1.0 / (rank+1)
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HIT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HIT_10 += 1
        if rank < 25:
            NDCG_25 += 1 / np.log2(rank + 2)
            HIT_25 += 1
        if rank < 50:
            NDCG_50 += 1 / np.log2(rank + 2)
            HIT_50 += 1

        # if valid_user % 500 == 0:
        #     print("user "+str(valid_user))

    AUC = AUC/valid_user
    MRR = MRR/valid_user
    NDCG_5 = NDCG_5/valid_user
    NDCG_10 = NDCG_10/valid_user
    NDCG_25 = NDCG_25/valid_user
    NDCG_50 = NDCG_50/valid_user
    HIT_5 = HIT_5/valid_user
    HIT_10 = HIT_10/valid_user
    HIT_25 = HIT_25/valid_user
    HIT_50 = HIT_50/valid_user
    return {
        'AUC': AUC, 'MRR': MRR,
        'NDCG_5': NDCG_5, 'NDCG_10': NDCG_10, 'NDCG_25': NDCG_25, 'NDCG_50': NDCG_50,
        'HIT_5': HIT_5, 'HIT_10': HIT_10, 'HIT_25': HIT_25, 'HIT_50': HIT_50,
    }


def sample_evaluate_valid_faster(model, d, sess):
    print("*** Launch sample_evaluate_valid_faster ****")
    item_to_eval = d.args.item_per_user + 1

    AUC = 0.0
    MRR = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_25 = 0.0
    NDCG_50 = 0.0
    HIT_5 = 0.0
    HIT_10 = 0.0
    HIT_25 = 0.0
    HIT_50 = 0.0
    valid_user = 0.0
    u = -1
    u_eval = -1
    user_to_eval = 500
    users = np.zeros((item_to_eval*user_to_eval,))
    prev_items = np.zeros((item_to_eval*user_to_eval, 1))
    if d.args.model in d.REBUSMODEL:
        list_fsub_items_id = np.zeros((item_to_eval*user_to_eval, d.args.L))
        list_fsub_items_values = np.zeros((item_to_eval*user_to_eval, d.args.L))
    else:
        list_fsub_items_id = None
        list_fsub_items_values = None
    pos_items = np.zeros((item_to_eval*user_to_eval,))
    list_prev_items_pos = np.zeros((item_to_eval*user_to_eval, d.args.max_lens))
    for u_temp in range(d.nb_users):
        # u = 0
        if d.valid_data[u_temp][0] != -1 or d.valid_data[u_temp][1] != -1:
            u += 1
            u_eval += 1
            valid_user += 1.0

            users_tmp = np.repeat(d.valid_user_id[u], item_to_eval)
            prev_items_tmp = np.repeat(d.valid_prev_item_id[u].reshape((1, 1)), item_to_eval, axis=0)
            if d.args.model in d.REBUSMODEL:
                list_fsub_items_id_tmp = np.repeat(d.valid_fsub_items_id[u].reshape((1, d.args.L)), item_to_eval, axis=0)
                list_fsub_items_values_tmp = np.repeat(d.valid_fsub_items_value[u].reshape((1, d.args.L)), item_to_eval, axis=0)
            pos_items_tmp = np.append(np.random.randint(0, d.nb_items, size=item_to_eval-1, dtype=np.int32), d.valid_pos_item_id[u])
            list_prev_items_pos_tmp = np.append(np.repeat(d.valid_list_prev_items_id[u].reshape((1, d.args.max_lens)), item_to_eval-1, axis=0), d.valid_list_prev_items_id_pos[u].reshape((1, d.args.max_lens)), axis=0)
            list_prev_items_pos_tmp[list_prev_items_pos_tmp == pos_items_tmp.reshape((pos_items_tmp.shape[0], 1))] = d.nb_items

            users[u_eval*item_to_eval:u_eval*item_to_eval+item_to_eval] = d.valid_user_id[u]
            prev_items[u_eval*item_to_eval:u_eval*item_to_eval+item_to_eval] = d.valid_prev_item_id[u]
            if d.args.model in d.REBUSMODEL:
                list_fsub_items_id[u_eval*item_to_eval:u_eval*item_to_eval+item_to_eval] = list_fsub_items_id_tmp
                list_fsub_items_values[u_eval*item_to_eval:u_eval*item_to_eval+item_to_eval] = list_fsub_items_values_tmp

            pos_items[u_eval*item_to_eval:u_eval*item_to_eval+item_to_eval] = pos_items_tmp
            list_prev_items_pos[u_eval*item_to_eval:u_eval*item_to_eval+item_to_eval] = list_prev_items_pos_tmp

        if u_eval == user_to_eval - 1 or u_temp == d.nb_users-1:
            predictions = model.predict(sess, users, prev_items, list_fsub_items_id, list_fsub_items_values, pos_items, list_prev_items_pos)
            predictions = predictions[0]

            if user_to_eval > d.nb_users - 1 - u_temp:
                user_to_eval = d.nb_users - u_temp - 1
            u_eval = -1
            users = np.zeros((item_to_eval*user_to_eval,))
            prev_items = np.zeros((item_to_eval*user_to_eval, 1))
            if d.args.model in d.REBUSMODEL:
                list_fsub_items_id = np.zeros((item_to_eval*user_to_eval, d.args.L))
                list_fsub_items_values = np.zeros((item_to_eval*user_to_eval, d.args.L))
            else:
                list_fsub_items_id = None
                list_fsub_items_values = None
            pos_items = np.zeros((item_to_eval*user_to_eval,))
            list_prev_items_pos = np.zeros((item_to_eval*user_to_eval, d.args.max_lens))
            for uu in range(int(predictions.shape[0]/item_to_eval)):
                count_auc_test = np.float32(predictions[uu*(item_to_eval)+item_to_eval-1] > predictions[uu*(item_to_eval):uu*(item_to_eval)+item_to_eval-1]).sum()

                rank = item_to_eval - (count_auc_test + 1)  # rank strat from 0 to item_to_eval-1

                AUC += count_auc_test / (item_to_eval - 1)  # We take off one item that corresponding to pos_item
                MRR += 1.0 / (rank+1)
                if rank < 5:
                    NDCG_5 += 1 / np.log2(rank + 2)
                    HIT_5 += 1
                if rank < 10:
                    NDCG_10 += 1 / np.log2(rank + 2)
                    HIT_10 += 1
                if rank < 25:
                    NDCG_25 += 1 / np.log2(rank + 2)
                    HIT_25 += 1
                if rank < 50:
                    NDCG_50 += 1 / np.log2(rank + 2)
                    HIT_50 += 1

    AUC = AUC/valid_user
    MRR = MRR/valid_user
    NDCG_5 = NDCG_5/valid_user
    NDCG_10 = NDCG_10/valid_user
    NDCG_25 = NDCG_25/valid_user
    NDCG_50 = NDCG_50/valid_user
    HIT_5 = HIT_5/valid_user
    HIT_10 = HIT_10/valid_user
    HIT_25 = HIT_25/valid_user
    HIT_50 = HIT_50/valid_user
    return {
        'AUC': AUC, 'MRR': MRR,
        'NDCG_5': NDCG_5, 'NDCG_10': NDCG_10, 'NDCG_25': NDCG_25, 'NDCG_50': NDCG_50,
        'HIT_5': HIT_5, 'HIT_10': HIT_10, 'HIT_25': HIT_25, 'HIT_50': HIT_50,
    }


def sample_evaluate_test(model, d, sess):
    item_to_eval = d.args.item_per_user + 1

    AUC = 0.0
    MRR = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_25 = 0.0
    NDCG_50 = 0.0
    HIT_5 = 0.0
    HIT_10 = 0.0
    HIT_25 = 0.0
    HIT_50 = 0.0
    valid_user = 0.0
    u = -1
    for u_temp in range(d.nb_users):
        if d.test_data[u_temp][0] == -1 or d.test_data[u_temp][1] == -1:
            continue
        else:
            u += 1
            valid_user += 1.0

        users_tmp = np.repeat(d.test_user_id[u], item_to_eval)
        prev_items_tmp = np.repeat(d.test_prev_item_id[u].reshape((1, 1)), item_to_eval, axis=0)
        if d.args.model in d.REBUSMODEL:
            list_fsub_items_id_tmp = np.repeat(d.test_fsub_items_id[u].reshape((1, d.args.L)), item_to_eval, axis=0)
            list_fsub_items_values_tmp = np.repeat(d.test_fsub_items_value[u].reshape((1, d.args.L)), item_to_eval, axis=0)
        else:
            list_fsub_items_id_tmp = None
            list_fsub_items_values_tmp = None

        pos_items_tmp = np.append(np.random.randint(0, d.nb_items, size=item_to_eval-1, dtype=np.int32), d.test_pos_item_id[u])
        list_prev_items_pos_tmp = np.append(np.repeat(d.test_list_prev_items_id[u].reshape((1, d.args.max_lens)), item_to_eval-1, axis=0), d.test_list_prev_items_id_pos[u].reshape((1, d.args.max_lens)), axis=0)
        list_prev_items_pos_tmp[list_prev_items_pos_tmp == pos_items_tmp.reshape((pos_items_tmp.shape[0], 1))] = d.nb_items
        predictions = model.predict(sess, users_tmp, prev_items_tmp, list_fsub_items_id_tmp, list_fsub_items_values_tmp, pos_items_tmp, list_prev_items_pos_tmp)

        predictions = predictions[0].reshape(item_to_eval)
        count_auc_test = np.float32(predictions[item_to_eval-1] > predictions[0:item_to_eval-1]).sum()
        rank = item_to_eval - (count_auc_test + 1)  # rank strat from 0 to item_to_eval-1

        AUC += count_auc_test / (item_to_eval - 1)  # We take off one item that corresponding to pos_item
        MRR += 1.0 / (rank+1)
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HIT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HIT_10 += 1
        if rank < 25:
            NDCG_25 += 1 / np.log2(rank + 2)
            HIT_25 += 1
        if rank < 50:
            NDCG_50 += 1 / np.log2(rank + 2)
            HIT_50 += 1

    AUC = AUC/valid_user
    MRR = MRR/valid_user
    NDCG_5 = NDCG_5/valid_user
    NDCG_10 = NDCG_10/valid_user
    NDCG_25 = NDCG_25/valid_user
    NDCG_50 = NDCG_50/valid_user
    HIT_5 = HIT_5/valid_user
    HIT_10 = HIT_10/valid_user
    HIT_25 = HIT_25/valid_user
    HIT_50 = HIT_50/valid_user
    return {
        'AUC': AUC, 'MRR': MRR,
        'NDCG_5': NDCG_5, 'NDCG_10': NDCG_10, 'NDCG_25': NDCG_25, 'NDCG_50': NDCG_50,
        'HIT_5': HIT_5, 'HIT_10': HIT_10, 'HIT_25': HIT_25, 'HIT_50': HIT_50,
    }


def evaluate_valid(model, d, sess):

    AUC = 0.0
    MRR = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_25 = 0.0
    NDCG_50 = 0.0
    HIT_5 = 0.0
    HIT_10 = 0.0
    HIT_25 = 0.0
    HIT_50 = 0.0
    valid_user = 0.0
    u = -1
    for u_temp in range(d.nb_users):
        if d.valid_data[u_temp][0] == -1 or d.valid_data[u_temp][1] == -1:
            continue
        else:
            u += 1
            valid_user += 1.0

        valid_data_set_eval_np = np.setdiff1d(d.np_list_items, d.valid_data_set_np[u_temp])  # We exclude item already seen by user

        item_to_eval = len(valid_data_set_eval_np) + 1  # All items that are not in user history except the pos_item

        users_tmp = np.repeat(d.valid_user_id[u], item_to_eval)
        prev_items_tmp = np.repeat(d.valid_prev_item_id[u].reshape((1, 1)), item_to_eval, axis=0)
        if d.args.model in d.REBUSMODEL:
            list_fsub_items_id_tmp = np.repeat(d.valid_fsub_items_id[u].reshape((1, d.args.L)), item_to_eval, axis=0)
            list_fsub_items_values_tmp = np.repeat(d.valid_fsub_items_value[u].reshape((1, d.args.L)), item_to_eval, axis=0)
        else:
            list_fsub_items_id_tmp = None
            list_fsub_items_values_tmp = None

        pos_items_tmp = np.append(valid_data_set_eval_np, d.valid_pos_item_id[u])

        list_prev_items_pos_tmp = np.append(np.repeat(d.valid_list_prev_items_id[u].reshape((1, d.args.max_lens)), item_to_eval-1, axis=0), d.valid_list_prev_items_id_pos[u].reshape((1, d.args.max_lens)), axis=0)
        list_prev_items_pos_tmp[list_prev_items_pos_tmp == pos_items_tmp.reshape((pos_items_tmp.shape[0], 1))] = d.nb_items
        predictions = model.predict(sess, users_tmp, prev_items_tmp, list_fsub_items_id_tmp, list_fsub_items_values_tmp, pos_items_tmp, list_prev_items_pos_tmp)

        predictions = predictions[0].reshape(item_to_eval)
        count_auc_test = np.float32(predictions[item_to_eval-1] > predictions[0:item_to_eval-1]).sum()
        rank = item_to_eval - (count_auc_test + 1)  # rank strat from 0 to item_to_eval-1

        AUC += count_auc_test / (item_to_eval - 1)  # We take off one item that corresponding to pos_item
        MRR += 1.0 / (rank+1)
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HIT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HIT_10 += 1
        if rank < 25:
            NDCG_25 += 1 / np.log2(rank + 2)
            HIT_25 += 1
        if rank < 50:
            NDCG_50 += 1 / np.log2(rank + 2)
            HIT_50 += 1

    AUC = AUC/valid_user
    MRR = MRR/valid_user
    NDCG_5 = NDCG_5/valid_user
    NDCG_10 = NDCG_10/valid_user
    NDCG_25 = NDCG_25/valid_user
    NDCG_50 = NDCG_50/valid_user
    HIT_5 = HIT_5/valid_user
    HIT_10 = HIT_10/valid_user
    HIT_25 = HIT_25/valid_user
    HIT_50 = HIT_50/valid_user
    # print('--- Validation Evaluation --- (AUC = {}), (NDCG_10 = {}), (HIT_10 = {})'.format(AUC, NDCG_10, HIT_10))
    return {
        'AUC': AUC, 'MRR': MRR,
        'NDCG_5': NDCG_5, 'NDCG_10': NDCG_10, 'NDCG_25': NDCG_25, 'NDCG_50': NDCG_50,
        'HIT_5': HIT_5, 'HIT_10': HIT_10, 'HIT_25': HIT_25, 'HIT_50': HIT_50,
        'valid_user': valid_user
    }


def evaluate_test(model, d, sess):

    AUC = 0.0
    MRR = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_25 = 0.0
    NDCG_50 = 0.0
    HIT_5 = 0.0
    HIT_10 = 0.0
    HIT_25 = 0.0
    HIT_50 = 0.0
    valid_user = 0.0
    topK_predictions_score = []
    topK_predictions_items = []
    topK_predictions_users = []
    u = -1
    for u_temp in range(d.nb_users):
        # u = 0
        if d.test_data[u_temp][0] == -1 or d.test_data[u_temp][1] == -1:
            # print(u)
            continue
        else:
            u += 1
            valid_user += 1.0

        test_data_set_eval_np = np.setdiff1d(d.np_list_items, d.test_data_set_np[u_temp])  # We exclude item already seen by user
        item_to_eval = len(test_data_set_eval_np) + 1  # All items that are not in user history except the pos_item

        users_tmp = np.repeat(d.test_user_id[u], item_to_eval)
        prev_items_tmp = np.repeat(d.test_prev_item_id[u].reshape((1, 1)), item_to_eval, axis=0)
        if d.args.model in d.REBUSMODEL:
            list_fsub_items_id_tmp = np.repeat(d.test_fsub_items_id[u].reshape((1, d.args.L)), item_to_eval, axis=0)
            list_fsub_items_values_tmp = np.repeat(d.test_fsub_items_value[u].reshape((1, d.args.L)), item_to_eval, axis=0)
        else:
            list_fsub_items_id_tmp = None
            list_fsub_items_values_tmp = None

        pos_items_tmp = np.append(test_data_set_eval_np, d.test_pos_item_id[u])

        list_prev_items_pos_tmp = np.append(np.repeat(d.test_list_prev_items_id[u].reshape((1, d.args.max_lens)), item_to_eval-1, axis=0), d.test_list_prev_items_id_pos[u].reshape((1, d.args.max_lens)), axis=0)
        list_prev_items_pos_tmp[list_prev_items_pos_tmp == pos_items_tmp.reshape((pos_items_tmp.shape[0], 1))] = d.nb_items

        predictions = model.predict(sess, users_tmp, prev_items_tmp, list_fsub_items_id_tmp, list_fsub_items_values_tmp, pos_items_tmp, list_prev_items_pos_tmp)

        predictions = predictions[0].reshape(item_to_eval)

        # Evaluation
        count_auc_test = np.float32(predictions[item_to_eval-1] > predictions[0:item_to_eval-1]).sum()
        rank = item_to_eval - (count_auc_test + 1)  # rank strat from 0 to item_to_eval-1

        AUC += count_auc_test / (item_to_eval - 1)  # We take off one item that corresponding to pos_item
        MRR += 1.0 / (rank+1)
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HIT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HIT_10 += 1
        if rank < 25:
            NDCG_25 += 1 / np.log2(rank + 2)
            HIT_25 += 1
        if rank < 50:
            NDCG_50 += 1 / np.log2(rank + 2)
            HIT_50 += 1

        # Prediction
        if (not d.grid_search) and (not d.args.production):
            sort_index = np.argsort(-predictions)
            topK_predictions_score.append(predictions[sort_index][0:d.args.prediction_TopN])
            topK_predictions_items.append(pos_items_tmp[sort_index][0:d.args.prediction_TopN])
            topK_predictions_users.append(u_temp)

    AUC = AUC/valid_user
    MRR = MRR/valid_user
    NDCG_5 = NDCG_5/valid_user
    NDCG_10 = NDCG_10/valid_user
    NDCG_25 = NDCG_25/valid_user
    NDCG_50 = NDCG_50/valid_user
    HIT_5 = HIT_5/valid_user
    HIT_10 = HIT_10/valid_user
    HIT_25 = HIT_25/valid_user
    HIT_50 = HIT_50/valid_user

    print("--- evaluate_test - valid_user {} ---".format(valid_user))

    if (not d.grid_search) and (d.args.production):
        topK_predictions_score, topK_predictions_items, topK_predictions_users = prediction_prod(model, d, sess)

    # print('--- Test Evaluation --- (AUC = {}), (NDCG_10 = {}), (HIT_10 = {})'.format(AUC, NDCG_10, HIT_10))
    return {
        'AUC': AUC, 'MRR': MRR,
        'NDCG_5': NDCG_5, 'NDCG_10': NDCG_10, 'NDCG_25': NDCG_25, 'NDCG_50': NDCG_50,
        'HIT_5': HIT_5, 'HIT_10': HIT_10, 'HIT_25': HIT_25, 'HIT_50': HIT_50,
        'valid_user': valid_user,
        'topK_predictions_score': topK_predictions_score,
        'topK_predictions_items': topK_predictions_items,
        'topK_predictions_users': topK_predictions_users,
    }


def evaluate_cold_start(model, d, sess):

    AUC = 0.0
    MRR = 0.0
    NDCG_5 = 0.0
    NDCG_10 = 0.0
    NDCG_25 = 0.0
    NDCG_50 = 0.0
    HIT_5 = 0.0
    HIT_10 = 0.0
    HIT_25 = 0.0
    HIT_50 = 0.0
    valid_user = 0.0
    topK_predictions_score = []
    topK_predictions_items = []
    topK_predictions_users = []
    u = -1
    for u_temp in range(d.nb_users_cold_start_user):
        # u = 0
        if d.cold_start_user_test_data[u_temp][0] == -1 or d.cold_start_user_test_data[u_temp][1] == -1:
            # print(u)
            continue
        else:
            u += 1
            valid_user += 1.0

        cold_start_user_test_data_set_eval_np = np.setdiff1d(d.np_list_items, d.cold_start_user_test_data_set_np[u_temp])  # We exclude item already seen by user
        item_to_eval = len(cold_start_user_test_data_set_eval_np) + 1  # All items that are not in user history except the pos_item

        users_tmp = np.repeat(d.cold_start_user_test_user_id[u], item_to_eval)
        prev_items_tmp = np.repeat(d.cold_start_user_test_prev_item_id[u].reshape((1, 1)), item_to_eval, axis=0)
        if d.args.model in d.REBUSMODEL:
            list_fsub_items_id_tmp = np.repeat(d.cold_start_user_test_fsub_items_id[u].reshape((1, d.args.L)), item_to_eval, axis=0)
            list_fsub_items_values_tmp = np.repeat(d.cold_start_user_test_fsub_items_value[u].reshape((1, d.args.L)), item_to_eval, axis=0)
        else:
            list_fsub_items_id_tmp = None
            list_fsub_items_values_tmp = None

        pos_items_tmp = np.append(cold_start_user_test_data_set_eval_np, d.cold_start_user_test_pos_item_id[u])

        list_prev_items_pos_tmp = np.append(np.repeat(d.cold_start_user_test_list_prev_items_id[u].reshape((1, d.args.max_lens)), item_to_eval-1, axis=0), d.cold_start_user_test_list_prev_items_id_pos[u].reshape((1, d.args.max_lens)), axis=0)
        list_prev_items_pos_tmp[list_prev_items_pos_tmp == pos_items_tmp.reshape((pos_items_tmp.shape[0], 1))] = d.nb_items
        predictions = model.predict(sess, users_tmp, prev_items_tmp, list_fsub_items_id_tmp, list_fsub_items_values_tmp, pos_items_tmp, list_prev_items_pos_tmp)

        predictions = predictions[0].reshape(item_to_eval)

        # Evaluation
        count_auc_test = np.float32(predictions[item_to_eval-1] > predictions[0:item_to_eval-1]).sum()
        rank = item_to_eval - (count_auc_test + 1)  # rank strat from 0 to item_to_eval-1

        AUC += count_auc_test / (item_to_eval - 1)  # We take off one item that corresponding to pos_item
        MRR += 1.0 / (rank+1)
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HIT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HIT_10 += 1
        if rank < 25:
            NDCG_25 += 1 / np.log2(rank + 2)
            HIT_25 += 1
        if rank < 50:
            NDCG_50 += 1 / np.log2(rank + 2)
            HIT_50 += 1

        if (not d.grid_search) and (not d.args.production):
            sort_index = np.argsort(-predictions)
            topK_predictions_score.append(predictions[sort_index][0:d.args.prediction_TopN])
            topK_predictions_items.append(pos_items_tmp[sort_index][0:d.args.prediction_TopN])
            topK_predictions_users.append(u_temp)

    AUC = AUC/valid_user
    MRR = MRR/valid_user
    NDCG_5 = NDCG_5/valid_user
    NDCG_10 = NDCG_10/valid_user
    NDCG_25 = NDCG_25/valid_user
    NDCG_50 = NDCG_50/valid_user
    HIT_5 = HIT_5/valid_user
    HIT_10 = HIT_10/valid_user
    HIT_25 = HIT_25/valid_user
    HIT_50 = HIT_50/valid_user

    print("--- evaluate_test - valid_user {} ---".format(valid_user))

    # print('--- Test Evaluation --- (AUC = {}), (NDCG_10 = {}), (HIT_10 = {})'.format(AUC, NDCG_10, HIT_10))
    return {
        'AUC': AUC, 'MRR': MRR,
        'NDCG_5': NDCG_5, 'NDCG_10': NDCG_10, 'NDCG_25': NDCG_25, 'NDCG_50': NDCG_50,
        'HIT_5': HIT_5, 'HIT_10': HIT_10, 'HIT_25': HIT_25, 'HIT_50': HIT_50,
        'valid_user': valid_user,
        'topK_predictions_score': topK_predictions_score,
        'topK_predictions_items': topK_predictions_items,
        'topK_predictions_users': topK_predictions_users,
    }


def prediction_prod(model, d, sess):
    print("----- Start Prediction Production -----")
    topK_predictions_score = []
    topK_predictions_items = []
    topK_predictions_users = []
    for u in range(d.nb_users_prod):
        prod_data_set_eval_np = np.setdiff1d(d.np_list_items, d.prod_data_set_np[u])

        item_to_eval = len(prod_data_set_eval_np)  # All items that are not in user history

        users_tmp = np.repeat(d.prod_user_id[u], item_to_eval)
        prev_items_tmp = np.repeat(d.prod_prev_item_id[u].reshape((1, 1)), item_to_eval, axis=0)
        if d.args.model in d.REBUSMODEL:
            list_fsub_items_id_tmp = np.repeat(d.prod_fsub_items_id[u].reshape((1, d.args.L)), item_to_eval, axis=0)
            list_fsub_items_values_tmp = np.repeat(d.prod_fsub_items_value[u].reshape((1, d.args.L)), item_to_eval, axis=0)
        else:
            list_fsub_items_id_tmp = None
            list_fsub_items_values_tmp = None
        pos_items_tmp = np.array(prod_data_set_eval_np)
        list_prev_items_pos_tmp = np.repeat(d.prod_list_prev_items_id[u].reshape((1, d.args.max_lens)), item_to_eval, axis=0)
        list_prev_items_pos_tmp[list_prev_items_pos_tmp == pos_items_tmp.reshape((pos_items_tmp.shape[0], 1))] = d.nb_items
        predictions = model.predict(sess, users_tmp, prev_items_tmp, list_fsub_items_id_tmp, list_fsub_items_values_tmp, pos_items_tmp, list_prev_items_pos_tmp)

        predictions = predictions[0].reshape(item_to_eval)

        sort_index = np.argsort(-predictions)
        topK_predictions_score.append(predictions[sort_index][0:d.args.prediction_TopN])
        topK_predictions_items.append(pos_items_tmp[sort_index][0:d.args.prediction_TopN])
        topK_predictions_users.append(u)

    return topK_predictions_score, topK_predictions_items,  topK_predictions_users


def find_path_stars(fsub_set, items_prev):
    sequence = ""
    path = []
    item_start = items_prev[-1]
    while True:
        if not items_prev:
            break

        item = str(items_prev.pop())
        if sequence == "":
            if item in fsub_set:
                sequence = item
                path.append(int(item))
        else:
            tmp_sequence = item + "-" + sequence
            if tmp_sequence in fsub_set:
                sequence = tmp_sequence
                path.append(int(item))

    if not path:
        path.append(item_start)

    return path


def find_markov_chains_k(items_prev, K):
    sequence = ""
    path = []
    item_start = items_prev[-1]
    count_insert_item = 0
    while True:
        if count_insert_item >= K:
            break
        if not items_prev:
            break

        item = str(items_prev.pop())
        count_insert_item += 1
        if sequence == "":
            sequence = item
            path.append(int(item))
        else:
            sequence = item + "-" + sequence
            path.append(int(item))

    return path


def damping_linear(x, current_fsub_size):
    return -(1/current_fsub_size) * x + 1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
