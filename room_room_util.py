import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import torch
import os
import pickle
from torch.autograd import Variable

total_reinitialization_count = 0

# data-loading related
def load_data(file_path):
    '''
    This function is to load the interaction data
    Data format: each line is an interaction between a user and an item
                 each line contains: user_id, item_id, timestamp(cardinal)
    :param file_path: file path of data
    :return: the needed information
    '''

    user_seq = []
    item_seq = []
    timestamp_seq = []
    event_seq = []
    duration_seq = []
    start_timestamp = None

    print('\n\nLoading the data...')

    with open(file_path, 'r') as f:
        [valida_idx, valida_len] = f.readline().strip().split(',')
        for line in f:
            row = line.strip().split(',')
            user_seq.append(row[0])
            item_seq.append(row[1])
            if start_timestamp is None:
                start_timestamp=int(row[2])
            timestamp_seq.append(int(row[2])-start_timestamp)
            event_seq.append(row[3])
            duration_seq.append(float(row[4]))

    user_seq = np.array(user_seq)
    item_seq = np.array(item_seq)
    duration_seq = np.array(duration_seq)
    timestamp_seq = np.array(timestamp_seq)

    print('Formating item sequence')
    nodeid = 0
    item2id = {}
    item_timediff_seq = []
    item_current_time = defaultdict(float)
    item_pos_timediff_seq = []
    item_pos_current_time = defaultdict(float)
    for idx,item in enumerate(item_seq):
        if item not in item2id:
            item2id[item] = nodeid
            nodeid += 1
        timestamp = timestamp_seq[idx]
        item_timediff_seq.append(timestamp - item_current_time[item])
        item_current_time[item] = timestamp
        # record pos timediffs
        if event_seq[idx] == 'pos':
            item_pos_timediff_seq.append(timestamp - item_pos_current_time[item])
            item_pos_current_time[item] = timestamp
        elif event_seq[idx] == 'neg':
            item_pos_timediff_seq.append(0)
        else:
            print('Event Error')
            import sys
            sys.exit()
    num_items = len(item2id)
    item_id_seq = [item2id[item] for item in item_seq]

    print('Formating user sequence')
    nodeid = 0
    user2id = {}
    user_timediff_seq = []
    user_current_time = defaultdict(float)
    user_previous_itemid_seq = []
    user_latest_itemid = defaultdict(lambda: num_items)
    user_pos_timediff_seq = []
    user_pos_current_time = defaultdict(float)
    for idx,user in enumerate(user_seq):
        if user not in user2id:
            user2id[user] = nodeid
            nodeid += 1
        timestamp = timestamp_seq[idx]
        user_timediff_seq.append(timestamp - user_current_time[user])
        user_current_time[user] = timestamp
        user_previous_itemid_seq.append(user_latest_itemid[user])
        user_latest_itemid[user]=item2id[item_seq[idx]]
        # record pos timediffs
        if event_seq[idx] == 'pos':
            user_pos_timediff_seq.append(timestamp - user_pos_current_time[user])
            user_pos_current_time[user] = timestamp
        elif event_seq[idx] == 'neg':
            user_pos_timediff_seq.append(0)
        else:
            print('Event Error')
            import sys
            sys.exit()
    user_id_seq = [user2id[user] for user in user_seq]

    user_timediff_seq = scale(np.array(user_timediff_seq)+1)
    item_timediff_seq = scale(np.array(item_timediff_seq)+1)
    user_pos_timediff_seq = scale(np.array(user_pos_timediff_seq) + 1)
    item_pos_timediff_seq = scale(np.array(item_pos_timediff_seq) + 1)

    print('*** The data has been loaded. ***')

    id2item = dict()
    for item in item2id:
        id2item[item2id[item]] = item
    id2user = dict()
    for user in user2id:
        id2user[user2id[user]] = user
    return [user2id, user_id_seq, user_pos_timediff_seq, user_previous_itemid_seq, \
            item2id, item_id_seq, item_pos_timediff_seq, \
            timestamp_seq, \
            int(valida_idx), int(valida_idx) + int(valida_len), start_timestamp,
            id2item, id2user, event_seq, duration_seq]

def reinitialize_tbatches():
    global current_tbatches_user, current_tbatches_item, \
        current_tbatches_previous_item
    global tbatchid_user, tbatchid_item, current_tbatches_user_timediffs, \
        current_tbatches_item_timediffs

    global current_tbatches_neg_user, current_tbatches_neg_item, \
        current_tbatches_neg_previous_item
    global current_tbatches_neg_user_timediffs, current_tbatches_neg_item_timediffs

    current_tbatches_user = defaultdict(list)
    current_tbatches_item = defaultdict(list)
    current_tbatches_previous_item = defaultdict(list)
    current_tbatches_neg_previous_item = defaultdict(list)
    current_tbatches_user_timediffs = defaultdict(list)
    current_tbatches_item_timediffs = defaultdict(list)
    current_tbatches_neg_user = defaultdict(list)
    current_tbatches_neg_item = defaultdict(list)
    current_tbatches_neg_user_timediffs = defaultdict(list)
    current_tbatches_neg_item_timediffs = defaultdict(list)

    tbatchid_item = defaultdict(lambda: -1)
    tbatchid_user = defaultdict(lambda: -1)

    global total_reinitialization_count
    total_reinitialization_count += 1

def generate_adj(item_history, num_users, old_adj):
    old_adj = old_adj.cpu().numpy()
    adj = np.eye(num_users)
    for item in item_history:
        item_history[item] = list(set(item_history[item]))
    for item in item_history:
        for i in item_history[item]:
            for j in item_history[item]:
                adj[i,j] = 1
    adj -= np.eye(num_users)
    old_adj -= np.diag(np.diag(old_adj))
    adj += 0.5*old_adj
    adj = normalize(adj, 'l1')
    adj += np.eye(num_users)

    return torch.from_numpy(adj)

def save_model(model, opt, arg, epoch, user_embd, item_embd, item_neg_embd,
               train_end_idx, model_type, room_count):
    print('*** Saving embeddings and model ***')
    state = {
        'user_embd': user_embd.cpu().data.numpy(),
        'item_embd': item_embd.cpu().data.numpy(),
        'item_neg_embd': item_neg_embd.cpu().data.numpy(),
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
        'train_end_idx': train_end_idx,
        'room_count': room_count.cpu().data.numpy()
    }
    model_path = 'saved_models_%s/depie_%.2f_unit' % (model_type, arg.time_unit)
    directory = os.path.join('./', model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, 'checkpoint.depie.ep%d.pth.tar' % epoch)
    torch.save(state, filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    del state

    print('*** Saved embeddings and model to file: %s ***\n\n'%filename)

def load_model(model, opt, epoch, arg, model_type):
    filename = './saved_models_%s/depie_%.2f_unit/checkpoint.depie.ep%d.pth.tar' % (model_type, arg.time_unit, epoch)
    # filename = './saved_models/depie/checkpoint.depie.ep%d.pth.tar' % epoch
    checkpoint = torch.load(filename)
    print('Loading trained model from %s \n'%(filename))
    user_embd = Variable(torch.from_numpy(checkpoint['user_embd']).cuda(0))
    item_embd = Variable(torch.from_numpy(checkpoint['item_embd']).cuda(0))
    item_neg_embd = Variable(torch.from_numpy(checkpoint['item_neg_embd']).cuda(0))

    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])
    room_count = torch.from_numpy(checkpoint['room_count']).cuda(0)

    return [model, opt, user_embd, item_embd, item_neg_embd, room_count]