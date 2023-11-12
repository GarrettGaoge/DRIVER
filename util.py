import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
import torch
import os
import pickle
from torch.autograd import Variable
import pickle as pkl

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

    with open(file_path, 'rb') as f:
        interactions = pkl.load(f)

    user_seq = []
    item_seq = []
    timestamp_seq = []
    for row in interactions:
        u, r, t = row
        user_seq.append(u)
        item_seq.append(r)
        timestamp_seq.append(t)

    print('Data Loaded!')

    user_seq = np.array(user_seq)
    item_seq = np.array(item_seq)
    timestamp_seq = np.array(timestamp_seq)

    num_items = -1
    for i in item_seq:
        if i > num_items:
            num_items = i
    num_items += 1

    num_users = -1
    for u in user_seq:
        if u > num_users:
            num_users = u
    num_users += 1

    item_timediff_seq = []
    item_current_time = defaultdict(float)
    for idx,item in enumerate(item_seq):
        timestamp = timestamp_seq[idx]
        item_timediff_seq.append(timestamp - item_current_time[item])
        item_current_time[item] = timestamp

    user_timediff_seq = []
    user_current_time = defaultdict(float)
    user_previous_itemid_seq = []
    user_latest_itemid = defaultdict(lambda: num_items-1)
    for idx,user in enumerate(user_seq):
        timestamp = timestamp_seq[idx]
        user_timediff_seq.append(timestamp - user_current_time[user])
        user_current_time[user] = timestamp
        user_previous_itemid_seq.append(user_latest_itemid[user])
        user_latest_itemid[user]=item_seq[idx]

    user_timediff_seq = scale(np.array(user_timediff_seq)+1)
    item_timediff_seq = scale(np.array(item_timediff_seq)+1)

    return [num_users, user_seq, user_timediff_seq, user_previous_itemid_seq,\
            num_items, item_seq, item_timediff_seq, timestamp_seq]

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

def save_model(model, opt, epoch, user_embd, item_embd, item_next_embd, room_count):
    print('*** Saving embeddings and model ***')
    state = {
        'user_embd': user_embd.cpu().data.numpy(),
        'item_embd': item_embd.cpu().data.numpy(),
        'item_next_embd': item_next_embd.cpu().data.numpy(),
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
        'room_count': room_count.cpu().data.numpy()
    }
    model_path = 'saved_models'
    directory = os.path.join('./', model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, 'checkpoint.DRIVER.ep%d.pth.tar' % epoch)
    torch.save(state, filename, pickle_protocol=pickle.HIGHEST_PROTOCOL)

    del state

    print('*** Saved embeddings and model to file: %s ***\n\n'%filename)

def load_model(model, opt, epoch):
    filename = './saved_models/checkpoint.DRIVER.ep%d.pth.tar' % (epoch)
    checkpoint = torch.load(filename)
    print('Loading trained model from %s \n'%(filename))
    user_embd = Variable(torch.from_numpy(checkpoint['user_embd']).cuda(0))
    item_embd = Variable(torch.from_numpy(checkpoint['item_embd']).cuda(0))
    item_neg_embd = Variable(torch.from_numpy(checkpoint['item_next_embd']).cuda(0))

    model.load_state_dict(checkpoint['state_dict'])
    opt.load_state_dict(checkpoint['optimizer'])
    room_count = torch.from_numpy(checkpoint['room_count']).cuda(0)

    return [model, opt, user_embd, item_embd, item_neg_embd, room_count]