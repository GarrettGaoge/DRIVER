import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import pickle as pkl

class NormalLinear(nn.Linear):
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

class DEPIE(nn.Module):
    def __init__(self, num_users, num_items, embd_size):
        super(DEPIE,self).__init__()

        # initialize the parameters
        self.embd_size = embd_size
        self.num_users = num_users
        self.num_items = num_items
        self.user_static_embd_size = num_users
        self.item_static_embd_size = num_items

        self.initial_user_embd = nn.Parameter(torch.Tensor(self.embd_size))
        self.initial_item_embd = nn.Parameter(torch.Tensor(self.embd_size))
        self.register_parameter('initial_user_embd', self.initial_user_embd)
        self.register_parameter('initial_item_embd', self.initial_item_embd)
        self.initial_item_next_embd = nn.Parameter(torch.Tensor(2*self.embd_size))
        self.register_parameter('initial_item_next_embd', self.initial_item_next_embd)

        rnn_input_size_items = rnn_input_size_users = self.embd_size

        self.item_rnn = nn.RNNCell(rnn_input_size_users, self.embd_size)
        self.user_rnn = nn.RNNCell(rnn_input_size_items, self.embd_size)

        self.prediction_layer = nn.Linear(self.user_static_embd_size +
                                          self.item_static_embd_size +
                                          self.embd_size * 3,
                                          self.item_static_embd_size + self.embd_size)

        self.embd_layer = NormalLinear(1, self.embd_size)
        self.user_att = nn.Linear(2*self.embd_size, 1)
        self.room_att = nn.Linear(2*self.embd_size, 1)

    def forward(self, user_embd, item_embd, timediffs, select=None):
        if select == 'item_update':
            item_embd_output = self.item_rnn(user_embd, item_embd)
            return F.normalize(item_embd_output)

        elif select == 'user_update':
            user_embd_output = self.user_rnn(item_embd, user_embd)
            return F.normalize(user_embd_output)

        elif select == 'item_next_update':
            item_embd_output = user_embd + item_embd.detach()
            return F.normalize(item_embd_output)

        elif select == 'project':
            user_projected_embd = self.context_convert(user_embd, timediffs)
            return user_projected_embd

    def context_convert(self, user_embd, timediffs):
        new_embd = user_embd * (1+self.embd_layer(timediffs))
        return new_embd

    def predict_item_embd(self, user_item_embd):
        X_out = self.prediction_layer(user_item_embd)
        return X_out

    def aggregate_function(self, item_embd, user_embd, item_cur_users, user2id, id2item,
                           current_item, current_user):
        if type(current_item).__name__ == 'list':
            aggred_item_list = []
            for idx, itemid in enumerate(current_item):
                userid = current_user[idx]
                other_users = item_cur_users[itemid]
                total_input = list()
                for other in other_users:
                    other_id = user2id[str(other)]
                    if other_id == userid:
                        continue
                    total_input.append(other_id)
                if len(total_input)==0:
                    aggred_user_embd = torch.zeros_like(user_embd[0,:]).unsqueeze(dim=0)
                else:
                    total_input = torch.LongTensor(total_input).cuda()
                    aggred_user_embd = self.aggregate_users(user_embd[total_input,:]).unsqueeze(dim=0)
                # pkl.dump(userid, f, pkl.HIGHEST_PROTOCOL)
                aggred_item_embd = self.combine_user_room(user_embd[userid],
                                                          item_embd[itemid,:],
                                                          aggred_user_embd)
                aggred_item_list.append(aggred_item_embd)
            return torch.cat(aggred_item_list, dim=0)

        other_users = item_cur_users[current_item]
        total_input = list()
        for other in other_users:
            other_id = user2id[str(other)]
            if other_id == current_user:
                continue
            total_input.append(other_id)
        total_input = torch.LongTensor(total_input).cuda()
        if len(total_input)==0:
            aggred_user_embd = torch.zeros_like(user_embd[0,:]).unsqueeze(dim=0)
        else:
            aggred_user_embd = self.aggregate_users(user_embd[total_input,:]).unsqueeze(dim=0)
        # if f:
        #     pkl.dump([current_user, current_item], f, pkl.HIGHEST_PROTOCOL)
        aggred_item_embd = self.combine_user_room(user_embd[current_user],
                                                  item_embd[current_item],
                                                  aggred_user_embd)
        return aggred_item_embd

    def combine_user_room(self, user, room, other_users):
        user = user.clone()
        room = room.clone()
        other_users = other_users.clone()
        user_at = nn.Sigmoid()(self.user_att(torch.cat((user.unsqueeze(dim=0),
                                                        other_users),
                                                       dim=1)))
        room_at = nn.Sigmoid()(self.room_att(torch.cat((user.unsqueeze(dim=0),
                                                          room.unsqueeze(dim=0)),
                                                         dim=1)))
        user_at = torch.exp(user_at)/(torch.exp(user_at)+torch.exp(room_at))
        room_at = torch.exp(room_at)/(torch.exp(user_at)+torch.exp(room_at))
        # if f:
        #     pkl.dump(user_at, f, pkl.HIGHEST_PROTOCOL)
        #     pkl.dump(room_at, f, pkl.HIGHEST_PROTOCOL)
        return user_at * other_users + room_at * room

    def aggregate_users(self, other_users):
        return torch.max(other_users, dim=0)[0]