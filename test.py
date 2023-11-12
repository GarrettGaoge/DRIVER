from util import *
import pickle
# import args
import os
from DRIVER import *
from torch import optim
from tqdm import trange
from collections import defaultdict
import sys
import multiprocessing
import random
import torch
import numpy as np

class Args():
    def __init__(self, train_file, lr, test_epoch, embd_size, time_unit, seed, gpuid):
        self.train_file = train_file
        self.lr = lr
        self.test_epoch = test_epoch
        self.embd_size = embd_size
        self.time_unit = time_unit
        self.seed = seed
        self.gpuid = gpuid

def test(arg):
    # dynamic relationship
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.gpuid)
    # arg = args.parse_args()
    model_type = 'DRIVER_%f_seed_%d' % (arg.lr, arg.seed)
    # model_type = 'DRIVER_more_%d_%f_seed_%d' % (idx, arg.lr, arg.seed)
    # model_type = 'DRIVER_embd_size=%d'%arg.embd_size
    output_file = 'results/performance_%s_ep_%d.txt' % (model_type, arg.test_epoch)
    # rel_file = open('relationships.pkl','wb')

    # load data
    [user2id, user_id_seq, user_timediff_seq, user_previous_itemid_seq,
     item2id, item_id_seq, item_timediff_seq, timestamp_seq,
     train_end_idx, test_start_idx,all_start_time,
     id2item, id2user, event_seq, duration_seq] = load_data(arg.train_file)

    with open('../data/room_state.pkl','rb') as f:
        room_state = pickle.load(f)

    num_items = len(item2id) + 1
    num_users = len(user2id)
    num_interactions = len(user_id_seq)
    test_end_idx = num_interactions

    old_rooms = defaultdict(dict)
    with trange(train_end_idx) as progress_bar:
        for j in progress_bar:
            userid = user_id_seq[j]
            itemid = item_id_seq[j]
            old_rooms[userid][itemid] = 1

    tbatch_timespan = arg.time_unit * 24 * 3600

    # load the model
    model = DRIVER(num_users, num_items, arg.embd_size).cuda()
    MSELoss = nn.MSELoss()

    lr = arg.lr
    opt = optim.Adam(model.parameters(), lr = lr, weight_decay=1e-5)

    model, opt, user_embeddings, item_embeddings, item_neg_embeddings, room_count\
        = load_model(model, opt, arg.test_epoch, arg, model_type)
    # pickle.dump(item_neg_embeddings.detach_().cpu().numpy(), rel_file, pickle.HIGHEST_PROTOCOL)

    user_embeddings_static = Variable(torch.eye(num_users).cuda())
    item_embeddings_static = Variable(torch.eye(num_items).cuda())

    # PERFORMANCE METRICS
    validation_ranks = []
    test_ranks = []
    new_validation_ranks = []
    new_test_ranks = []
    old_validation_ranks = []
    old_test_ranks = []
    high_validation_ranks, low_validation_ranks = [], []
    high_test_ranks, low_test_ranks = [], []

    opt.zero_grad()
    tbatch_start_time = None
    loss = 0
    # FORWARD PASS
    print("*** Making interaction predictions by forward pass (no t-batching) ***")

    # LOAD USER USAGE
    with open("./diffUsageUsers.pkl", "rb") as f:
        highUsageUsers = pickle.load(f)
        lowUsageUsers = pickle.load(f)

    user_vali_loss = defaultdict(list)
    user_vali_rank = defaultdict(list)
    room_change_point = defaultdict(int)
    room_aggred_point = defaultdict(int)
    item_cur_users = defaultdict(list)
    aggred_item_embd = item_embeddings.clone()
    last_true = dict()
    with trange(train_end_idx, test_start_idx) as progress_bar:
        for j in progress_bar:
            progress_bar.set_description('%dth interaction for validation' % j)

            # LOAD INTERACTION J
            userid = user_id_seq[j]
            itemid = item_id_seq[j]
            user_timediff = user_timediff_seq[j]
            item_timediff = item_timediff_seq[j]
            event = event_seq[j]
            timestamp = timestamp_seq[j]
            if not tbatch_start_time:
                tbatch_start_time = timestamp
            itemid_previous = user_previous_itemid_seq[j]

            item_cur_users[itemid] = room_state[int(id2item[itemid])][int(timestamp + all_start_time)]
            room_change_point[itemid] = int(timestamp + all_start_time)

            # LOAD USER AND ITEM EMBEDDING
            user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
            user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
            item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
            item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
            user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
            item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
            item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]
            item_neg_embeddings_previous = item_neg_embeddings[torch.cuda.LongTensor([itemid_previous])]

            # PROJECT USER EMBEDDING
            user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous,
                                                     timediffs=user_timediffs_tensor,select='project')
            if userid not in last_true:
                last_true[userid] = item_embedding_previous.detach()
            previous_room_state = last_true[userid].clone()
            user_item_embedding = torch.cat((user_projected_embedding,
                                             item_neg_embeddings_previous,
                                             item_embeddings_static[torch.cuda.LongTensor([itemid_previous])],
                                             user_embedding_static_input), dim=1)

            if event == 'pos':
                # PREDICT ITEM EMBEDDING
                predicted_item_embedding = model.predict_item_embd(user_item_embedding)
                # CONSTRUCT TRUE EMBEDDING
                true_item_embd = model.aggregate_function(item_embeddings.detach(),
                                                          user_embeddings.detach(),
                                                          item_cur_users,
                                                          user2id, id2item,
                                                          itemid, userid)
                last_true[userid] = true_item_embd.detach()
                # CALCULATE PREDICTION LOSS
                cur_loss = MSELoss(predicted_item_embedding,
                                torch.cat((true_item_embd, item_embedding_static_input), dim=1).detach())
                loss += cur_loss
                user_vali_loss[userid].append(cur_loss.item())
                # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS
                wait_for_aggred = []
                real_timestamp = int(timestamp+all_start_time)
                for idx in range(len(item_embeddings)-1):
                    if real_timestamp in room_state[int(id2item[idx])]:
                        wait_for_aggred.append(idx)
                        item_cur_users[idx] = room_state[int(id2item[idx])][real_timestamp]
                for idx in wait_for_aggred:
                    some_item = model.aggregate_function(item_embeddings.detach(), user_embeddings.detach(), item_cur_users,
                                                         user2id, id2item, idx, userid)
                    aggred_item_embd[idx] = some_item

                euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1),
                                                            torch.cat([aggred_item_embd, item_embeddings_static],
                                                                      dim=1)).squeeze(-1)

                # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
                true_item_distance = euclidean_distances[itemid]
                euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
                true_item_rank = np.sum(euclidean_distances_smaller) + 1

                validation_ranks.append(true_item_rank)
                if itemid in old_rooms[userid]:
                    old_validation_ranks.append(true_item_rank)
                else:
                    new_validation_ranks.append(true_item_rank)
                old_rooms[userid][itemid] = 1
                user_vali_rank[userid].append(true_item_rank)

                # SPLIT USER USAGE
                if userid in highUsageUsers:
                    high_validation_ranks.append(true_item_rank)
                else:
                    low_validation_ranks.append(true_item_rank)

                # UPDATE USER AND ITEM EMBEDDING
                user_embedding_output = model.forward(user_embedding_input, true_item_embd.detach(),
                                                      timediffs=user_timediffs_tensor,select='user_update')
                item_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                      timediffs=item_timediffs_tensor,select='item_update')
                next_input = torch.cat((user_embedding_input, item_embedding_input), dim=1)
                # pkl.dump([itemid, next_input], dr, pkl.HIGHEST_PROTOCOL)
                item_neg_embedding_output = model.forward(next_input,
                                                          item_neg_embeddings_previous*room_count[itemid_previous],
                                                          timediffs=None, select='item_next_update')
                room_count[itemid_previous] += 1
                item_neg_embedding_output /= room_count[itemid_previous]

                # SAVE EMBEDDINGS
                item_embeddings[itemid, :] = item_embedding_output.squeeze(0)
                user_embeddings[userid, :] = user_embedding_output.squeeze(0)
                item_neg_embeddings[itemid_previous, :] = item_neg_embedding_output.squeeze(0)

                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                loss += MSELoss(user_embedding_output, user_embedding_input.detach())
                loss += MSELoss(item_neg_embedding_output, item_neg_embeddings_previous.detach())
                #
                # if timestamp - tbatch_start_time > tbatch_timespan:
                #     loss.backward()
                #     opt.step()
                #     opt.zero_grad()
                #     loss = 0

                item_embeddings.detach_()
                user_embeddings.detach_()
                item_neg_embeddings.detach_()
                aggred_item_embd.detach_()

            elif event == 'neg':
                pass
            else:
                print('Event Error')
                sys.exit()

    tbatch_start_time = None
    loss = 0
    user_test_loss = defaultdict(list)
    user_test_rank = defaultdict(list)
    with trange(test_start_idx, test_end_idx) as progress_bar:
        for j in progress_bar:
            progress_bar.set_description('%dth interaction for validation' % j)

            # LOAD INTERACTION J
            userid = user_id_seq[j]
            itemid = item_id_seq[j]
            user_timediff = user_timediff_seq[j]
            item_timediff = item_timediff_seq[j]
            event = event_seq[j]
            timestamp = timestamp_seq[j]
            if not tbatch_start_time:
                tbatch_start_time = timestamp
            itemid_previous = user_previous_itemid_seq[j]

            item_cur_users[itemid] = room_state[int(id2item[itemid])][int(timestamp + all_start_time)]
            room_change_point[itemid] = int(timestamp + all_start_time)

            # LOAD USER AND ITEM EMBEDDING
            user_embedding_input = user_embeddings[torch.cuda.LongTensor([userid])]
            user_embedding_static_input = user_embeddings_static[torch.cuda.LongTensor([userid])]
            item_embedding_input = item_embeddings[torch.cuda.LongTensor([itemid])]
            item_embedding_static_input = item_embeddings_static[torch.cuda.LongTensor([itemid])]
            user_timediffs_tensor = Variable(torch.Tensor([user_timediff]).cuda()).unsqueeze(0)
            item_timediffs_tensor = Variable(torch.Tensor([item_timediff]).cuda()).unsqueeze(0)
            item_embedding_previous = item_embeddings[torch.cuda.LongTensor([itemid_previous])]
            item_neg_embeddings_previous = item_neg_embeddings[torch.cuda.LongTensor([itemid_previous])]

            # PROJECT USER EMBEDDING
            user_projected_embedding = model.forward(user_embedding_input, item_embedding_previous,
                                                     timediffs=user_timediffs_tensor,select='project')
            if userid not in last_true:
                last_true[userid] = item_embedding_previous.detach()
            previous_room_state = last_true[userid].clone()
            user_item_embedding = torch.cat((user_projected_embedding,
                                             item_neg_embeddings_previous,
                                             item_embeddings_static[torch.cuda.LongTensor([itemid_previous])],
                                             user_embedding_static_input), dim=1)

            if event == 'pos':
                # PREDICT ITEM EMBEDDING
                predicted_item_embedding = model.predict_item_embd(user_item_embedding)
                # CONSTRUCT TRUE EMBEDDING
                true_item_embd = model.aggregate_function(item_embeddings.detach(),
                                                          user_embeddings.detach(),
                                                          item_cur_users,
                                                          user2id, id2item,
                                                          itemid, userid)
                last_true[userid] = true_item_embd.detach()
                # CALCULATE PREDICTION LOSS
                cur_loss = MSELoss(predicted_item_embedding,
                                torch.cat((true_item_embd, item_embedding_static_input), dim=1).detach())
                loss += cur_loss
                user_test_loss[userid].append(cur_loss.item())
                # CALCULATE DISTANCE OF PREDICTED ITEM EMBEDDING TO ALL ITEMS
                wait_for_aggred = []
                real_timestamp = int(timestamp+all_start_time)
                for idx in range(len(item_embeddings)-1):
                    if real_timestamp in room_state[int(id2item[idx])]:
                        wait_for_aggred.append(idx)
                        item_cur_users[idx] = room_state[int(id2item[idx])][real_timestamp]
                for idx in wait_for_aggred:
                    some_item = model.aggregate_function(item_embeddings.detach(), user_embeddings.detach(), item_cur_users,
                                                         user2id, id2item, idx, userid)
                    aggred_item_embd[idx] = some_item

                euclidean_distances = nn.PairwiseDistance()(predicted_item_embedding.repeat(num_items, 1),
                                                            torch.cat([aggred_item_embd, item_embeddings_static],
                                                                      dim=1)).squeeze(-1)

                # CALCULATE RANK OF THE TRUE ITEM AMONG ALL ITEMS
                true_item_distance = euclidean_distances[itemid]
                euclidean_distances_smaller = (euclidean_distances < true_item_distance).data.cpu().numpy()
                true_item_rank = np.sum(euclidean_distances_smaller) + 1

                test_ranks.append(true_item_rank)
                if itemid in old_rooms[userid]:
                    old_test_ranks.append(true_item_rank)
                else:
                    new_test_ranks.append(true_item_rank)
                old_rooms[userid][itemid] = 1
                user_test_rank[userid].append(true_item_rank)

                # SPLIT USER USAGE
                if userid in highUsageUsers:
                    high_test_ranks.append(true_item_rank)
                else:
                    low_test_ranks.append(true_item_rank)

                # UPDATE USER AND ITEM EMBEDDING
                user_embedding_output = model.forward(user_embedding_input, true_item_embd.detach(),
                                                      timediffs=user_timediffs_tensor,select='user_update')
                item_embedding_output = model.forward(user_embedding_input, item_embedding_input,
                                                      timediffs=item_timediffs_tensor,select='item_update')
                next_input = torch.cat((user_embedding_input, item_embedding_input), dim=1)
                # pkl.dump([itemid, next_input], dr, pkl.HIGHEST_PROTOCOL)
                item_neg_embedding_output = model.forward(next_input,
                                                          item_neg_embeddings_previous*room_count[itemid_previous],
                                                          timediffs=None, select='item_next_update')
                room_count[itemid_previous] += 1
                item_neg_embedding_output /= room_count[itemid_previous]

                # SAVE EMBEDDINGS
                item_embeddings[itemid, :] = item_embedding_output.squeeze(0)
                user_embeddings[userid, :] = user_embedding_output.squeeze(0)
                item_neg_embeddings[itemid_previous, :] = item_neg_embedding_output.squeeze(0)

                # CALCULATE LOSS TO MAINTAIN TEMPORAL SMOOTHNESS
                # loss += MSELoss(item_embedding_output, item_embedding_input.detach())
                # loss += MSELoss(user_embedding_output, user_embedding_input.detach())
                # loss += MSELoss(item_neg_embedding_output, item_neg_embeddings_previous.detach())
                #

                # loss.backward()
                # opt.step()
                # opt.zero_grad()
                # loss = 0

                item_embeddings.detach_()
                user_embeddings.detach_()
                item_neg_embeddings.detach_()
                aggred_item_embd.detach_()

            elif event == 'neg':
                pass
            else:
                print('Event Error')
                sys.exit()

    # pickle.dump(item_neg_embeddings.detach_().cpu().numpy(), rel_file, pickle.HIGHEST_PROTOCOL)
    # rel_file.close()

    # DEFINE NGCF FUNCTION
    def ndcg(ranks, total=num_items):
        ranks = np.array(ranks)
        length = len(ranks)
        test_ranks = ranks[ranks <= total]
        ndcg = 1 / np.log2(test_ranks + 1)

        return sum(ndcg) / length

    # CALCULATE THE PERFORMANCE METRICS
    def performance(ranks,type='test',room='old'):
        performance_dict = dict()
        mrr = np.mean([1.0 / r for r in ranks])
        recalls = []
        nums = []
        for i in range(20):
            rec = sum(np.array(ranks)<=(i+1)) * 1.0 / len(ranks)
            recalls.append(rec)
            nums.append(sum(np.array(ranks)<=(i+1)) * 1.0)
        # rec10 = sum(np.array(ranks) <= 10) * 1.0 / len(ranks)

        ndcgs = []
        for i in range(20):
            ndi = ndcg(ranks, (i+1))
            ndcgs.append(ndi)
        performance_dict[type] = [mrr] + recalls + ndcgs + nums

        # PRINT AND SAVE THE PERFORMANCE METRICS
        fw = open(output_file, "a")
        metrics = ['Mean Reciprocal Rank']
        for i in range(20):
            metrics.append('Recall@%d'%(i+1))
        for i in range(20):
            metrics.append('NDCG@%d'%(i+1))
        # for i in range(20):
        #     metrics.append('Nums@%d'%(i+1))

        print('\n*** performance on %s rooms'%room)
        print('\n*** %s performance of epoch %d ***' % (type, arg.test_epoch))
        fw.write('\n*** %s performance of epoch %d ***\n' % (type, arg.test_epoch))
        for i in range(len(metrics)):
            print(metrics[i] + ': ' + str(performance_dict[type][i]))
            fw.write("%s: "%type + metrics[i] + ': ' + str(performance_dict[type][i]) + "\n")
        fw.flush()
        fw.close()

    print('Length of new rooms of validation ranks: %d'%len(new_validation_ranks))
    print('Length of new rooms of test ranks: %d'%len(new_test_ranks))
    print('\n')

    # performance(validation_ranks,'validation','all')
    # performance(new_validation_ranks,'validation','new')
    # performance(old_validation_ranks, 'validation', 'old')

    performance(test_ranks,'test','all')
    with open("./for_user_type.pkl", "wb") as f:
        pickle.dump(test_ranks, f, pickle.HIGHEST_PROTOCOL)
    # performance(new_test_ranks,'test','new')
    # performance(old_test_ranks, 'test', 'old')

    # print("Validation, high usage users:")
    # performance(high_validation_ranks, 'validation', 'all')
    # print("Validation, low usage users:")
    # performance(low_validation_ranks, 'validation', 'all')

    # print("Test, high usage users:")
    # performance(high_test_ranks, 'test', 'all')
    # print("Test, low usage users:")
    # performance(low_test_ranks, 'test', 'all')

def main():
    train_file = '../data/interactions.csv'
    # test_epoch = 59
    lr = 3e-4
    embd_size = 128
    time_unit = 1
    seed = 145
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    arg = Args(train_file, lr, 59, embd_size, time_unit, seed, 1)
    test(arg)

    # test different epoch
    # epochs = list(range(60))
    # def run_batch(eps):
    #     for idx, ep in enumerate(eps):
    #         arg = Args(train_file, lr, ep, embd_size, time_unit, seed, idx)
    #         proc = multiprocessing.Process(target=test, args=(arg,))
    #         proc.start()
    #
    # for i in range(10):
    #     cur_ep = epochs[i * 6:(i + 1) * 6]
    #     main_proc = multiprocessing.Process(target=run_batch, args=(cur_ep,))
    #     main_proc.start()
    #     main_proc.join()

    # arg = Args(train_file, lr, test_epoch, embd_size, time_unit, seed, 1)
    # test(arg)

    # gpuids = [0,1,5,3]
    # embd_sizes = [32,64,256,512]
    # for idx in range(len(gpuids)):
    #     embd_size = embd_sizes[idx]
    #     gpu = gpuids[idx]
    #     seed = 2
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     np.random.seed(seed)  # Numpy module.
    #     random.seed(seed)  # Python random module.
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     arg = Args(train_file, lr, test_epoch, embd_size, time_unit, seed, gpu)
    #     proc = multiprocessing.Process(target=test, args=(arg,))
    #     proc.start()

    # gpuids = [0]
    # seeds = [602]
    # 602, 593, 292, 755,
    # 86, 505, 258, 524,
    # 291, 45,
    # seeds = [887, 488, 306, 405, 576,
    #          597, 68, 244, 790, 454,
    #          760, 145, 318, 868, 338,
    #          911, 712, 586, 2, 7]
    # gpuids = [3,4,5,6,7] * 4

    # arg = Args(train_file, lr, test_epoch, embd_size, time_unit, 2, 7)
    # test(arg)

    # gpuids = [0, 1, 2, 3]
    # train_ind = [14,16,18,20]
    # for idx, gpu in enumerate(gpuids):
    #     train_file = '../data/interactions_%d.csv'%train_ind[idx]
    #     # seed = seeds[idx]
    #     seed = 2
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     np.random.seed(seed)  # Numpy module.
    #     random.seed(seed)  # Python random module.
    #     torch.manual_seed(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    #     arg = Args(train_file, lr, test_epoch, embd_size, time_unit, seed, gpu)
    #     proc = multiprocessing.Process(target=test, args=(arg,train_ind[idx]))
    #     proc.start()

    # # run five procs each time
    # def run_group(gpuids, seeds):
    #     for idx in range(len(gpuids)):
    #         gpuid = gpuids[idx]
    #         seed = seeds[idx]
    #         torch.manual_seed(seed)
    #         torch.cuda.manual_seed(seed)
    #         torch.cuda.manual_seed_all(seed)
    #         np.random.seed(seed)  # Numpy module.
    #         random.seed(seed)  # Python random module.
    #         torch.manual_seed(seed)
    #         torch.backends.cudnn.benchmark = False
    #         torch.backends.cudnn.deterministic = True
    #         arg = Args(train_file, lr, test_epoch, embd_size, time_unit, seed, gpuid)
    #         proc = multiprocessing.Process(target=test, args=(arg,))
    #         proc.start()
    # for idx in range(4):
    #     gpuid_group = gpuids[idx*5:(idx+1)*5]
    #     seed_group = seeds[idx*5:(idx+1)*5]
    #     main_proc = multiprocessing.Process(target=run_group, args=(gpuid_group, seed_group, ))
    #     main_proc.start()
    #     main_proc.join()

if __name__ == '__main__':
    main()