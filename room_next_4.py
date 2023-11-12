from util import *
import util as ut
# from tensorboardX import SummaryWriter
import args
import os
from room_next_DEPIE import *
from torch import optim
from tqdm import trange
from torch.autograd import Variable
from collections import defaultdict
import random
import multiprocessing
import time

# python3 DRIVER.py
# user average pooling to aggregate the next room embeddings

class Args():
    def __init__(self, train_file, embd_size, lr, epochs, time_unit, gpuid, seed):
        self.train_file = train_file
        self.embd_size = embd_size
        self.lr = lr
        self.epochs = epochs
        self.time_unit = time_unit
        self.gpuid = gpuid
        self.seed = seed

def train(arg):
    # Set log
    log = open("huajiao_next_only.log", 'a')
    log.write("Starting at:\n")
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n')

    # Set random seeds.
    seed = arg.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.gpuid)
    # DEFINE MODEL TYPE
    model_type = 'next_only_huajiao_%d'%seed
    # LOAD DATA
    [user2id, user_id_seq, user_timediff_seq, user_previous_itemid_seq,
     item2id, item_id_seq, item_timediff_seq, timestamp_seq,
     train_end_idx, test_start_idx, all_start_time,
     id2item, id2user, event_seq, duration_seq] = load_data(arg.train_file)
    # DATA INFO
    num_items = len(item2id) + 1
    num_users = len(user2id)
    num_interactions = len(user_id_seq)
    test_end_idx = num_interactions
    # TIME RANGE TO DO A UPDATE
    tbatch_timespan = arg.time_unit * 3600 * 24
    # CONSTRUCT MODEL AND LOSS FUNCTION
    model = DEPIE(num_users, num_items, arg.embd_size).cuda()
    MSELoss = nn.MSELoss()
    # INITIALIZATION
    initial_user_embd = nn.Parameter(F.normalize(torch.rand(arg.embd_size).cuda(), dim=0))
    initial_item_embd = nn.Parameter(F.normalize(torch.rand(arg.embd_size).cuda(), dim=0))
    initial_item_next_embd = nn.Parameter(F.normalize(torch.rand(2*arg.embd_size).cuda(), dim=0))
    model.initial_user_embd = initial_user_embd
    model.initial_item_embd = initial_item_embd
    model.initial_item_next_embd = initial_item_next_embd
    user_embd = initial_user_embd.repeat(num_users, 1)
    item_embd = initial_item_embd.repeat(num_items, 1)
    item_next_embd = initial_item_next_embd.repeat(num_items, 1)
    user_embd_static = Variable(torch.eye(num_users).cuda())
    item_embd_static = Variable(torch.eye(num_items).cuda())
    # OPTIM
    lr = arg.lr
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # BEGIN TRAINING
    print('*** Training the model. ***')
    with trange(arg.epochs) as epochs:
        for epoch in epochs:
            epochs.set_description('Epoch %d of %d' % (epoch, arg.epochs))
            opt.zero_grad()
            reinitialize_tbatches()
            total_loss, loss = 0, 0
            # INITIALIZATION FOR EACH EPOCH
            tbatch_start_time = None
            tbatch_to_insert = -1
            room_count = torch.ones((num_items, 1)).float().cuda()
            # ENUMERATE EACH SAMPLE
            with trange(train_end_idx) as train_samples:
                for train_sample in train_samples:
                    train_samples.set_description('Processed %d-th interactions' % train_sample)
                    # GET EACH SMAPLE'S INFO
                    event = event_seq[train_sample]
                    if event == 'neg':
                        continue
                    userid = user_id_seq[train_sample]
                    itemid = item_id_seq[train_sample]
                    user_timediff = user_timediff_seq[train_sample]
                    item_timediff = item_timediff_seq[train_sample]
                    # TBATCH
                    tbatch_to_insert = max(ut.tbatchid_user[userid], ut.tbatchid_item[itemid]) + 1
                    ut.tbatchid_user[userid] = tbatch_to_insert
                    ut.tbatchid_item[itemid] = tbatch_to_insert
                    ut.current_tbatches_user[tbatch_to_insert].append(userid)
                    ut.current_tbatches_item[tbatch_to_insert].append(itemid)
                    ut.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                    ut.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                    ut.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_seq[train_sample])

                    # SET FOR BEGIN TIME
                    timestamp = timestamp_seq[train_sample]
                    if tbatch_start_time is None:
                        tbatch_start_time = timestamp
                    # UPDATE
                    if (timestamp - tbatch_start_time > tbatch_timespan) | (train_sample == train_end_idx-1):
                        tbatch_start_time = timestamp
                        # ENUMERATE THE BATCHES
                        with trange(len(ut.current_tbatches_user)) as batches:
                            for batch in batches:
                                batches.set_description('Processed %d of %d T-batches'%(batch, len(ut.current_tbatches_user)))
                                # LOAD THE CURRENT TBATCH
                                tbatch_userids = torch.LongTensor(ut.current_tbatches_user[batch]).cuda()
                                tbatch_itemids = torch.LongTensor(ut.current_tbatches_item[batch]).cuda()

                                user_timediffs_tensor = Variable(torch.Tensor(ut.current_tbatches_user_timediffs[batch]).cuda()).unsqueeze(1)
                                item_timediffs_tensor = Variable(torch.Tensor(ut.current_tbatches_item_timediffs[batch]).cuda()).unsqueeze(1)

                                tbatch_itemids_previous = torch.LongTensor(ut.current_tbatches_previous_item[batch]).cuda()
                                item_embedding_previous = item_embd[tbatch_itemids_previous,:]
                                item_next_embedding_previous = item_next_embd[tbatch_itemids_previous,:]
                                user_embd_input = user_embd[tbatch_userids, :]
                                item_embd_input = item_embd[tbatch_itemids, :]

                                # PROJECT USER EMBEDDING
                                user_projected_embd = model.forward(user_embd_input, item_embedding_previous,
                                                                    timediffs=user_timediffs_tensor,
                                                                    select='project')

                                # CONSTRUCT EMBEDDING FOR PREDICTION(ONLY PREDICT POS ROOMS)
                                user_item_embd = torch.cat((user_projected_embd,
                                                            item_next_embedding_previous,
                                                            item_embd_static[tbatch_itemids_previous,:],
                                                            user_embd_static[tbatch_userids,:]), dim=1)
                                # PREDICT
                                predicted_item_embd = model.predict_item_embd(user_item_embd)
                                # CALCULATE THE PREDICTION LOSS
                                pred_loss = MSELoss(predicted_item_embd,
                                                    torch.cat((item_embd_input, item_embd_static[tbatch_itemids, :]), dim=1).detach())
                                loss += pred_loss
                                # UPDATE THE ITEM EMBEDDING AND USER EMBEDDING BY RNN
                                item_embd_out_pos = model.forward(user_embd_input,
                                                                  item_embd_input,
                                                                  timediffs=item_timediffs_tensor,
                                                                  select='item_update')
                                user_embd_out_pos = model.forward(user_embd_input,
                                                                  item_embd_input,
                                                                  timediffs=user_timediffs_tensor,
                                                                  select='user_update')
                                next_input = torch.cat((user_embd_input, item_embd_input), dim=1)
                                item_embd_out_next = model.forward(next_input,
                                                                   item_next_embedding_previous * room_count[tbatch_itemids_previous],
                                                                   timediffs=None,
                                                                   select='item_next_update')
                                room_count[tbatch_itemids_previous] += 1
                                item_embd_out_next /= room_count[tbatch_itemids_previous]

                                # UPDATE THE UPDATED EMBEDDING
                                item_embd[tbatch_itemids, :] = item_embd_out_pos
                                user_embd[tbatch_userids, :] = user_embd_out_pos
                                item_next_embd[tbatch_itemids_previous, :] = item_embd_out_next
                                # SMOOTHING
                                loss += MSELoss(item_embd_out_pos, item_embd_input.detach())
                                loss += MSELoss(user_embd_out_pos, user_embd_input.detach())
                                loss += MSELoss(item_embd_out_next, item_next_embedding_previous.detach())


                        # RECORD THE LOSS
                        total_loss += loss.item()
                        # BACK PROPAGATION
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                        # RE-INITIALIZATION FOR NEXT BATCH
                        loss = 0
                        item_embd.detach_()
                        user_embd.detach_()
                        item_next_embd.detach_()
                        reinitialize_tbatches()
                        tbatch_to_insert = -1
            # PRINT EPOCH LOSS
            print('\n\nTotal loss in epoch %d = %f' % (epoch, total_loss))
            # SAVE MODEL
            save_model(model, opt, arg, epoch, user_embd, item_embd, item_next_embd, train_end_idx, model_type, room_count)
            # RE-INITIALIZATION FOR NEXT EPOCH
            user_embd = model.initial_user_embd.repeat(num_users, 1)
            item_embd = model.initial_item_embd.repeat(num_items, 1)
            item_next_embd = model.initial_item_next_embd.repeat(num_items, 1)

            # logging
            log.write("Epoch %d is completed at:" % epoch)
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')
            log.flush()

    # writer.close()
    print('\n\n *** Training complete. ***')

    log.close()

def main():
    train_file = '../data/huajiao_interactions.csv'
    epochs = 60
    time_unit = 1
    lr = 3e-4
    gpuids = [0,1,2,3,4,5]*5
    seeds = [145, 2, 244, 258, 291,
             292, 306, 318, 338, 405,
             45, 454, 488, 505, 524,
             576, 586, 593, 597, 602,
             68, 7, 712, 755, 760,
             790, 86, 868, 887, 911]
    # seeds = [306, 593, 68, 755]

    # run 3 procs each time
    def run_group(gpuids, seeds):
        for idx in range(len(gpuids)):
            gpu = gpuids[idx]
            seed = seeds[idx]
            arg = Args(train_file, 128, lr, epochs, time_unit, gpu, seed)
            proc = multiprocessing.Process(target=train, args=(arg, ))
            proc.start()

    for idx in range(1):
        gpuid_group = gpuids[idx*1:(idx+1)*1]
        seed_group = seeds[idx*1:(idx+1)*1]
        main_proc = multiprocessing.Process(target=run_group, args=(gpuid_group, seed_group, ))
        main_proc.start()
        main_proc.join()

    # for idx in range(len(gpuids)):
    #     gpu = gpuids[idx]
    #     seed = seeds[idx]
    #     arg = Args(train_file, 128, lr, epochs, time_unit, gpu, seed)
    #     proc = multiprocessing.Process(target=train, args=(arg, ))
    #     proc.start()

if __name__ == '__main__':
    main()