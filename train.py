from util import *
import util as ut
import os
from DRIVER import *
from torch import optim
from tqdm import trange
from torch.autograd import Variable
from collections import defaultdict
import random
import time

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
    """
    To train the model
    """

    # SET LOG
    log = open("DRIVER-training.log",'a')
    log.write("Starting at:\n")
    log.write(time.asctime(time.localtime(time.time())))
    log.write('\n')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(arg.gpuid)

    # LOAD DATA
    [num_users, user_seq, user_timediff_seq, user_previous_itemid_seq,
     num_items, item_seq, item_timediff_seq, timestamp_seq] = load_data(arg.train_file)

    tbatch_timespan = arg.time_unit * 3600 * 24

    # CONSTRUCT MODEL AND LOSS FUNCTION
    model = DRIVER(num_users, num_items, arg.embd_size).cuda()
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

    # GET THE ROOM STATE
    with open('./data/room_state.pkl','rb') as f:
        room_state = pickle.load(f)

    # TRAINING
    print('*** Training the model. ***')
    with trange(arg.epochs) as epochs:
        for epoch in epochs:
            epochs.set_description('Epoch %d of %d' % (epoch, arg.epochs))
            opt.zero_grad()
            reinitialize_tbatches()
            total_loss, loss = 0, 0
            total_pred_loss = 0

            # RECORD WHICH USERS ARE IN EACH ROOM NOW
            item_cur_users = defaultdict(dict)

            # INITIALIZATION FOR EACH EPOCH
            tbatch_start_time = None
            room_count = torch.zeros((num_items,1)).cuda()
            # ENUMERATE EACH SAMPLE
            with trange(len(user_seq)) as train_samples:
                for train_sample in train_samples:
                    train_samples.set_description('Processed %d-th interactions' % train_sample)
                    # GET EACH SMAPLE'S INFO
                    userid = user_seq[train_sample]
                    itemid = item_seq[train_sample]
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
                    # GET CURRENT USERS IN THE ROOM
                    item_cur_users[tbatch_to_insert][itemid] = room_state[itemid][timestamp]
                    # UPDATE
                    if timestamp - tbatch_start_time > tbatch_timespan:
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
                                # GENERATE THE TRUE STATE FOR EACH ROOM
                                item_state = model.aggregate_function(user_embd.detach(),
                                                                      item_embd.detach(),
                                                                      item_cur_users[batch],
                                                                      ut.current_tbatches_item[batch],
                                                                      ut.current_tbatches_user[batch])
                                # CALCULATE THE PREDICTION LOSS
                                pred_loss = MSELoss(predicted_item_embd,
                                                    torch.cat((item_state,item_embd_static[tbatch_itemids,:]),dim=1))
                                loss += pred_loss
                                total_pred_loss += pred_loss.item()
                                # UPDATE THE ITEM EMBEDDING AND USER EMBEDDING BY RNN
                                item_embd_out_pos = model.forward(user_embd_input,
                                                                  item_embd_input,
                                                                  timediffs=item_timediffs_tensor,
                                                                  select='item_update')
                                user_embd_out_pos = model.forward(user_embd_input,
                                                                  item_state,
                                                                  timediffs=user_timediffs_tensor,
                                                                  select='user_update')

                                next_input = torch.cat((user_embd_input, item_embd_input), dim=1).detach()
                                item_embd_out_next = model.forward(next_input,
                                                                   item_next_embedding_previous * room_count[tbatch_itemids_previous],
                                                                   timediffs=None,
                                                                   select='item_next_update')
                                room_count[tbatch_itemids_previous] += 1
                                item_embd_out_next /= room_count[tbatch_itemids_previous]

                                # UPDATE THE UPDATED EMBEDDING
                                item_embd[tbatch_itemids,:] = item_embd_out_pos
                                user_embd[tbatch_userids,:] = user_embd_out_pos
                                item_next_embd[tbatch_itemids_previous,:] = item_embd_out_next
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
                        del item_cur_users
                        item_cur_users = defaultdict(dict)
                        tbatch_to_insert = -1
            # PRINT EPOCH LOSS
            print('\n\nTotal loss in epoch %d = %f' % (epoch, total_loss))

            # SAVE MODEL
            save_model(model, opt, epoch, user_embd, item_embd,
                       item_next_embd, room_count)
            # RE-INITIALIZATION FOR NEXT EPOCH
            user_embd = model.initial_user_embd.repeat(num_users, 1)
            item_embd = model.initial_item_embd.repeat(num_items, 1)
            item_next_embd = model.initial_item_next_embd.repeat(num_items, 1)

            log.write("Epoch %d is completed at:" % epoch)
            log.write("The loss in epoch %d is %.4f"%(epoch, total_loss))
            log.write(time.asctime(time.localtime(time.time())))
            log.write('\n')
            log.flush()
    # writer.close()
    print('\n\n *** Training complete. ***')
    log.close()

def main():
    train_file = './data/train.pkl'
    embd_size = 128
    epochs = 60
    time_unit = 1
    lr = 3e-4
    gpuid = 0
    seed = 2

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    arg = Args(train_file,
               embd_size,
               lr,
               epochs,
               time_unit,
               gpuid,
               seed)
    train(arg)

if __name__ == '__main__':
    main()