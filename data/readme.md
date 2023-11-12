# Data

File *train.pkl* contains all the training data.

With code:

'''
with open("./train.pkl") as f:
    train_data = pickle.load(f)
'''

You will get a list of interactions where each interaction is a list '[userid, itemid, timestamp]', 
which means the corresponding user entered the room at a specific timestamp. All the interactions 
are sorted in chronological order.