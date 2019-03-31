"""
Usage:
    training.py [options] <datafile>

Options:
    -c <checkpoint_folder>, --checkpoint-folder=<checkpoint_folder>
        Folder to store checkpoints [Default: checkpoints/]
    -n <n_epochs>, --n-epochs=<n_epochs>
        Number of epochs to train [Default: 4]
    -b <batch_size>, --batch-size=<batch_size>
        Batch size [Default: 500]
    -g, --use-gpu
        Use gpu for training
    --n-hiddens=<number_of_hidden_units>
        Number of hidden units in all layers [Default: 50]
    --learning-rate=<learning_rate>
        Learning rate [Default: 0.1]
"""
from docopt import docopt
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.utils.data as dutils
import torch.nn as nn
from torch import optim
from deepfc import network, data

args = docopt(__doc__)
print(args)

df = pd.read_csv(args['<datafile>'])
datasets = data.split_dataframe(df)

batch_size = int(args['--batch-size'])
n_epochs = int(args['--n-epochs'])
use_cuda = args['--use-gpu']
hidden_size = int(args['--n-hiddens'])
learning_rate = float(args['--learning-rate'])
checkpoint_folder = args['--checkpoint-folder']

n_stores = len(df['organization_unit_num'].unique())
n_weekday = len(df['week_day'].unique())
n_monthday = len(df['month_day'].unique())
n_features = len(df.columns) - 2  # ignore pos_item_quantity and sales_date

data_loader_train = dutils.DataLoader(data.StoreData(datasets.train),
                                      batch_size=batch_size,
                                      pin_memory=False)
data_loader_eval = dutils.DataLoader(data.StoreData(datasets.validation),
                                     batch_size=batch_size,
                                     pin_memory=False
                                     )

store_model = network.FeedforwardModel(n_stores, n_weekday, n_monthday,
                               n_features, hidden_size)

if use_cuda:
    store_model = store_model.cuda()
opt = optim.SGD(store_model.parameters(), lr=learning_rate)

# loss_func = BusinessLoss(2, 1)
loss_func = nn.MSELoss()
#loss_func = nn.PoissonNLLLoss()

# if use_cuda:
#  store_model = store_model.cuda()

loss_all = []
for epoch in range(n_epochs):
    total_loss = 0.

    with tqdm(total=len(data_loader_train) + len(data_loader_eval),
              bar_format="Train: {postfix[0]}, Valid: {postfix[1]}",
              postfix=[None, None]) as pbar:
        for batch, target in data_loader_train:
            opt.zero_grad()
            if use_cuda:
                batch = batch.cuda()
                target = target.cuda()
            pred = store_model(batch)
            loss_val = loss_func(pred, target)
            loss_val.backward()
            total_loss += loss_val.item()
            opt.step()
            pbar.postfix[0] = loss_val.item()
            pbar.update()

        with torch.no_grad():
            total_loss_eval = 0.
            for batch, target in (data_loader_eval):
                if use_cuda:
                    batch = batch.cuda()
                    target = target.cuda()
                pred = store_model(batch)
                loss_val = loss_func(pred, target)
                total_loss_eval += loss_val.item()

                pbar.postfix[1] = loss_val.item()
                pbar.update()
    loss_all.append((total_loss, total_loss_eval))

    print('Epoch {}, training loss {:.5g} eval loss {:.5g}'
          .format(epoch, total_loss, total_loss_eval))
    torch.save(store_model.state_dict(),
               os.path.join(checkpoint_folder, 'snapshot-{}.pth'.format(epoch)))
