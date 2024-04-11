import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import torch
from torch import nn
from utils import SignalDataset_iq, count_parameters
import argparse
from model_iq import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import time
import random

def train_transformer():
    if args.data == 'iq': 
        input_size = int(3200 / (args.src_time_step + args.trg_time_step))
    else: 
        input_size = 4096 
    input_dim = int(input_size / 2) 

    model = TransformerGenerationModel(input_dims=[input_dim, input_dim],
                             hidden_size=args.hidden_size,
                             embed_dim=args.embed_dim,
                             output_dim=args.output_dim,
                             num_heads=args.num_heads,
                             attn_dropout=args.attn_dropout,
                             relu_dropout=args.relu_dropout,
                             res_dropout=args.res_dropout,
                             out_dropout=args.out_dropout,
                             layers=args.nlevels,
                             attn_mask=args.attn_mask)
    if use_cuda:
        model = model.cuda()

    print("Model size: {0}".format(count_parameters(model)))

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=0)
    criterion= nn.CrossEntropyLoss(reduction="sum") 

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler, 
                'input_size': input_size, 
                'src_time_step': args.src_time_step,
                'trg_time_step': args.trg_time_step}
    return train_model(settings)


def train_model(settings):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    input_size = settings['input_size']
    src_time_step = settings['src_time_step']
    trg_time_step = settings['trg_time_step']


    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()

        for i_batch, (data_batched, label_batched) in enumerate(train_loader):
            cur_batch_size = len(data_batched) 
            src = data_batched[:, 0 : src_time_step, :].transpose(1, 0).float().cuda()
            trg = data_batched[:, src_time_step : , :].transpose(1, 0).float().cuda()
            trg_label = label_batched.cuda()
            model.zero_grad() 
            outputs = model(x=src, y=trg) 
            loss = criterion(outputs.double(), trg_label.long())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            epoch_loss += loss.detach().item() 

        avg_loss = epoch_loss / float(len(training_set))

        return avg_loss


    def evaluate(model, criterion):
        model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for i_batch, (data_batched, label_batched) in enumerate(test_loader):
                cur_batch_size = len(data_batched)
                src = data_batched[:, 0 : src_time_step, :].transpose(1, 0).float().cuda()
                trg = data_batched[:, src_time_step : , :].transpose(1, 0).float().cuda()
                trg_label = label_batched.cuda() 
                outputs = model(x=src, max_len=len(trg))
                loss = criterion(outputs.double(), trg_label.long())
                epoch_loss += loss.detach().item()
        avg_loss = epoch_loss / float(len(test_set))
        return avg_loss



    best_valid = 1e8
    for epoch in range(args.num_epochs):
        start = time.time() 

        train_loss = train(model, optimizer, criterion)
        print('Epoch {:2d} | Train Loss {:5.4f}'.format(epoch, train_loss))
        test_loss = evaluate(model, criterion)
        scheduler.step(test_loss)
        print("-"*50)
        print('Epoch {:2d} | Test  Loss {:5.4f}'.format(epoch, test_loss))
        print("-"*50)

        end = time.time()
        print("time: %d" % (end - start))



parser = argparse.ArgumentParser(description='Signal Data Analysis')
parser.add_argument('--model', type=str, default='Transformer',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--embed_dim', type=int, default=320,
                    help='dimension of real and imag embeddimg before transformer (default: 320)')
parser.add_argument('--data', type=str, default='iq')
parser.add_argument('--path', type=str, default='iq/',
                    help='path for storing the dataset')
parser.add_argument('--src_time_step', type=int, default=40)
parser.add_argument('--trg_time_step', type=int, default=24)
parser.add_argument('--attn_dropout', type=float, default=0.0,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.5,
                    help='output dropout')
parser.add_argument('--nlevels', type=int, default=6,
                    help='number of layers in the network (if applicable) (default: 6)')
parser.add_argument('--num_epochs', type=int, default=2000,
                    help='number of epochs (default: 2000)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--attn_mask', action='store_true',
                    help='use attention mask for Transformer (default: False)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip value (default: 0.35)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--hidden_size', type=int, default=2048,
                    help='hidden_size in transformer (default: 2048)')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

print(args)

# Assume cuda is used
use_cuda = True

"""
Data Loading
"""

torch.set_default_dtype(torch.float32)
total_time_step = args.src_time_step + args.trg_time_step
assert 3200 % total_time_step == 0, "3200 must be divisible by total_time_step in iq dataset"
args.output_dim = 3200//total_time_step
start = time.time()
print("Start loading the data....")
    
if args.data == 'iq': 
    training_set = SignalDataset_iq(args.path, time_step=total_time_step, train=True)
    test_set = SignalDataset_iq(args.path, time_step=total_time_step, train=False)
else:
    print("This file is for iq dataset only.")
    assert False
train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

end = time.time() 
print("Loading data time: %d" % (end - start))

train_transformer()
