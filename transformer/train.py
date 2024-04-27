import glob
import re
import torch
from torch import nn
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from utils import SignalDataset_music, StreamToLogger, get_env_variable, upload_blob
import argparse
from model import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import os
import time
import random
from torchmetrics.functional.classification import multilabel_average_precision
from sklearn.metrics import average_precision_score

from dotenv import load_dotenv

import mlflow

import logging
import datetime
import subprocess

if not load_dotenv():
    raise EnvironmentError("Missing .env file with valid key/values.")

# Accessing variables
mlflow_server_uri = get_env_variable('MLFLOW_SERVER_URI')
bucket_name = get_env_variable('BUCKET_NAME')

log_path = 'logs'
checkpoints_path = 'checkpoints'

mlflow.set_tracking_uri(uri=mlflow_server_uri)
mlflow.set_experiment("dl_signal")

def train_transformer():
    model = TransformerModel(
        time_step=args.time_step,
        input_dims=args.modal_lengths,
        hidden_size=args.hidden_size,
        embed_dim=args.embed_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        attn_dropout=args.attn_dropout,
        relu_dropout=args.relu_dropout,
        res_dropout=args.res_dropout,
        out_dropout=args.out_dropout,
        layers=args.nlevels,
        attn_mask=args.attn_mask,
        complex_mha=args.complex_mha,
        conj_attn=args.conj_attn,
        pre_ln=args.pre_ln,
        softmax=args.softmax,
        rescale=args.rescale,
        squared_norm=args.squared_norm,
        minus_im=args.minus_im,
        re=args.re,
    )
    if use_cuda:
        model = model.cuda()

    print("Model size: {0}".format(count_parameters(model)))

    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.lr, weight_decay=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    trainer = Trainer(settings)
    return trainer.train_model()

class Trainer:
    def __init__(self, settings):
        self.settings = settings

        self.best_eval_train_loss = float('inf')
        self.best_eval_train_loss_epoch = -1
        self.best_eval_train_aps = float('-inf')
        self.best_eval_train_aps_epoch = -1
        self.best_eval_validation_loss = float('inf')
        self.best_eval_validation_loss_epoch = -1
        self.best_eval_validation_aps = float('-inf')
        self.best_eval_validation_aps_epoch = -1

        self.training_step = 0
        self.epoch = 0

    def train_model(self):
        settings = self.settings

        squared_norm_part = "-squared_norm" if args.squared_norm else ""
        rescale_part = f"-rescale_{args.rescale}" if args.rescale != 1 else ""

        run_name = f"complex_mha-pre_ln-softmax{squared_norm_part}{rescale_part}"
        # run_name = None

        with mlflow.start_run(run_name=run_name, nested=nested_runs):
            run_start_time = time.time()
            model = settings['model']
            optimizer = settings['optimizer']
            criterion = settings['criterion']
            scheduler = settings['scheduler']
            model.to(device)

            mlflow.log_params(vars(args))

            mlflow.log_metric('parameters_total', count_parameters(model))
            mlflow.log_metric('parameters_attention_blocks', count_parameters(model.trans))

            def train(model, optimizer, criterion):
                epoch_loss = 0.0
                batch_size = args.batch_size
                num_batches = len(training_set) // batch_size
                total_batch_size = 0
                start_time = time.time()
                shape = (args.time_step, training_set.len, args.output_dim)
                true_vals = torch.zeros(shape)
                pred_vals = torch.zeros(shape)
                model.train()

                for i_batch, (batch_X, batch_y) in enumerate(train_loader):
                    model.zero_grad()
                    if args.variance_stats and i_batch == ((len(training_set) // batch_size) // 2) and self.epoch == 0:
                        model.trans.variance_stats = True
                    batch_X = batch_X.transpose(0, 1)
                    batch_y = batch_y.transpose(0, 1)
                    batch_X, batch_y = batch_X.float().to(device=device), batch_y.float().to(device=device)
                    preds = model(batch_X)
                    true_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = batch_y.detach()
                    pred_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = preds.detach()
                    loss = criterion(preds, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    total_batch_size += batch_size
                    epoch_loss += loss.item() * batch_size
                    current_loss = loss.item() * batch_size
                    avg_epoch_loss = epoch_loss / (i_batch + 1)
                    self.training_step = self.training_step + 1
                    print("[train_01]", "batch:", i_batch, "| epoch avg loss:", avg_epoch_loss, "| iteration loss:", current_loss)

                    if args.variance_stats and i_batch == ((len(training_set) // batch_size) // 2) and self.epoch == 0:
                        model.trans.variance_stats = False
                        stats = model.get_and_reset_stats()

                        for layer in range(stats.shape[0]):
                            for component in range(2):
                                for moment in range(3):
                                    mlflow.log_metric(f"variance_{'re' if component == 0 else 'im'}_m{moment}", stats[layer][component][moment], step=layer)
                        print("Recorded encoder layer variance statistics.")
                
                if args.og_aps:
                    eval_train_aps = average_precision_score(true_vals.flatten(), pred_vals.flatten())
                else:
                    eval_train_aps = multilabel_average_precision(pred_vals.view(-1, args.output_dim), true_vals.view(-1, args.output_dim).int(), args.output_dim, average="micro")

                eval_train_loss = epoch_loss / len(training_set)

                mlflow.log_metric("eval_train_loss", eval_train_loss, step=self.epoch)
                mlflow.log_metric("eval_train_aps", eval_train_aps, step=self.epoch)

                if eval_train_loss < self.best_eval_train_loss:
                    self.best_eval_train_loss = eval_train_loss
                    self.best_eval_train_loss_epoch = epoch

                mlflow.log_metric("best_eval_train_loss", self.best_eval_train_loss, step=self.epoch)
                mlflow.log_metric("best_eval_train_loss_epoch", self.best_eval_train_loss_epoch, step=self.epoch)

                if eval_train_aps > self.best_eval_train_aps:
                    self.best_eval_train_aps = eval_train_aps
                    self.best_eval_train_aps_epoch = epoch

                mlflow.log_metric("best_eval_train_aps", self.best_eval_train_aps, step=self.epoch)
                mlflow.log_metric("best_eval_train_aps_epoch", self.best_eval_train_aps_epoch, step=self.epoch)

                return eval_train_loss, eval_train_aps

            def evaluate(model, criterion):
                epoch_loss = 0.0
                batch_size = args.batch_size
                loader = validation_loader
                total_batch_size = 0
                shape = (args.time_step, validation_set.len, args.output_dim) 
                true_vals = torch.zeros(shape)
                pred_vals = torch.zeros(shape)
                model.eval()
                with torch.no_grad():
                    for i_batch, (batch_X, batch_y) in enumerate(loader):
                        batch_X = batch_X.transpose(0, 1)
                        batch_y = batch_y.transpose(0, 1)
                        batch_X, batch_y = batch_X.float().to(device=device), batch_y.float().to(device=device)
                        preds = model(batch_X)
                        true_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = batch_y.detach()
                        pred_vals[:, i_batch*batch_size:(i_batch+1)*batch_size, :] = preds.detach()
                        loss = criterion(preds, batch_y)
                        total_batch_size += batch_size
                        epoch_loss += loss.item() * batch_size
                        current_loss = loss.item() * batch_size
                        avg_epoch_loss = epoch_loss / (i_batch + 1)
                        print("[validation_01]", "batch:", i_batch, "| epoch avg loss:", avg_epoch_loss, "| iteration loss:", current_loss)

                    if args.og_aps:
                        eval_validation_aps = average_precision_score(true_vals.flatten(), pred_vals.flatten())
                    else:
                        eval_validation_aps = multilabel_average_precision(pred_vals.view(-1, args.output_dim), true_vals.view(-1, args.output_dim).int(), args.output_dim, average="micro")


                eval_validation_loss = epoch_loss / len(validation_set)

                mlflow.log_metric("eval_validation_loss", eval_validation_loss, step=self.epoch)
                mlflow.log_metric("eval_validation_aps", eval_validation_aps, step=self.epoch)

                if eval_validation_loss < self.best_eval_validation_loss:
                    self.best_eval_validation_loss = eval_validation_loss
                    self.best_eval_validation_loss_epoch = epoch

                mlflow.log_metric("best_eval_validation_loss", self.best_eval_validation_loss, step=self.epoch)
                mlflow.log_metric("best_eval_validation_loss_epoch", self.best_eval_validation_loss_epoch, step=self.epoch)

                if eval_validation_aps > self.best_eval_validation_aps:
                    self.best_eval_validation_aps = eval_validation_aps
                    self.best_eval_validation_aps_epoch = epoch

                mlflow.log_metric("best_eval_validation_aps", self.best_eval_validation_aps, step=self.epoch)
                mlflow.log_metric("best_eval_validation_aps_epoch", self.best_eval_validation_aps_epoch, step=self.epoch)

                return eval_validation_loss, eval_validation_aps
            
            old_sys_stdout = sys.stdout

            for epoch in range(starting_epoch, args.num_epochs):
                self.epoch = epoch

                if args.logging:
                    logging.root.handlers.clear()

                    current_time_utc = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    git_branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode()
                    git_short_commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode()
                    log_filename = f'training_log_{current_time_utc}_{git_short_commit_id}-{git_branch_name}_E{epoch:04}.log'

                    logging.basicConfig(level=logging.INFO, filename=f'{log_path}/{log_filename}', filemode='a')

                    stdout_logger = logging.getLogger('STDOUT')
                    stdout_logger.handlers.clear()

                    sl = StreamToLogger(stdout_logger, logging.INFO)
                    sys.stdout = sl

                    stdout_logger.info(args)
                    stdout_logger.info("Model size: {0}".format(count_parameters(model)))

                mlflow.log_metric('learning_rate', optimizer.param_groups[0]["lr"], step=self.epoch)

                start = time.time() 

                train_loss, acc_train = train(model, optimizer, criterion)
                print('Epoch {:2d} | Train Loss {:5.4f} | APS {:5.4f}'.format(epoch, train_loss, acc_train))
                validation_loss, acc_validation = evaluate(model, criterion)
                scheduler.step(validation_loss)
                print("-"*50)
                print('Epoch {:2d} | Validation Loss {:5.4f} | APS {:5.4f}'.format(epoch, validation_loss, acc_validation))
                print("-"*50)

                end = time.time()
                print("time: %d" % (end - start))

                current_time = time.time()
                elapsed_time = current_time - run_start_time

                examples_per_second = (self.training_step * args.batch_size) / elapsed_time
                mlflow.log_metric('examples_per_second', examples_per_second, step=self.epoch)

                if args.logging:
                    sl.close()
                    sys.stdout = old_sys_stdout

                    source_file_name = f'{log_path}/{log_filename}'
                    destination_blob_name = f'{log_path}/{log_filename}'
                    
                    upload_blob(bucket_name, source_file_name, destination_blob_name)

                
                # Save model state dictionary
                model_state = model.state_dict()

                # Save optimizer state dictionary
                optimizer_state = optimizer.state_dict()

                # Save scheduler state dictionary (if any)
                scheduler_state = scheduler.state_dict() if scheduler else None

                # Save everything to a dictionary
                checkpoint = {
                    'model': model_state,
                    'optimizer': optimizer_state,
                    'scheduler': scheduler_state
                }

                run = mlflow.active_run()

                filename = f'{run.info.run_id}_E{epoch:04}.pth'
                save_path = f'{checkpoints_path}/{filename}'

                blob_uri = f'https://storage.cloud.google.com/{bucket_name}/{save_path}'

                mlflow.log_text(blob_uri, save_path)

                torch.save(checkpoint, save_path)

                source_file_name = save_path
                destination_blob_name = save_path
                
                upload_blob(bucket_name, source_file_name, destination_blob_name)

print(sys.argv)
parser = argparse.ArgumentParser(description='Signal Data Analysis')
parser.add_argument('--attn_dropout', type=float, default=0.0,
                    help='attention dropout')
parser.add_argument('--attn_mask', action='store_true',
                    help='use attention mask for Transformer (default: False)')
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 16)')
parser.add_argument('--clip', type=float, default=0.35,
                    help='gradient clip value (default: 0.35)')
parser.add_argument('--complex_mha', action='store_true', dest='complex_mha',
                    help='use reformulated complex multiheaded attention')
parser.add_argument('--conj_attn', action='store_true', dest='conj_attn',
                    help='use reformulated complex conjugate attention')
parser.add_argument('--continue_from', type=str,
                    help='specify a checkpoint to continue from')
parser.add_argument('--data', type=str, default='music')
parser.add_argument('--embed_dim', type=int, default=320,
                    help='dimension of real and imag embeddimg before transformer (default: 320)')
parser.add_argument('--hidden_size', type=int, default=2048,
                    help='hidden_size in transformer (default: 2048)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--disable-logging', action='store_false', dest='logging',
                    help='disables logging')
parser.add_argument('--modal_lengths', nargs='+', type=int, default=[2048, 2048],
                    help='lengths of each modality (default: [2048, 2048])')
parser.add_argument('--minus_im', action='store_true', dest='minus_im',
                    help='subtract the imaginary part from the real part in the complex attention mechanism')
parser.add_argument('--model', type=str, default='Transformer',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--nlevels', type=int, default=6,
                    help='number of layers in the network (if applicable) (default: 6)')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs (default: 20)')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads for the transformer network')
parser.add_argument('--og_aps', action='store_true', dest='og_aps',
                    help='use the original APS function')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--out_dropout', type=float, default=0.5,
                    help='hidden layer dropout')
parser.add_argument('--output_dim', type=int, default=128,
                    help='dimension of output (default: 128)')
parser.add_argument('--path', type=str, default='music/',
                    help='path for storing the dataset')
parser.add_argument('--pre_ln', action='store_true', dest='pre_ln',
                    help='position Layer Norms before attention and FF')
parser.add_argument('--re', action='store_true', dest='re',
                    help='use only the real part in the attention mechanism')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='additional scaling factor for normalization of softmax dot product (default: 1.0)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--softmax', action='store_true', dest='softmax',
                    help='use softmax for attention scoring')
parser.add_argument('--squared_norm', action='store_true', dest='squared_norm',
                    help='square the real and imaginary parts before softmax in attention')
parser.add_argument('--time_step', type=int, default=64,
                    help='number of time step for each sequence(default: 64)')
parser.add_argument('--variance_stats', action='store_true', dest='variance_stats',
                    help='record encoder layer output activation variances')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

print(args)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
"""
Data Loading
"""
torch.set_default_dtype(torch.float32)
print("Start loading the data....")
start_time = time.time() 
if args.data == 'music':
    training_set = SignalDataset_music(args.path, args.time_step, mode='train')
    validation_set = SignalDataset_music(args.path, args.time_step, mode='validation')
elif args.data == 'iq':
    training_set = SignalDataset_iq(args.path, args.time_step, train=True)
    validation_set = SignalDataset_iq(args.path, args.time_step, train=False)
    print("This file is for music dataset only; use train_iq.py for training iq net.")
    assert False
print("Finish loading the data....")
train_loader = torch.utils.data.DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=True)

starting_epoch = 0
run_id=None
nested_runs = False
checkpoint=None

if args.continue_from:
    checkpoint_pattern = f'./checkpoints_{args.continue_from}_E*.pth'
    checkpoint_files = glob.glob(checkpoint_pattern)
    if checkpoint_files:
        
        # Find the latest file by sorting based on the numeric value in the filename
        checkpoint_files.sort(key=lambda x: int(re.findall("E(\d{4})\.pth", x)[0]))
        latest_checkpoint = checkpoint_files[-1]
        
        # Extracting both unique ID and epoch number from the filename
        match = re.search(r'checkpoints_([a-f0-9]+)_E(\d{4})\.pth', latest_checkpoint)
        if match:
            run_id = match.group(1)
            epoch_number = int(match.group(2))
            print(f"Found checkpoint for run_id={run_id} at epoch {epoch_number}")
        else:
            print("Error: Checkpoint filename format is incorrect.")
            raise ValueError("Filename pattern does not match expected format.")

        checkpoint = torch.load(latest_checkpoint, map_location=None) #torch.device('cpu'))

        starting_epoch = epoch_number + 1
        nested_runs=True
        print(f"Resuming training from epoch {starting_epoch}")
    else:
        print(f"No checkpoint found for run_id={args.continue_from}. Cancelling Run.")
        exit()

if nested_runs:
    with mlflow.start_run(run_id=run_id):
        # for squared_norm in [True]:
        #     args.squared_norm = squared_norm
        #     for rescale in [1, 2, 4]:
        #         args.rescale = rescale
        train_transformer()
else:
    train_transformer()