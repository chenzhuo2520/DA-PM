import os
import argparse
import datetime
import re
import time
import torch
import random
import yaml
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm.auto import tqdm
from path import Path
from sklearn.metrics import mean_absolute_error
from dataset.utils import fetch_mosi_datasets, fetch_mosei_datasets
from models.dpam import My_Model
from utils.eval_metrics import *
from utils.prepare_vocab import VocabHelp

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=2.5e-5, type=float, help='Learning Rate')
parser.add_argument('--warmup_proportion', default=0.1, type=float, help='Set Warmup Proportion')
parser.add_argument('--num_epochs', default=30, type=int, help='Set Number of Epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Set Batch Size')
parser.add_argument('--max_text_length', default=80, type=int, help='Set Max Text Length')
parser.add_argument('--max_time', default=10, type=int, help='Set Max Time')
parser.add_argument('--mode', default='train', type=str, choices=['train', 'test'], help='Train Or Test Mode')
parser.add_argument('--dataset_name', default='mosi', type=str, choices=['mosi', 'mosei'])
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--config', type=str, default='../configs/config_mosi.yaml')



args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def read_config(config_file, args):
    if not os.path.exists(config_file):
        raise FileNotFoundError(config_file)

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=loader)

    ## Override Args Values
    config['num_epochs'] = args.num_epochs
    config['MAX_TEXT_LENGTH'] = args.max_text_length
    config['MAX_TIME'] = args.max_time
    config['learning_rate'] = args.learning_rate
    config['warmup_proportion'] = args.warmup_proportion
    config['batch_size'] = args.batch_size
    config['dataset_name'] = args.dataset_name
    config['num_workers'] = args.num_workers
    return config


def set_random_seeds(seed):
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fetch_model_optim_sched(config, device):
    model = My_Model(device).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_proportion'] * config['num_train_optimization_steps'],
        num_training_steps=config['num_train_optimization_steps'],
    )

    return model, optimizer, scheduler


def train_loop(train_dataloader, model, criterion, optimizer, scheduler, device, config):
    training_loss = 0
    gt_sentiment_scores = None
    pred_scores = None
    model.train()
    start_time = time.monotonic()
    for step, batch in enumerate(tqdm(train_dataloader)):
        # audio_embed, vis_embed, input_ids, attention_mask, gt_scores, pos, id, text, text_len = batch
        audio_embed, vis_embed, input_ids, attention_mask, gt_scores, pos = batch
        audio_embed = audio_embed.to(device)
        vis_embed = vis_embed.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        gt_scores = gt_scores.reshape((-1, 1)).to(device)
        pos = pos.to(device)
        output = model(audio_embed, vis_embed, input_ids, attention_mask, pos)
        loss = criterion(output, gt_scores)


        if config['gradient_accumulation_step'] > 1:
            loss = loss / config['gradient_accumulation_step']

        loss.backward()

        training_loss += loss.item()

        if (step + 1) % config['gradient_accumulation_step'] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if pred_scores is None:
            pred_scores = output.detach().to(torch.device('cpu'))
            gt_sentiment_scores = gt_scores.detach().to(torch.device('cpu'))
        else:
            pred_scores = torch.vstack((pred_scores, output.detach().to(torch.device('cpu')))).to(
                torch.device('cpu'))
            gt_sentiment_scores = torch.vstack((gt_sentiment_scores, gt_scores.detach().to(torch.device('cpu')))).to(
                torch.device('cpu'))

    print("Training Batch time:", time.monotonic() - start_time)

    return training_loss / len(train_dataloader), gt_sentiment_scores, pred_scores


def validation_loop(val_dataloader, model, criterion, device, config):
    val_loss = 0
    gt_sentiment_scores = None
    pred_scores = None
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(tqdm(val_dataloader)):
            audio_embed, vis_embed, input_ids, attention_mask, gt_scores, pos = batch
            audio_embed = audio_embed.to(device)
            vis_embed = vis_embed.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            gt_scores = gt_scores.reshape((-1, 1)).to(device)
            pos = pos.to(device)
            output = model(audio_embed, vis_embed, input_ids, attention_mask, pos)
            loss = criterion(output, gt_scores)

            if config['gradient_accumulation_step'] > 1:
                loss = loss / config['gradient_accumulation_step']

            val_loss += loss.item()

            if pred_scores is None:
                pred_scores = output.detach().to(torch.device('cpu'))
                gt_sentiment_scores = gt_scores.detach().to(torch.device('cpu'))
            else:
                pred_scores = torch.vstack((pred_scores, output.detach().to(torch.device('cpu')))).to(
                    torch.device('cpu'))
                gt_sentiment_scores = torch.vstack(
                    (gt_sentiment_scores, gt_scores.detach().to(torch.device('cpu')))).to(torch.device('cpu'))

    return val_loss / len(val_dataloader), gt_sentiment_scores, pred_scores


def compute_metrics(pred_scores, gt_sentiment_scores):
    pred_scores = pred_scores.flatten().detach().numpy()
    gt_sentiment_scores = gt_sentiment_scores.flatten().detach().numpy()

    mae = mean_absolute_error(y_true=gt_sentiment_scores, y_pred=pred_scores)
    binary_acc = accuracy_score(y_true=gt_sentiment_scores > 0, y_pred=pred_scores > 0)
    f1 = f1_score(y_true=gt_sentiment_scores > 0, y_pred=pred_scores > 0)
    corr = np.corrcoef(pred_scores, gt_sentiment_scores)[0][1]

    return mae, binary_acc, f1, corr


def save_model(model):
    name = 'best_model'
    if not os.path.exists('pre_trained_models'):
        os.mkdir('pre_trained_models')
    torch.save(model.state_dict(), f'pre_trained_models/{name}.pt')


def main():
    ## Create Wandb object
    config = read_config('./configs/config_mosi.yaml', args)  # args.config)
    ## Set All Random Seeds
    set_random_seeds(config['seed'])

    criterion = torch.nn.MSELoss()

    ## Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Device: ", device)

    ##Create Output Directory if it doesn't exists
    config['model_internal_dir'] = datetime.datetime.now().strftime('%m%d%Y_%H%M%S')
    if not os.path.exists(config['model_output_dir']):
        os.mkdir(config['model_output_dir'])

    os.mkdir(Path(config['model_output_dir']) / config['model_internal_dir'])

    print("Model Dir: ", Path(config['model_output_dir']) / config['model_internal_dir'])

    if config['dataset_name'] == 'mosi':
        pos_vocab = VocabHelp.load_vocab('data/MOSI/vocab_pos.vocab')  # POS
    else:
        pos_vocab = VocabHelp.load_vocab('data/MOSEI/vocab_pos_mosei.vocab')  # POS
    print('Size of pos_vocab:', len(pos_vocab))
    ## Create All Data Loaders
    print("Preparing Datasets and DataLoaders")
    if config['dataset_name'] == 'mosi':
        train_dataset, val_dataset, test_dataset = fetch_mosi_datasets(config, pos_vocab)
    else:
        train_dataset, val_dataset, test_dataset = fetch_mosei_datasets(config, pos_vocab)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'],
                                               num_workers=config['num_workers'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'],
                                             num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'],
                                              num_workers=config['num_workers'])
    print("Datasets and DataLoaders Done! :D")

    config['num_train_optimization_steps'] = (
            int(len(train_dataset) / config['batch_size'] / config['gradient_accumulation_step']) * config[
        'num_epochs'])

    model, optimizer, scheduler = fetch_model_optim_sched(config, device)
    print("Model Created! :D")

    ## Training Loop Over Epochs
    print("Starting Training")
    best_mae = 1e8

    for epoch in range(config['num_epochs']):
        start = time.time()
        print(f"Epoch: {epoch}")
        tloss, gt_train_scores, train_preds = train_loop(train_loader, model, criterion, optimizer, scheduler, device,
                                                         config)
        vloss, gt_val_scores, val_preds = validation_loop(val_loader, model, criterion, device, config)
        test_loss, gt_test_scores, test_preds = validation_loop(test_loader, model, criterion, device, config)
        end = time.time()
        duration = end - start

        print("-" * 50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration,
                                                                                               vloss, test_loss))
        print("-" * 50)

        if test_loss < best_mae:
            best_epoch = epoch
            best_mae = test_loss
            eval_mosi(test_preds, gt_test_scores, True)
            best_results = test_preds
            best_truths = gt_test_scores
            print(f"Saved model at pre_trained_models/MM.pt!")
            save_model(model)
    print(f'Best epoch: {best_epoch}')
    eval_mosi(best_results, best_truths, True)


if __name__ == "__main__":
    main()
