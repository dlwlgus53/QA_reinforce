import utils
import time
import torch
import logging
import argparse
import datetime
from dataset import Dataset
import torch.nn as nn
import pdb
import init
from collections import OrderedDict
from trainer import valid, train, test
from torch.utils.data import DataLoader

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor

parser = argparse.ArgumentParser()

'''training'''
parser.add_argument('--max_length' ,  type = int, default=128, help = 'Input max length')
parser.add_argument('--do_train' ,  type = int, default=1)
parser.add_argument('--do_short' ,  type = int, default=1)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--test_batch_size' , type = int, default=16)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--base_trained', type = str, default = 't5-small', help =" pretrainned model from 🤗")
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')

'''saving'''
parser.add_argument('--detail_log' , type = int,  default = 0)
parser.add_argument('--save_prefix', type = str, help = 'prefix for all savings', default = '')

'''enviroment'''
parser.add_argument('--seed' ,  type = int, default=1,  help = 'Training seed')
parser.add_argument('--port' , type = int,  default = 12355, help = 'Port for multi-gpu enviroment')
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=4, type=int,help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,help='ranking within the nodes')

'''data'''
parser.add_argument('--dev_path' ,  type = str)
parser.add_argument('--train_path' , type = str)
parser.add_argument('--test_path' , type = str)
parser.add_argument('--data_rate' ,  type = float, default=0.01, help = 'Train data size(rate)')



args = parser.parse_args()
init.init_experiment(args)
logger = logging.getLogger("my")


def load_trained(model, optimizer = None):
    logger.info(f"Use pretrained model{args.pretrained_model}")
    state_dict = torch.load(args.pretrained_model)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.","") # [7:]remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    if optimizer != None:
        logger.info("load optimizer")
        opt_path = "./model/optimizer/" + args.pretrained_model[7:] #todo
        optimizer.load_state_dict(torch.load(opt_path))
    logger.info("load safely")
    
         
def get_loader(dataset,batch_size):
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle,  collate_fn=dataset.collate_fn)
    return loader       
       
def main_worker(model):
    
    batch_size = args.batch_size * args.gpus    
    train_dataset =Dataset(args, args.train_path, 'train')
    val_dataset =Dataset(args, args.dev_path, 'val')
    
    train_loader = get_loader(train_dataset, batch_size)
    dev_loader = get_loader(val_dataset, batch_size)
    optimizer = Adafactor(model.parameters(),lr=1e-3,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False)
    
    min_loss = float('inf')
    best_performance = {}

    logger.info("Training start")
    for epoch in range(args.max_epoch):
        train(args, model, train_loader, optimizer, train_dataset)
        loss = valid(args, model, dev_loader, args.data_rate, val_dataset)
        logger.info("Epoch : %d,  Loss : %.04f" % (epoch, loss))

        if loss < min_loss:
            logger.info("New best")
            min_loss = loss
            best_performance['min_loss'] = min_loss.item()
        torch.save(model.state_dict(), f"model/woz{args.save_prefix}/epoch_{epoch}r_{args.data_rate}loss_{loss:.4f}.pt")
        torch.save(optimizer.state_dict(), f"model/optimizer/woz{args.save_prefix}/epoch_{epoch}r_{args.data_rate}loss_{loss:.4f}.pt")
        logger.info("safely saved")
                
    logger.info(f"Best Score :  {best_performance}" )
    
def find_test_model(test_output_dir, cut_num = None):
    import os
    from operator import itemgetter
    fileData = []
    for fname in os.listdir(test_output_dir):
        if fname.startswith('epoch'):
            fileData.append(fname)

    fileData = sorted(fileData)
    if cut_num :
        return fileData[:cut_num]
    else:
        return fileData
                
        
def evaluate():
    if args.do_short: args.test_path = '../../woz-data/MultiWOZ_2.1/train_data0.001.json'
    
    test_dataset =Dataset(args, args.test_path, 'test')
    loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.test_batch_size, pin_memory=True,
        num_workers=0, shuffle=False, collate_fn=test_dataset.collate_fn)
    
    
    test_output_dir = f"model/woz{args.save_prefix}/"
    logger.info(f"Find model in {test_output_dir}")
    test_model_paths = find_test_model(test_output_dir)

    
    for test_model_path in test_model_paths:
        logger.info(f"User pretrained model{test_output_dir + test_model_path}")
        state_dict = torch.load(test_output_dir + test_model_path)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.","") # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v
        model = T5ForConditionalGeneration.from_pretrained(args.base_trained, return_dict=True).to('cuda')
        model.load_state_dict(new_state_dict)

            
        joint_goal_acc, slot_acc, F1, detail_wrong, loss = test(args, model, loader, test_dataset)
        
        logger.info(f'file {test_model_path} JGA : {joint_goal_acc} F1 : {F1} Slot Acc : {slot_acc} F1 : {F1} Loss : {loss}')
        
        if args.detail_log:
            utils.dict_to_json(detail_wrong, f'{args.save_prefix}{args.data_rate}.json')
    
    
    

if __name__ =="__main__":
    utils.makedirs("./logs"); utils.makedirs("./model/optimizer"); utils.makedirs("./out");
    utils.makedirs(f"model/optimizer/woz{args.save_prefix}")
    utils.makedirs(f"model/woz{args.save_prefix}")
    
    logger.info(f"{'-' * 30}")
    logger.info("Start New Trainning")
    start = time.time()
    logger.info(args)
    
    args.world_size = args.gpus * args.nodes 
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    if args.do_train:
        model = T5ForConditionalGeneration.from_pretrained(args.base_trained)
        if args.pretrained_model:
            load_trained(model)   
        model = nn.DataParallel(model).to("cuda")
        main_worker(model)
    evaluate()

    
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(f"take time : {result_list[0]}")
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")
    

