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
from trainer import tag, train, valid, test
from torch.utils.data import DataLoader

from transformers import T5Tokenizer, T5ForConditionalGeneration,Adafactor

parser = argparse.ArgumentParser()

'''training'''
parser.add_argument('--max_length' ,  type = int, default=128, help = 'Input max length')
parser.add_argument('--do_train' ,  type = int, default=1)
parser.add_argument('--do_short' ,  type = int, default=1)
parser.add_argument('--batch_size' , type = int, default=4)
parser.add_argument('--test_batch_size' , type = int, default=16)
parser.add_argument('--max_epoch' ,  type = int, default=1)
parser.add_argument('--DST_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--RW1_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--RW2_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--RW3_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--base_trained' , type = str,  default='t5-small', help = 'pretrainned model')
parser.add_argument('-patience', default=5, type=int,help='patience')


'''saving'''
parser.add_argument('--detail_log' , type = int,  default = 0)
parser.add_argument('--save_prefix', type = str, help = 'prefix for all savings', default = '')

'''enviroment'''
parser.add_argument('--seed' ,  type = int, default=1,  help = 'Training seed')
parser.add_argument('-n', '--nodes', default=1,type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=4, type=int,help='number of gpus per node')

'''data'''
parser.add_argument('--dev_path' ,  type = str)
parser.add_argument('--train_path' , type = str)
parser.add_argument('--test_path' , type = str)
parser.add_argument('--short_path' , type = str, default = "../woz_data/dev_data_short.json")
parser.add_argument('--ontology_path' , type = str)
parser.add_argument('--use_domain' , type = str)


args = parser.parse_args()
init.init_experiment(args)
logger = logging.getLogger("my")


def load_trained(model, model_path):
    logger.info(f"Use pretrained model{model_path}")
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.","") # [7:]remove 'module.' of DataParallel/DistributedDataParallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    logger.info("load safely")
    return model
    
         
def get_loader(dataset,batch_size):
    shuffle = False
    pin_memory = True
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=pin_memory,
        num_workers=0, shuffle=shuffle,  collate_fn=dataset.collate_fn)
    return loader       

def get_optimizer(model):
    opt = Adafactor(model.parameters(),lr=1e-3,
                    eps=(1e-30, 1e-3),
                    clip_threshold=1.0,
                    decay_rate=-0.8,
                    beta1=None,
                    weight_decay=0.0,
                    relative_step=False,
                    scale_parameter=False,
                    warmup_init=False)
    return opt

def train_worker(max_epoch, trainer, model, train_loader, dev_loader,  train_dataset, dev_dataset, optimizer,):
    min_loss = float('inf')
    best_performance = {}
    
    for epoch in range(max_epoch):
        trainer(args, model, train_loader, optimizer, train_dataset)
        loss = valid(args, model, dev_loader, args.data_rate, dev_dataset)
        logger.info("Epoch : %d,  Loss : %.04f" % (epoch, loss))
        if loss < min_loss:
            logger.info("New best")
            min_loss = loss
            best_performance['min_loss'] = min_loss.item()
            
        torch.save(model.state_dict(), f"model/woz{args.save_prefix}/epoch_{epoch}r_{args.data_rate}loss_{loss:.4f}.pt")
        logger.info("safely saved")
        
    return best_performance
           
def main_worker(DST_model, RW_models):
    batch_size = args.batch_size * args.gpus
    
    train_dataset =Dataset(args, args.train_path, 'train') # only unseen domain
    val_dataset =Dataset(args, args.dev_path, 'val')
    train_loader = get_loader(train_dataset, batch_size)
    val_loader = get_loader(val_dataset, batch_size)
    
    DST_model_opt = get_optimizer(DST_model)

    min_loss = float('inf')
    p =0
    
    loss, in_loss, out_loss, acc, in_acc, out_acc = valid(args, DST_model, val_loader)
    logger.info("pre-test,  Loss : %.04f, In_loss :%04f, Out_loss:%04f, acc:%04f, In_acc :%04f, Out_acc:%04f" 
            % ( loss, in_loss, out_loss, acc, in_acc, out_acc))
        
        
    for epoch in range(args.max_epoch):
        
        # 1. Tagging
        tagged_set = tag(args, DST_model, train_loader, train_dataset)
        train_dataset.update(tagged_set)
        train_loader = get_loader(train_dataset, batch_size)
        # 2. Reward

        reward = 0
        for RW_model in RW_models:    
            # reward += cal_reward(args, DST_model, train_loader)
            pass
            
        # 3. Train
        train(args, DST_model, train_loader, DST_model_opt)
        
        # 4. Validation
        loss, in_loss, out_loss, acc, in_acc, out_acc = valid(args, DST_model, val_loader)
        logger.info("Epoch : %d,  Loss : %.04f, In_loss :%04f, Out_loss:%04f, acc:%04f, In_acc :%04f, Out_acc:%04f" 
                    % (epoch, loss, in_loss, out_loss, acc, in_acc, out_acc))
        logger.info("patience : %d/%d" % (p, args.patience))
                    
        # 5. Save
        if min_loss>loss:
            p =0
            loss = min_loss
            torch.save(DST_model.state_dict(), f"model/woz{args.save_prefix}/epoch_{epoch}d_{args.use_domain}loss_{loss:.4f}.pt")
        else:
            p +=1
        if p > args.patience:
            logger.info("early stopping")
            break
        
        
        

    # logger.info(f"Best Score :  {best_performance}" )
    
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
    utils.makedirs("./logs");  utils.makedirs("./out");
    utils.makedirs(f"model/woz{args.save_prefix}")
    
    logger.info(f"{'-' * 30}")
    logger.info("Start New Trainning")
    start = time.time()
    logger.info(args)
    
    DST_model = T5ForConditionalGeneration.from_pretrained(args.base_trained)
    RW1_model = T5ForConditionalGeneration.from_pretrained(args.base_trained)
    RW2_model = T5ForConditionalGeneration.from_pretrained(args.base_trained)
    RW3_model = T5ForConditionalGeneration.from_pretrained(args.base_trained)
    
    DST_model = load_trained(DST_model, args.DST_model)   
    RW1_model = load_trained(RW1_model, args.RW1_model) 
    RW2_model = load_trained(RW2_model, args.RW2_model) 
    RW3_model = load_trained(RW3_model, args.RW3_model) 
    RW_models = [RW1_model, RW2_model, RW3_model]
    
    args.tokenizer = T5Tokenizer.from_pretrained(args.base_trained)
    
    DST_model = nn.DataParallel(DST_model).to("cuda")
    for i, RW_model in enumerate(RW_models):
        RW_model = nn.DataParallel(RW_model).to("cuda")
        RW_models[i] = RW_model
        
    main_worker(DST_model, RW_models)

    
    result_list = str(datetime.timedelta(seconds=time.time() - start)).split(".")
    logger.info(f"take time : {result_list[0]}")
    logger.info("End The Trainning")
    logger.info(f"{'-' * 30}")
    

