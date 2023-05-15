import torch
import pdb 
import json
import logging
import ontology
from utils import*
from collections import defaultdict

from utils import save_pickle

logger = logging.getLogger("my")


# TODO make it as class
def tag(args, model, train_loader, train_dataset):
    model.eval()
    tag_set = {} # dial_turn_slot_key : value
    logger.info("Tagging start")
    with torch.no_grad():
        for iter, batch in enumerate(train_loader):
            input_ids = batch["input"]["input_ids"].to("cuda")
            labels = batch["target"]["input_ids"].to("cuda")
            outputs_text = model.module.generate(input_ids=input_ids)
            text = args.tokenizer.batch_decode(outputs_text, skip_special_tokens = True)
            key_set = [f'{d}_{t}_{s}' for (d,t,s) in zip(batch['dial_id'], batch['turn_id'], batch['schema'])]
            tag_set.update({k:v for (k,v) in zip(key_set,text)})
            
            if (iter + 1) % 50 == 0:

                logger.info('step : {}/{} '.format(
                    iter, 
                    str(len(train_loader)),
                    )
                )
                
            
    return tag_set


def train(args, model, train_loader, optimizer):
    model.train()
    logger.info("Train start")
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input']['input_ids'].to('cuda')
        labels = batch['target']['input_ids'].to('cuda')
        
        outputs = model(input_ids=input_ids, labels=labels)
        outputs_text = model.module.generate(input_ids=input_ids)
        outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
        label_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in labels]
        loss =outputs.loss.mean()
        loss.backward()
        optimizer.step()
    
        if (iter + 1) % 50 == 0:
            logger.info(f"answer : {label_text[:3]}")
            logger.info(f"generated : {outputs_text[:3]}")
            logger.info('step : {}/{} Loss: {:.4f}'.format(
                iter, 
                str(len(train_loader)),
                loss.detach())
            )
        

def valid(args, model, val_loader):
    model.eval()
    loss_sum = 0
    logger.info("Validation start")
    with torch.no_grad():
        for iter,batch in enumerate(val_loader):
            input_ids = batch['input']['input_ids'].to('cuda')
            labels = batch['target']['input_ids'].to('cuda')
        
            outputs = model(input_ids=input_ids, labels=labels)
            outputs_text = model.module.generate(input_ids=input_ids)
            outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
            loss_sum += outputs.loss.mean().detach()
            if (iter + 1) % 50 == 0:
                logger.info('step : {}/{} Loss: {:.4f}'.format(
                iter, 
                str(len(val_loader)),
                outputs.loss.mean().detach()
                ))
           
    return  loss_sum/iter



def get_reward(args, model, train_loader):
    model.eval()
    loss_sum = 0
    logger.info("Validation start")
    with torch.no_grad():
        for iter,batch in enumerate(dev_loader):
            input_ids = batch['input']['input_ids'].to('cuda')
            labels = batch['target']['input_ids'].to('cuda')
        
            outputs = model(input_ids=input_ids, labels=labels)
            outputs_text = model.module.generate(input_ids=input_ids)
            outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
            loss_sum += outputs.loss.mean().detach()
            if (iter + 1) % 50 == 0:
                logger.info('step : {}/{} Loss: {:.4f}'.format(
                iter, 
                str(len(dev_loader)),
                outputs.loss.mean().detach()
                ))
           
    return  loss_sum/iter

def test(args, model, test_loader, test_dataset):
    belief_state= defaultdict(lambda : defaultdict(dict))# dial_id, # turn_id # schema
    
    model.eval()
    loss_sum = 0
    logger.info("Test start")
    with torch.no_grad():
        for iter,batch in enumerate(test_loader):
            outputs = model(input_ids=batch['input']['input_ids'].to('cuda'), labels=batch['target']['input_ids'].to('cuda'))
            outputs_text = model.generate(input_ids=batch['input']['input_ids'].to('cuda'))
            outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
            
            for idx in range(len(outputs_text)):
                dial_id = batch['dial_id'][idx]
                turn_id = batch['turn_id'][idx]
                schema = batch['schema'][idx]
                if turn_id not in belief_state[dial_id].keys():
                    belief_state[dial_id][turn_id] = {}
                if outputs_text[idx] == ontology.QA['NOT_MENTIONED'] : continue
    
                belief_state[dial_id][turn_id][schema] = outputs_text[idx]
                # test_dataset.belief_state[dial_id][turn_id][schema] = outputs_text[idx]
            

            if (iter + 1) % 50 == 0:
                logger.info('step : {}/{}'.format(
                iter+1, 
                str(len(test_loader)),
                ))
         
        with open('logs/pred_belief.json', 'w') as fp:
            json.dump(belief_state, fp, indent=4, ensure_ascii=False)
            

    
    
    test_file = json.load(open(args.test_path , "r"))

    joint_goal_acc, slot_acc, F1, detail_wrong = evaluate_metrics(belief_state,test_file ,  args.detail_log)

    loss_sum += outputs.loss.cpu()

    return  joint_goal_acc, slot_acc,  F1, detail_wrong, loss_sum/iter
        
        