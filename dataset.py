import re
import pdb
import json
import torch
import pickle
from tqdm import tqdm
import logging
from log_conf import init_logger
from collections import defaultdict
import random

logger = logging.getLogger("my")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, data_type, description = 'description1'):
        random.seed(args.seed)
        self.data_type = data_type
        self.tokenizer = args.tokenizer
        self.max_length = args.max_length
        self.use_domain = args.use_domain
        self.description = description
        self.ontology = json.load(open(args.ontology_path,"r"))
        self.pseudo_answer = None
        self.dict_answer_list  = {}
        
        raw_path = data_path
        if args.do_short :
            raw_path = args.short_path
                
        logger.info(f"load {self.data_type} raw file {raw_path}")
        raw_dataset = json.load(open(raw_path, "r"))
        turn_id, dial_id, question, schema, answer, context = self.seperate_data(
            raw_dataset
        )

        assert (
            len(turn_id) == len(dial_id) == len(question) == len(schema) == len(answer)
        )

        self.answer = answer  # for debugging
        # self.target = self.encode(answer) # TODO  not in here, do it later
        self.turn_id = turn_id
        self.dial_id = dial_id
        self.question = question
        self.schema = schema
        self.context = context
        
    def encode(self, texts, return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            # Truncate
            while True:
                tokenized = self.tokenizer.batch_encode_plus(
                    [text], padding=False, return_tensors=return_tensors
                )  # TODO : special token
                if len(tokenized['input_ids'][0]) > self.max_length:
                    idx = [m.start() for m in re.finditer("\[user\]", text)]
                    text = text[: idx[0]] + text[idx[1] :]  # delete one turn
                else:
                    break

            examples.append(tokenized)
        return examples

    def __len__(self):
        return len(self.dial_id)
    
    
    
    def update(self, pseudo_answer):
        logger.info(f"update the pseudo answer, length is {len(pseudo_answer)} ")
        self.pseudo_answer = pseudo_answer
        acc = [self.pseudo_answer[key] == self.dict_answer_list[key] for key in pseudo_answer.keys()]
        mask = [self.pseudo_answer[key] == self.ontology["NOT_MENTIONED"] for key in pseudo_answer.keys()]
        recall= []
        
        for idx, m in enumerate(mask):
            if m != True:
                recall.append(acc[idx])
        logger.info(f"acc {sum(acc)/len(acc):.4f}, recall {sum(recall)/len(recall):.4f} (true num : {(len(mask)-sum(mask))/len(mask):.4f})")
        
    def seperate_data(self, dataset):
        context = defaultdict(lambda: defaultdict(str))  # dial_id, # turn_id

        question = []
        answer = []
        schema = []
        dial_id = []
        turn_id = []

        for d_id in dataset.keys():
            dialogue = dataset[d_id]["log"]
            dialogue_text = ""
            for t_id, turn in enumerate(dialogue):
                
                dialogue_text += "[user] "
                dialogue_text += turn["user"]
                for key_idx, key in enumerate(self.ontology["all-domain"]):
                    if self.data_type != 'train' or (self.data_type == 'train' and self.use_domain in key):
                        q = self.ontology[key][self.description]
                        
                        if key in turn['belief']:  
                            a = turn['belief'][key]
                            if isinstance(a, list):
                                a = a[0] 
                        else:
                            a = self.ontology["NOT_MENTIONED"]

                        ### Important code!!! ###
                        if self.data_type == 'train' :
                            answer_save_key = f'{d_id}_{t_id}_{key}'
                            self.dict_answer_list[answer_save_key] = a 
                            a = 'None'
                            
                        
                        schema.append(key)
                        answer.append(a)
                        question.append(q)
                        dial_id.append(d_id)
                        turn_id.append(t_id)
                        context[d_id][t_id] = dialogue_text

                dialogue_text += "[sys] "
                dialogue_text += turn["response"]

        for_sort = [
            [t, d, q, s, a]
            for (t, d, q, s, a) in zip(turn_id, dial_id, question, schema, answer)
        ]
        
        sorted_items = sorted(for_sort, key=lambda x: (x[0], x[1])) # from easy to hard

        turn_id = [s[0] for s in sorted_items]
        dial_id = [s[1] for s in sorted_items]
        question = [s[2] for s in sorted_items]
        schema = [s[3] for s in sorted_items]
        answer = [s[4] for s in sorted_items]
        logger.info(f"{self.data_type} length : {len(turn_id)}")
        logger.info(f"use domain : {self.use_domain} ")
        logger.info(f"Use schemas: {set(schema)} ")
        return turn_id, dial_id, question, schema, answer, context

    def __getitem__(self, index):
        dial_id = self.dial_id[index]
        turn_id = self.turn_id[index]
        schema = self.schema[index]
        question = self.question[index]
        context = self.context[index]
        if self.data_type == 'train' and self.pseudo_answer:
            if index == 0:
                logger.info("use pseudo answer")
            key = f'{dial_id}_{turn_id}_{schema}'
            answer = self.pseudo_answer[key] 
        else:
            answer = self.answer[index]
        return {
            "answer": answer,
            "turn_id": turn_id,
            "question": question,
            "context": context,
            "dial_id": dial_id,
            "schema": schema,
        }

    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        """

        dial_id = [x["dial_id"] for x in batch]
        turn_id = [x["turn_id"] for x in batch]
        question = [x["question"] for x in batch]
        schema = [x["schema"] for x in batch]
        # target_list = [x["target"] for x in batch] # this is original
        answer = [x["answer"] for x in batch] # make it as like target

        history = [self.context[d][t] for (d, t) in zip(dial_id, turn_id)]
        input_source = [
            f"question: {q} context: {c}"
            for (q, c) in zip(
                question,
                history,
            )
        ]

        source = self.encode(input_source)
        target = self.encode(answer)
        
        source_list = [{k: v.squeeze() for (k, v) in s.items()} for s in source]
        target_list = [{k: v.squeeze() for (k, v) in s.items()} for s in target]

        pad_source = self.tokenizer.pad(source_list, padding=True)
        pad_target = self.tokenizer.pad(target_list, padding=True)

        return {
            "input": pad_source,
            "target": pad_target,
            "schema": schema,
            "dial_id": dial_id,
            "turn_id": turn_id,
        }


if __name__ == "__main__":
    import argparse

    init_logger(f"data_process.log")
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument("--do_short", type=int, default=1)
    parser.add_argument("--seed", type=float, default=1)
    parser.add_argument("--max_length", type=int, default=100)
    parser.add_argument("--except_domain", type=str, default="hotel")
    parser.add_argument("--description", type=str, default="description1")
    parser.add_argument("--ontology_path", type=str, default="./DST_model/QA.json")
    parser.add_argument('--short_path' , type = str, default = "../woz_data/dev_data_short.json")


    args = parser.parse_args()

    args.data_path = "../../woz-data/MultiWOZ_2.1/train_data.json"
    from transformers import T5Tokenizer

    args.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    dataset = Dataset(args, args.data_path, "train")
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn
    )
    t = args.tokenizer
    for batch in loader:
        for i in range(16):
            # print(t.decode(batch["input"]["input_ids"][i]))
            # print(t.decode(batch["target"]["input_ids"][i]))
            # print()
            pass

