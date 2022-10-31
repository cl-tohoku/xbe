
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import sys
import argparse
import matplotlib
import pdb
import numpy as np 
import time
import random
import time
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from apex import amp
from tqdm import trange
from torch.utils import data
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from dataset import *
from models import *
# crst info.
from models_crst_all import *
import crst_conf

from sklearn.metrics import average_precision_score
from apex.parallel import DistributedDataParallel
from scipy.special import softmax
class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)


def f1_score(output, label, rel_num):
    correct_by_relation = Counter()
    guess_by_relation = Counter()
    gold_by_relation = Counter()

    for i in range(len(output)):
        guess = output[i]
        gold = label[i]
        if gold == 0 and guess == 0:
            continue
        if gold == 0 and guess != 0:
            guess_by_relation[guess] += 1
        if gold != 0 and guess == 0:
            gold_by_relation[gold] += 1
        if gold != 0 and guess != 0:
            guess_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[gold] += 1
    
    f1_by_relation = Counter()
    recall_by_relation = Counter()
    prec_by_relation = Counter()
    for i in range(1, rel_num):
        recall = 0
        if gold_by_relation[i] > 0:
            recall = correct_by_relation[i] / gold_by_relation[i]
        precision = 0
        if guess_by_relation[i] > 0:
            precision = correct_by_relation[i] / guess_by_relation[i]
        if recall + precision > 0 :
            f1_by_relation[i] = 2 * recall * precision / (recall + precision)
        recall_by_relation[i] = recall
        prec_by_relation[i] = precision

    micro_f1 = 0
    if sum(guess_by_relation.values()) != 0 and sum(correct_by_relation.values()) != 0:
        recall = sum(correct_by_relation.values()) / sum(gold_by_relation.values())
        prec = sum(correct_by_relation.values()) / sum(guess_by_relation.values())    
        micro_f1 = 2 * recall * prec / (recall+prec)
    return micro_f1, f1_by_relation

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def train(args, model, train_dataloader, dev_dataloader, test_dataloader, devBagTest=None, testBagTest=None):
    # total step
    step_tot = len(train_dataloader) * args.max_epoch

    # optimizer
    if args.optim == "adamw":
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon, correct_bias=False)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=step_tot)
    elif args.optim == "sgd":
        params = model.parameters()
        optimizer = optim.SGD(params, args.lr)
    elif args.optim == "adam":
        params = model.parameters()
        optimizer = optim.Adam(params, args.lr)

    # amp training
    if args.optim == "adamw":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O0")

    model.train()
    model.zero_grad()

    print("Begin train...")
    print("We will train model in %d steps" % step_tot)
    global_step = 0
    best_dev_score = 0
    best_test_score = 0
    best_epoch = 0
    for i in range(args.max_epoch):
        for batch in train_dataloader:
            inputs = {
                "label":batch[0],
                "input_ids":batch[3],
                "mask":batch[4],
                "h_pos":batch[5],
                "t_pos":batch[6],
                'head_id':batch[7],
                'tail_id':batch[8],
                'sym_ids':batch[9],
                'sym_mask':batch[10],
                'sym_label':batch[11]
            }
            if args.use_bag:
                inputs["scope"] = batch[2]
            model.training = True
            model.train()
            loss, output = model(**inputs)
            if args.optim == "adamw":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            if args.optim == "adamw":
                scheduler.step()
            model.zero_grad()
            global_step += 1

            output = output.cpu().detach().numpy()
            label = batch[0].cpu().numpy()
            try:
                crr = (output == label).sum()
            except:
                print(crr)
                print(output)
                print(label)
                exit(0)
            tot = label.shape[0]

            sys.stdout.write("epoch: %d, loss: %.6f, acc: %.3f\r" % (i, loss, crr/tot))
            sys.stdout.flush()        
        
        with torch.no_grad():
            print("")
            print("deving....")
            model.training = False
            model.eval()

            if args.dataset == "semeval" or args.dataset == "tacred":
                eval_func = eval_F1
            elif args.dataset == "wiki80" or args.dataset == "chemprot":
                eval_func = eval_ACC
            elif args.dataset == "nyt" or args.dataset == "gids":
                eval_func = eval_AP
            
            score = eval_func(args, model, dev_dataloader)
            print('score',score)
            if score > best_dev_score:
                best_dev_score = score
                best_test_score = score#eval_func(args, model, test_dataloader)
                best_epoch = i
                print("Best Dev score: %.3f,\tTest score: %.3f" % (best_dev_score, best_test_score))
                if args.freeze_kg:
                    temp = '_bag_kg_frozen_crst_%s.mdl'%args.crst_mod if args.use_bag else '_kg_frozen_crst_%s.mdl'%args.crst_mod
                else:
                    temp = '_bag_kg_crst_%s.mdl'%args.crst_mod if args.use_bag else '_kg_crst_%s.mdl'%args.crst_mod
                if args.direct_feature:
                    temp = '_direct_'+temp
                if args.use_seg:
                    temp = '_seg'+temp 
                if args.ckpt_to_load != 'None':
                    torch.save(model.state_dict(), '../../save/nyt/'+args.prefix+str(args.w_symloss)+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp)
                else:
                    torch.save(model.state_dict(), '../../save/nyt/'+args.prefix+str(args.w_symloss)+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp)
            else:
                print("Dev score: %.3f" % score)
            print("-----------------------------------------------------------") 
    print("@RESULT: " + args.dataset +" Test score is %.3f" % best_test_score)
    f = open("../log/re_log", 'a+')
    temp = '_bag_kg_crst_%s.mdl'%args.crst_mod if args.use_bag else '_kg_crst_%s.mdl'%args.crst_mod
    if args.direct_feature:
        temp = '_direct_'+temp
    if args.use_seg:
        temp = '_seg'+temp 
    if args.ckpt_to_load == "None":
        f.write(args.prefix+str(args.w_symloss)+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'\t'+str(time.ctime())  +"\n")
    else:
        f.write(args.prefix+str(args.w_symloss)+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'\t' +str(time.ctime()) +"\n")
    f.write("@RESULT: Best Dev score is %.3f, Test score is %.3f\n, at epoch %d" % (best_dev_score, best_test_score, best_epoch))
    f.write("--------------------------------------------------------------\n")
    f.close()


def eval_AP(args, model, dataloader, return_output=False):
    tot_label = []
    tot_logits = []
    # tx info.
    tot_sent_tup = []
    tot_senti = []
    tot_att = []
    tokenizer = BertTokenizer.from_pretrained(args.surf_trf)
    for batch in dataloader:
        inputs = {
            "label":batch[0],
            "input_ids":batch[3],
            "mask":batch[4],
            "h_pos":batch[5],
            "t_pos":batch[6],
            'head_id':batch[7],
            'tail_id':batch[8],
            'sym_ids':batch[9],
            'sym_mask':batch[10],
            'sym_label':batch[11]
        }
        if args.use_bag:
            inputs["scope"] = batch[2]
        logits, output, cratt_s2t = model(**inputs)
        
        tot_label.extend(batch[0].cpu().detach().tolist())
        tot_logits.extend(logits.cpu().detach().tolist())
        # tx info.
        tot_sent_tup.extend(batch[12])
        tot_senti.extend(batch[3].cpu().detach().tolist())
        if args.crst_mod == 'resi':
            tot_att.extend(cratt_s2t.cpu()[:,1,:].detach().tolist())#cratt_s2t (B, 3, 60)
        
    tot_logits = np.array(tot_logits)
    tot_labels = np.zeros(tot_logits.shape)
    try:
        tot_labels[range(len(tot_labels)), tot_label] = 1
    except:
        print(tot_labels.shape,len(tot_label))
        print(logits.shape)
        exit(0)

    # tx info.
    tot_sent_tup_pred = []
    rows = []
        
    if not args.use_bag:
        test_scope = json.load(open('../../data/'+args.dataset+'/scope_test.json'))
        new_logits = np.zeros((len(test_scope), len(tot_labels[0])))
        new_labels = np.zeros((len(test_scope), len(tot_labels[0])))
        tot_logits = softmax(tot_logits,axis=1)
        for i in range(len(test_scope)):
            try:
                new_logits[i] = np.mean(tot_logits[test_scope[i][0]:test_scope[i][1]], axis=0)
                new_labels[i] = tot_labels[test_scope[i][0]]
            except IndexError:
                print(i)
                pass

            # tx info.
            pred_ri = np.argmax(np.mean(tot_logits[test_scope[i][0]:test_scope[i][1]], axis=0))
            pred_rs = args.id2rel[pred_ri]
            real_ri = np.argmax(tot_labels[test_scope[i][0]])
            for sent_tup in tot_sent_tup[test_scope[i][0]:test_scope[i][1]]:
                sent_tup_pred = "%s__%s" % (sent_tup, pred_rs)
                tot_sent_tup_pred.append(sent_tup_pred)
            if pred_ri == real_ri and real_ri != 0 and args.crst_mod == 'resi':
                for si in range(test_scope[i][0],test_scope[i][1]):
                    lwi = tot_senti[si]
                    att = tot_att[si]
                    sent_tup = tot_sent_tup[si]
                    lw = tokenizer.convert_ids_to_tokens(lwi)
                    row = dict(
                        tokens=lw,
                        attsc=att,
                        text=sent_tup)
                    rows.append(row)
                
        tot_logits = new_logits
        tot_labels = new_labels

    exclude_na_flatten_label = np.reshape(tot_labels[:,1:],-1)
    exclude_na_flatten_output = np.reshape(tot_logits[:,1:],-1)
    order = np.argsort(-exclude_na_flatten_output)
    p_100 = np.mean(exclude_na_flatten_label[order[:100]])
    p_200 = np.mean(exclude_na_flatten_label[order[:200]])
    p_300 = np.mean(exclude_na_flatten_label[order[:300]])
    p_500 = np.mean(exclude_na_flatten_label[order[:500]])
    p_1000 = np.mean(exclude_na_flatten_label[order[:1000]])
    p_2000 = np.mean(exclude_na_flatten_label[order[:2000]])
    print('p@100:' + str(p_100))
    print('p@200:' + str(p_200))
    print('p@300:' + str(p_300))
    print('p@500:' + str(p_500))
    print('p@1000:' + str(p_1000))
    print('p@2000:' + str(p_2000))
        
    tot_labels = tot_labels.astype(np.int)
    ap =  average_precision_score(tot_labels, tot_logits, average='micro')
    ap2 =  average_precision_score(tot_labels[:,1:], tot_logits[:,1:], average='micro')
    
    if return_output:
        return ap2, tot_logits, tot_labels, tot_sent_tup_pred, rows
    else:          
        return ap2
    
def eval_F1(args, model, dataloader):
    tot_label = []
    tot_output = []
    for batch in dataloader:
        inputs = {
            "label":batch[0],
            "input_ids":batch[3],
            "mask":batch[4],
            "h_pos":batch[5],
            "t_pos":batch[6],
            'head_id':batch[7],
            'tail_id':batch[8]
        }
        if args.use_bag:
            inputs["scope"] = batch[2]
        _, output = model(**inputs)
        tot_label.extend(batch[0].cpu().tolist())
        tot_output.extend(output.cpu().detach().tolist())
    f1, _ = f1_score(tot_output, tot_label, args.rel_num) 
    return f1
    

def eval_ACC(args, model, dataloader):
    tot = 0.0
    crr = 0.0
    for batch in dataloader:
        inputs = {
            "label":batch[0],
            "input_ids":batch[3],
            "mask":batch[4],
            "h_pos":batch[5],
            "t_pos":batch[6],
            'head_id':batch[7],
            'tail_id':batch[8],
            'sym_ids':batch[9],
            'sym_mask':batch[10],
            'sym_label':batch[11]
        }
        if args.use_bag:
            inputs["scope"] = batch[2]
        _, output = model(**inputs)
        output = output.cpu().detach().numpy()
        label = batch[0].cpu().numpy()
        crr += (output==label).sum()
        tot += label.shape[0]

        sys.stdout.write("acc: %.3f\r" % (crr/tot)) 
        sys.stdout.flush()

    return crr / tot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="latentRE")
    parser.add_argument("--batch_size_per_gpu", dest="batch_size_per_gpu", type=int, 
                        default=0, help="batch size pre gpu")
    parser.add_argument("--dataset", dest="dataset", type=str,
                        default='tacred',help='dataset to use')
    parser.add_argument("--lr", dest="lr", type=float,
                        default=3e-5, help='learning rate')
    parser.add_argument("--hidden_size", dest="hidden_size", type=int,
                        default=768,help='hidden size')
    parser.add_argument("--encoder", dest="encoder", type=str,
                        default='bert',help='encoder')
    parser.add_argument("--optim", dest="optim", type=str,
                        default='adamw',help='optimizer')
    
    
    parser.add_argument("--weight_decay", dest="weight_decay", type=float,
                        default=1e-5, help="weight decay")
    parser.add_argument("--adam_epsilon", dest="adam_epsilon", type=float,
                        default=1e-8, help="adam epsilon")
    parser.add_argument("--warmup_steps", dest="warmup_steps", type=int,
                        default=500, help="warmup steps")
    parser.add_argument("--max_grad_norm", dest="max_grad_norm", type=float,
                        default=1, help="max grad norm")
    parser.add_argument("--max_length", dest="max_length", type=int,
                        default=64, help="max sentence length")
    parser.add_argument("--ckpt_to_load", dest="ckpt_to_load", type=str,
                        default="None", help="ckpt to load")
    parser.add_argument("--entity_marker", action='store_true', 
                        help="if entity marker or cls")
    parser.add_argument("--train_prop", dest="train_prop", type=float,
                        default=1, help="train set prop")
    
    parser.add_argument("--mode", dest="mode",type=str, 
                        default="CM", help="{CM,OC,CT,OM,OT}")
    parser.add_argument("--max_epoch", dest="max_epoch", type=int, 
                        default=3, help="max epoch")

    parser.add_argument("--seed", dest="seed", type=int,
                        default=42, help="seed for network")
    parser.add_argument("--use_bag", dest="use_bag", action='store_true',
                        default=False, help="whether train in a bag of sentence setting")
    parser.add_argument("--bag_size", dest="bag_size", type=int,
                        default=0, help="size of each bag")
    parser.add_argument("--entity_embedding_load_path", dest="entity_embedding_load_path", type=str,
                        default=None, help="where pretrained entity embedding is stored")
    parser.add_argument("--direct_feature", dest="direct_feature", action='store_true',
                        default=False, help="whether directly use kg embedding as feature")

    parser.add_argument("--kg_method", dest="kg_method", type=str,
                        default=None, help="how entity embedding is trained")
    parser.add_argument("--prefix", dest="prefix", type=str,
                        default='TX', help="prefix of model name, TX, KG or TXKG")
    parser.add_argument("--freeze_entity", dest="freeze_entity", action='store_true',
                        default=False, help="whether freeze entity embedding during training")
    parser.add_argument("--test", dest="test", action='store_true',
                        default=False, help="whether test")
    parser.add_argument("--load", dest="load", action='store_true',
                        default=False, help="whether load")
    parser.add_argument("--use_seg", dest="use_seg", action='store_true',
                        default=False, help="whether use seg")

    # crst info.
    parser.add_argument("--crst_path", dest="crst_path",type=str,
                        default="", help="pretrained crst model such as KG encoder")
    parser.add_argument("--sym2id", dest="sym2id",type=str,
                        default="", help="the path to sym2id.json")
    parser.add_argument("--t5", dest="t5",type=str,
                        default="", help="t5 pretrained model 'razent/SciFive-base-Pubmed' or 't5-base'")
    parser.add_argument("--crst_mod", dest="crst_mod",type=str,
                        default="resi", help="{none, resi, gated}")
    parser.add_argument("--surf_trf", type=str,
                        default="bert-base-uncased", help="'bert-base-uncased' or 'dmis-lab/biobert-v1.1'")
    parser.add_argument("--mid_dim", type=int,
                        default=10, help="the Numb. of middle dim")
    parser.add_argument("--test_only", action='store_true',
                        default=False, help="whether test only")
    parser.add_argument("--w_symloss", type=str,
                        default='', help="weight for sym loss")
    parser.add_argument("--id2rel", type=dict,
                        default={}, help="id2rel dictionary")
    parser.add_argument("--dev_bio", action='store_true',
                        default=False, help="whether test only")
    parser.add_argument("--output_example", action='store_true',
                        default=False, help="whether output examples or not")
    parser.add_argument("--output_att", action='store_true',
                        default=False, help="whether output att info. or not")
    parser.add_argument("--freeze_kg", action='store_true',
                        default=False, help="whether freeze kg encoder or not")
    
    args = parser.parse_args()

    sym2id = json.load(open(os.path.join("../../data/"+args.dataset, "sym2id.json")))
    crst_conf.sym_size = len(sym2id)
    crst_conf.stitch_model = args.crst_mod
    crst_conf.max_length = args.max_length
    crst_conf.surface_transformer = args.surf_trf
    crst_conf.mid_dim = args.mid_dim
    
    # print args
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')

    # Warning
    print("*"*30)
    if args.dataset == 'semeval':
        print("Warning! The results reported on `semeval` may be different from our paper. Because we use the official evaluation script. See `finetune/readme` for more details.")
    print("*"*30)

    # set seed
    set_seed(args)
        
    if not os.path.exists("../log"):
        os.mkdir("../log")
    # params for dataloader
    rel2id = json.load(open(os.path.join("../../data/"+args.dataset, "rel2id.json")))
    args.rel_num = len(rel2id)
    args.id2rel = {ind:rel for rel, ind in rel2id.items()}
    ent2id = json.load(open(os.path.join("../../data/"+args.dataset, "entity2id.json")))
    args.entity_num = len(ent2id)
    args.entity_embedding_size = args.hidden_size*2
    if args.use_bag:
        model = REBagModel_KG_CRST(args, crst_conf=crst_conf)
    else:
        model = REModel_KG_CRST(args, crst_conf=crst_conf)
    if args.test or args.load:
        if args.freeze_kg:
            temp = '_bag_kg_frozen_crst_%s.mdl'%args.crst_mod if args.use_bag else '_kg_frozen_crst_%s.mdl'%args.crst_mod
        else:
            temp = '_bag_kg_crst_%s.mdl'%args.crst_mod if args.use_bag else '_kg_crst_%s.mdl'%args.crst_mod
        if args.direct_feature:
            temp = '_direct_'+temp
        if args.use_seg:
            temp = '_seg'+temp 
        if args.ckpt_to_load != 'None':
            print('loading',args.prefix+str(args.w_symloss)+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp)
            loaded = torch.load("../../save/nyt/"+args.prefix+str(args.w_symloss)+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp)
        else:
            print('loading',args.prefix+str(args.w_symloss)+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp)
            loaded = torch.load("../../save/nyt/"+args.prefix+str(args.w_symloss)+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp)
        new_loaded = {i.strip('module.'):loaded[i] for i in loaded}
        new_loaded_ = {}
        for i in new_loaded:
            if i == 'ntity_embedding.weight':
                i_ = "entity_embedding.weight"
                new_loaded_[i_] = new_loaded[i]
            else:
                new_loaded_[i] = new_loaded[i]
        model.load_state_dict(new_loaded_)
        
    model.cuda()

    test_set = REBagDataset_KG_CRST("../../data/"+args.dataset, "test.txt", args)
    test_dataloader = data.DataLoader(test_set, batch_size=args.batch_size_per_gpu, shuffle=False, collate_fn=test_set.collate_fn)
    
    
    if args.test:
        print("")
        print("testing....")
        model.training = False
        model.eval()

        if args.dataset == "semeval" or args.dataset == "tacred":
            eval_func = eval_F1
        elif args.dataset == "wiki80" or args.dataset == "chemprot":
            eval_func = eval_ACC
        elif args.dataset == "nyt" or args.dataset == "gids":
            eval_func = eval_AP
        
        score, tot_logits, tot_labels, tot_sample_tup_pred, att_info_rows = eval_func(args, model, test_dataloader, return_output=True)
        print('test_score', score)
        if args.ckpt_to_load != 'None':
            if args.freeze_kg:
                temp = '_bag_kg_frozen_crst_%s_'%args.crst_mod if args.use_bag else '_kg_frozen_crst_%s_'%args.crst_mod
            else:
                temp = '_bag_kg_crst_%s_'%args.crst_mod if args.use_bag else '_kg_crst_%s_'%args.crst_mod
            if args.direct_feature:
                temp = '_direct_'+temp
            if args.use_seg:
                temp = '_seg'+temp 
            np.save('../../result/nyt/'+args.prefix+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp+'logits.npy', tot_logits)
            np.save('../../result/nyt/'+args.prefix+args.dataset+'_'+args.ckpt_to_load+'_'+args.kg_method+temp+'labels.npy', tot_labels)
        else:
            if args.freeze_kg:
                temp = '_bag_kg_frozen_crst_%s_'%args.crst_mod if args.use_bag else '_kg_frozen_crst_%s_'%args.crst_mod
            else:
                temp = '_bag_kg_crst_%s_'%args.crst_mod if args.use_bag else '_kg_crst_%s_'%args.crst_mod
            if args.direct_feature:
                temp = '_direct_'+temp
            if args.use_seg:
                temp = '_seg'+temp 
            np.save('../../result/nyt/'+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'logits.npy', tot_logits)
            np.save('../../result/nyt/'+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'labels.npy', tot_labels)

            if args.output_example:
                with open('../../result/nyt/'+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'sample.txt', 'w') as fout:
                    for sample in tot_sample_tup_pred:
                        fout.write('%s\n' % sample)
            if args.output_att and args.crst_mod == 'resi':
                with open('../../result/nyt/'+args.prefix+args.dataset+'_'+'bert-base-uncased'+'_'+args.kg_method+temp+'att_info.json', 'w') as fout:
                    json.dump(att_info_rows, fout)
                
        exit(0)
        
    if args.train_prop == 1:
        print("Use all train data!")
        train_set = REBagDataset_KG_CRST("../../data/"+args.dataset, "train.txt", args)
    elif args.train_prop == 0.1:
        print("Use 10% train data!")
        train_set = REBagDataset_KG_CRST("../../data/"+args.dataset, "train_0.1.txt", args)
    elif args.train_prop == 0.01:
        print("Use 1% train data!")
        train_set = REBagDataset_KG_CRST("../../data/"+args.dataset, "train_0.01.txt", args)
    if args.dataset == 'nyt':
        dev_set = test_set
    else:
        dev_set = REBagDataset_KG("../../data/"+args.dataset, "dev.txt", args)
    dev_dataloader = data.DataLoader(dev_set, batch_size=args.batch_size_per_gpu, shuffle=False, collate_fn=dev_set.collate_fn)

    if not args.test_only:
        train_dataloader = data.DataLoader(train_set, batch_size=args.batch_size_per_gpu, shuffle=True, collate_fn=train_set.collate_fn)
        train(args, model, train_dataloader, dev_dataloader, test_dataloader)
    
    


