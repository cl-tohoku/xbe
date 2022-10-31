import torch
import pdb 
import torch.nn as nn
from transformers import BertModel
import numpy as np

from cross_stitch_model import CrossModel

class REModel_KG_CRST(nn.Module):
    """relation extraction model
    """
    def __init__(self, args, weight=None, crst_conf=None):
        super(REModel_KG_CRST, self).__init__()
        self.args = args
        self.kg_method = args.kg_method 
        self.training = True
        self.direct_feature = args.direct_feature 
        if weight is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            print("CrossEntropy Loss has weight!")
            self.loss = nn.CrossEntropyLoss(weight=weight)

        scale = 2 if args.entity_marker else 1
        if args.entity_embedding_load_path != None:
            pretrained_entity_embedding = torch.FloatTensor(np.load(args.entity_embedding_load_path))
            #print('forgot to add entity embedding!')
            self.entity_embedding = nn.Embedding.from_pretrained(pretrained_entity_embedding, freeze=args.freeze_entity)
        else:
            self.entity_embedding = nn.Embedding(args.entity_num,args.entity_embedding_size)
        if self.direct_feature and self.kg_method != 'None':
            if self.kg_method in ['ComplEx_cat','TransE']:
                self.rel_fc = nn.Linear(args.hidden_size*scale+2*self.entity_embedding.weight.shape[1], args.rel_num)
            elif self.kg_method in ['RotatE']:
                pass
            elif self.kg_method in ['TransE_re']:
                #self.rel_fc = nn.Linear(args.hidden_size*scale+self.entity_embedding.weight.shape[1], args.rel_num)
                if args.prefix == 'TX':
                    self.rel_fc = nn.Linear(args.hidden_size*scale+args.hidden_size*2+self.entity_embedding.weight.shape[1], args.rel_num)
                elif args.prefix == 'KG':
                    self.rel_fc = nn.Linear(args.hidden_size*2+self.entity_embedding.weight.shape[1], args.rel_num)
                elif args.prefix == 'TXKG':
                    self.rel_fc = nn.Linear(args.hidden_size*scale+args.hidden_size*4+self.entity_embedding.weight.shape[1], args.rel_num)
                    #self.rel_fc = nn.Linear(args.hidden_size*scale+args.hidden_size*4, args.rel_num)
                elif args.prefix == 'OKG':
                    self.rel_fc = nn.Linear(args.hidden_size*2+self.entity_embedding.weight.shape[1], args.rel_num)
                elif args.prefix == 'OHT':
                    self.rel_fc = nn.Linear(args.hidden_size*scale+self.entity_embedding.weight.shape[1], args.rel_num)
                elif args.prefix == 'OHTKG':
                    self.rel_fc = nn.Linear(args.hidden_size*scale+args.hidden_size*2+self.entity_embedding.weight.shape[1], args.rel_num)
                elif args.prefix == 'CHTKG':
                    self.rel_fc = nn.Linear(args.hidden_size*scale+args.hidden_size+args.hidden_size*2+self.entity_embedding.weight.shape[1], args.rel_num)
                elif args.prefix == 'CHT':
                    self.rel_fc = nn.Linear(args.hidden_size*scale+args.hidden_size+self.entity_embedding.weight.shape[1], args.rel_num)
            else:
                print('not defined method', self.kg_method)
                exit(0)
        else:
            self.rel_fc = nn.Linear(args.hidden_size*scale, args.rel_num)
        self.generate_rep = {'TransE':self.TransE,'TransE_re':self.TransE_re,'ComplEx_cat':self.ComplEx_cat}

        
        if args.ckpt_to_load == "cnn":
            self.bert = None
        elif args.ckpt_to_load != "None":
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            print("********* load from ckpt/"+args.ckpt_to_load+" ***********")
            ckpt = torch.load("../../../pretrain/ckpt/"+args.ckpt_to_load)
            self.bert.load_state_dict(ckpt["bert-base"])
        else:
            #self.bert = BertModel.from_pretrained('bert-base-uncased')
            print("*******No ckpt to load, Let's use bert base!*******")
        # CRST encoder
        self.crst_conf = crst_conf
        if args.crst_path != '':
            ckpt = torch.load(args.crst_path, map_location=torch.device('cuda:%s'%torch.cuda.current_device()))
            self.CRST = CrossModel(self.crst_conf)
            self.CRST.load_state_dict(ckpt['state_dict'], strict=False)
            if args.freeze_kg:
                for param in self.CRST.symbol_trf.parameters():
                    param.requires_grad=False
        else:
            self.CRST = CrossModel(self.crst_conf)
        
    def forward(self, label, input_ids, mask, h_pos, t_pos, head_id, tail_id, sym_ids, sym_mask, sym_label):
        # bert encode
        #outputs = self.bert(input_ids, mask)
        
        # CRST encode
        outputs = self.CRST(input_ids, mask, sym_ids, sym_mask, sym_label)

        # entity marker
        if self.args.entity_marker:
            indice = torch.arange(input_ids.size()[0])
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1) #(batch_size, hidden_size*2)
        else:
            #[CLS]
            state = outputs[0][:, 0, :] #(batch_size, hidden_size)

        # tx and kg rep
        max_h = torch.max(outputs[0], 1).values
        mean_h = torch.mean(outputs[0], 1)
        
        if self.args.prefix == 'TX':
            state = torch.cat([state, max_h, mean_h], axis=1)
        elif self.args.prefix == 'KG':
            state = torch.cat([outputs[1], outputs[2]], axis=1)
        elif self.args.prefix == 'TXKG':
            state = torch.cat([state, outputs[1], outputs[2], max_h, mean_h], axis=1)
        elif self.args.prefix == 'OKG':
            state = torch.cat([outputs[1], outputs[2]], axis=1)
        elif self.args.prefix == 'OHT':
            state = state
        elif self.args.prefix == 'OHTKG':
            state = torch.cat([state, outputs[1], outputs[2]], axis=1)
        elif self.args.prefix == 'CHTKG':
            cls = outputs[0][:, 0, :]
            state = torch.cat([state, cls, outputs[1], outputs[2]], axis=1)
        elif self.args.prefix == 'CHT':
            cls = outputs[0][:, 0, :]
            state = torch.cat([state, cls], axis=1)
            
        # linear map
        if self.direct_feature:   
            logits = self.kfeature(state, head_id, tail_id, self.kg_method)
            #logits = self.rel_fc(state)
        else:
            logits = self.rel_fc(state)
        _, output = torch.max(logits, 1)

        if self.training:
            loss = self.loss(logits, label)
            if self.args.prefix == 'TX':
                tot_loss = loss
            elif self.args.prefix in ['KG', 'TXKG']:
                if self.args.w_symloss != '':
                    tot_loss = loss + float(self.args.w_symloss)*outputs[4]
                else:
                    tot_loss = loss + outputs[4]
            elif self.args.prefix == 'OKG':
                tot_loss = loss
            elif self.args.prefix == 'OHT':
                tot_loss = loss
            elif self.args.prefix == 'OHTKG':
                tot_loss = loss + outputs[-1]
            elif self.args.prefix == 'CHTKG':
                tot_loss = loss + outputs[-1]
            elif self.args.prefix == 'CHT':
                tot_loss = loss
                
            return tot_loss, output
        else:
            cratt_t2s = outputs[5]
            return logits, output, cratt_t2s    
    def kfeature(self, rep, head_id, tail_id, kg_method):
        #print('head_id',head_id)
        #print('tail_id',tail_id)
        head = self.entity_embedding(head_id)
        tail = self.entity_embedding(tail_id)
        #print('head_sum',torch.sum(head))
        #print('tail_sum',torch.sum(tail))
        kg_rep = self.generate_rep[kg_method](rep, head, tail)
        logits = self.rel_fc(kg_rep)
        return logits
    def TransE(self, rep, head, tail):
        kg_rep = torch.cat([rep, head, tail], axis=1)
        return kg_rep
    def TransE_re(self, rep, head, tail):
        kg_rep = torch.cat([rep, tail-head], axis=1)
        return kg_rep
    def ComplEx_cat(self, rep, head, tail):
        kg_rep = torch.cat([rep, head, tail], axis=1)
        return kg_rep

    
class REBagModel_KG_CRST(nn.Module):
    """relation extraction model
    """
    def __init__(self, args, weight=None, crst_conf=None):
        super(REBagModel_KG_CRST, self).__init__()
        self.args = args 
        self.kg_method = args.kg_method
        self.training = True
        self.direct_feature = args.direct_feature
        self.use_seg = args.use_seg
        if weight is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            print("CrossEntropy Loss has weight!")
            self.loss = nn.CrossEntropyLoss(weight=weight)

        scale = 2 if args.entity_marker else 1
        self.rel_fc = nn.Linear(args.hidden_size*scale, args.rel_num)
        if args.entity_embedding_load_path != None:
            pretrained_entity_embedding = torch.FloatTensor(np.load(args.entity_embedding_load_path))
            self.entity_embedding = nn.Embedding.from_pretrained(pretrained_entity_embedding, freeze=args.freeze_entity)
        else:
            self.entity_embedding = nn.Embedding(args.entity_num,args.entity_embedding_size)
        if not self.direct_feature:
            if self.kg_method in ['ComplEx']:
                self.transfer_re = nn.Linear(args.hidden_size*scale, int(self.entity_embedding.weight.shape[1]/2))
                self.transfer_im = nn.Linear(args.hidden_size*scale, int(self.entity_embedding.weight.shape[1]/2))
            elif self.kg_method in ['RotatE']:
                self.transfer_phase = nn.Linear(args.hidden_size*scale, int(self.entity_embedding.weight.shape[1]/2))
            else:
                self.transfer = nn.Linear(args.hidden_size*scale, self.entity_embedding.weight.shape[1])
        if self.use_seg:
            self.fc1 = nn.Linear(args.hidden_size, args.hidden_size)
            if self.direct_feature and self.kg_method in ['TransE_re','transre_entity_name']:
                self.fc2 = nn.Linear(args.hidden_size, args.hidden_size*scale+self.entity_embedding.weight.shape[1])
            else:
                self.fc2 = nn.Linear(args.hidden_size, args.hidden_size*scale)
            self.fc1_att = nn.Linear(args.hidden_size, args.hidden_size)
            self.fc2_att = nn.Linear(args.hidden_size, args.hidden_size)

        #self.bert = BertModel.from_pretrained('bert-base-uncased')

        # CRST encoder
        self.crst_conf = crst_conf
        if args.crst_path != '':
            ckpt = torch.load(args.crst_path, map_location=torch.device('cuda:%s'%torch.cuda.current_device()))
            self.CRST = CrossModel(self.crst_conf)
            self.CRST.load_state_dict(ckpt['state_dict'], strict=False)
        else:
            self.CRST = CrossModel(self.crst_conf)
        
        self.drop = nn.Dropout()
        self.softmax = nn.Softmax(-1)
        if args.ckpt_to_load != "None":
            print("********* load from ckpt/"+args.ckpt_to_load+" ***********")
            ckpt = torch.load("../../../pretrain/ckpt/"+args.ckpt_to_load)
            self.bert.load_state_dict(ckpt["bert-base"])
        else:
            print("*******No ckpt to load, Let's use bert base!*******")
        self.generate_attn = {'TransE':self.TransE, 'DistMult':self.DistMult, 'ComplEx':self.ComplEx,'RotatE':self.RotatE}
        self.generate_rep = {'TransE_re':self.TransE_re,'transre_entity_name':self.TransE_re}
        self.pi = 3.14159265358979323846
    def forward(self, label, scope, input_ids, mask, h_pos, t_pos, head_id, tail_id, sym_ids, sym_mask, sym_label):
        # bert encode
        #outputs = self.bert(input_ids, mask)
        #print(label.shape, len(scope), input_ids.shape, mask.shape, h_pos.shape, t_pos.shape, head_id.shape, tail_id.shape)
        # entity marker
        #print(scope[-1][-1], input_ids.shape)

        # CRST encode
        outputs = self.CRST(input_ids, mask, sym_ids, sym_mask, sym_label)
        
        if self.args.entity_marker:
            indice = torch.arange(input_ids.size()[0]).cuda()
            h_state = outputs[0][indice, h_pos]
            t_state = outputs[0][indice, t_pos]
            state = torch.cat((h_state, t_state), 1) #(batch_size, hidden_size*2)
        else:
            #[CLS]
            state = outputs[0][:, 0, :] #(batch_size, hidden_size)
        if self.use_seg:
            logits = self.seg(outputs[0], state, scope, head_id, tail_id, self.kg_method)
        elif not self.direct_feature:   
            logits = self.katt(state, scope, head_id, tail_id, self.kg_method)
        else:
            logits = self.kfeature(state, scope, head_id, tail_id, self.kg_method)
        #print('logits sum',torch.sum(logits))
        #exit(0)
        _, output = torch.max(logits, 1)
        if self.training:
            #print('shape',logits.shape, label.shape)

            loss = self.loss(logits, label)
            return loss, output
        else:
            return logits, output
    def krep(self, rep, head_id, tail_id, kg_method):
        head = self.entity_embedding(head_id)
        tail = self.entity_embedding(tail_id)
        kg_rep = self.generate_rep[kg_method](rep, head, tail)
        return kg_rep
    def TransE_re(self, rep, head, tail):
        kg_rep = torch.cat([rep, tail-head], axis=1)
        return kg_rep
    def seg(self, seq, rep, scope, head_id, tail_id, kg_method):
        A = self.fc2_att(torch.tanh(self.fc1_att(seq)))
        P = torch.softmax(A, 1)
        U = torch.sum(P * seq, 1)
        if self.direct_feature and kg_method in ['TransE_re','transre_entity_name']:
            rep = self.krep(rep, head_id, tail_id, kg_method)
        G = torch.sigmoid(self.fc2(torch.tanh(self.fc1(U))))
        rep = G * rep
        
        bag_rep = []
        for i in range(len(scope)):
            bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
            bag_rep.append(bag_mat.mean(0))
        bag_rep = torch.stack(bag_rep, 0) # (B, H)
        bag_rep = self.drop(bag_rep)
        bag_logits = self.rel_fc(bag_rep)
        return bag_logits

    def att(self, rep, scope, label):
        query = torch.zeros((rep.size(0))).long()
        if torch.cuda.is_available():
            query = query.cuda()
        for i in range(len(scope)):
            query[scope[i][0]:scope[i][1]] = label[i]
        att_mat = self.rel_fc.weight.data[query]
        att_score = (rep*att_mat).sum(-1)
        bag_rep = []
        for i in range(len(scope)):
            bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
            softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) # (n)
            bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) # (n, 1) * (n, H) -> (n, H) -> (H)
        bag_rep = torch.stack(bag_rep, 0) # (B, H)
        bag_rep = self.drop(bag_rep)
        bag_logits = self.rel_fc(bag_rep)
        return bag_logits
    def katt(self, rep, scope, head_id, tail_id, kg_method):
        #print('head_id',head_id)
        #print('tail_id',tail_id)
        head = self.entity_embedding(head_id)
        tail = self.entity_embedding(tail_id)
        #print('head_sum',torch.sum(head))
        #print('tail_sum',torch.sum(tail))
        att_score = self.generate_attn[kg_method](rep, head, tail)
        #print('att_score sum',torch.sum(att_score))
        bag_rep = []
        for i in range(len(scope)):
            bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
            softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]]) # (n)
            bag_rep.append((softmax_att_score.unsqueeze(-1) * bag_mat).sum(0)) # (n, 1) * (n, H) -> (n, H) -> (H)
        bag_rep = torch.stack(bag_rep, 0) # (B, H)
        bag_rep = self.drop(bag_rep)
        bag_logits = self.rel_fc(bag_rep)
        return bag_logits
    def TransE(self, rep, head, tail):
        att_mat = tail-head
        att_score = (self.transfer(rep)*att_mat).sum(1)
        return att_score
    def DistMult(self, rep, head, tail):
        att_score = (self.transfer(rep)*head*tail).sum(1)
        return att_score
    def ComplEx(self, rep, head, tail):
        re_relation = self.transfer_re(rep)
        im_relation = self.transfer_im(rep)
        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        score = re_head * re_score + im_head * im_score
        score = score.sum(dim = 1)
        return score
    def RotatE(self, rep, head, tail):
        phase_relation = torch.tanh(self.transfer_phase(rep))
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        re_head, im_head = torch.chunk(head, 2, dim=1)
        re_tail, im_tail = torch.chunk(tail, 2, dim=1)
        re_score = re_relation * re_tail + im_relation * im_tail
        im_score = re_relation * im_tail - im_relation * re_tail
        re_score = re_score - re_head
        im_score = im_score - im_head
        score = torch.stack([re_score, im_score],dim=0)
        score = score.norm(dim=0)
        score = -score.sum(dim = 1)
        return score
    def att_test(self, rep, scope):
        bag_logits = []
        att_score = torch.matmul(rep, self.rel_fc.weight.data.transpose(0, 1)) # (nsum, H) * (H, N) -> (nsum, N)
        for i in range(len(scope)):
            bag_mat = rep[scope[i][0]:scope[i][1]] # (n, H)
            softmax_att_score = self.softmax(att_score[scope[i][0]:scope[i][1]].transpose(0, 1)) # (N, (softmax)n) 
            rep_for_each_rel = torch.matmul(softmax_att_score, bag_mat) # (N, n) * (n, H) -> (N, H)
            logit_for_each_rel = self.softmax(self.rel_fc(rep_for_each_rel)) # ((each rel)N, (logit)N)
            logit_for_each_rel = logit_for_each_rel.diag() # (N)
            bag_logits.append(logit_for_each_rel)
        bag_logits = torch.stack(bag_logits,0)
        return bag_logits
