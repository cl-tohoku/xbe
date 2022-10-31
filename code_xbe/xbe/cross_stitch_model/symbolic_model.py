import torch
from torch import nn

import pytorch_lightning as pl

class TripleTransformer(pl.LightningModule):

    def __init__(self, conf):
        super().__init__()
        from transformers import RobertaConfig
        from .modeling_roberta_interleaved import RobertaModel
        from transformers.models.roberta.modeling_roberta import RobertaLMHead
        trf_conf = RobertaConfig.from_pretrained(conf.symbolic_transformer)
        
        trf_conf.num_hidden_layers = conf.triple_trf_n_hidden
        trf_conf.num_attention_heads = conf.triple_trf_n_attn_heads
        trf_conf.max_position_embeddings = 32
        trf_conf.hidden_dropout_prob = conf.triple_trf_dropout
        
        self.vocab_size = conf.sym_size
        trf_conf.vocab_size = self.vocab_size
        self.trf = RobertaModel(trf_conf)
        assert conf.head == 'lmhead', 'todo: different heads'

        self.crit = nn.CrossEntropyLoss()
        self.encoder = self.trf
        
        self.head = RobertaLMHead(trf_conf)
        
    def forward(self, batch, return_instances=False):
        trf_out = self.trf(batch['sym_input_ids'])
        return self.predict(batch, trf_out, return_instances=return_instances)
    
    def predict(self, sym_labels, trf_out, return_instances=True):
        pred = self.head(trf_out.last_hidden_state)
        target = sym_labels
        loss = self.crit(pred.view(-1, self.vocab_size), target.view(-1))
        result = {'loss': loss}
        if return_instances:
            result['relation_pred'] = pred.detach()
            result['relation_target'] = target
        return result
