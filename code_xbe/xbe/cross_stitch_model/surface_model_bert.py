import torch
import pytorch_lightning as pl

class SurfaceTransformer(pl.LightningModule):
    
    def __init__(self, conf, cross_stitch_input_dim2=None):
        super().__init__()
        is_bert_variant = conf.surface_transformer.startswith('dmis-lab/biobert-v1.1')
        if is_bert_variant or conf.surface_transformer.startswith('bert'):
            
            if conf.model in ['none', 'resi', 'gated', 'crossmodel']:
                from .modeling_bert_interleaved import BertModel as model_cls
            else:
                from transformers import BertModel as model_cls
            
            self.trf = model_cls.from_pretrained(conf.surface_transformer)

            self.encoder = self.trf
            
        else:
            raise NotImplementedError('TODO')
        
        if conf.surface_freeze_encoder:
            for p in self.trf.BertEncoder.parameters():
                p.requires_grad = False
        self.crit = torch.nn.CrossEntropyLoss()
        
        
    def forward(self, batch, return_instances=False):
        out = self.trf(
            input_ids=batch['tx_input_ids'],
            attention_mask=batch['tx_attention_mask']
            )
        return self.result(batch, out, return_instances=return_instances)    
