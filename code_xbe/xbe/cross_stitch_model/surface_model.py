import torch

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.bart.modeling_bart import shift_tokens_right

import pytorch_lightning as pl

class SurfaceTransformer(pl.LightningModule):
    
    def __init__(self, conf, cross_stitch_input_dim2=None):
        super().__init__()
        is_t5_variant = conf.surface_transformer.startswith('razent/SciFive-base-Pubmed')
        if is_t5_variant or conf.surface_transformer.startswith('t5'):
            if conf.model in ['none', 'resi', 'gated', 'crossmodel']:
                from .modeling_t5_interleaved import T5ForConditionalGeneration as model_cls
            else:
                from transformers import T5ForConditionalGeneration as model_cls
            from transformers import T5Config
            trf_config = T5Config.from_pretrained(conf.surface_transformer)
            trf_config.force_bos_token_to_be_generated = True
            
            if conf.surface_random_init:
                self.trf = model_cls(trf_config)
            else:
                self.trf = model_cls.from_pretrained(
                    conf.surface_transformer, config=trf_config)
            self.predict = self.predict_seq2seq
            self.encoder = self.trf.encoder

            from .modeling_t5_interleaved import T5LMHead
            self.head = T5LMHead(trf_config)
            
        else:
            raise NotImplementedError('TODO')
        
        if conf.surface_freeze_encoder:
            for p in self.trf.encoder.parameters():
                p.requires_grad = False
        self.crit = torch.nn.CrossEntropyLoss()
        
        
    def forward(self, batch, return_instances=False):
        out = self.trf(
            input_ids=batch['tx_input_ids'],
            attention_mask=batch['tx_attention_mask'],
            decoder_input_ids=batch['tx_labels']
            )
        return self.result(batch, out, return_instances=return_instances)

    def result(self, tx_labels, out, return_instances=False):
        targets = tx_labels
        out.loss = self.crit(out.logits.flatten(0, 1), targets.flatten())

        result = {'loss': out.loss}
        if return_instances:
            result['tx_pred_ids'] = out.logits.argmax(-1)
            result['tx_target_input_ids'] = batch['tx_labels']
        return result

    def predict_seq2seq(self, tx_labels, tx_att, enc_out, return_instances=False):
        # https://discuss.huggingface.co/t/what-i-know-and-dont-know-about-sequence-to-sequence-batching/1046/4
        # https://github.com/huggingface/transformers/issues/5096
        decoder_input_ids = shift_tokens_right(
            tx_labels,
            self.trf.config.pad_token_id,
            self.trf.config.decoder_start_token_id
        )
        targets = tx_labels

        dec_out = self.trf.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc_out.last_hidden_state,
            encoder_attention_mask=tx_att,
            )
        
        lm_out = self.trf.lm_head(dec_out.last_hidden_state)
        
        logits = lm_out
        loss = self.crit(logits.flatten(0, 1), targets.flatten())

        out = Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=dec_out.past_key_values,
            decoder_hidden_states=dec_out.hidden_states,
            decoder_attentions=dec_out.attentions,
            cross_attentions=dec_out.cross_attentions,
            encoder_last_hidden_state=enc_out.last_hidden_state,
            encoder_hidden_states=enc_out.hidden_states,
            encoder_attentions=enc_out.attentions,
        )

        return self.result(tx_labels, out, return_instances=return_instances)

    
