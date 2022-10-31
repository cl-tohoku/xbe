from transformers import AdamW

#from .surface_model import SurfaceTransformer
from .surface_model_bert import SurfaceTransformer
from .symbolic_model import TripleTransformer

import pytorch_lightning as pl
import torchmetrics
import torch

class CrossModel(pl.LightningModule):
    
    def __init__(self, conf):
        super().__init__()
        self.surface_trf = SurfaceTransformer(conf)
        self.symbol_trf = TripleTransformer(conf)
        self.surface_loss_weight = conf.surface_loss_weight
        self.symbolic_loss_weight = conf.symbolic_loss_weight
        self.cross_stitch_active = True
        self.conf = conf

        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        
        from .layers import (
            GatedCrossStitch,
            GatedCrossStitch_Multihead,
            AverageCrossStitch,
            FirstTokenCrossStitch,
            NoCrossStitch,
            ResiCrossStitch,
            NoCrossAtt,
            FusionCross,
            Alignment,
            )
        stitch_model_cls = {
            'gated': GatedCrossStitch,
            'gated_multihead': GatedCrossStitch_Multihead,
            'average': AverageCrossStitch,
            'firsttoken': FirstTokenCrossStitch,
            'none': NoCrossStitch,
            'resi': ResiCrossStitch,
            'nocrossatt': NoCrossAtt,
            'fusion': FusionCross,
            }[self.conf.stitch_model]
        
        n_layers1 = len(set(conf.surface_cross_stitch_send_layers))
        n_layers2 = len(set(conf.symbol_cross_stitch_send_layers))
        n_layers3 = len(set(conf.surface_cross_stitch_receive_layers))
        n_layers4 = len(set(conf.symbol_cross_stitch_receive_layers))
        assert n_layers1 == n_layers2 == n_layers3 == n_layers4
        n_cross_stitch_layers = n_layers1
        if self.conf.stitch_model.startswith('nocrossatt') or self.conf.stitch_model.startswith('fusion'):
            self.cross_stitch = stitch_model_cls(
                n_cross_stitch_layers,
                input1_dim=self.surface_trf.trf.config.hidden_size,
                input2_dim=self.symbol_trf.trf.config.hidden_size,
                conf=self.conf
                )
        else:
            self.cross_stitch = stitch_model_cls(
                n_cross_stitch_layers,
                input1_dim=self.surface_trf.trf.config.hidden_size,
                input2_dim=self.symbol_trf.trf.config.hidden_size,
                attn_dim=conf.attn_dim,
                )

        if self.conf.cross_stitch_start_epoch > 0:
            self.cross_stitch_active = False
            for param in self.cross_stitch.parameters():
                param.requires_grad = False
                

        if self.conf.freeze_symbol_until_epoch > 0:
            for param in self.symbol_trf.parameters():
                param.requires_grad = False
        if self.conf.freeze_surface_until_epoch > 0:
            for param in self.surface_trf.parameters():
                param.requires_grad = False

        self.align = Alignment(
            repr1_dim=self.surface_trf.trf.config.hidden_size,
            repr2_dim=self.symbol_trf.trf.config.hidden_size,
            attn_dim=self.conf.alignment_attn_dim,
            )
                
    def forward(self, token, att_mask, sym, sym_mask, sym_labels, return_instances=False):
        """This method implements the main cross-encoder functionality.
        We have two transformer-based encoders, namely a symbol encoder
        (symbol_trf) and a surface encoder (surface_trf).  The forward
        methods of those two encoders are coroutines.
        The role of this method is to drive those two coroutines.
        That is, whenever one of the encoders reaches a layer specified
        as cross-stitch layer, we receive the current hidden states from
        that encoder layer, pass it into the cross stitch layer, and then
        send the updated hidden states to the other encoder, and vice versa.
        """
        stitch_idx = 0
        # initialize the two encoder coroutines
        symbol_send_layers = set(self.conf.symbol_cross_stitch_send_layers)
        symbol_rcv_layers = set(self.conf.symbol_cross_stitch_receive_layers)
        
        symbol_gen = self.symbol_trf.encoder(
            sym,
            sym_mask,
            cross_stitch_send_layers=symbol_send_layers,
            cross_stitch_receive_layers=symbol_rcv_layers,
            )
        
        surface_send_layers = set(self.conf.surface_cross_stitch_send_layers)
        surface_rcv_layers = set(self.conf.surface_cross_stitch_receive_layers)
        surface_gen = self.surface_trf.encoder(
            token,
            att_mask,
            cross_stitch_send_layers=surface_send_layers,
            cross_stitch_receive_layers=surface_rcv_layers,
            )
        
        symbol_repr = None
        surface_repr = None
        cross_stitch_results = {}
        
        cratt_t2s = None
        cratt_s2t = None
        for stitch_idx in range(self.cross_stitch.n_layers):
            # send current surface and symbol representations to the
            # encoding coroutines of their resepective encoders. The
            # encoders will process those representations layer-wise
            # until they reach a layer that is specified as "send" layer.
            # Once a "send" layer is reached, the coroutines will yield
            # the representations, which we receive here.
            
            surface_repr, symbol_repr = (
                surface_gen.send(surface_repr), symbol_gen.send(symbol_repr))
            
            if self.cross_stitch_active:
                # apply the cross stitch operation, i.e. take in two
                # representations, combine them, and then return them.
                
                cross_stitch_out = self.cross_stitch(
                    stitch_idx, surface_repr, symbol_repr)
                
                if self.conf.stitch_model == 'gated':
                    surface_repr = cross_stitch_out.output1
                    symbol_repr = cross_stitch_out.output2
                elif self.conf.stitch_model == 'resi':
                    surface_repr, symbol_repr, cratt_t2s, cratt_s2t = cross_stitch_out
                else:
                    surface_repr, symbol_repr = cross_stitch_out
                    
                if return_instances:
                    cross_stitch_results[stitch_idx] = {
                        'weight12': cross_stitch_out.weight12,
                        'weight21': cross_stitch_out.weight21,
                        'scores12': cross_stitch_out.scores12,
                        'scores21': cross_stitch_out.scores21,
                        }
                    
        # apply the remainder of the encoder layers, i.e., from the last
        # cross stich layer to the final layer. If there aren't any
        # cross-stitch layer, this is equivalent to the standard forward
        # methods in transformers
        
        symbol_out = symbol_gen.send(symbol_repr)
        surface_out = surface_gen.send(surface_repr)
        
        symbol_results = self.symbol_trf.predict(
            sym_labels, symbol_out, return_instances=False)
        
        #return surface_out.last_hidden_state, surface_results['loss'], symbol_results['loss']
        #return symbol_out.last_hidden_state, surface_results['loss'], symbol_results['loss']
        #hidden = torch.cat([surface_out.last_hidden_state, symbol_out.last_hidden_state], axis=1)#(B, tx_len+kg_len, dim)
        h_kg_h = symbol_out.last_hidden_state[:, 0, :]
        h_kg_r = symbol_out.last_hidden_state[:, 1, :]
        h_kg_t = symbol_out.last_hidden_state[:, 2, :]
        return surface_out.last_hidden_state, h_kg_h, h_kg_t, h_kg_r, symbol_results['loss'], cratt_s2t
