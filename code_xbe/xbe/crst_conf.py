"""
conf. for CRST (cross-stitch) encoder.
"""

surface_cross_stitch_send_layers = [4]
surface_cross_stitch_receive_layers = [5]
symbol_cross_stitch_send_layers = [4]
symbol_cross_stitch_receive_layers = [5]

surface_loss_weight = 0.7
symbolic_loss_weight = 0.3
alignment_loss_weight = 0.3

stitch_model = 'resi'
#stitch_model = 'none'
cross_stitch_start_epoch = 0
freeze_symbol_until_epoch = 0
freeze_surface_until_epoch = 0
attn_dim = 64
alignment_attn_dim = 64

#surface_transformer = 'razent/SciFive-base-Pubmed'
#surface_transformer = 't5-base'
surface_transformer = 'bert-base-uncased'
surface_random_init = False
surface_freeze_encoder = False

max_length=40

model = 'resi' #'resi', 'none' and 'gated'
#model = 'none'
lr = 2e-5

triple_trf_n_hidden = 6
triple_trf_dropout = 0.1
triple_trf_n_attn_heads = 12
head = 'lmhead'
pretrained_emb = False
symbolic_transformer = 'roberta-base'

gpu = 1
sym_size = 100000
bag_limt = 10 #the numb. of sentences in a bag

mid_dim = 10
