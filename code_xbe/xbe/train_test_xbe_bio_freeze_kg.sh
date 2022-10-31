CUDA_VISIBLE_DEVICES=0 python3 main_crst_bio_all.py \
        --seed 77777 \
        --lr 3e-5 --batch_size_per_gpu 100 --max_epoch 15 \
        --max_length 60 \
        --mode CM \
        --dataset bio \
        --entity_marker --ckpt_to_load None \
        --train_prop 1 \
        --bag_size 30 \
        --entity_embedding_load_path ../../data/bio/entity_embedding.npy \
        --kg_method TransE_re \
        --direct_feature \
        --freeze_entity \
        --prefix TXKG \
        --w_symloss 1.0 \
        --crst_mod resi \
	--freeze_kg \
        --crst_path ./ckpt_kg/bio-pre-kg.ckpt

CUDA_VISIBLE_DEVICES=0 python3 main_crst_bio_all.py \
        --seed 77777 \
        --lr 3e-5 --batch_size_per_gpu 100 --max_epoch 15 \
        --max_length 60 \
        --mode CM \
        --dataset bio \
        --entity_marker --ckpt_to_load None \
        --train_prop 1 \
        --bag_size 30 \
        --entity_embedding_load_path ../../data/bio/entity_embedding.npy \
        --kg_method TransE_re \
        --direct_feature \
        --freeze_entity \
        --prefix TXKG \
        --w_symloss 1.0 \
        --crst_mod resi \
        --freeze_kg \
        --crst_path ./ckpt_kg/bio-pre-kg.ckpt \
	--test --test_only
