# XBE
Codes and datasets for our paper **"Cross-stitching Text and Knowledge Graph Encoders for Distantly Supervised Relation Extraction"** (EMNLP 2022)
## Overview of XBE
  <img src="/xbe_overview.png" width="500">

### Setup
Install dependencies (consider using a virtual environment):
~~~~
pip install -r requirements.txt
~~~~
Then, follow the [instruction](https://github.com/NVIDIA/apex) to install **apex**.

### 1.Dataset:
We provide preprocessed NYT10 and Medline21.

Please download them from here: [NYT10](http://www.cl.ecei.tohoku.ac.jp/~dq/Data_for_EMNLP2022/nyt.zip) (i.e., `nyt.zip`) and [Medline21](http://www.cl.ecei.tohoku.ac.jp/~dq/Data_for_EMNLP2022/bio.zip) (i.e., `bio.zip`), and unzip them under `data/`

### 2.Pre-trained KG Encoder:
Please download our pre-trained KG encoders from: [NYT10 KG enc.](http://www.cl.ecei.tohoku.ac.jp/~dq/Data_for_EMNLP2022/nyt-pre-kg.ckpt) and [Medline21 KG enc.](http://www.cl.ecei.tohoku.ac.jp/~dq/Data_for_EMNLP2022/bio-pre-kg.ckpt), and put them under `code_xbe/xbe/ckpt_kg/`

### 3.Train:
Run the following script for training XBE on NYT-FB60K.
~~~
cd code_xbe/xbe/
bash train_xbe_nyt.sh
~~~
Run the following script for training XBE on Medline21.
~~~
cd code_xbe/xbe/
bash train_xbe_bio.sh
~~~

If you want to directly utilize our pre-trained XBE models, you can download them from: [NYT10 XBE](http://www.cl.ecei.tohoku.ac.jp/~dq/Data_for_EMNLP2022/TXKG0.6nyt_bert-base-uncased_TransE_re_direct__kg_crst_resi.mdl) and [Medline21 XBE](http://www.cl.ecei.tohoku.ac.jp/~dq/Data_for_EMNLP2022/TXKG1.0bio_bert-base-uncased_TransE_re_direct__kg_crst_bio_resi.mdl), and put them under `save/nyt/` and `save/bio/` respectively.
### 4.Test:
Please run the following script for testing the trained XBE model on NYT10 and Medline21 datasets respectively.
~~~~
cd code_xbe/xbe/
bash test_xbe_nyt.sh
~~~~
~~~~
cd code_xbe/xbe/
bash test_xbe_bio.sh
~~~~

### Ablation Study:
- Please run the following script for training and testing the XBE model without the cross-stitch mechanism.
  ~~~
  cd code_xbe/xbe/
  bash train_test_xbe_nyt_wo_xstitch.sh
  bash train_test_xbe_bio_wo_xstitch.sh
  ~~~
- Run the following script for the XBE model without Text encoder.
  ~~~
  cd code_xbe/xbe/
  bash train_test_xbe_nyt_wo_tx.sh
  bash train_test_xbe_bio_wo_tx.sh
  ~~~
- Run the following script for the XBE model without KG encoder.
  ~~~
  cd code_xbe/xbe/
  bash train_test_xbe_nyt_wo_kg.sh
  bash train_test_xbe_bio_wo_kg.sh
  ~~~
- Run the following script for the XBE model with frozen KG encoder.
  ~~~
  cd code_xbe/xbe/
  bash train_test_xbe_nyt_freeze_kg.sh
  bash train_test_xbe_bio_freeze_kg.sh
  ~~~
- Run the following script for the XBE model with the KG encoder that is randomly initialized.
  ~~~
  cd code_xbe/xbe/
  bash train_test_xbe_nyt_random_kg.sh
  bash train_test_xbe_bio_random_kg.sh
  ~~~

