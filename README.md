# XBE
Codes and datasets for our paper "Cross-stitching Text and Knowledge Graph Encoders for Distantly Supervised Relation Extraction" (EMNLP 2022)
## Overview of XBE
  <img src="/xbe_overview.png" width="500">

### Setup
Install dependencies (consider using a virtual environment):
~~~~
pip install -r requirements.txt
~~~~

### 1.Dataset:
We provide preprocessed NYT-FB60K (Han et al., 2018) and Medline21 (Dai et al., 2021).

Please download the NYT-FB60K dataset from and unzip it under data/

Similarly, please download the Medline21 dataset from and unzip it under data/

### 2.Pre-trained KG Encoder:
Please download the pre-trained KG encoder from and put them under 

### 3.Train:
Run the following script for training XBE on NYT-FB60K.
~~~

~~~
Run the following script for training XBE on Medline21.
~~~

~~~

If you want to directly utilize our pre-trained XBE models, you can download and from and put them under and respectively.
### 4.Test:
Please run the following script for testing the trained XBE model on NYT-FB60K and Medline21 datasets respectively.
~~~~

~~~~
~~~~

~~~~

### Ablation Study:
Please run the following script for training and testing the XBE model without the cross-stitch mechanism.

Run the following script for the XBE model without Text encoder.

Run the following script for the XBE model without KG encoder.

Run the following script for the XBE model with frozen KG encoder.

Run the following script for the XBE model with the KG encoder that is randomly initialized.

