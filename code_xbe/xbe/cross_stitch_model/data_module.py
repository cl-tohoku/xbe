import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import T5Tokenizer

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)

class UGDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            symbols: list(),
            tx_tokenizer: T5Tokenizer,
            max_token_len: int,
            is_testing: bool
            ):

        self.data = data
        self.symbol2id = {symbol:ind for ind, symbol in enumerate(symbols)}
        self.tx_tokenizer = tx_tokenizer
        self.max_token_len = max_token_len
        self.n_symbos = len(self.symbol2id)
        self.is_testing = is_testing

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        sym_h = data_row['symbol_head']
        sym_t = data_row['symbol_tail']
        sym_r = data_row['symbol_rel']
        
        tx_h = data_row['tx_head']
        tx_t = data_row['tx_tail']
        tx_r = data_row['tx_rel']

        #tx_r_mask = tx_r.replace('[HEAD]', '<extra_id_0>').replace('[TAIL]', '<extra_id_1>')
        #tx_r_label = tx_r.replace('[HEAD]', tx_h).replace('[TAIL]', tx_t)

        tx_r_mask = tx_r.replace('[HEAD]', tx_h).replace('[TAIL]', tx_t)
        tx_r_label = ' '.join(sym_r.split('_'))
        tx_r_label = '%s %s %s' % (tx_h, tx_r_label, tx_t)
        
        hi = self.symbol2id[sym_h]
        ri = self.symbol2id[sym_r]
        ti = self.symbol2id[sym_t]
        sym_mask = self.symbol2id['[MASK]']

        """
        mask_kg = torch.randint(3, (1,))
        if self.is_testing:
            #symbol_input_ids = torch.tensor([hi, sym_mask, ti])
            symbol_input_ids = [torch.tensor([hi, sym_mask, ti]),
                                torch.tensor([sym_mask, sym_mask, sym_mask]),
                                torch.tensor([sym_mask, sym_mask, sym_mask])][mask_kg]
        else:
            symbol_input_ids = [torch.tensor([hi, sym_mask, ti]),
                                torch.tensor([sym_mask, sym_mask, sym_mask]),
                                torch.tensor([sym_mask, sym_mask, sym_mask])][mask_kg] 
        """
        
        symbol_input_ids = torch.tensor([hi, sym_mask, ti])
        #symbol_input_ids = torch.tensor([sym_mask, sym_mask, sym_mask])
        symbol_attention_mask = torch.tensor([1, 1, 1])
        symbol_labels = torch.tensor([hi, ri, ti])
        
        source_encoding = self.tx_tokenizer(
            tx_r_mask,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        target_encoding=self.tx_tokenizer(
            tx_r_label,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        tx_labels = target_encoding['input_ids']
        tx_labels[tx_labels==0] = -100
        
        return dict(
            kg_triplet='%s %s %s' % (sym_h, sym_r, sym_t),
            tx_triplet=tx_r_label,
            sym_input_ids=symbol_input_ids,
            sym_attention_mask=symbol_attention_mask,
            tx_input_ids=source_encoding['input_ids'].flatten(),
            tx_attention_mask=source_encoding['attention_mask'].flatten(),
            sym_labels=symbol_labels,
            tx_labels=tx_labels.flatten(),
            alignment=symbol_labels
            )

class UGDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame,
            tx_tokenizer: T5Tokenizer,
            symbols: list(),
            batch_size: int=8,
            max_token_len: int=60
            ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tx_tokenizer = tx_tokenizer,
        self.symbols = symbols
        self.max_token_len = max_token_len
        self.n_symbols = len(symbols)
        
    def setup(self):
        self.train_dataset = UGDataset(self.train_df, self.symbols, self.tx_tokenizer[0], self.max_token_len, is_testing=False)
        self.test_dataset = UGDataset(self.test_df, self.symbols, self.tx_tokenizer[0], self.max_token_len, is_testing=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True
            )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=4
            )


    
        
