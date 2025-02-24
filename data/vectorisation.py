import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Function to create n-grams from sequences
def seq2ngrams(seqs, n=3):
    return [[seq[i:i+n] for i in range(len(seq) - n + 1)] for seq in seqs]

# Function to tokenise the input dataset
def tokenisation(df, maxlen_seq, device):
    # Set maximum sequence length and filter sequences accordingly
    input_seqs, target_sst3_seqs, target_sst8_seqs = df[['seq', 'sst3', 'sst8']][(df.len <= maxlen_seq) & (~df.has_nonstd_aa)].values.T

    # Convert input sequences to n-grams
    input_grams = seq2ngrams(input_seqs)

    # Tokenize input n-grams
    tokenizer_encoder = Tokenizer()
    tokenizer_encoder.fit_on_texts(input_grams)
    # Convert input data to sequences of integers
    input_data = tokenizer_encoder.texts_to_sequences(input_grams)
    # Pad input sequences to ensure uniform length
    input_data = sequence.pad_sequences(input_data, maxlen=maxlen_seq, padding='post')
    # Convert to tensor and move to device
    input_data = torch.tensor(input_data, dtype=torch.long).to(device)

    # Tokenize target data for sst3
    tokenizer_decoder_sst3 = Tokenizer(char_level=True)
    tokenizer_decoder_sst3.fit_on_texts(target_sst3_seqs)
    target_data_sst3 = tokenizer_decoder_sst3.texts_to_sequences(target_sst3_seqs)
    target_data_sst3 = sequence.pad_sequences(target_data_sst3, maxlen=maxlen_seq, padding='post')
    target_data_sst3 = torch.tensor(target_data_sst3, dtype=torch.long).to(device)

    # Tokenize target data for sst8
    tokenizer_decoder_sst8 = Tokenizer(char_level=True)
    tokenizer_decoder_sst8.fit_on_texts(target_sst8_seqs)
    target_data_sst8 = tokenizer_decoder_sst8.texts_to_sequences(target_sst8_seqs)
    target_data_sst8 = sequence.pad_sequences(target_data_sst8, maxlen=maxlen_seq, padding='post')
    target_data_sst8 = torch.tensor(target_data_sst8, dtype=torch.long).to(device)

    n_words = len(tokenizer_encoder.word_index) + 1
    n_tags_sst3 = len(tokenizer_decoder_sst3.word_index) + 1
    n_tags_sst8 = len(tokenizer_decoder_sst8.word_index) + 1
    
    return input_data, target_data_sst3, target_data_sst8, n_words, n_tags_sst3, n_tags_sst8, target_sst3_seqs, target_sst8_seqs