import torch
import torch.nn as nn
import math
import torch.nn.functional as F


# Define positional encoding to add spatial information to embeddings
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, maxlen_seq):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, maxlen_seq, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(maxlen_seq, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        # Multi-head attention layer
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        # Normalization layer
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        # Fully connected layers for feedforward network
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        # Normalization layer
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Self-attention and residual connection
        attn_output, _ = self.multi_head_attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
         # Feedforward network and residual connection
        ff_output = self.fc2(F.relu(self.fc1(x)))
        x = self.layer_norm2(x + ff_output)
        return x

# Define the main model class with embedding, CNN, LSTM, and transformers
class TransformerModel(nn.Module):
    def __init__(self, n_words, n_tags_sst3, n_tags_sst8, embed_dim, num_heads, ff_dim, maxlen_seq, num_encoder_layers=6, dropout=0.0):
        super(TransformerModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(n_words, embed_dim)
        
        # CNN Layer -- 1D convolution layer to capture local features by applying convolution across the input embeddings after the embedding layer
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=3, padding=1)
        
        # RNN Layer (LSTM) layer follows the CNN to capture sequential dependencies
        # Bidirectional to capture both past and future context, the output from the bidirectional LSTM has double the embedding dimension, so need to reduce back to the original embedding size
        self.lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, bidirectional=True)
        
        # Linear layer to reduce bidirectional LSTM output to the original embedding size
        self.linear_bi_lstm = nn.Linear(embed_dim * 2, embed_dim)
        
        # Positional encoding layer
        self.positional_encoding = PositionalEncoding(embed_dim, maxlen_seq)
        
        # Stacked transformer encoder layers (dynamically set based on hyperparameter)
        self.encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_encoder_layers)
        ])
        
        # Classifier layers for sst3 and sst8
        self.classifier_sst3 = nn.Linear(embed_dim, n_tags_sst3)
        self.classifier_sst8 = nn.Linear(embed_dim, n_tags_sst8)
    
    def forward(self, x):
        x = self.embedding(x)  # x: [batch_size, seq_len, embed_dim]
        
        # CNN layer:permute to [batch_size, embed_dim, seq_len] for 1D convolution
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)  #permute back to [batch_size, seq_len, embed_dim]
        
        # RNN layer (LSTM)
        x, _ = self.lstm(x)
        
        # Linear layer to project bidirectional LSTM output back to embedding dimension
        x = self.linear_bi_lstm(x)
        
        # Positional encoding
        x = x.permute(1, 0, 2)  # Change to [seq_len, batch_size, embed_dim] for transformer input
        x = self.positional_encoding(x)
        
        # Transformer encoder layers
        for encoder in self.encoders:
            x = encoder(x)
        
        # Classifiers for SST tasks
        output_sst3 = self.classifier_sst3(x)
        output_sst8 = self.classifier_sst8(x)
        
        return output_sst3, output_sst8