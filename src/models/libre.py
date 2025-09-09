import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch_geometric.nn import GATConv, global_mean_pool

class LigandEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=3, dropout=0.2):
        super(LigandEncoder, self).__init__()
        self.dropout = dropout
        
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * heads, output_dim, heads=1, concat=False, dropout=dropout)
        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.leaky_relu(self.gat1(x, edge_index), negative_slope=0.2)
        x = F.leaky_relu(self.gat2(x, edge_index), negative_slope=0.2)
        x = F.leaky_relu(self.gat3(x, edge_index), negative_slope=0.2)

        x = global_mean_pool(x, batch)

        return x

class ResidueFC(nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(ResidueFC, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64)  
        )
        
    def forward(self, x):
        
        x = self.fc_layers(x)
        
        return x   
    
class ResidueCNN(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, dropout=0.2):
        super(ResidueCNN, self).__init__()
        
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=output_dim, 
                                kernel_size=kernel_size, stride=1,
                                padding='same', padding_mode='zeros')

        self.batch_norm = nn.BatchNorm1d(output_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
          
    def forward(self, x):
        
        x = self.conv1d(x.permute(0, 2, 1))  
        x = self.gelu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        
        return x
    

class ResidueLSTM(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(ResidueLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=input_dim//2, 
                            num_layers = num_layers, bidirectional = True, 
                            batch_first = True)
        
        self.layer_norm = nn.LayerNorm(input_dim)
            
    def forward(self, x):
        
        x, _ = self.lstm(x)    
        x = self.layer_norm(x)  
        
        return x

class ResidueEncoder(nn.Module):
    def __init__(self, input_dim=1024, dropout=0.2):
        super(ResidueEncoder, self).__init__()
        
        self.residue_cnn_1 = ResidueCNN(input_dim=input_dim, output_dim=256,
                                        dropout=dropout)
        
   
        self.residue_cnn_2 = ResidueCNN(input_dim=256, output_dim=64, 
                                        dropout=dropout)
        
        self.residue_lstm = ResidueLSTM(input_dim=64)
        
        
    def forward(self, embeddings):

        cnn_out_1 = self.residue_cnn_1(embeddings) 
        cnn_out_2 = self.residue_cnn_2(cnn_out_1) 
        
        final_residue_embedding = self.residue_lstm(cnn_out_2) 
        
        return final_residue_embedding
    
class BRPredictor(nn.Module):
    def __init__(self,  use_cnn_lstm=True, use_ligand=True, 
                        residue_input_dim=1024, ligand_input_dim=40, 
                        ligand_hidden_dim=128, dropout=0.2):
        super(BRPredictor, self).__init__()
        
        

        if use_cnn_lstm:
            self.residue_encoder = ResidueEncoder(input_dim=residue_input_dim, 
                                                  dropout=dropout)
        else:
            self.residue_encoder = ResidueFC(input_dim=residue_input_dim, 
                                             dropout=dropout)

        if use_ligand:
            self.ligand_encoder = LigandEncoder(input_dim=ligand_input_dim, hidden_dim=ligand_hidden_dim, 
                                                output_dim=64)
        else:
            self.ligand_encoder = None
        
        self.attn_r2l = MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.attn_l2r = MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(64)      

        self.fc1 = nn.Linear(64, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  
        
        self.dropout = nn.Dropout(dropout)
        
        self.gelu = nn.GELU()
        
    def forward(self, residue_embeddings, ligand_data=None):

        residue_out = self.residue_encoder(residue_embeddings) 
        
        if self.ligand_encoder and ligand_data is not None:            

            ligand_out = self.ligand_encoder(ligand_data) 
            ligand_out = ligand_out.unsqueeze(1)
               
            r2l_out, _ = self.attn_r2l(query=residue_out, key=ligand_out, value=ligand_out)  
            l2r_out, _ = self.attn_l2r(query=ligand_out, key=residue_out, value=residue_out)  
            
            residue_out = self.layer_norm(residue_out + r2l_out)
            ligand_out = self.layer_norm(ligand_out + l2r_out)
            
            binding_embedding = residue_out * ligand_out  

        else:
            binding_embedding = residue_out

        x = self.gelu(self.fc1(binding_embedding))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        
        affinity_score = self.fc3(x)  
        
        return binding_embedding, affinity_score