import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedGraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(GatedGraphConv, self).__init__()
        self.U = nn.Linear(node_dim, hidden_dim, bias=False)
        self.V = nn.Linear(node_dim, hidden_dim, bias=False)
        self.A = nn.Linear(node_dim, hidden_dim, bias=False)
        self.B = nn.Linear(node_dim, hidden_dim, bias=False)
        self.C = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.bn_node = nn.BatchNorm1d(hidden_dim)
        self.bn_edge = nn.BatchNorm1d(hidden_dim)

    def forward(self, h, e, adj):

        e = self.C(e) + self.A(h.unsqueeze(2)) + self.B(h.unsqueeze(1))
        e = self.bn_edge(e.view(-1, e.size(-1))).view(e.size()) 
        e = F.relu(e)
        adj_augmented = adj.unsqueeze(3)
        e = e * adj_augmented
        e_weight = torch.sigmoid(e)
        e_norm = F.softmax(e_weight, dim=2)
        h_in = self.U(h)
        h_neighbors = self.V(h)


        m = torch.einsum('bmnd,bnd->bmnd', e_norm, h_neighbors)
        m = m.sum(dim=2) 
        m = self.bn_node(m.view(-1, m.size(-1))).view(m.size())  
        h = h_in + F.relu(m)
        
        return h, e_norm

class GatedGCNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(self.__class__, self).__init__()
        self.hidden_dim = hidden_dim
        self.gatedgcn = GatedGraphConv(hidden_dim, hidden_dim, hidden_dim)
    
    def forward(self, node_embed, res_edge, Adj):
        node_embed, res_edge_new = self.gatedgcn(node_embed, res_edge, Adj)
        return node_embed, res_edge_new