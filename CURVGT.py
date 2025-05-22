from ast import pattern
import os
import torch
import argparse
import numpy as np
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch.optim.lr_scheduler import StepLR
from torch_scatter import scatter

from graphTransformers import TransformerLayer_GT, TransformerLayer_Fourier, TransformerLayer_Galerkin
from ResNET import BasicBlock

def parse_args():
    parser = argparse.ArgumentParser(description="CURVGT model training")
    parser.add_argument('--model_name', type=str, required=True, help="Model name")
    parser.add_argument('--datasets_function', type=str, required=True, help="Datasets function`s name")
    parser.add_argument('--datasets', type=str, required=True, help="Datasets name")
    parser.add_argument('--gpu', type=int, default=1, help="GPU device number to use, default is 1")
    parser.add_argument('--batch_size', type=int, default=10, help="Batch size for training, default is 10")
    parser.add_argument('--modes', type=int, default=21, help="Number of modes to select in the feature decomposition, default is 21")
    parser.add_argument('--test_pattern', type=int, default=1, help="0: Run test with model loaded, 1: Normal training")
    return parser.parse_args()
args = parse_args()
model_name = args.model_name
datasets_function = args.datasets_function
datasets = args.datasets
gpu = args.gpu
batch_size = args.batch_size
modes = args.modes
test_pattern = args.test_pattern
learning_rate = 1e-2

def read_2d_list_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        return [[int(item) for item in line.strip().split()] for line in lines]

# define paths
edge_attr_path = f'./dataset/{datasets_function}/geometry_processed_docs/edge_attr.pt'
vec_normal_path = f'./dataset/{datasets_function}/geometry_processed_docs/vec_normal.pt'
grad_matrix_path = f'./dataset/{datasets_function}/geometry_processed_docs/gradMatrix.pt'
mass_path = f'./dataset/{datasets_function}/geometry_processed_docs/mass.pt'
vertex_to_voronoi_path = f'./dataset/{datasets_function}/geometry_processed_docs/vertexToVoronoi.txt'
boundary_vertices_path = f'./dataset/{datasets_function}/geometry_processed_docs/boundaryVertices.pt'
inv_metrics_path = f'./dataset/{datasets_function}/geometry_processed_docs/inv_metrics.pt'
pd_path = f'./dataset/{datasets_function}/geometry_processed_docs/PD.pt'
eigenvec_path = f'./dataset/{datasets_function}/geometry_processed_docs/normalized_eigenvec.pt'

train_data_path = f'./dataset/{datasets_function}/wrinkle/train_data(.1).pt'
test_data_path = f'./dataset/{datasets_function}/wrinkle/test_data(.1).pt'

tree_mask0d4_path = f'./dataset/{datasets_function}/sub_tree_partitions/tree_mask0d4.pt'
tree_mask1d4_path = f'./dataset/{datasets_function}/sub_tree_partitions/tree_mask1d4.pt'
tree_mask2d4_path = f'./dataset/{datasets_function}/sub_tree_partitions/tree_mask2d4.pt'

tree_roots0d4_path = f'./dataset/{datasets_function}/sub_tree_partitions/tree_roots0d4.pt'
tree_roots1d4_path = f'./dataset/{datasets_function}/sub_tree_partitions/tree_roots1d4.pt'
tree_roots2d4_path = f'./dataset/{datasets_function}/sub_tree_partitions/tree_roots2d4.pt'

tree_brodcastmap_path  = f'./dataset/{datasets_function}/sub_tree_partitions/tree0d4_broadcastmap.pt'
tree_brodcastmap_path1 = f'./dataset/{datasets_function}/sub_tree_partitions/tree1d4_broadcastmap.pt'
tree_brodcastmap_path2 = f'./dataset/{datasets_function}/sub_tree_partitions/tree2d4_broadcastmap.pt'

option_mask_path = f'./dataset/{datasets_function}/geometry_processed_docs/option_mask.pt'
HyperPT_path = f'./dataset/{datasets_function}/geometry_processed_docs/HyperPT.pt'
H2frame_path = f'./dataset/{datasets_function}/geometry_processed_docs/H2frame.pt'

GaussianCurvature_path = f'./dataset/{datasets_function}/geometry_processed_docs/GaussianCurvature.pt'

# load data
edge_attrs = torch.load(edge_attr_path)
vertex_normals = torch.load(vec_normal_path)
G, M = torch.load(grad_matrix_path).to_dense(), torch.diag(torch.load(mass_path).to_dense().flatten())
vertices_to_voronoi = read_2d_list_from_txt(vertex_to_voronoi_path)
boundary_mask = torch.load(boundary_vertices_path).flatten() 

metrics, PD = torch.load(inv_metrics_path), torch.load(pd_path)

eigenvecs = torch.load(eigenvec_path)
indices = torch.linspace(0, len(eigenvecs) - 1, steps=modes).long()

eigenvecs = eigenvecs[:modes] 

max_voronoi_count = max(len(v) for v in vertices_to_voronoi)
vertices_to_voronoi_tensor = torch.full((len(vertices_to_voronoi), max_voronoi_count), -1, dtype=torch.long)
for i, voronoi_list in enumerate(vertices_to_voronoi):
    vertices_to_voronoi_tensor[i, :len(voronoi_list)] = torch.tensor(voronoi_list, dtype=torch.long)
mask = vertices_to_voronoi_tensor != -1  # mask contains True and False values
degree = mask.sum(dim=1).float().unsqueeze(1) #so that its shape is (|V|,1)

v_num = vertex_normals.shape[0]
e_num = edge_attrs.shape[0]
f_num = 4802
mu = 5

train_data, test_data = torch.load(train_data_path),torch.load(test_data_path)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

# move to gpu
edge_attrs = edge_attrs.float().to(device)
vertex_normals = vertex_normals.float().to(device)
G, M = G.float().to(device), M.float().to(device)
mask, degree, boundary_mask = mask.to(device), degree.float().to(device),boundary_mask.to(device)
metrics, PD = metrics.float().to(device), PD.float().to(device)

tree_mask = torch.load(tree_mask0d4_path).long().to(device)
tree_mask1 = torch.load(tree_mask1d4_path).long().to(device)
tree_mask2 = torch.load(tree_mask2d4_path).long().to(device)

tree_roots = torch.load(tree_roots0d4_path).long()
tree_roots1 = torch.load(tree_roots1d4_path).long()
tree_roots2 = torch.load(tree_roots2d4_path).long()

tree_brodcastmap = torch.load(tree_brodcastmap_path).long().to(device)
tree_brodcastmap1 = torch.load(tree_brodcastmap_path1).long().to(device)
tree_brodcastmap2 = torch.load(tree_brodcastmap_path2).long().to(device)

option_mask = torch.load(option_mask_path).long().to(device)
HyperPT = torch.load(HyperPT_path).float().to(device)
H2frame = torch.load(H2frame_path).float().to(device)

GaussianCurvature = torch.load(GaussianCurvature_path).float().to(device)

import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


ACTIVATION = {'gelu':nn.GELU(),'tanh':nn.Tanh(),'sigmoid':nn.Sigmoid(),'relu':nn.ReLU(),'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus(),'ELU':nn.ELU()}

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu'):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            self.act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.linear_pre = nn.Linear(n_input, n_hidden)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])


    def forward(self, x):
        x = self.act(self.linear_pre(x))
        for i in range(self.n_layers):
            x = self.act(self.linears[i](x)) + x

        x = self.linear_post(x)
        return x

class CurvAwareAttn(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add',heads=1, treeRoots=tree_roots,treeMasks=tree_mask,broadcastmaps=0,isYoung=False,isComplete=False):
        super(CurvAwareAttn, self).__init__()  
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.heads = heads
        self.linear = Linear(3,3) 
        self.att = torch.nn.Parameter(torch.Tensor(heads, 2 * out_channels))
        self.treeMask = treeMasks

        self.k = torch.nn.Parameter(torch.ones(v_num, 3,3)).to(device) 
        self.attn = torch.nn.Parameter(torch.ones(v_num, 1,2,3)).to(device)
        self.k2 = torch.nn.Parameter(torch.ones(v_num, 3,3)).to(device)  

        mask = torch.zeros_like(self.k, dtype=torch.bool).to(device)
        mask[treeRoots] = True
        self.k.register_hook(lambda grad: grad * mask.float())

        mask2 = torch.zeros_like(self.attn, dtype=torch.bool).to(device)
        mask2[treeRoots] = True
        self.attn.register_hook(lambda grad: grad * mask2.float()) 

        mask3 =  torch.zeros_like(self.k2, dtype=torch.bool).to(device)
        mask3[treeRoots] = True
        self.k2.register_hook(lambda grad: grad * mask3.float()) 

        self.layernormA = nn.LayerNorm([e_num,3,2])
        self.layernormB = nn.LayerNorm([e_num,2])
        self.layernormC = nn.LayerNorm([e_num,2])
        
        self.broadcastmaps = broadcastmaps

        ###init para PT requires
        self.cur_k = self.k.repeat(batch_size, 1, 1)
        self.cur_attn = self.attn.repeat(batch_size, 1, 1,1)
        self.cur_k2 = self.k2.repeat(batch_size, 1, 1)

        self.e1s, self.e2s, self.e3s = edge_attrs[:,11:14].repeat(batch_size, 1),edge_attrs[:,14:17].repeat(batch_size, 1),edge_attrs[:,17:20].repeat(batch_size, 1)

        Theta = edge_attrs[:,9:10].repeat(batch_size, 1)
        Theta_masked = Theta*treeMasks.repeat(batch_size).view(-1,1)
        self.cos_unmasked, self.sin_unmasked = torch.cos(Theta), torch.sin(Theta)
        self.cos_masked, self.sin_masked = torch.cos(Theta_masked), torch.sin(Theta_masked)

        self.cur_broadcastmaps = self.broadcastmaps.repeat(batch_size)
        incre_mask = (self.cur_broadcastmaps != -1).long()
        incre = torch.concat([torch.ones_like(self.broadcastmaps) * self.broadcastmaps.shape[0] * i for i in range(batch_size)], dim=0).long()       
        self.cur_broadcastmaps += incre_mask*incre

        self.H2frame = H2frame.repeat(batch_size,1,1)
        self.transform_matrix = HyperPT.repeat(batch_size, 1,1)
        self.option_mask = option_mask.repeat(batch_size)

        self.isYoung = isYoung
        self.isComplete = isComplete
        self.youngLinearTransform = nn.Linear(3,3)

        self.layernorm1 = nn.LayerNorm(e_num)
        self.linear1 = nn.Linear(2,1)

        self.c = torch.nn.Parameter(torch.ones(2)).to(device)

    def forward(self, x, edge_index, istraining):
        self.istraining= istraining
        return self.propagate(edge_index, x=x)

    def parallel_transport(self, repeat_times,x_j,mask):

        Theta = edge_attrs[:,9:10].repeat(repeat_times, 1)*mask.view(-1,1)
        e1s, e2s, e3s =  edge_attrs[:,11:14].repeat(repeat_times, 1),edge_attrs[:,14:17].repeat(repeat_times, 1),edge_attrs[:,17:20].repeat(repeat_times, 1)
        cos, sin = torch.cos(Theta),torch.sin(Theta)
        a, b = torch.einsum('ij,ij->i',e1s,x_j).view(-1,1),torch.einsum('ij,ij->i',e2s,x_j).view(-1,1)
        pt_ed = a*cos*e1s+a*sin*e3s+b*e2s
        return pt_ed

    def parallel_transport_unmasked(self, repeat_times,x_j):
        #PT with positive Gaussian curvature
        e1s, e2s, e3s =  self.e1s[:repeat_times*e_num],self.e2s[:repeat_times*e_num],self.e3s[:repeat_times*e_num]
        cos, sin = self.cos_unmasked[:repeat_times*e_num],self.sin_unmasked[:repeat_times*e_num]
        a, b = torch.einsum('ij,ij->i',e1s,x_j).view(-1,1),torch.einsum('ij,ij->i',e2s,x_j).view(-1,1)
        pt_ed = a*cos*e1s+a*sin*e3s+b*e2s
    
        #PT with negative Gaussian Curvature, projection along principal directions
        x_dir, y_dir = self.H2frame[:repeat_times*e_num,0], self.H2frame[:repeat_times*e_num,1]
        transform_matrix = self.transform_matrix[:repeat_times*e_num]
        cur_option_mask = self.option_mask[:repeat_times*e_num]
    
        a, b = torch.einsum('ij,ij->i',x_dir,x_j).view(-1,1),torch.einsum('ij,ij->i',y_dir,x_j).view(-1,1)
        local_coord = torch.cat((a,b),dim=-1)
        pt_ed2 = torch.einsum('ijk,ik->ij',transform_matrix, local_coord)

        pt_ed2 = x_dir*pt_ed2[:,0].unsqueeze(1) + y_dir * pt_ed2[:,1].unsqueeze(1)

        strictly_pos = (cur_option_mask == 1).long().unsqueeze(1)
        strictly_neg = (cur_option_mask == -1).long().unsqueeze(1)
        strictly_zero = (cur_option_mask == 0).long().unsqueeze(1)

        return pt_ed*strictly_pos + pt_ed2*strictly_neg + x_j*strictly_zero

    def parallel_transport_masked(self, repeat_times,x_j):
        e1s, e2s, e3s =  self.e1s[:repeat_times*e_num],self.e2s[:repeat_times*e_num],self.e3s[:repeat_times*e_num]
        cos, sin = self.cos_masked[:repeat_times*e_num],self.sin_masked[:repeat_times*e_num]
        a, b = torch.einsum('ij,ij->i',e1s,x_j).view(-1,1),torch.einsum('ij,ij->i',e2s,x_j).view(-1,1)
        pt_ed = a*cos*e1s+a*sin*e3s+b*e2s
    
        #PT with negative Gaussian Curvature, projection along principal directions
        x_dir, y_dir = self.H2frame[:repeat_times*e_num,0], self.H2frame[:repeat_times*e_num,1]
        transform_matrix = self.transform_matrix[:repeat_times*e_num]
        cur_option_mask = self.option_mask[:repeat_times*e_num]

        a, b = torch.einsum('ij,ij->i',x_dir,x_j).view(-1,1),torch.einsum('ij,ij->i',y_dir,x_j).view(-1,1)
        local_coord = torch.cat((a,b),dim=-1)
        pt_ed2 = torch.einsum('ijk,ik->ij',transform_matrix, local_coord)

        pt_ed2 = x_dir*pt_ed2[:,0].unsqueeze(1) + y_dir * pt_ed2[:,1].unsqueeze(1)
        #if masked, then the transform matrix degenerates into an identity matrix
        cur_tree_mask = self.treeMask.repeat(repeat_times).unsqueeze(1)
        pt_ed2 =  x_j*(cur_tree_mask+1)%2 + pt_ed2*cur_tree_mask

        strictly_pos = (cur_option_mask == 1).long().unsqueeze(1)
        strictly_neg = (cur_option_mask == -1).long().unsqueeze(1)
        strictly_zero = (cur_option_mask == 0).long().unsqueeze(1)
        return pt_ed*strictly_pos + pt_ed2*strictly_neg + x_j*strictly_zero

    def message(self, x_i, x_j, edge_index):
        repeat_times = edge_index.shape[1]//(edge_attrs.shape[0])

        pt = self.parallel_transport_unmasked(repeat_times,x_j)
        if not self.isComplete and not self.isYoung:
            cur_broadcastmaps = self.cur_broadcastmaps[:repeat_times*e_num]

            extN = vertex_normals.repeat(batch_size, 1)
            def proj_back(z):
                projection = torch.einsum('ij,imj->im', extN, z)
                return z- (projection * extN).unsqueeze(1)

            cur_k = self.cur_k[:repeat_times*e_num]
            cur_k = proj_back(cur_k)
            broadcast = torch.stack([self.parallel_transport_masked(repeat_times,cur_k[edge_index[0]][:,i,:]) for i in range(3)],dim=-1)[cur_broadcastmaps]
            for _ in range(3): #broadcast 3+1 times in total
                broadcast = torch.stack([self.parallel_transport_masked(repeat_times,broadcast[:,i,:]) for i in range(3)], dim=-1)[cur_broadcastmaps]

            g, pds = metrics.repeat(repeat_times,1,1)[edge_index[1]], PD.repeat(repeat_times,1,1)[edge_index[1]]
            a = torch.einsum('ijk,ikl->ijl',pds, broadcast/broadcast.norm()).transpose(1,2)
            b = torch.einsum('ijk,ik->ij',pds,pt)

            final_pt = torch.einsum('ijk,ij,ik->i',g,a[:,0],b)
            final_pt = final_pt.unsqueeze(-1)*broadcast[:,1]
            
            cur_attn = self.cur_attn[:repeat_times*e_num]

            broadcast = torch.stack([self.parallel_transport_masked(repeat_times,cur_attn[edge_index[0]][:,i,0,:]) for i in range(1)],dim=-1)[cur_broadcastmaps]
            for _ in range(3): #broadcast 3+1 times in total
                broadcast = torch.stack([self.parallel_transport_masked(repeat_times,broadcast[:,i,:]) for i in range(1)], dim=-1)[cur_broadcastmaps]
            
            a1 = torch.einsum('ijk,ikl->ijl',pds, broadcast/broadcast.norm()).transpose(1,2)

            broadcast = torch.stack([self.parallel_transport_masked(repeat_times,cur_attn[edge_index[0]][:,i,1,:]) for i in range(1)],dim=-1)[cur_broadcastmaps]
            for _ in range(3): #broadcast 3+1 times in total
                broadcast = torch.stack([self.parallel_transport_masked(repeat_times,broadcast[:,i,:]) for i in range(1)], dim=-1)[cur_broadcastmaps]
            a2 = torch.einsum('ijk,ikl->ijl',pds, broadcast/broadcast.norm()).transpose(1,2)            

            a1 = a1/a1.norm()
            a2 = a2/a2.norm()

            attn = torch.stack([a1,a2], dim=-2).view(-1,2,2)

            b1 = torch.einsum('ijk,ik->ij',pds,x_i)
            b2 = torch.einsum('ijk,ik->ij',pds,pt)
        
            alpha1, alpha2 = torch.einsum('ijk,ij,ik->i',g,b1, attn[:,0]), torch.einsum('ijk,ij,ik->i',g,b2, attn[:,1])
            alpha = self.linear1(torch.stack([alpha1,alpha2], dim=-1)).flatten()
            
            alpha = softmax(alpha, edge_index[0])

            return final_pt * alpha.unsqueeze(-1) 
        
        elif self.isYoung:
            final_pt = self.youngLinearTransform(pt)

            alpha = (torch.cat([x_i, pt], dim=-1)  * self.att.flatten().unsqueeze(1).repeat(1,3).view(self.heads,-1)).sum(dim=-1)
            alpha = F.dropout(alpha, p=.2, training=self.istraining)
            alpha = F.relu(alpha)
            alpha = softmax(alpha, edge_index[0])
            return final_pt * alpha.unsqueeze(-1)        
        
        else:
            cur_broadcastmaps = self.cur_broadcastmaps[:repeat_times*e_num]
            cur_k = self.cur_k[:repeat_times*e_num]
            broadcast = torch.stack([self.parallel_transport_masked(repeat_times,
                                    cur_k[edge_index[0]][:,i,:]) for i in range(3)],dim=-1)
            broadcast = broadcast[cur_broadcastmaps]
            for _ in range(3): #broadcast 3+1 times in total
                broadcast = torch.stack([self.parallel_transport_masked(repeat_times,
                                                                broadcast[:,i,:]) for i in range(3)], dim=-1)
                broadcast = broadcast[cur_broadcastmaps]
            
            g, pds = metrics.repeat(repeat_times,1,1)[edge_index[1]], PD.repeat(repeat_times,1,1)[edge_index[1]]
            a = torch.einsum('ijk,ikl->ijl',pds, broadcast/broadcast.norm()).transpose(1,2)
            b = torch.einsum('ijk,ik->ij',pds,pt)
            final_pt = torch.einsum('ijk,imj,ik->im',g,a,b)

    def update(self, aggr_out, x,edge_index=None):
        repeat_times = edge_index.shape[1]//(edge_attrs.shape[0]) 
        cur_broadcastmaps = self.cur_broadcastmaps[:repeat_times*e_num]

        extN = vertex_normals.repeat(batch_size, 1)
        def proj_back(z):
            projection = torch.einsum('ij,imj->im', extN, z)
            return z- (projection * extN).unsqueeze(1)
        
        cur_k = self.cur_k2[:repeat_times*e_num]
        
        cur_k = proj_back(cur_k)
        broadcast = torch.stack([self.parallel_transport_masked(repeat_times,cur_k[edge_index[0]][:,i,:]) for i in range(3)],dim=-1)[cur_broadcastmaps]
        for _ in range(3): #broadcast 3+1 times in total
            broadcast = torch.stack([self.parallel_transport_masked(repeat_times,broadcast[:,i,:]) for i in range(3)], dim=-1)[cur_broadcastmaps]
        
        x = x[edge_index[1]]

        g, pds = metrics.repeat(repeat_times,1,1)[edge_index[1]], PD.repeat(repeat_times,1,1)[edge_index[1]]
        a = torch.einsum('ijk,ikl->ijl',pds, broadcast/broadcast.norm()).transpose(1,2)
        b = torch.einsum('ijk,ik->ij',pds,x)

        final = torch.einsum('ijk,ij,ik->i',g,a[:,0],b)
        final = final.unsqueeze(-1)*broadcast[:,1]
        final = scatter(final, edge_index[1], dim=0, dim_size=v_num*repeat_times, reduce='mean')

        extN = vertex_normals.repeat(final.shape[0]//v_num, 1)
        final =  final- (torch.einsum('ij,ij->i', extN, final).unsqueeze(-1) * extN)

        the_end = True
        if the_end: 
            b = torch.einsum('ijk,ik->ij',PD.repeat(repeat_times,1,1),self.c[0]*final+self.c[1]*aggr_out)
            final = torch.einsum('ijk,ij,ik->i',g,a[:,2],b[edge_index[1]])
            final = scatter(final, edge_index[1], dim=0, dim_size=v_num*repeat_times, reduce='mean')
            return final.unsqueeze(-1).repeat(1,3)

        return self.c[0]*final+self.c[1]*aggr_out
    

class CURVGT(torch.nn.Module):
    def __init__(self, input=2, hidden_channels=3, heads=1,model_type='mlp'):
        super(CURVGT, self).__init__()
        self.model_type = model_type
        heads = 3
        self.layernorm1 = nn.LayerNorm([v_num, 1])
        self.layernorm11 = nn.LayerNorm([v_num, 1])
        self.layernorm12 = nn.LayerNorm([v_num, 1])
            
        self.layernorm2 = nn.LayerNorm([v_num])
        self.layernorm3 = nn.LayerNorm([v_num])
        self.layernorm4 = nn.LayerNorm([v_num])
        self.layernorm5 = nn.LayerNorm([v_num])

        if model_type == 'CURVGT' or model_type == 'YoungCURVGT':
            if model_type == 'CURVGT':
                self.CURVGT = CurvAwareAttn(hidden_channels, 1,treeRoots=tree_roots, treeMasks=tree_mask,broadcastmaps=tree_brodcastmap).cuda(device)
                self.CURVGT1 = CurvAwareAttn(hidden_channels, 1,treeRoots=tree_roots1, treeMasks=tree_mask1,broadcastmaps=tree_brodcastmap1).cuda(device)
                self.CURVGT2 = CurvAwareAttn(hidden_channels, 1,treeRoots=tree_roots2, treeMasks=tree_mask2,broadcastmaps=tree_brodcastmap2).cuda(device)
            else:
                self.CURVGT = CurvAwareAttn(hidden_channels, 1,treeRoots=tree_roots, treeMasks=tree_mask,broadcastmaps=tree_brodcastmap,isYoung=True).cuda(device)
                self.CURVGT1 = CurvAwareAttn(hidden_channels, 1,treeRoots=tree_roots1, treeMasks=tree_mask1,broadcastmaps=tree_brodcastmap1,isYoung=True).cuda(device)
                self.CURVGT2 = CurvAwareAttn(hidden_channels, 1,treeRoots=tree_roots2, treeMasks=tree_mask2,broadcastmaps=tree_brodcastmap2,isYoung=True).cuda(device)       
            self.linear = Linear(hidden_channels, 1).cuda(device)
            self.linear1 = Linear(hidden_channels, 1).cuda(device)
            self.linear2 = Linear(hidden_channels, 1).cuda(device)
            self.gcn1forgeo = GCNConv(5, hidden_channels*heads)
            self.gcn2forgeo = GCNConv(hidden_channels*heads,9)

            self.scaling = nn.Parameter(torch.randn(1,1))

            if datasets_function == "datasets_3":   # heterogenous heat equation
                self.gat1 = GATConv(6, hidden_channels, heads=heads, concat=True)
                self.gat2 = GATConv(hidden_channels * heads, 9, heads=heads, concat=False)
                self.gtforCURVGT1 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=hidden_channels*heads)
                self.gtforCURVGT2 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=1)
                self.final_linear = Linear(2,1)
                self.final_mlp1 = MLP(9, 5, 1)


            elif datasets_function == "datasets_wave":   # wave equation
                self.gat1 = GATConv(3, hidden_channels, heads=heads, concat=True)
                self.gat2 = GATConv(hidden_channels * heads, 9, heads=heads, concat=False)
                self.gat3 = GATConv(5, hidden_channels, heads=heads, concat=True)
                self.gat4 = GATConv(hidden_channels * heads, 9, heads=heads, concat=False)
                self.gtforCURVGT1 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=hidden_channels*heads)
                self.gtforCURVGT2 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=1)
                self.gtforCURVGT3 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=hidden_channels*heads)
                self.gtforCURVGT4 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=1)
            
                self.final_linear2 = Linear(13,1)
                self.final_linear3 = Linear(13,1)
                self.final_linear4 = Linear(4,1)

                self.final_mlp1 = MLP(13, 5, 1)
                self.final_mlp2 = MLP(13, 5, 1)

                self.final_linear = Linear(13,1)

            elif datasets_function == "datasets_plbo":
                self.gat1 = GATConv(5, hidden_channels, heads=heads, concat=True)
                self.gat2 = GATConv(hidden_channels * heads, 9, heads=heads, concat=False)

                self.gtforCURVGT1 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=hidden_channels*heads)
                self.gtforCURVGT2 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=1)

                self.final_mlp1 = MLP(9, 5, 1)

                self.final_linear = Linear(2,1)

            else:
                self.gat1 = GATConv(5, hidden_channels, heads=heads, concat=True)
                self.gat2 = GATConv(hidden_channels * heads, 9, heads=heads, concat=False)
                self.gtforCURVGT1 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=hidden_channels*heads)
                self.gtforCURVGT2 = TransformerLayer_GT(hidden_channels*heads, heads=1, hidden_dim=hidden_channels, output_dim=1)

                self.final_mlp1 = MLP(9, 5, 1)

                self.final_linear = Linear(2,1)
            
            self.extN = vertex_normals.repeat(batch_size, 1)
            self.extG, self.extM = torch.block_diag(*[G for _ in range(batch_size)]), torch.block_diag(
                *[M for _ in range(batch_size)])
            self.ext_eigenvecs = eigenvecs.repeat(1, batch_size)
            self.ext_mask, self.ext_degree = mask.repeat(batch_size, 1), degree.repeat(batch_size, 1)

            self.ext_vertices_to_voronoi_tensor = vertices_to_voronoi_tensor.repeat(batch_size, 1)
            incre_mask = (self.ext_vertices_to_voronoi_tensor != -1).long()
            incre = torch.concat([torch.ones_like(vertices_to_voronoi_tensor) * vertices_to_voronoi_tensor.shape[0] * i for i in range(batch_size)], dim=0).long()
            self.ext_vertices_to_voronoi_tensor += incre_mask*incre
        elif model_type == 'SIMPLE_GCN':
            hidden_channels = 16
            self.gcn1 = GCNConv(input, hidden_channels)
            self.gcn2 = GCNConv(hidden_channels, hidden_channels)
            self.blocks = nn.ModuleList([BasicBlock(hidden_channels, hidden_channels) for i in range(3)])
            self.fc = nn.Linear(hidden_channels, 1)

        elif model_type == 'RESNET':
            num_blocks=32
            self.blocks = nn.ModuleList()
            out_channels = 1
            hidden_channels = hidden_channels
            self.input_liner = nn.Linear(input, hidden_channels)
            for _ in range(num_blocks-1):
                self.blocks.append(BasicBlock(hidden_channels, hidden_channels))
            self.fc = nn.Linear(hidden_channels, out_channels)
   
        elif model_type == 'NaiveCURVGT':
            heads = 3
            self.gat1 = GATConv(input+1, hidden_channels, heads=heads, concat=True)
            self.gat2 = GATConv(hidden_channels * heads, 9, heads=heads, concat=False)
            
            if datasets_function == "datasets_3":  
                self.final_linear = Linear(12,1)
            elif datasets_function == "datasets_wave": 
                self.final_linear = Linear(13,1)
            else:
                self.final_linear = Linear(11,1)

        print(model_type)

    def forward(self, data, istraining=True):
        x, edge_index = data.x, data.edge_index
        
        # extend in consistency with batchsize
        current_batchsize = x.shape[0] // vertex_normals.shape[0]
        if self.model_type == 'CURVGT' or self.model_type == 'YoungCURVGT':
            extN = vertex_normals.repeat(current_batchsize, 1)
            extG, extM = torch.block_diag(*[G for _ in range(current_batchsize)]), torch.block_diag(*[M for _ in range(current_batchsize)])
            ext_mask, ext_degree = mask.repeat(current_batchsize, 1), degree.repeat(current_batchsize, 1)
            
            v_nums,f_nums = v_num,f_num*3

            extN,extG,extM,ext_eigenvecs,ext_mask,ext_degree = (self.extN[:current_batchsize*v_nums],
                                                                self.extG[:current_batchsize*f_nums,:current_batchsize*v_nums],
                                                                self.extM[:current_batchsize*v_nums,:current_batchsize*v_nums],
                                                                self.ext_eigenvecs[:,:current_batchsize*v_nums],
                                                                self.ext_mask[:current_batchsize*v_nums],
                                                                self.ext_degree[:current_batchsize*v_nums])
            ext_vertices_to_voronoi_tensor = self.ext_vertices_to_voronoi_tensor[:current_batchsize*v_nums]

            ################## without modes
            if datasets_function == 'datasets_wave':
                vector_fields = (extG @  x[:,0]).view(-1, 3)
                vector_fields = torch.sum(vector_fields[ext_vertices_to_voronoi_tensor, :] * ext_mask.unsqueeze(2).repeat(1, 1, 3), dim=1)
                vector_fields /= ext_degree.repeat(1, 3)

                V0 = self.CURVGT(vector_fields.view(-1, 3), edge_index, istraining)
                V0 = self.linear(V0).view(-1, 1)
                V0 = self.layernorm1(V0.view(current_batchsize,-1,1)).view_as(V0)
                V0 = F.relu(V0)

                vector_fields = (extG @  x[:, 2]).view(-1, 3)
                vector_fields = torch.sum(vector_fields[ext_vertices_to_voronoi_tensor, :] * ext_mask.unsqueeze(2).repeat(1, 1, 3), dim=1)
                vector_fields /= ext_degree.repeat(1, 3)

                V1 = self.CURVGT1(vector_fields.view(-1, 3), edge_index, istraining)
                V1 = self.linear1(V1).view(-1, 1)
                V1 = self.layernorm11(V1.view(current_batchsize,-1,1)).view_as(V1)
                V1 = F.relu(V1)

                V2 = self.CURVGT2(vector_fields.view(-1, 3), edge_index, istraining)
                V2 = self.linear2(V2).view(-1, 1)
                V2 = self.layernorm12(V2.view(current_batchsize,-1,1)).view_as(V2)
                V2 = F.relu(V2)


            else:
                vector_fields = (extG @  x[:,0]).view(-1, 3)
                vector_fields = torch.sum(vector_fields[ext_vertices_to_voronoi_tensor, :] * ext_mask.unsqueeze(2).repeat(1, 1, 3), dim=1)
                vector_fields /= ext_degree.repeat(1, 3)

                V0 = self.CURVGT(vector_fields.view(-1, 3), edge_index, istraining)
                V0 = self.linear(V0).view(-1, 1)
                V0 = self.layernorm1(V0.view(current_batchsize,-1,1)).view_as(V0)

                V1 = self.CURVGT1(vector_fields.view(-1, 3), edge_index, istraining)
                V1 = self.linear1(V1).view(-1, 1)
                V1 = self.layernorm11(V1.view(current_batchsize,-1,1)).view_as(V1)

                V2 = self.CURVGT2(vector_fields.view(-1, 3), edge_index, istraining)
                V2 = self.linear2(V2).view(-1, 1)
                V2 = self.layernorm12(V2.view(current_batchsize,-1,1)).view_as(V2)

            if datasets_function == 'datasets_wave':
                y = torch.concat([data.x[:,0:2],V0], dim=1)
                y = self.gat1(y, edge_index)
                y = self.gtforCURVGT1(y)
                y = self.layernorm2(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y = F.relu(y)
                y = self.gat2(y, edge_index)
                y = self.gtforCURVGT2(y)
                y = self.layernorm3(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y1 = self.final_mlp1(torch.concat([x, y], dim=1))

                y = torch.concat([data.x[:,2:4],V1,V1,V2], dim=1)
                y = self.gat3(y, edge_index)
                y = self.gtforCURVGT3(y)
                y = self.layernorm4(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y = F.relu(y)
                y = self.gat4(y, edge_index)
                y = self.gtforCURVGT4(y)
                y = self.layernorm5(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y2 = self.final_mlp2(torch.concat([x, y], dim=1))
                return self.final_linear4(torch.concat([x[:,0:1],x[:,2:3], y1,y2], dim=1))       
            elif  datasets_function == 'datasets_plbo':            
                y = torch.concat([x,V0,V1,V2], dim=1)
                y = self.gat1(y, edge_index)
                y = self.gtforCURVGT1(y)
                y = self.layernorm2(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y = F.relu(y)

                y = self.gat2(y, edge_index)

                y = self.gtforCURVGT2(y)
                y = self.layernorm3(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y = self.final_mlp1(y)
                return self.final_linear(torch.concat([x[:,0:1], y], dim=1))                                
            else:
                y = torch.concat([x,V0,V1,V2], dim=1)
                y = self.gat1(y, edge_index)
                y = self.gtforCURVGT1(y)
                y = self.layernorm2(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y = F.relu(y)

                y = self.gat2(y, edge_index)

                y = self.gtforCURVGT2(y)
                y = self.layernorm3(y.view(current_batchsize,y.shape[0]//current_batchsize,-1).transpose(1,2)).transpose(1,2)
                y = y.reshape(-1, y.shape[2])
                y = self.final_mlp1(y)

                return self.final_linear(torch.concat([x[:,0:1], y], dim=1))



def one_go(model_type=f"{model_name}"):
    model = CURVGT(model_type=model_type).to(device)
    best_model = model.state_dict()
    best_loss = 1e9

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    def L2Criterion(y,y_pred, mu=0):
        current_batchsize = y.shape[0]//M.shape[0]
        weighted_mse_loss = (y-y_pred).transpose(1,0) @ torch.block_diag(*[M]*current_batchsize) @ (y-y_pred)  
        boundary_loss = torch.sum(boundary_mask.repeat(current_batchsize)*(y-y_pred).flatten()**2)
        return weighted_mse_loss + mu * boundary_loss
    def L2Criterion2(y,y_pred,mu=0):
        current_batchsize = y.shape[0]//M.shape[0]
        return  (y-y_pred).transpose(1,0) @ torch.block_diag(*[M]*current_batchsize) @ (y-y_pred)  
    
    def H1Criterion(y,y_pred,mu=0):
        current_batchsize = y.shape[0]//M.shape[0]
        extG, extM = torch.block_diag(*[G for _ in range(current_batchsize)]), torch.block_diag(*[M for _ in range(current_batchsize)])

        ext_vertices_to_voronoi_tensor = vertices_to_voronoi_tensor.repeat(current_batchsize, 1)
        incre_mask = (ext_vertices_to_voronoi_tensor != -1).long()
        incre = torch.concat([torch.ones_like(vertices_to_voronoi_tensor) * vertices_to_voronoi_tensor.shape[0] * i for i in range(current_batchsize)], dim=0).long()
        ext_vertices_to_voronoi_tensor += incre_mask*incre
        
        delta_y = y-y_pred
        delta = (extG@delta_y).view(-1,3)
        delta = delta[ext_vertices_to_voronoi_tensor,:]
        delta = torch.mean(delta,dim=1)
        delta = torch.sum(delta*delta,dim=-1).unsqueeze(1)

        h1_loss = torch.sum(extM @ delta).view(-1,1) 
        l2_loss = delta_y.transpose(1,0) @ extM @ delta_y   
        return h1_loss+l2_loss
    
    scheduler = StepLR(optimizer, step_size=50, gamma=0.95) 

    def train(collect_data=False):
        model.train()
        total_loss = 0
        graph_cnt = 0
                    
        for data in train_loader:
            data = data.to(device) 
            if datasets_function == "datasets_3": 
                data.x, data.y = data.x[:, 0:3].float(), data.y.float()
                data.x[:,1] /= torch.max(torch.abs(data.x[:,1]))
            elif datasets_function == "datasets_wave":
                data.x, data.y = data.x[:, 0:4].float(), data.y.float()
                data.x[:,1] /= torch.max(torch.abs(data.x[:,1]))
                data.x[:,3] /= torch.max(torch.abs(data.x[:,3]))
            else:
                data.x, data.y = data.x[:, 0:2].float(), data.y.float()
            if torch.isnan(data.x).any():
                print([torch.isnan(data.x[:,i]).any() for i in range(3)])
            optimizer.zero_grad()  
            output = model(data, istraining=True) 
            graph_cnt += output.shape[0]//M.shape[0]

            loss = L2Criterion2(output, data.y.unsqueeze(1), mu) 
            loss.backward()        
            optimizer.step()      

            total_loss += loss.item()

        return total_loss / graph_cnt

    def test_H1():
        model.eval()
        total_loss = 0
        graph_cnt = 0
        with torch.no_grad():  
            onlyone = False
            for data in test_loader:
                data = data.to(device)
                if datasets_function == "datasets_3":
                    data.x, data.y = data.x[:, 0:3].float(), data.y.float()
                    data.x[:,1] /= torch.max(torch.abs(data.x[:,1]))
                elif datasets_function == "datasets_wave": 
                    data.x, data.y = data.x[:, 0:4].float(), data.y.float()
                    data.x[:,1] /= torch.max(torch.abs(data.x[:,1]))
                    data.x[:,3] /= torch.max(torch.abs(data.x[:,3]))
                else:
                    data.x, data.y = data.x[:, 0:2].float(), data.y.float()
                output = model(data, istraining=False)
                graph_cnt += output.shape[0]//M.shape[0]
                loss = H1Criterion(output, data.y.unsqueeze(1))  
                total_loss += loss.item()
                if onlyone:
                    onlyone = False
                    outputAVector(data.y.flatten(), 'groundtruth.txt')
                    outputAVector(output.flatten(), 'novel.txt')
        return total_loss / graph_cnt

    def test_L2():
        model.eval()
        total_loss = 0
        graph_cnt = 0
        with torch.no_grad(): 
            onlyone = False
            for data in test_loader:
                data = data.to(device)
                if datasets_function == "datasets_3": 
                    data.x, data.y = data.x[:, 0:3].float(), data.y.float()
                    data.x[:,1] /= torch.max(torch.abs(data.x[:,1]))
                elif datasets_function == "datasets_wave": 
                    data.x, data.y = data.x[:, 0:4].float(), data.y.float()
                    data.x[:,1] /= torch.max(torch.abs(data.x[:,1]))
                    data.x[:,3] /= torch.max(torch.abs(data.x[:,3]))
                else:
                    data.x, data.y = data.x[:, 0:2].float(), data.y.float()
                output = model(data, istraining=False)
                graph_cnt += output.shape[0]//M.shape[0]
                loss = L2Criterion2(output, data.y.unsqueeze(1))  
                total_loss += loss.item()
                if onlyone:
                    onlyone = False
                    outputAVector(data.y.flatten(), 'groundtruth.txt')
                    outputAVector(output.flatten(), 'novel.txt')
        return total_loss / graph_cnt

    def outputAVector(tensor,path):
        np.savetxt(path, tensor.cpu().numpy(), fmt='%.6f')

    train_losses = []
    train_epoch = 300
    best_epoch = -1

    for epoch in range(0, train_epoch):
        train_loss = train()

        if torch.isnan(torch.tensor(train_loss)).any():
            raise ValueError("NAN")
    
        scheduler.step()
    
        train_losses.append(train_loss)
    
        if train_loss < best_loss:
            best_loss = train_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'./dataset/{datasets_function}/best_model/best_model_{model_name}.pt')

        if epoch % 1 == 0:
            test_loss_H1, test_loss_L2 = test_H1(), test_L2()
            print(f'({model_name})Epoch {epoch}, Train Loss: {train_loss:.8f}, Test Loss(H1): {test_loss_H1:.8f}, Test Loss(L2): {test_loss_L2:.8f}')
    
    best_model = torch.load(f'./dataset/{datasets_function}/best_model/best_model_{model_name}.pt')
    model.load_state_dict(best_model)
    print(f"-----------------({model_name})(best_epoch, best_train_loss):", best_epoch, best_loss)
    
    final_test_loss_H1 = test_H1()
    final_test_loss_L2 = test_L2()

    return torch.tensor(train_losses).view(1, -1), best_model, final_test_loss_H1, final_test_loss_L2


def evaluate_model():
    # Load the best model saved during training
    model = CURVGT(model_type=model_name).to(device)
    best_model = torch.load(f'./dataset/{datasets_function}/best_model/best_model_{model_name}.pt')  # Load saved model
    model.load_state_dict(best_model)
    model.eval()  # Set model to evaluation mode

    # Define loss functions (same as in one_go function)
    def L2Criterion(y, y_pred, mu=0):
        current_batchsize = y.shape[0] // M.shape[0]
        weighted_mse_loss = (y - y_pred).transpose(1, 0) @ torch.block_diag(*[M] * current_batchsize) @ (y - y_pred)
        boundary_loss = torch.sum(boundary_mask.repeat(current_batchsize) * (y - y_pred).flatten() ** 2)
        return weighted_mse_loss + mu * boundary_loss

    def H1Criterion(y, y_pred, mu=0):
        current_batchsize = y.shape[0] // M.shape[0]
        extG, extM = torch.block_diag(*[G for _ in range(current_batchsize)]), torch.block_diag(*[M for _ in range(current_batchsize)])

        ext_vertices_to_voronoi_tensor = vertices_to_voronoi_tensor.repeat(current_batchsize, 1)
        incre_mask = (ext_vertices_to_voronoi_tensor != -1).long()
        incre = torch.concat([torch.ones_like(vertices_to_voronoi_tensor) * vertices_to_voronoi_tensor.shape[0] * i for i in range(current_batchsize)], dim=0).long()
        ext_vertices_to_voronoi_tensor += incre_mask * incre
        
        delta_y = y - y_pred
        delta = (extG @ delta_y).view(-1, 3)
        delta = delta[ext_vertices_to_voronoi_tensor, :]
        delta = torch.mean(delta, dim=1)
        delta = torch.sum(delta * delta, dim=-1).unsqueeze(1)

        h1_loss = torch.sum(extM @ delta).view(-1, 1)
        l2_loss = delta_y.transpose(1, 0) @ extM @ delta_y
        return h1_loss + l2_loss

    # Initialize test losses
    total_loss_H1 = 0
    total_loss_L2 = 0
    graph_cnt = 0
    
    with torch.no_grad():  # Disable gradient computation during evaluation
        for data in test_loader:
            data = data.to(device)
            
            if datasets_function == "datasets_3":  # For anisotropic heat equation
                data.x, data.y = data.x[:, 0:3].float(), data.y.float()
                data.x[:, 1] /= torch.max(torch.abs(data.x[:, 1]))
            elif datasets_function == "datasets_wave":  # For wave equation
                data.x, data.y = data.x[:, 0:4].float(), data.y.float()
                data.x[:, 1] /= torch.max(torch.abs(data.x[:, 1]))
                data.x[:, 3] /= torch.max(torch.abs(data.x[:, 3]))
            else:  # Default case
                data.x, data.y = data.x[:, 0:2].float(), data.y.float()

            output = model(data, istraining=False)  # Get model output
            
            # Compute losses
            loss_H1 = H1Criterion(output, data.y.unsqueeze(1))
            loss_L2 = L2Criterion(output, data.y.unsqueeze(1))

            total_loss_H1 += loss_H1.item()
            total_loss_L2 += loss_L2.item()
            graph_cnt += output.shape[0] // M.shape[0]

    # Calculate average losses
    avg_loss_H1 = total_loss_H1 / graph_cnt
    avg_loss_L2 = total_loss_L2 / graph_cnt

    print(f"Evaluation Results for {model_name}:")
    print(f"Avg Test Loss (H1): {avg_loss_H1:.8f}")
    print(f"Avg Test Loss (L2): {avg_loss_L2:.8f}")

    return avg_loss_H1, avg_loss_L2  # Return evaluation results

if test_pattern == 1:
    train_loss_tensor, best_model, final_test_loss_H1, final_test_loss_L2 = one_go()
    best_train_loss = min(train_loss_tensor.flatten())
    print(f"\n-----------------({model_name})Initial Best Test Loss(H1): {final_test_loss_H1:.8f}, Best Test Loss(L2): {final_test_loss_L2:.8f}")
    torch.save(best_model, f'./dataset/{datasets_function}/best_model/best_model_{model_name}.pt')
    print(f"({model_name})Best Model Saved with Train Loss: {best_train_loss:.8f}")
elif test_pattern == 0:
    final_test_loss_H1, final_test_loss_L2 = evaluate_model()
    print(f"Loaded model {model_name}. Test Loss (H1): {final_test_loss_H1:.8f}, Test Loss (L2): {final_test_loss_L2:.8f}")
else:
    print("Invalid test_pattern value. Please use 0 or 1.")    