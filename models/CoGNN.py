import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch.nn import Module, Dropout, LayerNorm, Identity
import torch.nn.functional as F
from typing import Tuple
import numpy as np

from helpers.classes import GumbelArgs, EnvArgs, ActionNetArgs, Pool, DataSetEncoders
from models.temp import TempSoftPlus
from models.action import ActionNet


class CoGNN(Module):
    def __init__(self, gumbel_args: GumbelArgs, env_args: EnvArgs, action_args: ActionNetArgs, pool: Pool):
        super(CoGNN, self).__init__()
        self.env_args = env_args
        self.learn_temp = gumbel_args.learn_temp
        if gumbel_args.learn_temp:
            self.temp_model = TempSoftPlus(gumbel_args=gumbel_args, env_dim=env_args.env_dim)
        self.temp = gumbel_args.temp

        self.num_layers = env_args.num_layers
        self.env_net = env_args.load_net()
        self.use_encoders = env_args.dataset_encoders.use_encoders()

        layer_norm_cls = LayerNorm if env_args.layer_norm else Identity
        self.hidden_layer_norm = layer_norm_cls(env_args.env_dim)
        self.skip = env_args.skip
        self.dropout = Dropout(p=env_args.dropout)
        self.drop_ratio = env_args.dropout
        self.act = env_args.act_type.get()
        self.in_act_net = ActionNet(action_args=action_args)
        self.out_act_net = ActionNet(action_args=action_args)

        # Encoder types
        self.dataset_encoder = env_args.dataset_encoders
        self.env_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=env_args.env_dim, model_type=env_args.model_type)
        self.act_bond_encoder = self.dataset_encoder.edge_encoder(emb_dim=action_args.hidden_dim, model_type=action_args.model_type)

        # Pooling function to generate whole-graph embeddings
        self.pooling = pool.get()

    def forward(self, x: Tensor, edge_index: Adj, pestat, edge_attr: OptTensor = None, batch: OptTensor = None,
                edge_ratio_node_mask: OptTensor = None) -> Tuple[Tensor, Tensor]:
        result = 0

        calc_stats = edge_ratio_node_mask is not None
        if calc_stats:
            edge_ratio_edge_mask = edge_ratio_node_mask[edge_index[0]] & edge_ratio_node_mask[edge_index[1]]
            edge_ratio_list = []

        # bond encode
        if edge_attr is None or self.env_bond_encoder is None:
            env_edge_embedding = None
        else:
            env_edge_embedding = self.env_bond_encoder(edge_attr)
        if edge_attr is None or self.act_bond_encoder is None:
            act_edge_embedding = None
        else:
            act_edge_embedding = self.act_bond_encoder(edge_attr)

        # node encode  
        x = self.env_net[0](x, pestat)  # (N, F) encoder
        if not self.use_encoders:
            x = self.dropout(x)
            x = self.act(x)

        for gnn_idx in range(self.num_layers):
            x = self.hidden_layer_norm(x)

            # action
            in_logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                        act_edge_attr=act_edge_embedding)  # (N, 2)
            out_logits = self.out_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
                                          act_edge_attr=act_edge_embedding)  # (N, 2)

            temp = self.temp_model(x=x, edge_index=edge_index,
                                   edge_attr=env_edge_embedding) if self.learn_temp else self.temp
            # in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=True)
            # out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=True)

            in_probs = torch.cat((F.gumbel_softmax(logits=in_logits[:,0:2], tau=temp, hard=True)[:,0:1], F.gumbel_softmax(logits=in_logits[:,2:4], tau=temp, hard=True)[:,0:1]), dim=1)
            out_probs = torch.cat((F.gumbel_softmax(logits=out_logits[:,0:2], tau=temp, hard=True)[:,0:1], F.gumbel_softmax(logits=out_logits[:,2:4], tau=temp, hard=True)[:,0:1]), dim=1)

            # logits = self.in_act_net(x=x, edge_index=edge_index, env_edge_attr=env_edge_embedding,
            #                             act_edge_attr=act_edge_embedding)  # (N, 4)
            # probs = F.gumbel_softmax(logits=logits, tau=temp, hard=True)
            # print(logits.shape, probs.shape)
            # 0 - isolate, 1 - listen, 2 - broadcast, 3 - standard, 4 - distribute
            # in_probs = probs[:, 1] + probs[:, 3]
            # out_probs = probs[:, 2] + probs[:, 3]

            # new_edge_index = self.create_new_edges(edge_index=edge_index, distribute=probs[:,4])

            # edge_weight = self.create_edge_weight(edge_index=new_edge_index,
                                                  # keep_in_prob=in_probs, keep_out_prob=out_probs)

            # in_probs = F.gumbel_softmax(logits=in_logits, tau=temp, hard=False)
            # out_probs = F.gumbel_softmax(logits=out_logits, tau=temp, hard=False)
            # in_probs = F.softmax(input=in_logits/temp)
            # out_probs = F.softmax(input=in_logits/temp)
            # print('!!!!!!!!!!!!!!!!!!!!!!!')
            # edge_weight = self.create_edge_weight(edge_index=edge_index,
            #                                       keep_in_prob=in_probs[:, 0], keep_out_prob=out_probs[:, 0])
            edge_weight = self.create_edge_weight(edge_index=edge_index,
                                                  keep_in_prob=in_probs, keep_out_prob=out_probs)

            # environment
            out = self.env_net[1 + gnn_idx](x=x, edge_index=edge_index, edge_weight=edge_weight,
                                            edge_attr=env_edge_embedding)
            out = self.dropout(out)
            out = self.act(out)

            if calc_stats:
                edge_ratio = edge_weight[edge_ratio_edge_mask].sum() / edge_weight[edge_ratio_edge_mask].shape[0]
                edge_ratio_list.append(edge_ratio.item())

            if self.skip:
                x = x + out
            else:
                x = out

        x = self.hidden_layer_norm(x)
        x = self.pooling(x, batch=batch)
        x = self.env_net[-1](x)  # decoder
        result = result + x

        if calc_stats:
            edge_ratio_tensor = torch.tensor(edge_ratio_list, device=x.device)
        else:
            edge_ratio_tensor = -1 * torch.ones(size=(self.num_layers,), device=x.device)
        return result, edge_ratio_tensor

    def create_edge_weight(self, edge_index: Adj, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        # return edge_in_prob * edge_out_prob
        return (edge_in_prob * edge_out_prob).mean(dim=1)

    # def create_new_edges(self, edge_index: Adj, distribute: Tensor):
    #     from torch_geometric.utils import to_networkx, from_networkx
    #     import networkx as nx
    #     from typing import Tuple

    #     # # Convert to NetworkX for easier connected component analysis
    #     # if isinstance(edge_index, tuple):
    #     #     edge_index_tensor = edge_index[0]  # If it's a tuple, get the edge indices
    #     # else:
    #     edge_index_tensor = edge_index
            
    #     num_nodes = distribute.size(0)
        
    #     # Create NetworkX graph
    #     nx_graph = nx.Graph()
    #     nx_graph.add_nodes_from(range(num_nodes))
        
    #     # Add edges to the graph
    #     edges = edge_index_tensor.t().tolist()  # Convert to list of [source, target] pairs
    #     nx_graph.add_edges_from(edges)
        
    #     # Create a subgraph of distribute nodes
    #     distribute_nodes = torch.where(distribute == 1)[0].tolist()
    #     distribute_subgraph = nx_graph.subgraph(distribute_nodes)
        
    #     # Find connected components in the distribute subgraph
    #     connected_components = list(nx.connected_components(distribute_subgraph))
        
    #     # For each connected component, find adjacent non-distribute nodes
    #     new_edges = []
    #     for component in connected_components:
    #         # Get non-distribute neighbors of this component
    #         neighbors = set()
    #         for node in component:
    #             for neighbor in nx_graph.neighbors(node):
    #                 if neighbor not in distribute_nodes:  # If it's a non-distribute node
    #                     neighbors.add(neighbor)
                        
    #         # Connect all pairs of these neighbors
    #         neighbors = list(neighbors)
    #         for i in range(len(neighbors)):
    #             for j in range(i+1, len(neighbors)):
    #                 new_edges.append([neighbors[i], neighbors[j]])
    #                 new_edges.append([neighbors[j], neighbors[i]])  # Add both directions for directed graphs
        
    #     # If there are no new edges, return the original edge_index
    #     if not new_edges:
    #         return edge_index
        
    #     # Convert new_edges to tensor
    #     new_edges_tensor = torch.tensor(new_edges, dtype=edge_index_tensor.dtype, device=edge_index_tensor.device).t()
        
    #     # Combine with original edges
    #     combined_edges = torch.cat([edge_index_tensor, new_edges_tensor], dim=1)
        
    #     # Remove duplicate edges
    #     combined_edges = torch.unique(combined_edges, dim=1)
        
    #     # # If the original edge_index was a tuple with edge attributes, handle accordingly
    #     # if isinstance(edge_index, tuple) and len(edge_index) > 1:
    #     #     # Create new attributes for the new edges (zeros or ones depending on the use case)
    #     #     orig_num_edges = edge_index[0].size(1)
    #     #     new_num_edges = combined_edges.size(1) - orig_num_edges
            
    #     #     # Create new edge attributes (setting to ones as default, modify as needed)
    #     #     new_edge_attr = torch.ones((new_num_edges, edge_index[1].size(1)), 
    #     #                               dtype=edge_index[1].dtype, 
    #     #                               device=edge_index[1].device)
            
    #     #     # Combine with original edge attributes
    #     #     combined_attr = torch.cat([edge_index[1], new_edge_attr], dim=0)
            
    #     #     return (combined_edges, combined_attr)
        
    #     return combined_edges

    

    

    # def create_new_edges(self, edge_index: Adj, distribute: torch.Tensor, num_edges: int = 100) -> Adj:
    #     """
    #     Ultra-fast tensor-based implementation that adds random edges between nodes connected to distribute nodes.
    #     No uniqueness check is performed to maximize speed.
        
    #     Args:
    #         edge_index: Tensor of shape [2, num_edges] containing the edges of the graph
    #         distribute: Tensor of size n, where 1 indicates a distribute node and 0 indicates a non-distribute node
    #         num_edges: Number of new random edges to add (default: 100)
        
    #     Returns:
    #         Updated edge_index with random edges added
    #     """
    #     import torch
    #     from torch_geometric.typing import Adj

    #     # Extract edge indices if it's a tuple with edge attributes
    #     if isinstance(edge_index, tuple):
    #         edge_index_tensor = edge_index[0]
    #         has_edge_attr = True
    #         edge_attr = edge_index[1]
    #     else:
    #         edge_index_tensor = edge_index
    #         has_edge_attr = False
        
    #     device = edge_index_tensor.device
        
    #     # Get distribute nodes
    #     distribute_mask = distribute == 1
        
    #     # If no distribute nodes, return original edge_index
    #     if not torch.any(distribute_mask):
    #         return edge_index
        
    #     # Find neighbors of distribute nodes (tensor operations)
    #     # First, create masks for distribute source and target nodes
    #     src_is_distribute = distribute_mask[edge_index_tensor[0]]
    #     dst_is_distribute = distribute_mask[edge_index_tensor[1]]
        
    #     # Get edges where source is distribute and target is not
    #     edges_src_distribute = edge_index_tensor[:, src_is_distribute & ~dst_is_distribute]
    #     # Get edges where target is distribute and source is not
    #     edges_dst_distribute = edge_index_tensor[:, ~src_is_distribute & dst_is_distribute]
        
    #     # Extract non-distribute neighbors (target nodes where source is distribute)
    #     neighbors_from_src = edges_src_distribute[1]
    #     # Extract non-distribute neighbors (source nodes where target is distribute)
    #     neighbors_from_dst = edges_dst_distribute[0]
        
    #     # Combine all neighbors
    #     all_neighbors = torch.cat([neighbors_from_src, neighbors_from_dst])
        
    #     # Get unique neighbors
    #     unique_neighbors = torch.unique(all_neighbors)
        
    #     # If we don't have enough neighbors, return original edge_index
    #     if unique_neighbors.size(0) <= 1:
    #         return edge_index
        
    #     # Generate random indices for selecting neighbors
    #     num_neighbors = unique_neighbors.size(0)
        
    #     # Simple random sampling for source and destination nodes
    #     num_edges_to_create = min(num_edges, num_neighbors * (num_neighbors - 1) // 2)
        
    #     # Generate random indices for src and dst nodes
    #     # We'll sample twice as many as needed (for bidirectional edges)
    #     src_indices = torch.randint(0, num_neighbors, (num_edges_to_create,), device=device)
    #     dst_indices = torch.randint(0, num_neighbors, (num_edges_to_create,), device=device)
        
    #     # Make sure src and dst are different
    #     same_mask = src_indices == dst_indices
    #     while same_mask.any():
    #         # Re-sample for indices that are the same
    #         dst_indices[same_mask] = torch.randint(0, num_neighbors, (same_mask.sum(),), device=device)
    #         same_mask = src_indices == dst_indices
        
    #     # Convert indices to actual node IDs
    #     src_nodes = unique_neighbors[src_indices]
    #     dst_nodes = unique_neighbors[dst_indices]
        
    #     # Create new edges (both directions for undirected graph)
    #     new_src = torch.cat([src_nodes, dst_nodes])
    #     new_dst = torch.cat([dst_nodes, src_nodes])
    #     new_edges = torch.stack([new_src, new_dst], dim=0)
        
    #     # Combine with original edges
    #     combined_edges = torch.cat([edge_index_tensor, new_edges], dim=1)
        
    #     # If the original edge_index had edge attributes, handle them
    #     if has_edge_attr:
    #         # Create new edge attributes (all ones)
    #         new_attr = torch.ones((new_edges.size(1), edge_attr.size(1)), 
    #                             dtype=edge_attr.dtype, 
    #                             device=device)
            
    #         # Combine with original attributes
    #         combined_attr = torch.cat([edge_attr, new_attr], dim=0)
            
    #         return (combined_edges, combined_attr)
        
    #     return combined_edges