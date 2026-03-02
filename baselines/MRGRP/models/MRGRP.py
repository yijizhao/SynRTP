import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.mlp import mlp, mlp_res, QuantileMLP
from .layers.embedding import PointEmb, DenseMatEmb3
from .layers.PointAttention import PointAttention3
from .layers.utility_funcs import pad_roll, get_edge_features
from typing import Dict, Union
from typing import Optional,List,Tuple
from torch import Tensor


class MRGRP(nn.Module):

    def __init__(self, context_c_sizes, wb_c_sizes, pickup_c_sizes, flags, args, output_emb_size=256, batch_size=2048):

        super(MRGRP, self).__init__()

        self.args = args

        self.hidden_size = 256
        
        self.point_emb = PointEmb(context_c_sizes[:3], wb_c_sizes, pickup_c_sizes, output_emb_size, args)
        self.rp_point_unified_emb_mlp = mlp(295,[self.hidden_size, self.hidden_size])

        self.etr_rnn_cell = nn.LSTM(output_emb_size+3, self.hidden_size, batch_first=True)  # Assuming input_size and hidden_size are both 256
        self.route_rnn_cell = nn.LSTM(output_emb_size+1, self.hidden_size, batch_first=True)  # Assuming input_size and hidden_size are both 256
        self.add_online = 0
        
        if self.add_online == 1:
            self.route_rnn_cell_online = nn.LSTM(output_emb_size, self.hidden_size, batch_first=True)  # Assuming input_size and hidden_size are both 256
            self.online_rank_emb = nn.Embedding(12, 16)
            self.online_rp_emb = mlp(16+self.hidden_size,[self.hidden_size])
            self.new_query_mlp = mlp(self.hidden_size*3, [self.hidden_size*2])
        
        self.dense_mat_emb = DenseMatEmb3(hidden_size=self.hidden_size)
        self.point_attention = PointAttention3(hidden_size=self.hidden_size)
        self.fp_mlp = self.create_mlp(self.hidden_size, [64, 32, 1])  # Assuming the input size is 256
        self.etr_mlp = self.create_mlp(self.hidden_size, [64, 32])

        self.head_etr_pickup = mlp_res(32, 64, 32)
        self.head_etr_deliver = mlp_res(32, 64, 32)
        self.etr_pickup = mlp_res(32, 64, 32)
        self.etr_deliver = mlp_res(32, 64, 32)  

        self.etr_quantile = QuantileMLP(32, 9)
        self.etr_state_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.etr_state_c = nn.Parameter(torch.Tensor(self.hidden_size))
        self.rp_state_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.rp_state_c = nn.Parameter(torch.Tensor(self.hidden_size))
        self.online_state_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.online_state_c = nn.Parameter(torch.Tensor(self.hidden_size))

        self.graph_size = 25

        self.context_n_feature_len = int(flags.context_n_feature_len)  # [bs, n]
        self.context_c_feature_len = int(flags.context_c_feature_len)  # [bs, n]
        self.wb_c_feature_len = int(flags.wb_c_feature_len)
        self.wb_time_feature_len = int(flags.wb_time_feature_len)
        self.pickup_n_feature_len = int(flags.pickup_n_feature_len)
        self.pickup_c_feature_len = int(flags.pickup_c_feature_len)
        self.deliver_n_feature_len = int(flags.deliver_n_feature_len)
        self.da_n_feature_len = int(flags.da_n_feature_len)


        nn.init.uniform_(self.etr_state_h)
        nn.init.uniform_(self.etr_state_c)
        nn.init.uniform_(self.rp_state_h)
        nn.init.uniform_(self.rp_state_c)
        nn.init.uniform_(self.online_state_h)
        nn.init.uniform_(self.online_state_c)



    def forward(self,features):
        graph_size = self.graph_size
        context_n_feature_len = self.context_n_feature_len  # [bs, n]
        context_c_feature_len = self.context_c_feature_len  # [bs, n]
        wb_c_feature_len = self.wb_c_feature_len
        wb_time_feature_len = self.wb_time_feature_len
        pickup_n_feature_len = self.pickup_n_feature_len
        pickup_c_feature_len = self.pickup_c_feature_len
        deliver_n_feature_len = self.deliver_n_feature_len
        da_n_feature_len = self.da_n_feature_len
        if self.training:
            batch_size = features.shape[0]
            wb_info = features[:,1250:1300].view(batch_size, graph_size, 2).long() 
            wb_ids = wb_info[:, :, 0] 
            point_types = wb_info[:, :, 1]

            # Rider and path features
            context_c_features = features[:,3615:3618].long()  
            context_n_features = features[:,3618:3623] 

            # Order features
            wb_c_features = features[:,0:125].view(batch_size, graph_size, wb_c_feature_len).long()
            wb_time_features = features[:,1175:1250].view(batch_size, graph_size, wb_time_feature_len)

            # Node features
            pickup_c_features = features[:,125:250].view(batch_size, graph_size, pickup_c_feature_len).long()
            pickup_n_features = features[:,250:725].view(batch_size, graph_size, pickup_n_feature_len)
            deliver_n_features = features[:,725:1025].view(batch_size, graph_size, deliver_n_feature_len)
            da_n_features = features[:,1025:1150].view(batch_size, graph_size, da_n_feature_len)

            # Distance matrices
            geohash_dist_mat = features[:,2157:2886].view(batch_size, graph_size + 2, graph_size + 2)
            line_dist_mat = features[:,1428:2157].view(batch_size, graph_size + 2, graph_size + 2)
            angle_mat = features[:,2886:3615].view(batch_size, graph_size + 2, graph_size + 2)

            # Labels (only for training)
            labels = features[:,1300:1425].view(batch_size, graph_size, 5).long()
            route_labels = labels[:, :, 0]
            etr_labels = labels[:, :, 1]
            route_origin = labels[:, :, 2]
            time_origin = labels[:, :, 3]
            # Mask
            # batch_size = int(torch.tensor(batch_size_).item())
            point_masks = wb_ids <= 0 #[b,12]
            extend_point_masks = F.pad(point_masks, (2, 0), "constant", 0.0)
            point_shift_ids = pad_roll(wb_ids, shift=1, input_dim=2)
            point_shift_same_id = point_shift_ids == wb_ids
            point_shift_types = pad_roll(point_types, shift=1)
            point_shift_pickup = point_shift_types == -1
            current_point_masks = self.get_available_masks(point_masks, point_shift_same_id, point_shift_pickup)

            edge_fea = get_edge_features(line_dist_mat, wb_time_features, wb_info, extend_point_masks, self.args)

            # ETR Embedding
            etr_point_embedding, rp_point_embedding = self.point_emb(context_n_features, context_c_features[:, :3], wb_c_features,
                    pickup_n_features, pickup_c_features, deliver_n_features, da_n_features,
                    geohash_dist_mat, line_dist_mat, angle_mat, extend_point_masks, point_types, edge_fea)
            
                        
            etr_state = (self.etr_state_h.expand(batch_size, -1).contiguous(),
                    self.etr_state_c.expand(batch_size, -1).contiguous())
            route_state = (self.rp_state_h.expand(batch_size, -1).contiguous(),
                    self.rp_state_c.expand(batch_size, -1).contiguous())

            predict_logits = []
            predict_selections = []
            predict_etr = []

            label_selections = []  
            label_etr = []

            prev_selection = torch.tensor([-10], dtype=torch.long).to(etr_point_embedding.device)
            
            current_selection = torch.tensor([-10], dtype=torch.long).to(etr_point_embedding.device)
            dense_mat_features = self.dense_mat_emb(geohash_dist_mat, point_masks)
            selection_scatter_base = torch.arange(graph_size).unsqueeze(0).repeat(batch_size, 1).to(etr_point_embedding.device)

            graph_size_int = int(graph_size.item()) if isinstance(graph_size, torch.Tensor) else int(graph_size)

            for i in range(graph_size_int):
                quantile_size = 9
                if len(predict_etr) > 0:
                    predict_etr_ = predict_etr[-1].view(batch_size, 1, 1, quantile_size)
                else:
                    predict_etr_ = torch.zeros(batch_size, 1, 1, quantile_size).to(etr_point_embedding.device)
                dynamic_feature = self.get_route_dynamic_features(geohash_dist_mat, prev_selection, current_selection, point_masks, wb_time_features, predict_etr_, line_dist_mat=line_dist_mat, angle_mat=angle_mat)
                # Concatenate and pass through MLP
                point_features = self.rp_point_unified_emb_mlp(torch.cat([rp_point_embedding[:, 2:], dynamic_feature], dim=-1))


                query_input = torch.cat([etr_state[0], route_state[0]], dim=-1)  # Assuming etr_state and route_state are tuples of (h, c)

                next_logits = self.point_attention(query_input, point_features)
                next_logits = next_logits.view(batch_size, graph_size)
                next_logits_ = torch.where(current_point_masks, torch.ones_like(next_logits) * -1e9, next_logits)
                selected_idx = torch.argmax(next_logits_, dim=1)
                selected_idx = torch.where(torch.all(current_point_masks, dim=1),
                                        torch.ones_like(selected_idx) * (graph_size - 1), selected_idx) 
                selected_idx = selected_idx.view(batch_size, 1)
                predict_logits.append(next_logits)
                predict_selections.append(selected_idx)
                # Generate label index
                prev_selection = current_selection

                label_idx = (route_labels - 2 == i).long()
                max_idx = torch.argmax(label_idx, dim=1)
                max_idx = torch.where(torch.all(label_idx <= 0, dim=1), torch.ones_like(max_idx) * -1, max_idx) 
                current_selection = max_idx.view(batch_size, 1)
                label_selections.append(current_selection)
                current_selection = torch.where(current_selection < 0,
                                                torch.ones_like(current_selection) * (graph_size - 1), current_selection)
                selection_indices = torch.cat([torch.arange(batch_size).view(batch_size, 1).long().to(etr_point_embedding.device), current_selection], dim=-1)

                # Generate label ETR Perform advanced indexing to gather elements similar to tf.gather_nd
                current_etr = etr_labels[selection_indices[:, 0], selection_indices[:, 1]]
                label_etr.append(current_etr)
                
                dense_mat_features = self.dense_mat_emb(geohash_dist_mat, point_masks)
                current_selection_mask = torch.repeat_interleave(current_selection, repeats=graph_size, dim=1)
                # adding_mask = torch.equal(selection_scatter_base, current_selection_mask)
                adding_mask = (selection_scatter_base == current_selection_mask)
                point_masks = torch.logical_or(adding_mask, point_masks)


                current_point_masks = self.get_available_masks(point_masks, point_shift_same_id, point_shift_pickup)
                # Update ETR hidden state
                etr_dynamic_features = self.get_etr_dynamic_features(geohash_dist_mat, prev_selection, current_selection,
                                                                        line_dist_mat=line_dist_mat, angle_mat=angle_mat)
                # Gather elements from etr_point_embeddings using selection_indices
                etr_selected_point_embeddings = etr_point_embedding[selection_indices[:, 0], selection_indices[:, 1]]
                # Concatenate etr_selected_point_embeddings with etr_dynamic_features
                etr_point_features = torch.cat([etr_selected_point_embeddings, etr_dynamic_features], dim=-1)
                etr_lstm_output, etr_state = self.etr_recurrence(etr_point_features, etr_state)
                if i == 0:
                    pickup_predict = self.head_etr_pickup(etr_lstm_output)
                    deliver_predict = self.head_etr_deliver(etr_lstm_output)
                else:
                    pickup_predict = self.etr_pickup(etr_lstm_output)
                    deliver_predict = self.etr_deliver(etr_lstm_output)
                current_point_types = point_types[selection_indices[:, 0], selection_indices[:, 1]]
                quantile_input = torch.where(current_point_types.reshape(-1,1) == -1, pickup_predict, deliver_predict)
                quantile_output = self.etr_quantile(quantile_input)
                predict_etr.append(quantile_output)
                
                # Update Route hidden state
                route_point_features = rp_point_embedding[selection_indices[:, 0], selection_indices[:, 1], 2:]
                # route_point_features = torch.gather(rp_point_embedding[:, 2:], dim=1, index=selection_indices)
                route_point_features = torch.cat([route_point_features, etr_dynamic_features], dim=-1)
                # route_lstm_output, route_state = self.route_rnn_cell(route_point_features, route_state)
                route_lstm_output, route_state = self.route_rnn_cell(route_point_features.unsqueeze(1), (route_state[0].unsqueeze(0), route_state[1].unsqueeze(0)))
                route_lstm_output = route_lstm_output.squeeze(1)
                route_state = (route_state[0].squeeze(0), route_state[1].squeeze(0))

            # assert False
            result_logits = torch.stack(predict_logits, dim=1).view(batch_size, graph_size*graph_size)  # [1024, 144]
            result_selections = torch.stack(predict_selections, dim=1).view(batch_size, graph_size)  # [1024, 12]  torch.Size([512, 25])
            result_label_selections = torch.stack(label_selections, dim=1).view(batch_size, graph_size)  # [1024, 12]
            result_etr = torch.stack(predict_etr, dim=1).view(batch_size, graph_size * 9)  # [1024, 108]
            result_label_etr = torch.stack(label_etr, dim=1)  # [1024, 12]
           
            return {
                "logits": result_logits, 
                "selections": result_selections, 
                "label_selections": result_label_selections, 
                "etr": result_etr, 
                "label_etr": result_label_etr, 
             } 

        else:

            batch_size = features.shape[0]
            wb_info = features[:,1250:1300].view(batch_size, graph_size, 2).long() 
            wb_ids = wb_info[:, :, 0]
            point_types = wb_info[:, :, 1]

            # Rider and path features
            context_c_features = features[:,3615:3618].long()

            context_n_features = features[:,3618:3623]

            # Order features
            wb_c_features = features[:,0:125].view(batch_size, graph_size, wb_c_feature_len).long()
            wb_time_features = features[:,1175:1250].view(batch_size, graph_size, wb_time_feature_len)

            # Node features
            pickup_c_features = features[:,125:250].view(batch_size, graph_size, pickup_c_feature_len).long()
            pickup_n_features = features[:,250:725].view(batch_size, graph_size, pickup_n_feature_len)
            deliver_n_features = features[:,725:1025].view(batch_size, graph_size, deliver_n_feature_len)
            da_n_features = features[:,1025:1150].view(batch_size, graph_size, da_n_feature_len)

            # Distance matrices
            geohash_dist_mat = features[:,2157:2886].view(batch_size, graph_size + 2, graph_size + 2)
            line_dist_mat = features[:,1428:2157].view(batch_size, graph_size + 2, graph_size + 2)
            angle_mat = features[:,2886:3615].view(batch_size, graph_size + 2, graph_size + 2)

            # Labels (only for training)
            labels = features[:,1300:1425].view(batch_size, graph_size, 5).long()
            route_labels = labels[:, :, 0]
            etr_labels = labels[:, :, 1]
            route_origin = labels[:, :, 2]
            time_origin = labels[:, :, 3]

            
            point_masks = wb_ids <= 0 #[b,12]
            extend_point_masks = F.pad(point_masks, (2, 0), "constant", 0.0)
            point_shift_ids = pad_roll(wb_ids, shift=1, input_dim=2)
            point_shift_same_id = point_shift_ids == wb_ids
            point_shift_types = pad_roll(point_types, shift=1)
            point_shift_pickup = point_shift_types == -1
            current_point_masks = self.get_available_masks(point_masks, point_shift_same_id, point_shift_pickup)

            edge_fea = get_edge_features(line_dist_mat, wb_time_features, wb_info, extend_point_masks, self.args)



            # ETR Embedding
            etr_point_embedding, rp_point_embedding = self.point_emb(context_n_features, context_c_features[:, :3], wb_c_features,
                    pickup_n_features, pickup_c_features, deliver_n_features, da_n_features,
                    geohash_dist_mat, line_dist_mat, angle_mat, extend_point_masks, point_types, edge_fea)
            
            etr_state = (self.etr_state_h.expand(batch_size, -1).contiguous(),
                    self.etr_state_c.expand(batch_size, -1).contiguous())
            route_state = (self.rp_state_h.expand(batch_size, -1).contiguous(),
                    self.rp_state_c.expand(batch_size, -1).contiguous())

            route_state_online = (self.online_state_h.expand(batch_size, -1).contiguous(),
                    self.online_state_c.expand(batch_size, -1).contiguous())            

            predict_logits = []
            predict_selections = []
            predict_etr = []

            label_selections = []  
            label_etr = []

            prev_selection = torch.tensor([-10], dtype=torch.long).to(etr_point_embedding.device)

            current_selection = torch.tensor([-10], dtype=torch.long).to(etr_point_embedding.device)
            dense_mat_features = self.dense_mat_emb(geohash_dist_mat, point_masks)
            selection_scatter_base = torch.arange(graph_size).unsqueeze(0).repeat(batch_size, 1).to(etr_point_embedding.device)
            graph_size_int = int(graph_size.item()) if isinstance(graph_size, torch.Tensor) else int(graph_size)
            for i in range(graph_size_int):
                # Select next point index
                quantile_size = 9
                if len(predict_etr) > 0:
                    predict_etr_ = predict_etr[-1].view(batch_size, 1, 1, quantile_size)
                else:
                    predict_etr_ = torch.zeros(batch_size, 1, 1, quantile_size).to(etr_point_embedding.device)
                dynamic_feature = self.get_route_dynamic_features(geohash_dist_mat, prev_selection, current_selection, point_masks, wb_time_features, predict_etr_, line_dist_mat=line_dist_mat, angle_mat=angle_mat)
                # Concatenate and pass through MLP
                point_features = self.rp_point_unified_emb_mlp(torch.cat([rp_point_embedding[:, 2:], dynamic_feature], dim=-1))

                query_input = torch.cat([etr_state[0], route_state[0]], dim=-1)  # Assuming etr_state and route_state are tuples of (h, c)


                next_logits = self.point_attention(query_input, point_features)

                next_logits = next_logits.view(batch_size, graph_size)
                next_logits_ = torch.where(current_point_masks, torch.ones_like(next_logits) * -1e9, next_logits)
                selected_idx = torch.argmax(next_logits_, dim=1)
                selected_idx = torch.where(torch.all(current_point_masks, dim=1),
                                        torch.ones_like(selected_idx) * (graph_size - 1), selected_idx) 
                selected_idx = selected_idx.view(batch_size, 1)
                predict_logits.append(next_logits)
                predict_selections.append(selected_idx)
                # Generate label index
                prev_selection = current_selection

                label_idx = (route_labels - 2 == i).long()
                max_idx = torch.argmax(label_idx, dim=1)
                max_idx = torch.where(torch.all(label_idx <= 0, dim=1), torch.ones_like(max_idx) * -1, max_idx) 
                current_selection_label = max_idx.view(batch_size, 1)
                label_selections.append(current_selection_label)
                current_selection_label = torch.where(current_selection_label < 0,
                                                torch.ones_like(current_selection_label) * (graph_size - 1), current_selection_label)
                selection_indices_label = torch.cat([torch.arange(batch_size).view(batch_size, 1).long().to(etr_point_embedding.device), current_selection_label], dim=-1)

                # Generate label ETR Perform advanced indexing to gather elements similar to tf.gather_nd
                current_etr_label = etr_labels[selection_indices_label[:, 0], selection_indices_label[:, 1]]
                label_etr.append(current_etr_label)

                current_selection = predict_selections[-1]
                current_selection = torch.where(current_selection < 0,
                                                torch.ones_like(current_selection) * (graph_size - 1), current_selection)
                selection_indices = torch.cat([torch.arange(batch_size).view(batch_size, 1).long().to(etr_point_embedding.device), current_selection], dim=-1)
                # Update masks
                dense_mat_features = self.dense_mat_emb(geohash_dist_mat, point_masks)
                current_selection_mask = torch.repeat_interleave(current_selection, repeats=graph_size, dim=1)
                # adding_mask = torch.equal(selection_scatter_base, current_selection_mask)
                adding_mask = (selection_scatter_base == current_selection_mask)
                point_masks = torch.logical_or(adding_mask, point_masks)
                current_point_masks = self.get_available_masks(point_masks, point_shift_same_id, point_shift_pickup)
                # Update ETR hidden state
                etr_dynamic_features = self.get_etr_dynamic_features(geohash_dist_mat, prev_selection, current_selection,
                                                                        line_dist_mat=line_dist_mat, angle_mat=angle_mat)
                # Gather elements from etr_point_embeddings using selection_indices
                etr_selected_point_embeddings = etr_point_embedding[selection_indices[:, 0], selection_indices[:, 1]]
                # Concatenate etr_selected_point_embeddings with etr_dynamic_features
                etr_point_features = torch.cat([etr_selected_point_embeddings, etr_dynamic_features], dim=-1)
                etr_lstm_output, etr_state = self.etr_recurrence(etr_point_features, etr_state)
                if i == 0:
                    pickup_predict = self.head_etr_pickup(etr_lstm_output)
                    deliver_predict = self.head_etr_deliver(etr_lstm_output)
                else:
                    pickup_predict = self.etr_pickup(etr_lstm_output)
                    deliver_predict = self.etr_deliver(etr_lstm_output)
                current_point_types = point_types[selection_indices[:, 0], selection_indices[:, 1]]
                quantile_input = torch.where(current_point_types.reshape(-1,1) == -1, pickup_predict, deliver_predict)
                quantile_output = self.etr_quantile(quantile_input)
                predict_etr.append(quantile_output)

                # Update Route hidden state
                route_point_features = rp_point_embedding[selection_indices[:, 0], selection_indices[:, 1], 2:]
                # route_point_features = torch.gather(rp_point_embedding[:, 2:], dim=1, index=selection_indices)
                route_point_features = torch.cat([route_point_features, etr_dynamic_features], dim=-1)
                # route_lstm_output, route_state = self.route_rnn_cell(route_point_features, route_state)
                route_lstm_output, route_state = self.route_rnn_cell(route_point_features.unsqueeze(1), (route_state[0].unsqueeze(0), route_state[1].unsqueeze(0)))
                route_lstm_output = route_lstm_output.squeeze(1)
                route_state = (route_state[0].squeeze(0), route_state[1].squeeze(0))
            
            result_logits = torch.stack(predict_logits, dim=1).view(batch_size, graph_size*graph_size)  
            result_selections = torch.stack(predict_selections, dim=1).view(batch_size, graph_size) 
            result_label_selections = torch.stack(label_selections, dim=1).view(batch_size, graph_size)  
            result_etr = torch.stack(predict_etr, dim=1).view(batch_size, graph_size * 9)  
            result_label_etr = torch.stack(label_etr, dim=1)  

            return {
                "logits": result_logits, 
                "selections": result_selections, 
                "label_selections": result_label_selections, 
                "etr": result_etr, 
                "label_etr": result_label_etr, 
             } 
        
    def get_available_masks(self, point_selected_masks, point_shift_same_id, point_shift_pickup):
        target_delivery = torch.logical_and(point_shift_same_id, point_shift_pickup)  
        pickup_exist = torch.logical_not(pad_roll(point_selected_masks, input_dim=2, shift=1)) 
        delivery_mask = torch.logical_and(target_delivery, pickup_exist) 
        current_point_masks = torch.logical_or(delivery_mask, point_selected_masks)  
        return current_point_masks
    
    def create_mlp(self, input_size, layer_sizes):
        layers = []
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            input_size = size
        return nn.Sequential(*layers)

    def get_route_dynamic_features(self, point_dist_mat, prev_selection, current_selection, point_masks, time_features, predict_etr_, line_dist_mat, angle_mat):

        a_f = torch.tensor([-10]).to(point_dist_mat.device)
        dist_flag = 0 if torch.all(current_selection == a_f) else 1
        dist_feature_list = []
        batch_size, _ = point_masks.size()
        if torch.all(prev_selection == a_f)  :
            prev_selection = torch.ones((batch_size, 1), dtype=torch.int64).to(point_dist_mat.device)
            if dist_flag == 0:
                prev_selection = prev_selection * -2
            else :
                prev_selection = prev_selection * -1

        if torch.all(current_selection == a_f):
            current_selection = torch.ones((batch_size, 1), dtype=torch.int64).to(point_dist_mat.device)
            if dist_flag == 0:
                current_selection = current_selection * -2
            else :
                current_selection = current_selection * -1

        prev_dist_features = self.get_dist_features(point_dist_mat, prev_selection, point_masks, flag=dist_flag)
        current_dist_features = self.get_dist_features(point_dist_mat, current_selection, point_masks)
        dist_feature_list.extend(prev_dist_features)
        dist_feature_list.extend(current_dist_features)

        # if line_dist_mat is not None and angle_mat is not None:
        line_prev_dist_features = self.get_dist_features(line_dist_mat, prev_selection, point_masks, flag=dist_flag)
        line_current_dist_features = self.get_dist_features(line_dist_mat, current_selection, point_masks)
        dist_feature_list.extend(line_prev_dist_features)
        dist_feature_list.extend(line_current_dist_features)

        radian_mat = angle_mat / 180.0 * 3.1415926
        u_features = torch.mul(line_dist_mat, torch.cos(radian_mat))
        v_features = torch.mul(line_dist_mat, torch.sin(radian_mat))
        uv_prev_dist_features = self.get_uv_dist_features(u_features, v_features, prev_selection, point_masks, flag=dist_flag)
        uv_current_dist_features = self.get_uv_dist_features(u_features, v_features, current_selection, point_masks)
        dist_feature_list.extend(uv_prev_dist_features)
        dist_feature_list.extend(uv_current_dist_features)


        current_time_features = self.get_time_features(time_features, predict_etr_)
        
        return torch.cat([torch.stack(dist_feature_list, dim=-1)] + current_time_features, dim=-1)

    def get_time_features(self, time_features, predict_etr_, quantile_size: int =9):
        batch_size,graph_size = time_features.size(0),time_features.size(1)
        feature_size = 3
        time_features = time_features.unsqueeze(-1).repeat(1, 1, 1, 9)
        time_features = time_features.view(batch_size, graph_size, feature_size, quantile_size)

        time_features = time_features - predict_etr_
        
        time_features = time_features.view(batch_size, graph_size, feature_size * quantile_size)

        abs_time_feature = time_features / 3600.0
        abs_time_feature = torch.where(abs_time_feature > 1.0, torch.ones_like(abs_time_feature).to(abs_time_feature.device), abs_time_feature)
        abs_time_feature = torch.where(abs_time_feature < -1.0, torch.ones_like(abs_time_feature).to(abs_time_feature.device) * -1.0, abs_time_feature)


        return [abs_time_feature]
    
    def get_dist_features(self, point_dist_mat, selection, point_masks, flag: int = 1):
        batch_size, graph_size = point_masks.size()
        if torch.all(selection == torch.tensor([-10]).to(point_dist_mat.device)):
            if flag == 0:
                selection = torch.full((batch_size, 1), -2, dtype=torch.int64).to(point_dist_mat.device)
            else :
                selection = torch.full((batch_size, 1), -1, dtype=torch.int64).to(point_dist_mat.device)


        selection = selection.unsqueeze(-1).expand(-1, -1, point_dist_mat.shape[-1])
        current_dist = torch.gather(point_dist_mat, 1, selection + 2)
        current_dist = current_dist.view(batch_size, graph_size + 2)[:, 2:]
        current_dist = torch.where(point_masks, torch.full_like(current_dist, 1e9), current_dist)

        # Features
        abs_dist_feature = current_dist / 5000.0
        abs_dist_feature = torch.where(abs_dist_feature > 1.0, torch.ones_like(abs_dist_feature).to(abs_dist_feature.device), abs_dist_feature)
        relative_dist = torch.where(current_dist > 0, current_dist, torch.full_like(current_dist, 1e9))
        min_dist = torch.min(relative_dist, dim=1, keepdim=True)[0] + 1.0
        relative_dist_feature = current_dist / min_dist / 20.0
        relative_dist_feature = torch.where(relative_dist_feature > 1.0, torch.ones_like(relative_dist_feature).to(relative_dist_feature.device), relative_dist_feature)
        


        return [abs_dist_feature, relative_dist_feature]

    def get_uv_dist_features(self, u_mat, v_mat, selection, point_masks, flag: int = 1):
        batch_size, graph_size = point_masks.size()
        if selection is None and flag == 0:
            selection = torch.full((batch_size, 1), -2, dtype=torch.int64).to(u_mat.device)
        elif selection is None:
            selection = torch.full((batch_size, 1), -1, dtype=torch.int64).to
        selection = selection.unsqueeze(-1).expand(-1, -1, u_mat.shape[-1])
        u_dist = torch.gather(u_mat, 1, selection + 2)
        u_dist = u_dist.view(batch_size, graph_size + 2)[:, 2:]
        u_dist = torch.where(point_masks, torch.full_like(u_dist, 1e9), u_dist)
        u_abs_dist_feature = u_dist / 5000.0
        u_abs_dist_feature = torch.where(u_abs_dist_feature > 1.0, torch.ones_like(u_abs_dist_feature).to(u_abs_dist_feature.device), u_abs_dist_feature)
        u_abs_dist_feature = torch.where(u_abs_dist_feature < -1.0, torch.full_like(u_abs_dist_feature, -1.0), u_abs_dist_feature)

        v_dist = torch.gather(v_mat, 1, selection + 2)
        v_dist = v_dist.view(batch_size, graph_size + 2)[:, 2:]
        v_dist = torch.where(point_masks, torch.full_like(v_dist, 1e9), v_dist)
        v_abs_dist_feature = v_dist / 5000.0
        v_abs_dist_feature = torch.where(v_abs_dist_feature > 1.0, torch.ones_like(v_abs_dist_feature).to(u_abs_dist_feature.device), v_abs_dist_feature)
        v_abs_dist_feature = torch.where(v_abs_dist_feature < -1.0, torch.full_like(v_abs_dist_feature, -1.0), v_abs_dist_feature)


        return [u_abs_dist_feature, v_abs_dist_feature]

    def get_etr_dynamic_features(self, point_dist_mat, prev_selection, current_selection, line_dist_mat, angle_mat):
        batch_size = current_selection.size(0)
        if torch.all(prev_selection == torch.tensor([-10]).to(point_dist_mat.device)):
            prev_selection = torch.full((batch_size, 1), -1, dtype=torch.int64).to(point_dist_mat.device)
        dist_indices = torch.cat(
            [torch.arange(batch_size, dtype=torch.int64).view(batch_size, 1).to(point_dist_mat.device),prev_selection + 2, current_selection + 2], dim=-1) #N*3

        i, j, k = dist_indices[:, 0], dist_indices[:, 1], dist_indices[:, 2]
        etr_dist_feature = point_dist_mat[i, j, k] / 3000.0
        etr_dist_feature = torch.clamp(etr_dist_feature, min=0, max=5.0)
        radian_mat = angle_mat / 180.0 * 3.1415926
        u_features = torch.mul(line_dist_mat, torch.cos(radian_mat))
        v_features = torch.mul(line_dist_mat, torch.sin(radian_mat))
        # u_dist_feature = torch.gather(u_features.view(-1), 0, dist_indices.view(-1)).view(batch_size, -1) / 3000.0
        u_dist_feature = u_features[i, j, k]/ 3000.0
        u_dist_feature = torch.clamp(u_dist_feature, min=-5.0, max=5.0)
        # v_dist_feature = torch.gather(v_features.view(-1), 0, dist_indices.view(-1)).view(batch_size, -1) / 3000.0
        v_dist_feature = v_features[i, j, k]/ 3000.0
        v_dist_feature = torch.clamp(v_dist_feature, min=-5.0, max=5.0)
        return torch.cat([etr_dist_feature.view(-1,1), u_dist_feature.view(-1,1), v_dist_feature.view(-1,1)], dim=1)
        # return etr_dist_feature.view(batch_size, 1)
    
    def etr_recurrence(self, input, state:Tuple[Tensor, Tensor]):
        lstm_output, state = self.etr_rnn_cell(input.unsqueeze(1), (state[0].unsqueeze(0), state[1].unsqueeze(0)))
        lstm_output = lstm_output.squeeze(1)
        state = (state[0].squeeze(0), state[1].squeeze(0))
        etr_predict = self.etr_mlp(lstm_output)
        return etr_predict, state

    def route_recurrence(self, input, state):
        lstm_output, state = self.route_rnn_cell(input, state)
        return lstm_output, state






        
