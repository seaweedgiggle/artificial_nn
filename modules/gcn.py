import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.parameter import Parameter
import math

class ClsAttention(nn.Module):

    def __init__(self, feat_size, num_classes):
        super().__init__()
        self.feat_size = feat_size
        self.num_classes = num_classes
        # 用1x1卷积将1024个channel变成num_classes个
        self.channel_w = nn.Conv2d(feat_size, num_classes, 1, bias=False)

    def forward(self, feats):
        # (bs, 512, 49*n) -> (bs, 30, 512)
        
        batch_size, feat_size, HW = feats.size()
        att_maps = self.channel_w(feats.unsqueeze(3))
        # (bs, num_classes, 49*n, 1) -> (bs, num_classes, 49*n) ( 49*n 是经过了softmax之后的，已经变成了概率值)
        att_maps = torch.softmax(att_maps.view(batch_size, self.num_classes, -1), dim=2)
        feats_t = feats.permute(0, 2, 1)
        # feats与att_maps矩阵相乘 (bs, num_classes, 49*n) * (bs, 49*n, 512) -> (bs, num_classes, 512)
        cls_feats = torch.bmm(att_maps, feats_t)
        return cls_feats


class GCLayer(nn.Module):

    def __init__(self, in_size, state_size):
        super().__init__()
        self.condense = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.condense_norm = nn.BatchNorm1d(state_size)
        self.fw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.fw_norm = nn.BatchNorm1d(state_size)
        self.bw_trans = nn.Conv1d(in_size, state_size, 1, bias=False)
        self.bw_norm = nn.BatchNorm1d(state_size)
        self.update = nn.Conv1d(3 * state_size, in_size, 1, bias=False)
        self.update_norm = nn.BatchNorm1d(in_size)
        self.relu = nn.ReLU(inplace=True)
        # v2:
        self.dropout = nn.Dropout(0.5)

    def forward(self, states, fw_A, bw_A):
        # states: batch size x feat size x nodes
        condensed = self.relu(self.condense_norm(self.condense(states)))
        fw_msg = self.relu(self.fw_norm(self.fw_trans(states).bmm(fw_A)))
        bw_msg = self.relu(self.bw_norm(self.bw_trans(states).bmm(bw_A)))
        updated = self.update_norm(self.update(torch.cat((condensed, fw_msg, bw_msg), dim=1)))
        updated = self.relu(self.dropout(updated) + states)
        return updated

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(8, in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters_xavier()

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        x = x.permute(0,2,1)
        self.weight = Parameter(torch.FloatTensor(x.size()[0], self.in_features, self.out_features).to('cuda'))
        self.reset_parameters_xavier()
        support = torch.bmm(x, self.weight)
        output = torch.bmm(adj, support)
        
        if self.bias is not None:
            return  (output + self.bias).permute(0,2,1)
        else:
            return  output.permute(0,2,1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvolutionLayer(nn.Module):

    def __init__(self,in_size, state_size):
        super(GraphConvolutionLayer, self).__init__()
        self.in_size = in_size
        self.state_size = state_size

        self.condense = nn.Conv1d(in_size, state_size, 1)
        self.condense_norm = nn.BatchNorm1d(state_size)

        self.gcn_forward = GraphConvolution(in_size, state_size)
        self.gcn_backward = GraphConvolution(in_size, state_size)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.conv1d = nn.Conv1d(3*state_size, in_size, 1, bias=False)
        self.norm = nn.BatchNorm1d(in_size)

        self.test_conv = nn.Conv1d(state_size, in_size, 1, bias=False)
    def forward(self, x, fw_A, bw_A):
        
        states = x
        condensed_message = self.relu(self.condense_norm(self.condense(x)))
        fw_message = self.relu(self.gcn_forward(x, fw_A))
        bw_message = self.relu(self.gcn_backward(x, bw_A))
        update = torch.cat((condensed_message, fw_message, bw_message),dim=1)
        x = self.norm(self.conv1d(update))
        x = self.relu(x+states)

        return x

class GCN(nn.Module):

    def __init__(self, in_size, state_size):
        super(GCN, self).__init__()

        # in_size:1024, state_size:256
        self.gcn1 = GraphConvolutionLayer(in_size, state_size)
        self.gcn2 = GraphConvolutionLayer(in_size, state_size)
        self.gcn3 = GraphConvolutionLayer(in_size, state_size)

    def forward(self, states, fw_A, bw_A):
        # states: batch_size * feature_size(in_size) * number_classes
        states = states.permute(0,2,1)
        # states: batch_size * number_classes * feature_size(in_size)
        states = self.gcn1(states, fw_A, bw_A)
        states = self.gcn2(states, fw_A, bw_A)
        states = self.gcn3(states, fw_A, bw_A)
        
        return states.permute(0,2,1)

class GCNFeatureExtractor(nn.Module):

    def __init__(self, num_classes, fw_adj, bw_adj):
        super().__init__()
        self.num_classes = num_classes
        feat_size = 512
        self.cls_atten = ClsAttention(feat_size, num_classes)
        
        self.gcn = GCN(feat_size, 256)

        self.fc2 = nn.Linear(feat_size, num_classes)

        fw_D = torch.diag_embed(fw_adj.sum(dim=1))   # 出度
        bw_D = torch.diag_embed(bw_adj.sum(dim=1))   # 入度
        inv_sqrt_fw_D = fw_D.pow(-0.5)
        inv_sqrt_fw_D[torch.isinf(inv_sqrt_fw_D)] = 0
        inv_sqrt_bw_D = bw_D.pow(-0.5)
        inv_sqrt_bw_D[torch.isinf(inv_sqrt_bw_D)] = 0
        
        self.fw_A = inv_sqrt_fw_D.mm(fw_adj).mm(inv_sqrt_fw_D) # fw_A = D^{0.5} * A * D^{0.5}
        self.bw_A = inv_sqrt_bw_D.mm(bw_adj).mm(inv_sqrt_bw_D)
    
    
    # 1. mask_fill 把 mask 的地方补 0
    # 2. 计算每个样本下没有 masked 掉的
    # n 表示 每个样本有的图片数
    # 就是说有四张图片，但是有的样本可能不足四张，所以输出的 49*n 的 n 中其实有一部分是无效的
    # 计算 global_feats 和 cls_feats 的时候要进行处理    
    
    def forward(self, encoder_feats):
        batch_size = encoder_feats.size(0)
        # (20, 20) -> (bs, 20, 20)
        fw_A = self.fw_A.repeat(batch_size, 1, 1)
        bw_A = self.bw_A.repeat(batch_size, 1, 1)
        
        # (bs, 49*n, 512) -> (bs, 512, 49*n)
        encoder_feats = encoder_feats.permute(0, 2, 1)
        
        # (bs, 512, 49*n) -> (bs, 512)
        global_feats = encoder_feats.mean(dim=2)
        
        # (bs, 512, 49*n) -> (bs, num_classes, 512)
        cls_feats = self.cls_atten(encoder_feats)
        
        # (bs, 1, 512).concat((bs, num_classes, 512)) -> (bs, num_classes + 1, 512)
        # 也就是 node_feats[:, 0, :] 代表 global_feats
        node_feats = torch.cat((global_feats.unsqueeze(1), cls_feats), dim=1)

        node_feats = node_feats.contiguous()
        
        # (bs, 21, 512) -> (bs, 21, 512)
        node_states = self.gcn(node_feats, fw_A, bw_A)
        
        # (bs, 21, 512) -> (bs, 512)
        # global_states = node_states.mean(dim=1)
#         print(node_states.shape)
        return node_states
