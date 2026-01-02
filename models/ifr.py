import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import device
from config import args
from typing import List, Dict

from models.losses import symmetric_CE


def compute_prototypes(embeddings:torch.Tensor, labels:torch.Tensor):
    '''
    Args:
        embeddings: [n_embeddings, c, h, w]
        labels: [n_embeddings, ]
    '''
    unique_labels = torch.range(start=0, end=torch.max(labels)).unsqueeze(dim=1).type_as(labels)    # [n_cls, 1]
    matrix = unique_labels.eq(labels.reshape(1, list(labels.shape)[0])).type_as(embeddings)
    flatten_prototypes = torch.matmul(matrix, embeddings.flatten(1)) / matrix.sum(dim=1, keepdim=True)
    _, c, h, w = list(embeddings.shape)
    prototypes = torch.reshape(flatten_prototypes, shape=(list(flatten_prototypes.shape)[0], c, h, w))
    return prototypes


class AttentionHead(nn.Module):
    def __init__(self, args, in_dim:int=512, is_head:bool=True) -> None:
        '''
        Args:
            out_dim: output dimension of the head layer;
            in_dim: input dimension of the head layer;
            typical_atten: Whether to use typical attention modules, which includes key, query and values layer;
        '''
        super(AttentionHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = args['out_dim']
        self.is_head = is_head
        
        self.query_head = nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=1, stride=1, bias=False)
        self.key_head = nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=1, stride=1, bias=False)
        self.value_head = nn.Conv2d(in_channels=self.in_dim, out_channels=self.out_dim, kernel_size=1, stride=1, bias=False)
        
        if is_head:
            self.bn = nn.BatchNorm2d(self.out_dim)
            self.transform_head = nn.Conv2d(in_channels=self.out_dim, out_channels=self.out_dim, kernel_size=1, stride=1, bias=False)

    def reset_params(self) -> None:
        self.query_head.weight = nn.Parameter(torch.eye(self.out_dim, self.in_dim).unsqueeze(-1).unsqueeze(-1)*args['gain'])
        self.key_head.weight = nn.Parameter(torch.eye(self.out_dim, self.in_dim).unsqueeze(-1).unsqueeze(-1)*args['gain'])
        self.value_head.weight = nn.Parameter(torch.eye(self.out_dim, self.in_dim).unsqueeze(-1).unsqueeze(-1)*args['gain'])
        
        if self.is_head:
            nn.init.constant_(self.bn.weight, 1.)
            nn.init.constant_(self.bn.bias, 0.)
            self.transform_head.weight = nn.Parameter(torch.eye(self.out_dim, self.out_dim).unsqueeze(-1).unsqueeze(-1))

    def embed(self, context_x:torch.Tensor, aug_context:torch.Tensor) -> torch.Tensor:
        
        context_queries = self.query_head(context_x)
        aug_keys = self.key_head(aug_context)
        aug_values = self.value_head(aug_context)

        context_reconst = self.compute_attention(queries=context_queries, keys=aug_keys, values=aug_values)
        context_features = context_x + args['scale_reconst']*context_reconst   # original features fusion
        
        return context_features
    
    def forward_pass(self, context_x, context_y, aug_context) -> torch.Tensor:
        context_features = self.embed(context_x, aug_context)
        if self.is_head:
            context_features = self.bn(context_features)
            context_features = self.transform_head(F.adaptive_avg_pool2d(context_features, (1, 1)))
        
        prototypes = compute_prototypes(context_features, context_y) # [n_classes, c, h, w]
        
        dist_res = F.cosine_similarity(context_features.flatten(1).unsqueeze(1), 
                                         prototypes.flatten(1).unsqueeze(0),
                                         dim=-1,
                                         eps=1e-30)*10
        return dist_res


    def pred(self, target_x:torch.Tensor, aug_context:torch.Tensor, context_x:torch.Tensor, context_y:torch.Tensor) -> torch.Tensor:
        context_features = self.embed(context_x, aug_context)
        if self.is_head:
            context_features = self.bn(context_features)
            context_features = self.transform_head(F.adaptive_avg_pool2d(context_features, (1, 1)))
        
        target_features = self.embed(target_x, aug_context)
        if self.is_head:
            target_features = self.bn(target_features)
            target_features = self.transform_head(F.adaptive_avg_pool2d(target_features, (1, 1)))
        
        prototypes = compute_prototypes(context_features, context_y)
        
        dist_res = F.cosine_similarity(target_features.flatten(1).unsqueeze(1), 
                                         prototypes.flatten(1).unsqueeze(0),
                                         dim=-1,
                                         eps=1e-30)*10
        
        return dist_res
    
    def compute_attention(self, queries, keys, values) -> torch.Tensor:
        n_q, c_q, h_q, w_q = list(queries.shape)
        n_v, c_v, h_v, w_v = list(values.shape)

        flatten_queries = torch.reshape(queries, shape=(-1, c_q))    # [n_supp*h*w, c]
        flatten_keys = torch.reshape(keys, shape=(-1, c_v))             # [n_clusters*h*w, c]
        flatten_values = torch.reshape(values, shape=(-1, c_v))
        d_scale = torch.rsqrt(torch.tensor(self.out_dim).type_as(queries)).to(device)

        inner_prod = torch.matmul(flatten_queries, flatten_keys.t())    # [n_supp*h*w, n_clusters*h*w]
        inner_logits = d_scale * inner_prod

        inner_logits = torch.reshape(inner_logits, shape=(n_q, h_q*w_q, n_v*h_v*w_v))
        max_logits, _ = torch.max(inner_logits, dim=1, keepdim=True)
        inner_logits = inner_logits - max_logits

        exp_logits = torch.exp(inner_logits)
        softmax_logits = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

        res = torch.matmul(softmax_logits.reshape((n_q*h_q*w_q, n_v*h_v*w_v)), flatten_values)
        return res.reshape(shape=queries.shape)


def pa(features, vars):
    return F.conv2d(features, vars[0])


def _run_evaluation(context_feat:torch.Tensor, 
                    aug_context_feat:torch.Tensor, 
                    context_labels:torch.Tensor, 
                    target_feat:torch.Tensor, 
                    target_labels:torch.Tensor, 
                    backbone:nn.Module, 
                    attention_head:AttentionHead,
                    feat_linear_vars:List[torch.Tensor],
                    proto_linear_vars:List[torch.Tensor]):
    
    attention_head.eval()
    
    with torch.no_grad():
        fused_context_feat = attention_head.embed(context_feat, aug_context_feat)
        fused_target_feat = attention_head.embed(target_feat, aug_context_feat)

        prototypes = compute_prototypes(fused_context_feat, context_labels)
        target_feats = pa(F.adaptive_avg_pool2d(fused_target_feat, (1, 1)), feat_linear_vars)
        proto_feats = pa(F.adaptive_avg_pool2d(prototypes, (1, 1)), proto_linear_vars)
        
        _, dynamic_dict = symmetric_CE(target_feats.flatten(1), proto_feats.flatten(1), target_labels)
        
        return dynamic_dict
        

def pa_adaptation(context_feat:torch.Tensor, 
                  aug_context_feat:torch.Tensor, 
                  context_labels:torch.Tensor, 
                  target_feat:torch.Tensor, 
                  target_labels:torch.Tensor, 
                  backbone:nn.Module, 
                  attention_head:AttentionHead,
                  dataset_name:str,
                  max_iter:int=40):
    
    """
    PA adaptation func()
    Args:
        context_feat: [context_batch_size, c, h, w]
        aug_context_feat: [context_batch_size, c, h, w], derived from context_feat,
        context_labels: [context_batch_size, ]
        target_feat: [target_batch_size, c, h, w]
        target_labels: [target_batch_size, ]
        backbone: nn.Module, pretrained backbone, used to embed the context and target features
        attention_head: AttentionHead
        dataset_name: str
        max_iter: int
    """
    data_recorder = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }
    
    feat_dim = 512
    feat_linear_vars = [torch.eye(feat_dim, feat_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)]
    proto_linear_vars = [torch.eye(feat_dim, feat_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True)]
    attention_head.reset_params()
    attention_head.to(device)
    
    if dataset_name in ["traffic_sign", "mnist"]:
        lr = 5e-3
    else:
        lr = 1e-3
    
    weight_decay = 0.0 if dataset_name in ["traffic_sign", "mnist"] else 0.1
    
    optimizer = torch.optim.Adam([
        {'params': feat_linear_vars},
        {'params': proto_linear_vars},
        {'params': attention_head.parameters()},
    ], lr=lr, weight_decay=weight_decay)
    
    for i in range(max_iter):
        
        # eval the performance before the first step
        eval_dynamic_dict = _run_evaluation(context_feat, aug_context_feat, context_labels, 
                                            target_feat, target_labels, 
                                            backbone, attention_head, feat_linear_vars, proto_linear_vars)
        data_recorder['val_losses'].append(eval_dynamic_dict['loss'])
        data_recorder['val_accs'].append(eval_dynamic_dict['acc'])
        
        attention_head.train()
        optimizer.zero_grad()
        
        fused_context_feat = attention_head.embed(context_feat, aug_context_feat)
        prototypes = compute_prototypes(fused_context_feat, context_labels)
        target_feats = pa(F.adaptive_avg_pool2d(fused_context_feat, (1, 1)), feat_linear_vars)
        proto_feats = pa(F.adaptive_avg_pool2d(prototypes, (1, 1)), proto_linear_vars)
        loss, train_dynamic_dict = symmetric_CE(target_feats.flatten(1), proto_feats.flatten(1), context_labels)
        
        data_recorder['train_losses'].append(train_dynamic_dict['loss'])
        data_recorder['train_accs'].append(train_dynamic_dict['acc'])
        loss.backward()
        optimizer.step()
        
        if i == max_iter - 1:
            eval_dynamic_dict = _run_evaluation(context_feat, aug_context_feat, context_labels, 
                                                target_feat, target_labels, 
                                                backbone, attention_head, feat_linear_vars, proto_linear_vars)
            data_recorder['val_losses'].append(eval_dynamic_dict['loss'])
            data_recorder['val_accs'].append(eval_dynamic_dict['acc'])
    return data_recorder