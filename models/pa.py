
import torch
import numpy as np
import math
import torch.nn.init as init

from models.model_utils import sigmoid, cosine_sim
from models.losses import prototype_loss
from utils import device
import torch.nn.functional as F

#from config import args

def apply_selection(features, vartheta):
    """
    Performs pre-classifier alignment of features (feature adaptation) via a linear transformation.
    """

    #features = features.unsqueeze(-1).unsqueeze(-1)
    features = F.conv2d(features, vartheta[0]).flatten(1)

    return features


def pa(context_features, context_labels, target_features, target_labels, max_iter=40, ad_opt='linear', lr=0.1, distance='cos'):
    """
    PA method: learning a linear transformation per task to adapt the features to a discriminative space 
    on the support set during meta-testing
    """
    input_dim = context_features.size(1)
    output_dim = input_dim
    stdv = 1. / math.sqrt(input_dim)
    vartheta = []
    if ad_opt == 'linear':
        vartheta.append(torch.eye(output_dim, input_dim).unsqueeze(-1).unsqueeze(-1).to(device).requires_grad_(True))

    optimizer = torch.optim.Adadelta(vartheta, lr=lr)
    
    tmp_recorder = {
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }
    
    for i in range(max_iter):
        
        with torch.no_grad():
            selected_context = apply_selection(context_features, vartheta)
            selected_target = apply_selection(target_features, vartheta)
            loss, val_stat, _ = prototype_loss(selected_context, context_labels,
                                               selected_target, target_labels, distance=distance)
            tmp_recorder['val_losses'].append(val_stat['loss'])
            tmp_recorder['val_accs'].append(val_stat['acc'])

        optimizer.zero_grad()
        selected_context = apply_selection(context_features, vartheta)
        loss, train_stat, _ = prototype_loss(selected_context, context_labels,
                                            selected_context, context_labels, distance=distance)
        tmp_recorder['train_losses'].append(train_stat['loss'])
        tmp_recorder['train_accs'].append(train_stat['acc'])
        loss.backward()
        optimizer.step()
        
        if i == max_iter - 1:
            with torch.no_grad():
                selected_context = apply_selection(context_features, vartheta)
                selected_target = apply_selection(target_features, vartheta)
                loss, val_stat, _ = prototype_loss(selected_context, context_labels,
                                                   selected_target, target_labels, distance=distance)
                tmp_recorder['val_losses'].append(val_stat['loss'])
                tmp_recorder['val_accs'].append(val_stat['acc'])

    return vartheta, tmp_recorder
