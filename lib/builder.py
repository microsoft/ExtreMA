import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import torch.distributed as dist

class ExtreMA(nn.Module):
    """
    Build the model
    """
    def __init__(self, base_encoder, ema_encoder, proj_dim=256, mlp_dim=4096, T=1., mask_ratio=0.8, num_masks=1, disjoint=True):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(ExtreMA, self).__init__()

        self.T = T
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks
        self.disjoint_sampling = disjoint
        
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = ema_encoder(num_classes=mlp_dim)
        self.base_encoder.student=True
        self.momentum_encoder.student=False

        hidden_dim = self.base_encoder.norm.weight.data.shape[0]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer
        # projectors
        self.base_encoder.head = self._build_proj(3, hidden_dim, mlp_dim, proj_dim)
        self.momentum_encoder.head = self._build_proj(3, hidden_dim, mlp_dim, proj_dim)
        # predictor
        self.predictor = self._build_pred(2, proj_dim, mlp_dim, proj_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_pred(self, num_layers, input_dim, mlp_dim, output_dim):
        layers = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            layers.append(nn.Linear(dim1, dim2, bias=True))

            if l < num_layers - 1:
                layers.append(nn.LayerNorm(dim2))
                layers.append(nn.GELU())
            
        mlp = nn.Sequential(*layers)
        mlp.apply(self._init_weights)
        return mlp

    def _build_proj(self, num_layers, input_dim, mlp_dim, output_dim):
        layers = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            layers.append(nn.Linear(dim1, dim2, bias=True))

            if l < num_layers - 1:
                layers.append(nn.LayerNorm(dim2))
                layers.append(nn.GELU())

        mlp = nn.Sequential(*layers)
        mlp.apply(self._init_weights)
        return mlp

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def byol_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        loss = ((q - k) ** 2).sum(dim=-1)
        return loss.mean() 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'base_encoder.' + k for k in self.base_encoder.no_weight_decay()}

    @torch.no_grad()
    def generate_mask(self, x, num_masks, mask_ratio=0.):
        if mask_ratio > 0:
            if self.disjoint_sampling:
                view_size = int(196 * (1 - mask_ratio))
                B = x.size(0)
                device = x.get_device()
                noise = torch.rand(B, 196, device=device) 
                mask_index = torch.argsort(noise, dim=1)  # consider cls token
                masks = []
                for i in range(num_masks):
                    # 196 patches are hard-coded
                    mask = mask_index[:, view_size*i:view_size*(i+1)]
                    mask = mask.long()
                    masks.append(mask)
            else:
                masks = []
                for i in range(num_masks):
                    # 196 patches are hard-coded
                    B = x.size(0)
                    device = x.get_device()
                    noise = torch.rand(B, 196, device=device)
                    mask_index = torch.argsort(noise, dim=1)
                    mask = mask_index[:, :int(196*(1-mask_ratio))] # consider the cls token
                    mask = mask.long()
                    masks.append(mask)
        else:
            masks = None

        return masks

    def forward(self, x, m, loss='byol'):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: ema momentum
        Output:
            loss
        """
        if isinstance(x, list):
            x1 = x[0]
            x2 = x[1]
        else:
            x1 = x
            x2 = x

        # compute features
        B,_,_,_ = x1.size()
        device = x1.get_device()

        mask_s = self.generate_mask(x1, self.num_masks, self.mask_ratio)
        mask_t = self.generate_mask(x1, 1, 0.) 

        q1 = self.predictor(self.base_encoder(x1, mask_s, self.mask_ratio))
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            k2 = self.momentum_encoder(x2, mask_t)
            
        if loss == "byol":
            q1 = torch.chunk(q1, self.num_masks, dim=0)
            loss = 0.
            for q1i in q1:
                loss += self.byol_loss(q1i, k2)
            return loss / self.num_masks 

        elif loss == 'infonce':
            q1 = torch.chunk(q1, self.local_crops, dim=0)
            loss = 0.
            for q1i in q1:
                loss += self.contrastive_loss(q1i, k2)
            return loss / self.num_masks
      
# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# utils
@torch.no_grad()
def mean_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # print(tensor.size())
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.mean(torch.cat(tensors_gather,dim=0))
    return output
