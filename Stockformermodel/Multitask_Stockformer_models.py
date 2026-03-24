import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def maml_init_(module):
    if isinstance(module, nn.Linear):  # Broader linear layer initialization
        nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LSTM):  # Initialize RNN-like layers
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    elif isinstance(module, nn.Module):  # Check custom modules
        if hasattr(module, 'end_emb_class'):
            maml_init_(module.end_emb_class)
        if hasattr(module, 'end_emb_regress'):
            maml_init_(module.end_emb_regress)
    else:
        # Generic initialization
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    return module

class temporalEmbedding(nn.Module):
    def __init__(self, D):
        super(temporalEmbedding, self).__init__()
        self.ff_te = FeedForward([55,D,D])

    def forward(self, TE, T=50):
        '''
        TE:[B,T,2]
        '''
        dayofweek = torch.empty(TE.shape[0], TE.shape[1], 5, device=TE.device) # [B,T,5]
        timeofday = torch.empty(TE.shape[0], TE.shape[1], T, device=TE.device) # [B,T,50]
        for i in range(TE.shape[0]):
            dayofweek[i] = F.one_hot(TE[..., 0][i].to(torch.int64) % 5, 5)
        for j in range(TE.shape[0]):
            timeofday[j] = F.one_hot(TE[..., 1][j].to(torch.int64) % 50, T)
        TE = torch.cat((dayofweek, timeofday), dim=-1) # [B,T,55]
        TE = TE.unsqueeze(dim=2) # [B,T,1,55]
        TE = self.ff_te(TE) # [B,T,1,F]

        return TE  # [B,T,1,F]

class FeedForward(nn.Module):
    def __init__(self, fea, res_ln=False):
        super(FeedForward, self).__init__()
        self.res_ln = res_ln
        self.L = len(fea) - 1
        self.linear = nn.ModuleList([nn.Linear(fea[i], fea[i+1]) for i in range(self.L)])
        self.ln = nn.LayerNorm(fea[self.L], elementwise_affine=False)

    def forward(self, inputs):
        x = inputs
        for i in range(self.L):
            x = self.linear[i](x)
            if i != self.L-1:
                x = F.relu(x)
        if self.res_ln:
            x += inputs
            x = self.ln(x)
        return x

class sparseSpatialAttention(nn.Module):
    def __init__(self, features, h, d, s, dropout=0.2):
        super(sparseSpatialAttention, self).__init__()
        self.qfc = nn.Linear(features, features)
        self.kfc = nn.Linear(features, features)
        self.vfc = nn.Linear(features, features)
        self.ofc = nn.Linear(features, features)

        self.h = h
        self.d = d
        self.s = s
        self.attn_dropout = nn.Dropout(dropout)

        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = nn.Sequential(nn.Linear(features, features),
                                 nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(features, features))

        self.proj = nn.Linear(d, 1)

    def forward(self, x, adjgat):
        '''
        [B,T,N,D]
        '''
        # add spatial positional encoding
        x_ = x + adjgat

        Q = self.qfc(x_)
        K = self.kfc(x_)
        V = self.vfc(x_)

        B, T, N, D = Q.shape

        Q_K = torch.matmul(Q, K.transpose(-2, -1))

        Q_K /= (self.d ** 0.5)

        attn = self.attn_dropout(torch.softmax(Q_K, dim=-1))

        # Copy operation: for each stock n, find which stock attends most to n,
        # then copy that stock's attended value. Uses gather instead of expand
        # to avoid O(B*T*N*N*D) memory allocation.
        attended = torch.matmul(attn, V)  # [B, T, N, D]
        cp = attn.argmax(dim=-2)  # [B, T, N] — which stock attends most to each n
        cp_expanded = cp.unsqueeze(-1).expand(B, T, N, D)  # [B, T, N, D]
        value = torch.gather(attended, dim=2, index=cp_expanded)  # [B, T, N, D]

        value = self.ofc(value) + x_
        value = self.ln(value)
        return self.ff(value)

class temporalAttention(nn.Module):
    def __init__(self, features, h, d, dropout=0.2):
        super(temporalAttention, self).__init__()
        self.qfc = FeedForward([features,features])
        self.kfc = FeedForward([features,features])
        self.vfc = FeedForward([features,features])
        self.ofc = FeedForward([features,features])
        self.h = h
        self.d = d
        self.attn_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features,features,features], True)

    def forward(self, x, te, Mask=True):
        '''
        x:[B,T,N,F]
        te:[B,T,N,F]
        '''
        x += te

        query = self.qfc(x).permute(0,2,1,3) #[B,T,N,F]
        key = self.kfc(x).permute(0,2,3,1) #[B,T,N,F]
        value = self.vfc(x).permute(0,2,1,3) #[B,T,N,F]

        attention = torch.matmul(query, key) # [k*B,N,T,T]
        attention /= (self.d ** 0.5) # scaled

        if Mask:
            batch_size = x.shape[0]
            num_steps = x.shape[1]
            num_vertexs = x.shape[2]
            mask = torch.ones(num_steps, num_steps, device=x.device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attention) # [k*B,N,T,T]
            attention = torch.where(mask, attention, zero_vec)

        attention = self.attn_dropout(F.softmax(attention, -1)) # [k*B,N,T,T]

        value = torch.matmul(attention, value).permute(0,2,1,3) # [k*B,N,T,d]
        value = self.ofc(value)
        value += x

        value = self.ln(value)

        return self.ff(value)

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()

class temporalConvNet(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(temporalConvNet, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)
    
    def forward(self, xh):
        xh = self.tcn(xh.transpose(1,3)).transpose(1,3)
        return xh

class adaptiveFusion(nn.Module):
    def __init__(self, features, h, d, dropout=0.2):
        super(adaptiveFusion, self).__init__()
        self.qlfc = FeedForward([features,features])
        self.klfc = FeedForward([features,features])
        self.vlfc = FeedForward([features,features])
        self.khfc = FeedForward([features,features])
        self.vhfc = FeedForward([features,features])
        self.ofc = FeedForward([features,features])
        self.h = h
        self.d = d
        self.attn_dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(features, elementwise_affine=False)
        self.ff = FeedForward([features,features,features], True)

    def forward(self, xl, xh, te, Mask=True):
        '''
        xl: [B,T,N,F]
        xh: [B,T,N,F]
        te: [B,T,1,F]
        '''
        xl += te
        xh += te

        query = self.qlfc(xl).permute(0,2,1,3) # [B,T,N,F]
        keyh = torch.relu(self.khfc(xh)).permute(0,2,3,1) # [B,T,N,F]
        valueh = torch.relu(self.vhfc(xh)).permute(0,2,1,3) # [B,T,N,F]

        attentionh = torch.matmul(query, keyh) # [k*B,N,T,T]
        
        if Mask:
            batch_size = xl.shape[0]
            num_steps = xl.shape[1]
            num_vertexs = xl.shape[2]
            mask = torch.ones(num_steps, num_steps, device=xl.device) # [T,T]
            mask = torch.tril(mask) # [T,T]
            mask = torch.unsqueeze(torch.unsqueeze(mask, dim=0), dim=0) # [1,1,T,T]
            mask = mask.repeat(self.h * batch_size, num_vertexs, 1, 1) # [k*B,N,T,T]
            mask = mask.to(torch.bool)
            zero_vec = (-2 ** 15 + 1)*torch.ones_like(attentionh) # [k*B,N,T,T]
            attentionh = torch.where(mask, attentionh, zero_vec)
        attentionh /= (self.d ** 0.5) # scaled
        attentionh = self.attn_dropout(F.softmax(attentionh, -1)) # [k*B,N,T,T]


        value = torch.matmul(attentionh, valueh).permute(0,2,1,3)
        value = self.ofc(value)
        value = value + xl #+ xh

        value = self.ln(value)

        return self.ff(value)

class dualEncoder(nn.Module):
    def __init__(self, features, h, d, s):
        super(dualEncoder, self).__init__()
        self.tcn = temporalConvNet(features)
        self.tatt = temporalAttention(features, h, d)
        
        self.ssal = sparseSpatialAttention(features, h, d, s)
        self.ssah = sparseSpatialAttention(features, h, d, s)
        
        
    def forward(self, xl, xh, te, adjgat):
        xl = self.tatt(xl, te)
        xh = self.tcn(xh)
        

        spa_statesl = self.ssal(xl,adjgat)
        spa_statesh = self.ssah(xh,adjgat)
        xl = spa_statesl + xl
        xh = spa_statesh + xh
        
        return xl, xh


class StockformerBackbone(nn.Module):
    def __init__(self, infea, outfea, L, h, d, s, T1, T2, dev, noise_std=0.01):
        super(StockformerBackbone, self).__init__()
        self.noise_std = noise_std
        self.outfea = outfea
        self.adjgat_proj = None  # lazy init when adjgat dim != outfea
        self.start_emb_l = FeedForward([infea, outfea, outfea])
        self.start_emb_h = FeedForward([infea, outfea, outfea])
        self.te_emb = temporalEmbedding(outfea)

        self.dual_encoder = nn.ModuleList([dualEncoder(outfea, h, d, s) for i in range(L)])
        self.adaptive_fusion = adaptiveFusion(outfea, h, d)

        self.pre_l = nn.Conv2d(T1, T2, (1,1))
        self.pre_h = nn.Conv2d(T1, T2, (1,1))

    def forward(self, xl, xh, te, bonus, indicator, adjgat):
        # Project adjgat to model dimension if needed
        if adjgat.shape[-1] != self.outfea:
            if self.adjgat_proj is None or self.adjgat_proj.in_features != adjgat.shape[-1]:
                self.adjgat_proj = nn.Linear(adjgat.shape[-1], self.outfea, bias=False).to(adjgat.device)
            adjgat = self.adjgat_proj(adjgat)
        '''
        x:[B,T,N]
        indicator:[B,T,N]
        bonus:[B,T,N,D2]
        '''
        xl, xh, indicator = xl.unsqueeze(-1), xh.unsqueeze(-1), indicator.unsqueeze(-1)
        xl = torch.concat([xl,indicator,bonus],dim = -1)
        xh = torch.concat([xh,indicator,bonus],dim = -1)
        xl, xh, TE = self.start_emb_l(xl), self.start_emb_h(xh), self.te_emb(te)

        # Gaussian noise injection during training (acts as implicit regularization)
        if self.training and self.noise_std > 0:
            xl = xl + torch.randn_like(xl) * self.noise_std
            xh = xh + torch.randn_like(xh) * self.noise_std

        for enc in self.dual_encoder:
            xl, xh = enc(xl, xh, TE[:,:xl.shape[1],:,:], adjgat)
        
        hat_y_l = self.pre_l(xl)
        hat_y_h = self.pre_h(xh)

        hat_y = self.adaptive_fusion(hat_y_l, hat_y_h, TE[:,xl.shape[1]:,:,:])

        return hat_y, hat_y_l
    
class StockformerOutput(nn.Module):
    def __init__(self, outfea, outfea_class, outfea_regress, dev):
        super(StockformerOutput, self).__init__()
        # Classification output layer; dimension by number of classes
        self.end_emb_class = FeedForward([outfea, outfea, outfea_class])
        # Regression output layer; dimension by number of regression targets
        self.end_emb_regress = FeedForward([outfea, outfea, outfea_regress])
        
    def forward(self, hat_y, hat_y_l):
        '''
        x:[B,T,N]
        hat_y, hat_y_l: two feature representations, both used for classification and regression
        '''
        # Classification output
        hat_y_class = self.end_emb_class(hat_y)
        hat_y_l_class = self.end_emb_class(hat_y_l)
        # Regression output
        hat_y_regress = self.end_emb_regress(hat_y)
        hat_y_l_regress = self.end_emb_regress(hat_y_l)
        
        return hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress


class Stockformer(torch.nn.Module):
    def __init__(self, infea, outfea, outfea_class, outfea_regress, L, h, d, s, T1, T2, dev):
        super(Stockformer, self).__init__()
        self.features = StockformerBackbone(infea, outfea, L, h, d, s, T1, T2, dev)
        self.classifier = StockformerOutput(outfea, outfea_class, outfea_regress, dev)
        maml_init_(self.classifier)

    def forward(self, xl, xh, te, bonus, indicator, adjgat):
        hat_y, hat_y_l = self.features(xl, xh, te, bonus, indicator, adjgat)
        hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = self.classifier(hat_y, hat_y_l)
        return hat_y_class, hat_y_l_class, hat_y_regress.squeeze(-1), hat_y_l_regress.squeeze(-1)
