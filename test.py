from tab import TabTransformer

import torch
import torch.nn    as nn


x = torch.randn(64,20)

net = TabTransformer(
            num_features = 20,
            dim = 64,                           # dimension, paper set at 32
                       # binary prediction, but could be anything
            depth = 6,                          # depth, paper recommended 6
            heads = 8,                          # heads, paper recommends 8
            attn_dropout = 0.0,                 # post-attention dropout
            ff_dropout = 0.0,                   # feed forward dropout

            mlp_act = nn.ReLU()               # activation for final mlp, defaults to relu, but could be anything else (selu etc)
            
        )

y = net(x)
print(y.shape)