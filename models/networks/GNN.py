import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GNN(nn.Module):
    def __init__(self, n_F, n_gnn = 1):
        '''
            Time complexity: O(NNF+NFF)
        '''
        super().__init__()
        self.relu = nn.ReLU()
        self.n_gnn = n_gnn
        self.W = nn.ParameterList([ nn.Parameter(torch.empty((n_F, n_F))) for _ in range(n_gnn)])
        self.bn = nn.ModuleList([ LayerNorm(n_F) for _ in range(n_gnn) ])

        ## init self.W
        for i in range(n_gnn):
            torch.nn.init.xavier_uniform_(self.W[i])
    
    def forward(self, A, X, reverse=False):
        '''helper:
            Time complexity: O(NNF+NFF)
            A: adjacent matrix bs, N, N
            X: features bs, N, F

            return bs, N, F
        '''
        A = nn.Softmax(dim=-1)(A) ## Laplacian Smoothing http://www4.comp.polyu.edu.hk/~csxmwu/papers/AAAI-2018-GCN.pdf
        if reverse: A = 1.0 - A
        for i in range(self.n_gnn):
            node = torch.matmul(A, self.bn[i](X) )
            X = self.relu(torch.matmul(node, self.W[i])) 
        return X


class SpatialGNN(nn.Module):
    def __init__(self, n_F, h, w, n_gnn = 1, distance = "manhattan"):
        '''helper
            h, w: the height, width of the input features (bs, F, h, w)
            distance: manhattan or euler for generate distance-related adjacent matrix (default: manhattan)
        '''
        super().__init__()
        ## solve for spatial-related Adjacent matrix A
        N = int(h*w)
        self.A = nn.Parameter(torch.empty((N,N)), requires_grad=False)
        if distance=="manhattan":
            dist = lambda x, y: float(abs(x//w-y//w)+abs(x%w-y%w))
        elif distance=="euler":
            dist = lambda x, y: math.sqrt(float((x//w-y//w)**2+(x%w-y%w)**2))
        else:
            raise "unknow distance options!"
        for i in range(N):
            for j in range(i, N):
                self.A[i, j] = -dist(i,j)
                self.A[j, i] = -dist(j,i)
        ## n_gnn
        self.gnn = GNN(n_F=n_F, n_gnn=n_gnn)
    
    def forward(self, X, channels_last=True, reverse=False):
        ''' helper:
            channels_last: channels is in the last dim (dim=-1)?
            X: bs, F, N when channels_last=False
            X: bs, N, F when channels_last=True

            return X bs, N, F
        '''
        # print('self.A', self.A)
        if channels_last==False:
            X = X.permute(0, 2, 1) ## -> bs, N, F
        return self.gnn(self.A, X, reverse)



if __name__ == "__main__":

    ### Test O2OAtten
    # SP = torch.rand(40, 27, 64, 64)
    # x = torch.rand(40, 512, 16, 16)
    SP = torch.rand(4, 10, 27, 64, 64)
    x = torch.rand(4, 10, 512, 16, 16)

    # x = torch.rand(4, 25, 64, 64).cuda()

    backbone_img_layer_infos = [
        [64, 7, 2, 3], ###inchannel 64, n_class -> inchannel, W,H:128->64
        [3, 2, 1], ### for maxpool, W,H:64->32
        [3, 64, 3, 2, 1, 1],  ###outchannel 64, num_block 3, at first block, a down_sample is used to change inchannel to outchannel(64), W,H:32->16
        [4, 128, 3, 1, 1, 1],  ###outchannel 128
        [6, 256, 3, 1, 1, 1],  ###outchannel 256
        [3, 512, 3, 1, 1, 1]  ###outchannel 512, in Encoder and Decoder the W and H keep same, only the channel is changed.
    ]

    backbone_word_layer_infos = [
        [64, 7, 2, 3],
        [3, 2, 1],
        [3, 64, 3, 2, 1, 1],
        [4, 128, 3, 1, 1, 1],
        [6, 256, 3, 1, 1, 1],
        [3, 512, 3, 1, 1, 1]
    ]

    upblock_layer_infos = [
        [4, 128, 3, 1, 1, 0, 1],
        [3, 64, 3, 2, 1, 1, 1],
        [4, 64, 3, 2, 1, 1, 1]
    ]

    unet_upblock_layer_infos = [
        [3, 512, 3, 1, 1, 0, 1],
        [6, 256, 3, 1, 1, 0, 1],
        [4, 128, 3, 1, 1, 0, 1],
        [3, 64, 3, 2, 1, 1, 1],
        [4, 64, 3, 2, 1, 1, 1]
    ]

    n_classes, embed_dim, num_heads, batch_sz = 27, 512, 8, 4
    # n_classes, embed_dim, num_heads, batch_sz = 25, 25, 5, 4

    # model = AE(16*16, 27, 512, 512, 512, pool=(1, 1), factor=1)
    # model = ResNetAE(backbone_img_layer_infos, n_classes, upblock_layer_infos)
    model = SSCNavResNetUNetAE(backbone_img_layer_infos, n_classes, unet_upblock_layer_infos)

    # for i in range(10):
    #     model_out = model(x, rgb)
    model_out = model(SP)

    print('model_out shape', model_out.shape)

    # writer = SummaryWriter('./tb/hrnet')
    # writer.add_graph(hrnet, x)
    # writer.close()
