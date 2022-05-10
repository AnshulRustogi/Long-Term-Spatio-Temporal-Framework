import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.graph import Graph

class Model(nn.Module):

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.edge_type = 2

        temporal_kernel_size = 9
        spatial_kernel_size = A.size(0) + self.edge_type
        st_kernel_size = (temporal_kernel_size, spatial_kernel_size)

        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        self.class_layer_0 = StgcnBlock(in_channels, 64, st_kernel_size, self.edge_type, stride=1, residual=False, **kwargs)
        self.class_layer_1 = StgcnBlock(64, 64, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_2 = StgcnBlock(64, 64, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_3 = StgcnBlock(64, 128, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.class_layer_4 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_5 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_6 = StgcnBlock(128, 256, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.class_layer_7 = StgcnBlock(256, 256, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.class_layer_8 = StgcnBlock(256, 256, st_kernel_size, self.edge_type, stride=1, **kwargs)

        self.recon_layer_0 = StgcnBlock(256, 128, st_kernel_size, self.edge_type, stride=1, **kwargs)
        self.recon_layer_1 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.recon_layer_2 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.recon_layer_3 = StgcnBlock(128, 128, st_kernel_size, self.edge_type, stride=2, **kwargs)
        self.recon_layer_4 = StgcnBlock(128, 128, (3, spatial_kernel_size), self.edge_type, stride=2, **kwargs)
        self.recon_layer_5 = StgcnBlock(128, 128, (5, spatial_kernel_size), self.edge_type, stride=1, padding=False, residual=False, **kwargs)
        self.recon_layer_6_encoder_q = StgcnReconBlock(128+3, 30, (1, spatial_kernel_size), self.edge_type, stride=1, padding=False, residual=False, activation=None, **kwargs)
        self.recon_layer_6_encoder_k = StgcnReconBlock(128+3, 30, (1, spatial_kernel_size), self.edge_type, stride=1, padding=False, residual=False, activation=None, **kwargs)
        #########################

        self.dropout_classifier = nn.Dropout(0.60, inplace=False)
        self.dropout_1 = nn.Dropout(0.50, inplace=False)
        self.dropout_2 = nn.Dropout(0.75, inplace=False)
        self.dropout_3 = nn.Dropout(0.25, inplace=False)



        self.K =  32768                   #queue_size
        self.m =  0.999                   #momentum
        self.T =  0.07                   #Temperature
        for param_q, param_k in zip(self.recon_layer_6_encoder_q.parameters(), self.recon_layer_6_encoder_k.parameters()):
            param_k.data.copy_(param_q.data)    # initialize
            param_k.requires_grad = False       # not update by gradient
            param_k.requires_grad_ = False
        # create the queue
        self.register_buffer("queue", torch.randn(self.K, 3, 10, 25))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        ##########################
        #self.norm = nn.BatchNorm2d(3, affine=False).cuda()
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in range(9)])
            self.edge_importance_recon = nn.ParameterList([nn.Parameter(torch.ones(self.A.size())) for i in range(9)])
        else:
            self.edge_importance = [1] * (len(self.st_gcn_networks)+len(self.st_gcn_recon))
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, x_target, x_last, A_act, lamda_act):
        N, C, T, V, M = x.size()
        #x_recon = x[:,:,:,:,0]                                  # [2N, 3, 300, 25]
        x = x.permute(0, 4, 3, 1, 2).contiguous()               # [N, 2, 25, 3, 300]
        x = x.view(N * M, V * C, T)                             # [2N, 75, 300]
        x_last = x_last.permute(0,4,1,2,3).contiguous().view(-1,3,1,25)

        x_bn = self.data_bn(x)
        x_bn = x_bn.view(N, M, V, C, T)
        x_bn = x_bn.permute(0, 1, 3, 4, 2).contiguous()
        x_bn = x_bn.view(N * M, C, T, V)

        h0, _ = self.class_layer_0(x_bn, self.A * self.edge_importance[0], A_act, lamda_act)       # [2N, 64, 300, 25]
        h0, _ = self.class_layer_1(h0, self.A * self.edge_importance[1], A_act, lamda_act)         # [2N, 64, 300, 25]
        h0, _ = self.class_layer_1(h0, self.A * self.edge_importance[1], A_act, lamda_act)         # [2N, 64, 300, 25]
        h0, _ = self.class_layer_2(h0, self.A * self.edge_importance[2], A_act, lamda_act)         # [2N, 64, 300, 25]
        h0 = self.dropout_1(h0)
        h0, _ = self.class_layer_3(h0, self.A * self.edge_importance[3], A_act, lamda_act)         # [2N, 128, 150, 25]
        h0, _ = self.class_layer_4(h0, self.A * self.edge_importance[4], A_act, lamda_act)         # [2N, 128, 150, 25]
        h0, _ = self.class_layer_5(h0, self.A * self.edge_importance[5], A_act, lamda_act)         # [2N, 128, 150, 25]
        h0 = self.dropout_3(h0)
        h0, _ = self.class_layer_6(h0, self.A * self.edge_importance[6], A_act, lamda_act)         # [2N, 256, 75, 25]
        h0, _ = self.class_layer_7(h0, self.A * self.edge_importance[7], A_act, lamda_act)         # [2N, 256, 75, 25]
        h8, _ = self.class_layer_8(h0, self.A * self.edge_importance[8], A_act, lamda_act)         # [2N, 256, 75, 25]

        x_class = F.avg_pool2d(h8, h8.size()[2:])
        x_class = x_class.view(N, M, -1, 1, 1).mean(dim=1)
        x_class = self.fcn(self.dropout_classifier(x_class))
        x_class = x_class.view(x_class.size(0), -1)

        #
        r0, _ = self.recon_layer_0(h8, self.A*self.edge_importance_recon[0], A_act, lamda_act)                          # [2N, 128, 75, 25]
        r0, _ = self.recon_layer_1(r0, self.A*self.edge_importance_recon[1], A_act, lamda_act)                          # [2N, 128, 38, 25]
        r0, _ = self.recon_layer_2(r0, self.A*self.edge_importance_recon[2], A_act, lamda_act)                          # [2N, 128, 19, 25]
        r0, _ = self.recon_layer_3(r0, self.A*self.edge_importance_recon[3], A_act, lamda_act)                          # [2N, 128, 10, 25]
        r0, _ = self.recon_layer_4(r0, self.A*self.edge_importance_recon[4], A_act, lamda_act)                          # [2N, 128, 5, 25]
        r5, _ = self.recon_layer_5(r0, self.A*self.edge_importance_recon[5], A_act, lamda_act)                          # [2N, 128, 1, 25]

        r5_key = r5.clone().detach().cuda() + (0.1**0.5)*torch.randn(N*M, 128, 1, 25).cuda()
        r5_key.requires_grad = True
        r5_key.requires_grad_ = True
        r5, r5_key = self.dropout_2(r5), self.dropout_2(r5_key)
        r6, _ = self.recon_layer_6_encoder_q(torch.cat((r5, x_last),1), self.A*self.edge_importance_recon[6], A_act, lamda_act)   # [2N, 64, 1, 25]

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            r6_key, _ = self.recon_layer_6_encoder_k(torch.cat((r5_key, x_last),1), self.A*self.edge_importance_recon[6], A_act, lamda_act)   # [2N, 64, 1, 25]

        #Generating Positive Pair
        pred = x_last.squeeze().repeat(1,10,1) + r6.squeeze()                                                  # [2N, 30, 25]
        pred_key = x_last.squeeze().repeat(1,10,1) + r6_key.squeeze()

        #Final Positive Pair
        pred = pred.contiguous().view(-1, 3, 10, 25)[::2]                                                      # [N, 3, 10, 25]
        #*******Will get stored in memory bank******* and used a negative pair for other batches
        pred_key = pred_key.contiguous().view(-1, 3, 10, 25)[::2]                                              # [N, 3, 10, 25]
        #
        #pred = pred.contiguous().view(-1, 3, 10, 25)
        x_target = x_target.permute(0,4,1,2,3).contiguous().view(-1,3,10,25)

        #Storing the pair in queue
        #loss_contrastive = self.loss_infoNCE(pred, pred_key)
        #self._dequeue_and_enqueue(pred_key)
        return x_class, pred, pred_key, x_target[::2]

    def loss_infoNCE_not_using(self, query, key):
        query = query.clone().detach()
        key = key.clone().detach()
        batch_size,_,_,_ = query.size()
        l_pos_MSE = self.loss_similarity(query, key)
        l_neg_MSE = 0
        maxItem = self.K//batch_size - 1
        start = 0
        for i in range(maxItem):
            negative_pair = self.queue[(start*batch_size):((start+1)*batch_size),:,:,:]
            l_neg_MSE += self.loss_similarity(query, negative_pair)
            start += 1
        loss_contrastive = -torch.log(1e-12+(l_pos_MSE/(l_neg_MSE)))
        return loss_contrastive

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        gpu_index = keys.device.index
        self.queue[(ptr + batch_size * gpu_index):(ptr + batch_size * (gpu_index + 1)),:,:,:] = keys

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.recon_layer_6_encoder_q.parameters(), self.recon_layer_6_encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def update_ptr(self, batch_size):
        assert self.K % batch_size == 0 #  for simplicity
        self.queue_ptr[0] = (self.queue_ptr[0] + batch_size) % self.K

    def extract_feature(self, x):

        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class StgcnBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 edge_type=2,
                 t_kernel_size=1,
                 stride=1,
                 padding=True,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        if padding == True:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0,0)

        self.gcn = SpatialGcn(in_channels=in_channels,
                              out_channels=out_channels,
                              k_num=kernel_size[1],
                              edge_type=edge_type,
                              t_kernel_size=t_kernel_size)
        self.tcn = nn.Sequential(nn.BatchNorm2d(out_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(out_channels,
                                           out_channels,
                                           (kernel_size[0], 1),
                                           (stride, 1),
                                           padding),
                                 nn.BatchNorm2d(out_channels),
                                 nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=(stride, 1)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, B, lamda_act):

        res = self.residual(x)
        x, A = self.gcn(x, A, B, lamda_act)
        x = self.tcn(x) + res

        return self.relu(x), A

class StgcnReconBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 edge_type=2,
                 t_kernel_size=1,
                 stride=1,
                 padding=True,
                 dropout=0,
                 residual=True,
                 activation='relu'):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        if padding == True:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (0,0)

        self.gcn_recon = SpatialGcnRecon(in_channels=in_channels,
                                         out_channels=out_channels,
                                         k_num=kernel_size[1],
                                         edge_type=edge_type,
                                         t_kernel_size=t_kernel_size)
        self.tcn_recon = nn.Sequential(nn.BatchNorm2d(out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(in_channels=out_channels,
                                                          out_channels=out_channels,
                                                          kernel_size=(kernel_size[0], 1),
                                                          stride=(stride, 1),
                                                          padding=padding,
                                                          output_padding=(stride-1,0)),
                                       nn.BatchNorm2d(out_channels),
                                       nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                             out_channels=out_channels,
                                                             kernel_size=1,
                                                             stride=(stride, 1),
                                                             output_padding=(stride-1,0)),
                                          nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.activation = activation

    def forward(self, x, A, B, lamda_act):

        res = self.residual(x)
        x, A = self.gcn_recon(x, A, B, lamda_act)
        x = self.tcn_recon(x) + res
        if self.activation == 'relu':
            x = self.relu(x)
        else:
            x = x

        return x, A

class SpatialGcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 k_num,
                 edge_type=2,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels*k_num,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x, A, B, lamda_act):

        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num,  kc//self.k_num, t, v)
        x1 = x[:,:self.k_num-self.edge_type,:,:,:]
        x2 = x[:,-self.edge_type:,:,:,:]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x_sum = x1+x2*lamda_act

        return x_sum.contiguous(), A

class SpatialGcnRecon(nn.Module):

    def __init__(self, in_channels, out_channels, k_num, edge_type=3,
                 t_kernel_size=1, t_stride=1, t_padding=0, t_outpadding=0, t_dilation=1,
                 bias=True):
        super().__init__()

        self.k_num = k_num
        self.edge_type = edge_type
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels*k_num,
                                         kernel_size=(t_kernel_size, 1),
                                         padding=(t_padding, 0),
                                         output_padding=(t_outpadding, 0),
                                         stride=(t_stride, 1),
                                         dilation=(t_dilation, 1),
                                         bias=bias)

    def forward(self, x, A, B, lamda_act):

        x = self.deconv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.k_num,  kc//self.k_num, t, v)
        x1 = x[:,:self.k_num-self.edge_type,:,:,:]
        x2 = x[:,-self.edge_type:,:,:,:]
        x1 = torch.einsum('nkctv,kvw->nctw', (x1, A))
        x2 = torch.einsum('nkctv,nkvw->nctw', (x2, B))
        x_sum = x1+x2*lamda_act

        return x_sum.contiguous(), A

