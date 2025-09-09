import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pywt

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[(day_emb[:, :, :] * self.time).type(torch.LongTensor)]
        time_day = time_day.transpose(1, 2).contiguous()

        week_emb = x[..., 2]
        time_week = self.time_week[(week_emb[:, :, :]).type(torch.LongTensor)]
        time_week = time_week.transpose(1, 2).contiguous()

        tem_emb = time_day + time_week

        tem_emb = tem_emb.permute(0,3,1,2)

        return tem_emb

class TATT_1(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TATT_1, self).__init__()

        self.conv1 = nn.Conv2d(c_in, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(num_nodes, 1, kernel_size=(1, 1),
                            stride=(1, 1), bias=False)
        self.w = nn.Parameter(torch.rand(num_nodes, c_in), requires_grad=True)
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.zeros(tem_size, tem_size), requires_grad=True)
        self.v = nn.Parameter(torch.rand(tem_size, tem_size), requires_grad=True)
        nn.init.xavier_uniform_(self.v)
        # nn.init.xavier_uniform_(self.b)
        self.bn = nn.BatchNorm1d(tem_size)

    def forward(self, seq):

        seq = seq.transpose(3, 2)

        seq = seq.permute(0, 1, 3, 2).contiguous()
        c1 = seq.permute(0, 1, 3, 2)  # b,c,n,l->b,c,l,n
        f1 = self.conv1(c1).squeeze()  # b,l,n

        c2 = seq.permute(0, 2, 1, 3)  # b,c,n,l->b,n,c,l
        f2 = self.conv2(c2).squeeze(axis=1)  # b,c,n  [50, 1, 12]

        logits = torch.sigmoid(torch.matmul(torch.matmul(f1, self.w), f2) + self.b)
        logits = torch.matmul(self.v, logits)
        logits = logits.permute(0, 2, 1).contiguous()

        logits = self.bn(logits).permute(0, 2, 1).contiguous()

        coefs = torch.softmax(logits, -1)
        T_coef = coefs.transpose(-1, -2)

        x_1 = torch.einsum('bcnl,blq->bcnq', seq, T_coef)

        return x_1

class DualChannelLearner(nn.Module):
    def __init__(self, features=128, layers=4, length=12, num_nodes=170, dropout=0.1):
        super(DualChannelLearner, self).__init__()


        self.low_freq_layers = nn.ModuleList([
            TATT_1(features, num_nodes, length) for _ in range(layers)
        ])


        kernel_size = int(length / layers + 1)
        dilation_size = None
        self.high_freq_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    features, 
                    features, 
                    (1, kernel_size)),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for _ in range(layers)
        ])

        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, XL, XH):
        res_xl = XL
        res_xh = XH

        for layer in self.low_freq_layers:
            XL = layer(XL)
        XL = XL + res_xl


        XH = nn.functional.pad(XH, (1, 0, 0, 0))  
        for layer in self.high_freq_layers:
            XH = layer(XH)  
        #print(XH.shape)
        XH = XH + res_xh

        alpha_sigmoid = torch.sigmoid(self.alpha)
        output = alpha_sigmoid * XL + (1 - alpha_sigmoid) * XH

        return output 
    
class Diffusion_GCN(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention——> [B, N, N]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = ModuleList(
                [GraphAttentionLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = GraphAttentionLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x    

class Graph_Generator(nn.Module):
    def __init__(self, channels=128, num_nodes=170, length=12, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.E_s = nn.Parameter(torch.randn(channels, num_nodes))
        self.length = length
        self.E_tod = nn.Parameter(torch.randn(1, channels, 1, length))
        self.E_dow = nn.Parameter(torch.randn(1, channels, 1, length))
        nn.init.xavier_uniform_(self.E_s)
        nn.init.xavier_uniform_(self.E_tod)
        nn.init.xavier_uniform_(self.E_dow)
        self.fc = nn.Linear(2,1)
        
    def forward(self, x):
        #E_t = X*E_w*E_d
        E_t = x*self.E_tod*self.E_tod

        E_d = torch.tanh(
            torch.einsum("bcnt, cm->bnm", x, self.E_s).contiguous()
        )

        A_adp = torch.softmax(
            F.relu(
                torch.einsum("bnc, bmc->bnm", E_d, E_d).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        
        return (A_adp.mean(dim=0) > 0.5).float()


class HybridGraphLearner(nn.Module):
    def __init__(self, channels=128, layers = 2, num_nodes=170, length=12, diffusion_step=1, dropout=0.1, emb=None, device=None):
        super().__init__()
        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.layers = layers
        self.generator = Graph_Generator(channels, num_nodes, length, diffusion_step, dropout)
        
        self.gcn = nn.ModuleList([
            Diffusion_GCN(channels, 
                          diffusion_step, 
                          dropout)
            for _ in range(self.layers)
        ])
        self.gat = nn.ModuleList([
            GAT(channels, 
                channels, 
                dropout, 
                alpha=0.2, 
                nheads=1)
            for _ in range(self.layers)
        ])
        
        self.device = device
        self.alpha = nn.Parameter(torch.tensor(-5.0)) 
        self.emb =  nn.Parameter(torch.randn(channels, num_nodes, 12))
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 6).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(6, num_nodes).to(device), requires_grad=True).to(device)
        
        self.conv_gcn = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        
        self.conv_gat = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        
        self.conv_fusion = nn.Conv2d(in_channels=channels,
                                    out_channels=channels,
                                    kernel_size=(1, 1))
        self.alpha = nn.Parameter(torch.tensor(-5.0))
    def forward(self, x):
        
        A_adp = F.softmax(
                    F.relu(torch.mm(self.nodevec1, self.nodevec2)),
                    dim=1).unsqueeze(0)
        skip = x
        x = self.conv(x)
        A_dyn = self.generator(x)
        for gcn_layer in self.gcn:
            x_gcn = gcn_layer(x, A_adp) 
        x_gcn = self.conv_gcn(x_gcn)
            #print(x_gcn.shape)
        for gat_layer in self.gat:
            x_gat = gat_layer(x.transpose(1, 3), A_dyn).transpose(1, 3)
        x_gat = self.conv_gat(x_gat)
        #spatial gate
        alpha_sigmoid = torch.sigmoid(self.alpha)
        x = alpha_sigmoid * x_gcn + (1 - alpha_sigmoid) * x_gat
        x = self.conv_fusion(x)
        x = x*self.emb + skip
        return x



class HSTGNN(nn.Module):
    def __init__(
        self, device, input_dim, num_nodes, channels, granularity, dropout=0.1
    ):
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.output_len = 12
        diffusion_step = 1

        self.Temb = TemporalEmbedding(granularity, channels)

        self.start_conv_l = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )
        self.start_conv_h = nn.Conv2d(
            in_channels=input_dim, out_channels=channels, kernel_size=(1, 1)
        )


        self.layers = 4
        self.DCL = DualChannelLearner(
            features = channels, 
            layers = self.layers, 
            length = self.output_len, 
            num_nodes = self.num_nodes, 
            dropout=0.1
        )
        
        self.HGL = HybridGraphLearner(
            channels=channels*2, 
            layers = self.layers, 
            num_nodes=num_nodes, 
            length = self.output_len, 
            diffusion_step=1, 
            dropout=0.1, 
            emb=None,
            device=self.device
        )

        self.glu = GLU(channels*2, dropout)

        self.regression_layer = nn.Conv2d(
            channels*2, self.output_len, kernel_size=(1, self.output_len)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input):
        
        input_data = input
        #decoupling layer
        residual_cpu = input_data.cpu()
        residual_numpy = residual_cpu.detach().numpy()
        coef = pywt.wavedec(residual_numpy, 'db1', level=2)
        coefl = [coef[0]] + [None] * (len(coef) - 1)
        coefh = [None] + coef[1:]
        xl = pywt.waverec(coefl, 'db1')
        xh = pywt.waverec(coefh, 'db1')
        xl = torch.from_numpy(xl).to(self.device)
        xh = torch.from_numpy(xh).to(self.device)
        #start conv
        x = input
        x_l = self.start_conv_l(xl + x)
        x_h = self.start_conv_h(xh + x)
        #DCL
        x = self.DCL(x_l, x_h)
        time_emb = self.Temb(input.permute(0, 3, 2, 1))
        x = torch.cat([x] + [time_emb], dim=1)
        #HGL
        x = self.HGL(x)
        #Output layer
        x_final = self.glu(x) + x
        prediction = self.regression_layer(F.relu(x_final))
        return prediction
    
