
import torch
import torch.nn as nn
import torch.nn.functional as F 


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out- 
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d. 
        This MLP has 3 layers.
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3
    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        return x 


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 


class Crop_enc(nn.Module):
    def __init__(self, in_dim=4, out_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, crop):
        emb = self.encoder(crop)
        return emb  # batchsize * 512


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = torch.nn.Identity()

        enc_dim = 512
        self.projector1 = projection_MLP(prev_dim)
        self.projector2 = projection_MLP(prev_dim + enc_dim)
        self.projector3 = projection_MLP(prev_dim + enc_dim)
        self.projector4 = projection_MLP(prev_dim + enc_dim)

        self.predictor1 = prediction_MLP()
        self.predictor2 = prediction_MLP()
        self.predictor3 = prediction_MLP()
        self.predictor4 = prediction_MLP()

        self.color_enc2 = Crop_enc(out_dim=enc_dim)
        self.color_enc3 = Crop_enc(out_dim=enc_dim)
        self.color_enc4 = Crop_enc(out_dim=enc_dim)

    def forward(self, x1_1, x1_2, x2_1, x2_2, x3_1, x3_2, x4_1, x4_2, color_enc, epoch):

        _, feature_list1_1, _, _, _ = self.encoder(x1_1, level=1)
        _, feature_list1_2, _, _, _ = self.encoder(x1_2, level=1)
        _, _, feature_list2_1, _, _ = self.encoder(x2_1, level=2)
        _, _, feature_list2_2, _, _ = self.encoder(x2_2, level=2)
        _, _, _, feature_list3_1, _ = self.encoder(x3_1, level=3)
        _, _, _, feature_list3_2, _ = self.encoder(x3_2, level=3)
        _, _, _, _, feature_list4_1 = self.encoder(x4_1, level=4)
        _, _, _, _, feature_list4_2 = self.encoder(x4_2, level=4)

        # # encoding color
        co2_1, co2_2 = self.color_enc2(color_enc[0]), self.color_enc2(color_enc[1])
        co3_1, co3_2 = self.color_enc3(color_enc[2]), self.color_enc3(color_enc[3])
        co4_1, co4_2 = self.color_enc4(color_enc[4]), self.color_enc4(color_enc[5])

        # # projection head
        z1_1, z1_2 = self.projector1(feature_list1_1), self.projector1(feature_list1_2)
        z2_1, z2_2 = self.projector2(torch.cat([feature_list2_1, co2_1], dim=-1)), self.projector2(torch.cat([feature_list2_2, co2_2], dim=-1))
        z3_1, z3_2 = self.projector3(torch.cat([feature_list3_1, co3_1], dim=-1)), self.projector3(torch.cat([feature_list3_2, co3_2], dim=-1))
        z4_1, z4_2 = self.projector4(torch.cat([feature_list4_1, co4_1], dim=-1)), self.projector4(torch.cat([feature_list4_2, co4_2], dim=-1))
        
        # predictor
        p1_1, p1_2 = self.predictor1(z1_1), self.predictor1(z1_2)
        p2_1, p2_2 = self.predictor2(z2_1), self.predictor2(z2_2)
        p3_1, p3_2 = self.predictor3(z3_1), self.predictor3(z3_2)
        p4_1, p4_2 = self.predictor4(z4_1), self.predictor4(z4_2)
        
        # loss
        L1 = D(p1_1, z1_2) / 2 + D(p1_2, z1_1) / 2
        L2 = D(p2_1, z2_2) / 2 + D(p2_2, z2_1) / 2
        L3 = D(p3_1, z3_2) / 2 + D(p3_2, z3_1) / 2
        L4 = D(p4_1, z4_2) / 2 + D(p4_2, z4_1) / 2

        L = L1 + L2 + L3 + L4
        
        return {'loss': L, 'loss1': L1, 'loss2': L2, 'loss3': L3, 'loss4': L4}

