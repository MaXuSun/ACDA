import torch
from torch import dtype, nn
import torch.nn.functional as F

"""attention 模型"""
class PAM_Module(nn.Module):
    def __init__(self, num, sizes,mode=None):
        super(PAM_Module, self).__init__()
        self.sizes = sizes
        self.mode = mode
        for i in range(num):
            setattr(self, "query" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))
            setattr(self, "value" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))
            setattr(self, "key" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))

    def forward(self, feat_sources, feat_targets):
        """calculate the attention weight and alpha"""
        ret_feats, ret_alphas = [], []
        for i, query in enumerate(feat_targets):
            Bt, Ct, Ht, Wt = query.size()
            pro_query = getattr(self, "query"+str(i))(query).view(Bt, -1, Ht*Wt).permute(0, 2, 1)
            attentions, means = [], []
            for j, key in enumerate(feat_sources):
                pro_key = getattr(self, "key"+str(j))(key).view(Bt, -1, Ht*Wt)
                energy = torch.bmm(pro_query, pro_key)
                means.append(energy.mean().item())
                attentions.append(torch.softmax(energy, dim=-1))
            
            ret_alphas.append(torch.softmax(torch.tensor(means),dim=0))
            attention = torch.stack(attentions, dim=0).sum(0)
            value = getattr(self, "value"+str(i))(query).view(Bt, -1, Ht*Wt)
            out = torch.bmm(value, attention.permute(0, 2, 1)).view(Bt, Ct, Ht, Wt)
            ret_feats.append(out)
                
        ret_alphas = torch.stack(ret_alphas,dim=0)
        return ret_feats, ret_alphas


class CAM_Module(nn.Module):
    def __init__(self, num, sizes,mode=None):
        super(CAM_Module, self).__init__()
        self.sizes = sizes
        self.mode = mode
        for i in range(num):
            setattr(self, "value" + str(i),
                    nn.Conv2d(in_channels=sizes[1], out_channels=sizes[1], kernel_size=1))

    def forward(self, feat_sources, feat_targets):
        ret_feats, ret_alphas = [], []
        for i, query in enumerate(feat_targets):
            Bt, Ct, Ht, Wt = query.size()
            pro_query = query.view(Bt, Ct, -1)
            attentions, means = [], []
            for j, key in enumerate(feat_sources):
                pro_key = key.view(Bt, Ct, -1).permute(0, 2, 1)
                energy = torch.bmm(pro_query, pro_key) 
                means.append(energy.mean().item())
                attentions.append(torch.softmax(energy, dim=-1))

            ret_alphas.append(torch.softmax(torch.tensor(means),dim=0))
            attention = torch.stack(attentions, dim=0).sum(0)
            value = getattr(self, "value"+str(i))(query).view(Bt, Ct, -1)
            out = torch.bmm(attention, value).view(Bt, Ct, Ht, Wt)
            ret_feats.append(out)

        ret_alphas = torch.stack(ret_alphas,dim=0)
        return ret_feats, ret_alphas
        # return _,ret_alphas


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""

    def __init__(self, s_shape, t_shape, factor=1):
        super(ConvReg, self).__init__()
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(
                s_C, t_C//factor, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented(
                'student size {}, teacher size {}'.format(s_H, t_H))

    def forward(self, x):
        x = self.conv(x)
        return x


class Fit(nn.Module):
    def __init__(self, s_shape, t_shape, factor=1):
        super(Fit, self).__init__()
        _, s_C, s_H, s_W = s_shape
        _, t_C, t_H, t_W = t_shape
        if s_H == 2*t_H:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(
                s_C, t_C//factor, kernel_size=4, stride=2, padding=1)
        elif s_H == t_H:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=1, stride=1, padding=0)
        else:
            self.conv = nn.Conv2d(
                s_C, t_C//factor, kernel_size=(1+s_H-t_H, 1+s_W-t_W))

    def forward(self, x):
        x = self.conv(x)
        return x

class Project(nn.Module):
    def __init__(self, origin_sizes, new_size=torch.Size([-1, 16, 14, 14]), factor=1):
        super(Project, self).__init__()
        for i, size_o in enumerate(origin_sizes):
            setattr(self, "target"+str(i),
                    Fit(size_o, new_size, factor=factor))
            setattr(self, "source"+str(i),
                    Fit(size_o, new_size, factor=factor))

    def forward(self, feat_sources, feat_targets):
        new_feat_sources, new_feat_targets = [], []
        for i, source in enumerate(feat_sources):
            new_feat_sources.append(getattr(self, "source" + str(i))(source))
        for i, target in enumerate(feat_targets):
            new_feat_targets.append(getattr(self, "target" + str(i))(target))
        return new_feat_sources, new_feat_targets


class DAAttention(nn.Module):
    def __init__(self, origin_sizes, new_size=torch.Size([-1, 32, 7, 7]), factor=1):
        super(DAAttention, self).__init__()
        self.pro = Project(origin_sizes, new_size=new_size,
                           factor=factor)      

        self.layer_num = len(origin_sizes)

        self.pam = PAM_Module(self.layer_num,new_size,self.mode)
        self.cam = CAM_Module(self.layer_num,new_size,self.mode)

        self.C = new_size[1]
        self.H = new_size[2]
        self.W = new_size[3]

    def forward(self, feat_sources, feat_targets):
        new_feat_sources, new_feat_targets = self.pro(feat_sources, feat_targets)     
        
        feat_pam,alpha_pam = self.pam(new_feat_sources,new_feat_targets)
        feat_cam,alpha_cam = self.cam(new_feat_sources,new_feat_targets)
        
        ret_alpha = None
        ret_targets,ret_sources = [],[]
        
        for i in range(self.layer_num):
            ret_targets.append(((feat_pam[i]+feat_cam[i])*0.5).view(-1,self.C*self.H*self.W))
            ret_alpha = (alpha_cam+alpha_pam)*0.5
            
            ret_sources.append(new_feat_sources[i].view(-1,self.C*self.H*self.W))
            ret_targets.append(new_feat_targets[i].view(-1,self.C*self.H*self.W))

        return ret_sources, ret_alpha, ret_targets

