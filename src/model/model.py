from . import resnet 
from utils import utils
from . import utils as model_utils
import torch.nn as nn
import torch.nn.functional as F

backbones = [resnet]

class DomainModule(nn.Module):
    def __init__(self, num_domains, **kwargs):
        super(DomainModule, self).__init__()
        self.num_domains = num_domains           
        self.domain = 0                        
    def set_domain(self, domain=0):             
        assert(domain < self.num_domains), \
              "The domain id exceeds the range (%d vs. %d)" \
              % (domain, self.num_domains)
        self.domain = domain

class BatchNormDomain(DomainModule):
    def __init__(self, in_size, num_domains, norm_layer, **kwargs):
        super(BatchNormDomain, self).__init__(num_domains)
        self.bn_domain = nn.ModuleDict() 
        for n in range(self.num_domains):
            self.bn_domain[str(n)] = norm_layer(in_size, **kwargs)

    def forward(self, x):
        out = self.bn_domain[str(self.domain)](x)
        return out
class FC_BN_ReLU_Domain(nn.Module):
    def __init__(self, in_dim, out_dim, num_domains_bn):
        super(FC_BN_ReLU_Domain, self).__init__()
        """
        fc --> BatchNorm1d(){‘0’，“1”} --> relu
        """
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = BatchNormDomain(out_dim, num_domains_bn, nn.BatchNorm1d)
        self.relu = nn.ReLU(inplace=True)
        self.bn_domain = 0

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class DANet(nn.Module):
    def __init__(self, num_classes, feature_extractor='resnet101', 
                 fx_pretrained=True, frozen=[], 
                 num_domains_bn=2, dropout_ratio=(0.5,)):
        super(DANet, self).__init__()
        self.feature_extractor = utils.find_class_by_name(
               feature_extractor, backbones)(pretrained=fx_pretrained, 
               frozen=frozen, num_domains=num_domains_bn)

        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn
        feat_dim = self.feature_extractor.out_dim
        self.in_dim = feat_dim

        self.FC = nn.ModuleDict()
        self.dropout = nn.ModuleDict()
        self.dropout['logits'] = nn.Dropout(p=dropout_ratio[0])
        self.FC['logits'] = nn.Linear(self.in_dim, num_classes)

        for key in self.FC:
            for m in self.FC[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), \
               "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def forward(self, x):
        att_feat,feat = self.feature_extractor(x)
        feat = feat.view(-1, self.in_dim)

        to_select = {}
        to_select['feat'] = feat

        to_select["layer2"] = att_feat[0]
        to_select["layer3"] = att_feat[1]
        to_select["layer4"] = att_feat[2]

        x = feat
        for key in self.FC:
            x = self.dropout[key](x)
            x = self.FC[key](x)
            to_select[key] = x

        to_select['probs'] = F.softmax(x, dim=1)

        return to_select

def danet(num_classes, feature_extractor, fx_pretrained=True, 
          frozen=[], dropout_ratio=0.5, state_dict=None, 
          fc_hidden_dims=[], num_domains_bn=1, **kwargs):

    model = DANet(feature_extractor=feature_extractor, 
                num_classes=num_classes, frozen=frozen, 
                fx_pretrained=fx_pretrained, 
                dropout_ratio=dropout_ratio, 
                fc_hidden_dims=fc_hidden_dims,
                num_domains_bn=num_domains_bn, **kwargs)

    if state_dict is not None:
        model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model