import torch
import torch.nn.functional as F
import torch.nn as nn

def cosine(x, y):
    x = nn.functional.normalize(x, dim=1)
    y = nn.functional.normalize(y, dim=1)
    similarity = torch.einsum("nc,ck->nk", [x, y.T])
    distances = 1 - similarity
    return distances


# For euclidean distance compute_mode='use_mm_for_euclid_dist' or 'use_mm_for_euclid_dist_if_necessary'
# may provide a speedup but then the distance to the sample itself might not be zero.
# This is due to the precision error and the different equations solved in them vs the 'donot_use_mm_for_euclid_dist'
# When using mm approach the equation solved is x^2+y^2-2xy vs (x-y)^2
# This may cause precision errors and infact can also result in negative distances, though that is internally handled by pytorch.
# Source: https://github.com/pytorch/pytorch/issues/42479#issuecomment-668896488
def euclidean(x, y):
    distances = torch.cdist(x, y, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
    return distances


class Block(nn.Module):
    def __init__(self,  out_features, nfea,device):
        super(Block, self).__init__()
        self.S_norm = nn.BatchNorm1d(out_features, momentum=0.6).to(device)
        self.S = nn.Linear(out_features, out_features).to(device)

        self.U_norm = nn.BatchNorm1d(nfea, momentum=0.6).to(device)
        self.U = nn.Linear(nfea, out_features).to(device)

        self.device = device
    def forward(self, input, view):
        input1 = self.S(self.S_norm(input))
        input2 = self.U(self.U_norm(view))
        output = input1 + input2
        return output
class FusionLayer(nn.Module):
    def __init__(self, num_views, fusion_type, in_size, hidden_size=32):
        super(FusionLayer, self).__init__()
        self.fusion_type = fusion_type
        if self.fusion_type == 'weight':
            self.weight = nn.Parameter(torch.ones(num_views) / num_views, requires_grad=True)
        if self.fusion_type == 'attention':
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 1, bias=False)
            )

    def forward(self, emb_list,weight):
        if self.fusion_type == "average":
            common_emb = sum(emb_list) / len(emb_list)
        elif self.fusion_type == "weight":
            # weight = F.softmax(self.weight, dim=0)
            common_emb = sum([w * e for e, w in zip(weight, emb_list)])
        elif self.fusion_type == 'attention':
            emb_ = torch.stack(emb_list, dim=1)
            w = self.encoder(emb_)
            weight = torch.softmax(w, dim=1)
            common_emb = (weight * emb_).sum(1)
        else:
            sys.exit("Please using a correct fusion type")
        return common_emb


class SparseInferenceModule(nn.Module):
    def __init__(self, nfeats, n_view,n_classes, device, thre=0.01,blocks=1, fusion_type='weight',hidden_dim=32):
        super(SparseInferenceModule, self).__init__()
        self.beta=0.1
        self.n_classes = n_classes
        self.blocks = blocks
        self.device=device
        self.n_view=n_view
        for i in range(n_view):
            exec('self.block{}=Block(hidden_dim,{},device)'.format(i, nfeats[i] ))
        self.theta = nn.Parameter(torch.FloatTensor([thre]), requires_grad=True).to(device)

        self.ZZ_init = nn.ModuleList([nn.Linear(feat,hidden_dim).to(device) for feat in nfeats])
        self.classifier=nn.Linear(hidden_dim, self.n_classes).to(device)
        self.fusionlayer = FusionLayer(n_view, fusion_type, self.n_classes, hidden_size=64)
        self.TCPConfidenceLayer = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, 1),nn.Sigmoid()) for _ in range(self.n_view)])
        self.device=device
        self.centers = torch.zeros(self.n_classes,hidden_dim).requires_grad_(requires_grad=False).to(device)
        self.centers_views = torch.zeros(self.n_view,self.n_classes,hidden_dim).requires_grad_(requires_grad=False).to(device)
        self.weight= torch.zeros(self.n_view,1).to(device)
        self.euclidean_dist_obj = torch.nn.PairwiseDistance(p=2).to(device)
    def soft_threshold(self, u):
        return F.selu(u - self.theta) - F.selu(-1.0 * u - self.theta)
    def update_centers(self, FV, true_label):
        FV = FV[:true_label.shape[0]].detach()
        deltas = FV - self.centers[true_label, :].requires_grad_(requires_grad=False)
        for cls_no in set(true_label.tolist()):
            #            print(self.centers[cls_no].shape,deltas[true_label==cls_no].shape,torch.mean(self.beta * deltas[true_label==cls_no],dim=0).shape)
            self.centers[cls_no] += self.beta * torch.mean(
                deltas[true_label == cls_no], dim=0
            )

    def update_view_centers(self, FV, true_label,view):
        FV = FV[:true_label.shape[0]].detach()
        deltas = FV - self.centers_views[view][true_label, :].requires_grad_(requires_grad=False)
        for cls_no in set(true_label.tolist()):
            #            print(self.centers[cls_no].shape,deltas[true_label==cls_no].shape,torch.mean(self.beta * deltas[true_label==cls_no],dim=0).shape)
            self.centers_views[view][cls_no] += self.beta * torch.mean(
                deltas[true_label == cls_no], dim=0
            )

    def forward(self, features, true_label):
        TCPConfidence = dict()
        output_z = 0
        for j in range(self.n_view):
            output_z += self.ZZ_init[j](features[j] / 1.0)
        for i in range(0, self.blocks):
            Z = list()
            TCPConfidence = dict()
            for view in range(0,self.n_view):
                exec('Z.append(self.block{}(output_z, features[view]))'.format(view))
                TCPConfidence[view] = self.TCPConfidenceLayer[view](Z[view])
                Z[view]=Z[view]*TCPConfidence[view]
                Z[view]=self.soft_threshold(Z[view])
                self.update_view_centers(Z[view], true_label,view)
            self.caculate_weight()
            output_z=self.soft_threshold(self.fusionlayer(Z,self.weight))
        logit=self.classifier(output_z)

        return logit,TCPConfidence

    def caculate_weight(self):

        d_views = torch.zeros(self.n_view, 1).to(self.device)
        for view in range(self.n_view):
            h= torch.zeros( 1).to(self.device)
            i=0
            d_view=torch.zeros(int(self.n_classes*(self.n_classes-1)/2))
            for cls_no in range(self.n_classes-1):
                for cls_no2 in  range(cls_no+1, self.n_classes):
                    d_view[i]=self.euclidean_dist_obj(self.centers_views[view][cls_no, :], self.centers_views[view][cls_no2, :])
                    i+=1
            d_views[view]=min(d_view)
        d_views_=torch.reciprocal(d_views)
        d_view_hat=d_views_/ torch.norm( d_views_, p=1, dim=0)

        self.weight=torch.exp(-d_view_hat)/torch.exp(-d_view_hat).sum()


    def infer(self, features):
        TCPConfidence = dict()
        output_z = 0
        for j in range(self.n_view):
            output_z += self.ZZ_init[j](features[j] / 1.0)
        for i in range(0, self.blocks):
            Z = list()
            TCPConfidence = dict()
            for view in range(0,self.n_view):
                exec('Z.append(self.block{}(output_z, features[view]))'.format(view))
                TCPConfidence[view] = self.TCPConfidenceLayer[view](Z[view])
                Z[view]=Z[view]*TCPConfidence[view]

            output_z=self.soft_threshold(self.fusionlayer(Z,self.weight))
        logit=self.classifier(output_z)

        return logit

