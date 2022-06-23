import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input.view(-1, 4), dim=-1)
 
        target = target.view(-1, 1)
 
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        return loss


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.ce = nn.CrossEntropyLoss()
#         self.focal_loss = FocalLoss(4)

    def forward(self, input, target, mask, cls_prob, cls_label): #todo
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        nll_loss = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        nll_loss = torch.sum(nll_loss) / torch.sum(mask)
        
        cls_prob = cls_prob.view(-1, 4)
        cls_label = cls_label.view(-1)
        cls_loss = self.ce(cls_prob, cls_label)
#         cls_loss = self.focal_loss(cls_prob, cls_label)
        
        nll_loss = nll_loss.mean()
        cls_loss = cls_loss.mean()
        loss = nll_loss + 0.2*cls_loss
#         loss = nll_loss
        
#         print(nll_loss.item(), cls_loss.item())

        return loss, nll_loss, cls_loss


def compute_loss(output, reports_ids, reports_masks, cls_prob, cls_label):
    criterion = LanguageModelCriterion()
    loss, nll_loss, cls_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:], cls_prob, cls_label)
    return loss, nll_loss, cls_loss