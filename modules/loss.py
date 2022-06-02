import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target, mask, cls_prob, cls_label): #todo
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        nll_loss = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        nll_loss = torch.sum(nll_loss) / torch.sum(mask)
        
        cls_prob = cls_prob.view(-1, 4)
        cls_label = cls_label.view(-1)
        cls_loss = self.ce(cls_prob, cls_label)
        
        nll_loss = nll_loss.mean()
        cls_loss = cls_loss.mean()
        loss = nll_loss + cls_loss
        
#         print(nll_loss.item(), cls_loss.item())

        return loss, nll_loss, cls_loss


def compute_loss(output, reports_ids, reports_masks, cls_prob, cls_label):
    criterion = LanguageModelCriterion()
    loss, nll_loss, cls_loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:], cls_prob, cls_label)
    return loss, nll_loss, cls_loss