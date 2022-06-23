from modules import visual_extractor
import torch


def build_optimizer(args, model):
    # ve_params = list(map(id, model.visual_extractor.parameters()))
    # ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())

    ve_params = list(map(id, model.visual_extractor.parameters()))

#     lp_params = list(map(id, model.visual_extractor.model[6][-1].parameters())) + \
#                 list(map(id, model.visual_extractor.model[7][-1].parameters()))

    dec_params = list(map(id, model.encoder_decoder.model.decoder.parameters()))

    enc_params = filter(lambda x: id(x) not in dec_params, model.encoder_decoder.parameters())

    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.encoder_decoder.model.decoder.parameters(), 'lr': args.lr_dec},
         {'params': enc_params, 'lr': args.lr_enc},
         {'params': model.visual_extractor.parameters(), 'lr' : args.lr_ve}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(optimizer, args.step_size, args.gamma)
    return lr_scheduler
