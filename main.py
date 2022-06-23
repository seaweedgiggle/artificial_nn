import torch
import argparse
import numpy as np
import os
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from modules.evaluator import Evaluator
from models.r2gen import R2GenModel
from memDrivenModel.models.r2gen import MemR2GenModel
from tensorboardX import SummaryWriter

def parse_agrs():
    parser = argparse.ArgumentParser()
    
    # model_name = 'final_mimic_224_gamma1_bs12'
    parser.add_argument('--model_name', type=str, default='test', help='the name of the model')
    #tensorboard settings

    parser.add_argument('--tb_dir', type=str, default='tb/iu_xray/', help='the path to save tensorboard')

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation_label.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=16, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=12, help='the number of samples for a batch')
    parser.add_argument('--img_size', type=int, default=224)

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')
#     parser.add_argument('--att', type=str, default='SimAM')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=2, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=1, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    parser.add_argument('--lstm_dim', type=int, default=512, help='the hidden dim of lstm decoder.')
    parser.add_argument('--max_gen_length', type=int, default=120, help='max length of generated caption')
    parser.add_argument('--sqrt', action='store_true')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--devices', type=str, default="0,1,2,3", help='cuda visible devices')
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=60, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_enc', type=float, default=1e-3, help='the learning rate for the pretrained resnet.')
    parser.add_argument('--lr_dec', type=float, default=5e-4, help='the learning rate for the laplacian pyramid.')
    parser.add_argument('--lr_ve', type=float, default=5e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=20, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9223, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--output_att_map', action='store_true', help='only available when test_only is True.')
    parser.add_argument('--att_path', type=str, default='output_att/', help='the path to save the attention maps.')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--reports_path', type=str, default='reports/')
    parser.add_argument('--mirqi', action='store_true')
    parser.add_argument('--eval_step', type=int, default=5)

    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_agrs()
    args.tb_dir = args.tb_dir + args.model_name
    args.save_dir = args.save_dir + args.model_name
    args.reports_path = args.reports_path + args.model_name + '.csv'
    
    print(args)
    if not args.save and not args.test_only:
        args.save_dir = 'results/test'
        args.tb_dir = 'tb/test'
        args.save_path = 'reports/test'
        print('\nTesting... Not saving models.\n')

    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # tensorboard
    tb_writer = SummaryWriter(args.tb_dir)

    # create tokenizer
    tokenizer = Tokenizer(args)
    
    # create adjacent matrix
    with open( 'data/matrix.txt','r') as matrix_file:
        adjacency_matrix = [[int(num) for num in line.split(',')] for line in matrix_file]
    
    args.num_classes = 20
    fw_adj = torch.tensor(adjacency_matrix, dtype=torch.float, device="cuda")
    bw_adj = fw_adj.t()
    identity_matrix = torch.eye(args.num_classes+1, device='cuda')
    fw_adj = fw_adj.add(identity_matrix)
    bw_adj = bw_adj.add(identity_matrix)
    args.fw_adj = fw_adj
    args.bw_adj = bw_adj

    # create data loader
    # if not args.test_only:
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    print("The sizes of datasets of Train/Val/Test are: ", len(train_dataloader.dataset), len(val_dataloader.dataset), len(test_dataloader.dataset))

    # else:
    #     test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    #     print("The size of Test Set is: ", len(test_dataloader.dataset))

    # build model architecture
    model = R2GenModel(args, tokenizer)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

    # get function handles of loss and metrics
    criterion = compute_loss 
    metrics = compute_scores 

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model) 
    lr_scheduler = build_lr_scheduler(args, optimizer)
 
    # build trainer and start to train
    if not args.test_only:
        trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader, tb_writer)
        trainer.train()
    
    else:
        evaluator = Evaluator(model, metrics, test_dataloader, args)
        evaluator.evaluate()


if __name__ == '__main__':
    main()
