import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf
import tensorboardX
from sklearn.metrics import roc_auc_score
from scipy.special import softmax
from modules.labelprocess import Runner

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)
        self.eval_step = args.eval_step

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            
            if epoch % self.eval_step == 0:
                self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, self.args.dataset_name+'.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except RuntimeError:
            for k, v in list(checkpoint['state_dict'].items()):
                new_k = k[7:]
                checkpoint['state_dict'][new_k] = v
                del checkpoint['state_dict'][k]
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        print("Current learning rate:", self.optimizer.state_dict()['param_groups'][0]['lr'])

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader, tb_writer):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.tb_writer = tb_writer
        self.multi_gpu = args.n_gpu > 1
        self.reports_path = args.reports_path
        self.mirqi = args.mirqi

    def _train_epoch(self, epoch):
        train_loss = 0
        train_nll_loss = 0
        train_cls_loss = 0
        self.model.train()
        start_time = time.time()
        for batch_idx, (images_id, images, reports_ids, reports_masks, img_padding_mask, label) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)
            label = label.to(self.device)
            if img_padding_mask is not None:
                img_padding_mask = img_padding_mask.to(self.device)
                
            # 这里的 loss 需要修改
            output, classify_output = self.model(images, label, reports_ids, mode='train', img_mask=img_padding_mask)
            
            loss, nll_loss, cls_loss = self.criterion(output, reports_ids, reports_masks, classify_output, label)
            train_loss += loss.item()
            train_nll_loss += nll_loss.item()
            train_cls_loss += cls_loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {
            'train_loss': train_loss / len(self.train_dataloader),
            'train_nll_loss': train_nll_loss / len(self.train_dataloader),
            'train_cls_loss': train_cls_loss / len(self.train_dataloader)
        }
        stop_time = time.time()
        self.tb_writer.add_scalar('train_loss', train_loss, epoch)
        self.tb_writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], epoch)
        print('time elapsed : ' + str(stop_time - start_time) + ' s')
        
        if epoch % self.eval_step == 0:
            self.model.eval()

            auc_eval = [0 for _ in range(20)]
            # (12, 20) -> (n, 20)
            labels = [torch.zeros((1, 20), dtype=int).cpu(), 
                      torch.ones((1, 20), dtype=int).cpu(),
                     2 * torch.ones((1, 20), dtype=int).cpu(),
                     3 * torch.ones((1, 20), dtype=int).cpu()]
            # (12, 20, 4) -> (n, 20, 4)
            cls_probs = [softmax(torch.rand((1, 20, 4)), -1).to(self.device), 
                         softmax(torch.rand((1, 20, 4)), -1).to(self.device),
                        softmax(torch.rand((1, 20, 4)), -1).to(self.device),
                        softmax(torch.rand((1, 20, 4)), -1).to(self.device)]

            eval_loss = 0
            # 验证
            with torch.no_grad():
                val_gts, val_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks, img_padding_mask, label) in enumerate(self.val_dataloader):
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    if img_padding_mask is not None:
                        img_padding_mask = img_padding_mask.to(self.device)
                    output, cls_prob = self.model(images, label, mode='sample', img_mask=img_padding_mask)

    #                 print(cls_prob.max(-1)[1], label)

                    #后面这里加入分类结果的验证代码
                    labels.append(label)
                    cls_probs.append(cls_prob)

                    # reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    if not self.multi_gpu:
                        reports = self.model.tokenizer.decode_batch(output)
                        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    else:
                        reports = self.model.module.tokenizer.decode_batch(output)
                        ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                    print('eval', reports)
                    # loss = self.criterion(output, reports_ids, reports_masks)
                    # eval_loss += loss.item()

                    # 插文本，一维列表，元素是报告
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)

                if self.mirqi:
                #生成两个csv，处理，几百个
                    labels_csv = pd.DataFrame(data = val_res)
                    labels_csv.to_csv('./val_res.csv',index=False,header=None)
                    labels_csv = pd.DataFrame(data = val_gts)
                    labels_csv.to_csv('./val_gts.csv',index=False,header=None)
                    print()
                    print("Ready to make labels")
                    print()
                    os.system('./ascript.sh')
                    
                    temp = Runner()
                    temp.run()


                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                           {i: [re] for i, re in enumerate(val_res)})
                log.update(**{'val_' + k: v for k, v in val_met.items()}) 
                self.tb_writer.add_scalar('eval_loss', eval_loss, epoch)
                for key in val_met:
                    self.tb_writer.add_scalar('val_'+key, val_met[key], epoch)

                labels = torch.cat(labels, 0)  # (bs, 20)
                cls_probs = torch.cat(cls_probs, 0) # (bs, 20, 4)
                for i in range(20):
                    y_true = labels[:, i]
                    ones = torch.sparse.torch.eye(4)
                    y_true = ones.index_select(0, y_true).numpy()
                    y_score = softmax(cls_probs[:, i, :].squeeze(1).cpu().numpy(), -1)

    #                 try:
                    auc_eval[i] += roc_auc_score(y_true, y_score, multi_class='ovr', average='macro').mean()
    #                 except:
    #                     pass
                print("epoch {}:".format(epoch))
                total_AUC = 0
                for i in range(20):
                    print("valid: class {} AUC {:.4f}".format(i, auc_eval[i]))
                    total_AUC += auc_eval[i]
                print("valid: average AUC {:.4f}".format(total_AUC / 20))

            auc_test = [0 for _ in range(20)]

            # (12, 20) -> (n, 20)
            labels = [torch.zeros((1, 20), dtype=int).cpu(), 
                      torch.ones((1, 20), dtype=int).cpu(),
                     2 * torch.ones((1, 20), dtype=int).cpu(),
                     3 * torch.ones((1, 20), dtype=int).cpu()]
            # (12, 20, 4) -> (n, 20, 4)
            cls_probs = [softmax(torch.rand((1, 20, 4)), -1).to(self.device), 
                         softmax(torch.rand((1, 20, 4)), -1).to(self.device),
                        softmax(torch.rand((1, 20, 4)), -1).to(self.device),
                        softmax(torch.rand((1, 20, 4)), -1).to(self.device)]
            self.model.eval()

            with torch.no_grad():
                test_gts, test_res = [], []
                for batch_idx, (images_id, images, reports_ids, reports_masks, img_padding_mask, label) in enumerate(self.test_dataloader):
                    images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                        self.device), reports_masks.to(self.device)
                    if img_padding_mask is not None:
                        img_padding_mask = img_padding_mask.to(self.device)
                    output, cls_prob = self.model(images, label, mode='sample', img_mask=img_padding_mask)

                    #后面这里加入分类结果的验证代码
                    labels.append(label)
                    cls_probs.append(cls_prob)

                    # reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                    if not self.multi_gpu:
                        reports = self.model.tokenizer.decode_batch(output)
                        ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                    else:
                        reports = self.model.module.tokenizer.decode_batch(output)
                        ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                    print('test', reports)

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                for key in test_met:
                    self.tb_writer.add_scalar('test_'+key, test_met[key], epoch)

                labels = torch.cat(labels, 0)
                cls_probs = torch.cat(cls_probs, 0)
                for i in range(20):
                    # 输入为 (bs, 4)
                    y_true = labels[:, i]
                    ones = torch.sparse.torch.eye(4)
                    y_true = ones.index_select(0, y_true).numpy()
                    y_score = softmax(cls_probs[:, i, :].squeeze(1).cpu().numpy(), -1)
    #                 print(y_true.shape, y_score.shape)
                    auc_test[i] += roc_auc_score(y_true, y_score, multi_class='ovr', average='macro').mean()

                print("epoch {}:".format(epoch))
                total_AUC = 0
                for i in range(20):
                    print("test: class {} AUC {:.4f}".format(i, auc_test[i]))
                    total_AUC += auc_test[i]
                print("test: average AUC {:.4f}".format(total_AUC / 20))
        
        self.lr_scheduler.step()
        return log