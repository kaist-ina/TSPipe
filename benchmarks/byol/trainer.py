import os
import math
import multiprocessing
from tqdm import tqdm
from argparse import Namespace
from typing import Iterable, Optional
mp = multiprocessing.get_context("spawn")

from utils import _create_model_training_folder

import torch
import torch.nn.functional as F
import torchvision
from torch.nn.parameter import Parameter
from torch.utils.tensorboard import SummaryWriter

from tspipe import TSPipe
from tspipe.profiler import profile_semantic
from tspipe.dataloader import FastDataLoader, DummyInputGenerator

class BYOLTrainer:
    def __init__(self, online_network, target_network, predictor, optimizer, device, scheduler, **params):
        self.online_network = online_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.device = device
        self.predictor = predictor
        self.max_epochs = params['max_epochs']
        self.writer = SummaryWriter()
        self.m = params['m']
        self.batch_size = params['batch_size']
        self.num_workers = params['num_workers']
        self.checkpoint_interval = params['checkpoint_interval']
        self.image_x = eval(params['input_shape'])[0]
        self.scheduler = scheduler
        _create_model_training_folder(self.writer, files_to_same=["./config/config.yaml", "main.py", 'trainer.py'])
        
        self.dummy_input = True if params['dummy_input'] == True else False

        if self.dummy_input:
            print("Warning: Dummy Input Enabled.")
            
    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    def train(self, train_dataset):

        train_loader = FastDataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True, pin_memory=True)

        niter = 0
        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        self.initializes_target_network()
        batch_id = 0
        for epoch_counter in range(self.max_epochs):
            if self.dummy_input:
                dummy_input_gen = DummyInputGenerator(self.batch_size, input_shape=self.image_x)
                pbar = tqdm(dummy_input_gen)
            else:
                pbar = tqdm(train_loader)

            for (batch_view_1, batch_view_2), _ in pbar:
                batch_id += 1

                profile_semantic(niter, 0, 0, False, None, 0, 'copy')
                batch_view_1 = batch_view_1.to(self.device)
                profile_semantic(niter, 0, 0, False, None, 0, 'copy_finish')
                profile_semantic(niter, 1, 0, False, None, 0, 'copy')
                batch_view_2 = batch_view_2.to(self.device)
                profile_semantic(niter, 1, 0, False, None, 0, 'copy_finish')

                if niter == 0:
                    grid = torchvision.utils.make_grid(batch_view_1[:32])
                    self.writer.add_image('views_1', grid, global_step=niter)

                    grid = torchvision.utils.make_grid(batch_view_2[:32])
                    self.writer.add_image('views_2', grid, global_step=niter)

                loss = self.update(batch_view_1, batch_view_2, niter)
                self.writer.add_scalar('loss', loss, global_step=niter)
                # torch.cuda.nvtx.range_push("BackwardCompute")
                profile_semantic(niter, 0, 0, False, None, 0, 'backward')
                self.optimizer.zero_grad()
                loss.backward()
                profile_semantic(niter, 0, 0, False, None, 0, 'backward_finish')
                profile_semantic(niter, 0, 0, False, None, 0, 'optimize')
                
                self.optimizer.step()
                # torch.cuda.nvtx.range_pop()

                self._update_target_network_parameters()  # update the key encoder
                profile_semantic(niter, 0, 0, False, None, 0, 'optimize_finish')
                pbar.set_postfix({'loss': loss, 'batch_id': batch_id})
                niter += 1
                if batch_id % 100 == 0:
                    self.save_model(os.path.join(model_checkpoints_folder, f'model_batch{batch_id}_part0.pt'))

                if batch_id > 1:
                    loss_fn = torch.nn.MSELoss(reduction='sum')

                
            print("End of epoch {}".format(epoch_counter))
            if self.scheduler is not None:
                self.scheduler.step()

        # save checkpoints
        self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))

    def update(self, batch_view_1, batch_view_2, niter = 0):
        # compute query feature
        profile_semantic(niter, 0, 0, False, None, 0, 'compute')
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        profile_semantic(niter, 0, 0, False, None, 0, 'compute_finish')
        profile_semantic(niter, 1, 0, False, None, 0, 'compute')
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
        profile_semantic(niter, 1, 0, False, None, 0, 'compute_finish')

        # compute key features
        with torch.no_grad():
            profile_semantic(niter, 0, 0, True, None, 0, 'compute')
            targets_to_view_2 = self.target_network(batch_view_1)
            profile_semantic(niter, 0, 0, True, None, 0, 'compute_finish')
            profile_semantic(niter, 1, 0, True, None, 0, 'compute')
            targets_to_view_1 = self.target_network(batch_view_2)
            profile_semantic(niter, 1, 0, True, None, 0, 'compute_finish')

        profile_semantic(niter, 0, 0, False, None, 0, 'loss')
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        profile_semantic(niter, 0, 0, False, None, 0, 'loss')
        return loss.mean()

    def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)



class DummyBYOLTrainer(BYOLTrainer):
    def train(self, train_dataset):
        train_loader = FastDataLoader(train_dataset, batch_size=self.batch_size,
                                  num_workers=self.num_workers, drop_last=False, shuffle=True)

        self.initializes_target_network()

        for epoch_counter in range(self.max_epochs):
            pbar = tqdm(train_loader)

            for (batch_view_1, batch_view_2), _ in pbar:
                # do nothing
                pass
            print("End of epoch {}".format(epoch_counter))


class TSPipeBYOLTrainer(BYOLTrainer):
    def __init__(self, online_network, target_network, predictor, optimizer: torch.optim.Optimizer, device, scheduler, **params):
        super().__init__(online_network, target_network, predictor, optimizer, device, scheduler, **params)
        self.optimizer = optimizer
        self.online_network = online_network
        self.target_network = target_network
        self.predictor_network = predictor
        self.dummy_input = True if params['dummy_input'] == True else False
        self.image_x = eval(params['input_shape'])[0]
        self.scheduler = scheduler
        self.params = params

        if self.dummy_input:
            print("Warning: Dummy Input Enabled.")

    @staticmethod
    def contrastive_loss(online_view_1, online_view_2, target_view_1, target_view_2, args: Namespace, extra_args: Namespace):
        loss = TSPipeBYOLTrainer.regression_loss(online_view_1, target_view_2)
        loss += TSPipeBYOLTrainer.regression_loss(online_view_2, target_view_1)
        return loss.mean()

    @staticmethod
    def calculate_target_network_parameters(m, online_new_param, target_param:Optional[Iterable[Parameter]] = None):
        """
        Momentum update of the key encoder
        """
        
        @torch.no_grad()
        def calc():
            result = []
            for param_q, param_k in zip(online_new_param, target_param):
                detached = param_k.clone().detach()
                detached = detached * m + param_q.data * (1. - m)
                result.append(detached)
            return result
        return calc()

    def train(self, train_dataset):

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        self.initializes_target_network()
        
        
        initial_lr = self.optimizer.param_groups[0]['lr']
        print(f"initial_lr : {initial_lr}")
        initial_momentum = self.params['m']
        print(f"initial_momentum : {initial_momentum}")

        warmup_epochs = 10
        lr = self.adjust_learning_rate(1, warmup_epochs = warmup_epochs, initial_lr = initial_lr)
        m = self.adjust_moco_momentum(0, initial_momentum)

        self.tspipe = TSPipe(self.online_network,
                                    self.target_network,
                                    self.predictor_network,
                                    self.optimizer,
                                    TSPipeBYOLTrainer.contrastive_loss,
                                    TSPipeBYOLTrainer.calculate_target_network_parameters,
                                    self.m,
                                    model_checkpoints_folder
        )

        if self.tspipe.is_primary:
            # prepare dataloader
            if self.dummy_input:
                train_loader = DummyInputGenerator(self.batch_size, input_shape=self.image_x)
            else:
                train_loader = FastDataLoader(train_dataset, batch_size=self.batch_size,
                                            num_workers=self.num_workers, drop_last=False, shuffle=True, pin_memory=False)

            iters_per_epoch = len(train_loader)
            print(f"iters_per_epoch : {iters_per_epoch}")
            niter = 0
            for epoch_counter in range(self.max_epochs):                
                pbar = tqdm(train_loader)
                for (batch_view_1, batch_view_2), _ in pbar:
                    if niter == 0:
                        grid = torchvision.utils.make_grid(batch_view_1[:32])
                        self.writer.add_image('views_1', grid, global_step=niter)

                        grid = torchvision.utils.make_grid(batch_view_2[:32])
                        self.writer.add_image('views_2', grid, global_step=niter)

                    loss = self.tspipe.feed(batch_view_1.share_memory_(), batch_view_2.share_memory_())
                    if loss is not None:
                        self.writer.add_scalar('loss', loss, global_step=niter)
                        pbar.set_postfix({'loss': loss, 'batch_id': niter})
                    niter += 1
                print("End of epoch {}".format(epoch_counter))
                self.tspipe.feed_epoch()
                
                lr = self.adjust_learning_rate(epoch_counter+1, warmup_epochs = warmup_epochs, initial_lr = initial_lr)
                m = self.adjust_moco_momentum(epoch_counter, initial_momentum)
                self.tspipe.update_lr(lr)
                self.tspipe.update_momentum(m)
                self.writer.add_scalar('learning_rate', lr, global_step=niter)
                self.writer.add_scalar('momentum', m, global_step=niter)

            self.tspipe.stop()

            # save checkpoints
            print("Saving checkpoints...")
            self.save_model(os.path.join(model_checkpoints_folder, 'model.pth'))
            print("Saving checkpoints OK")
    
    def adjust_learning_rate(self, epoch, warmup_epochs, initial_lr):
        """Decays the learning rate with half-cycle cosine after warmup"""
        if epoch < warmup_epochs:
            lr = initial_lr * epoch / warmup_epochs
        else:
            lr = initial_lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (self.params['max_epochs'] - warmup_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def adjust_moco_momentum(self, epoch, initial_momentum):
        """Adjust moco momentum based on current epoch"""
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / self.params['max_epochs'])) * (1. - initial_momentum)
        return m
    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))

        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        return loss.mean()