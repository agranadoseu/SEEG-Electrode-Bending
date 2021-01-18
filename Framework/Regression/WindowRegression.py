"""
Regression using CNN

Written by:
Alejandro Granados ( PhD MSc DIC BSc )
School of Biomedical Engineering and Patient Sciences
King's College London, 2020

Contact:
alejandro.granados@kcl.ac.uk
agranados.eu@gmail.com
"""

import os
import sys
import time
import pickle
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

import torch
from torch.utils import data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from Framework.Tools import FileSystem
from Framework.Regression.RegressionModel import RegressionModel
from Framework.Regression.NeuralNet import SimpleNet3D
from Framework.Regression.NeuralNet import ComplexNet3D
from Framework.Regression.RunningAverageMeter import RunningAverageMeter


class WindowRegression(RegressionModel):
    # attributes
    gpu = 0
    batch_size = 1#100
    num_workers = 1#6
    niters = 200
    val_freq = 5

    learning_rate = 0.001
    num_forwards_passes = 200
    weight_decay = 1e-3
    step_decay = 20

    checkpoint_dir = None
    best_val_loss = None
    best_val_itr = None

    def __init__(self, channels=1, window_size=None, labels=None):
        super().__init__()

        # # dict: 'training', 'validation', 'test'
        # self.datasets = datasets
        self.channels = channels
        self.window_size = window_size
        self.labels = labels    # lu, gu, vector

        # gpu support
        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')

        # checkpoint_dir
        self.filesystem = FileSystem.FileSystem()
        self.checkpoint_dir = os.path.join(os.getcwd(), 'cnn')
        if not os.path.exists(self.checkpoint_dir):
            self.filesystem.create_dir(self.checkpoint_dir)
        self.timestamp = None

        # data generator
        # params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': self.num_workers}
        # self.generators = {'training': data.DataLoader(self.datasets['training'], **params),
        #                    'testing': data.DataLoader(self.datasets['testing'], **params),
        #                    'validation': data.DataLoader(self.datasets['validation'], **params)}

        # create model
        self.model = self.create()
        self.loss_fn = None

        # visualisation
        #self.fig, self.ax = self.viz_pred_init()

    def init_train(self, datasets=None, mcdropout=None, fold=None, niters=None, valfreq=None, lr=None):
        # dict: 'training', 'validation', 'test'
        self.datasets = datasets
        self.fold = fold
        self.mcdropout = mcdropout
        self.niters = niters
        self.val_freq = valfreq
        self.learning_rate = lr

        # data generator
        params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': self.num_workers}
        self.generators = {'training': data.DataLoader(self.datasets['training'], **params),
                           'validation': data.DataLoader(self.datasets['validation'], **params),
                           'testing': data.DataLoader(self.datasets['testing'], **params)}

        # visualisation
        # self.fig, self.ax = self.viz_pred_init()

    def init_test(self, datasets=None):
        # dict: 'test'
        self.datasets = datasets

        # data generator
        params = {'batch_size': self.batch_size, 'shuffle': True, 'num_workers': self.num_workers}
        self.generators = {'testing': data.DataLoader(self.datasets['testing'], **params)}

        # loss function
        self.loss_fn = torch.nn.MSELoss()

    def init_loss_function(self):
        # loss function
        self.loss_fn = torch.nn.MSELoss()

    def save_state(self, itr=None):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'f'+str(self.fold)+'-i'+str(itr)+'.pth'))

    def load_state(self, timestamp=None, filename=None):
        self.model.load_state_dict(torch.load(os.path.join(self.checkpoint_dir, timestamp + '/' + filename)))
        self.model.eval()

        print('Model state_dict loaded:')
        # for param_tensor in self.model.state_dict():
        #     print('     {}   {}'.format(param_tensor, self.model.state_dict()[param_tensor].size()))

    def create(self):
        # model = SimpleNet3D(window=self.window_size, channels=self.channels).cuda()

        model = None
        if self.labels == 'lu':
            model = SimpleNet3D(window=self.window_size, channels=self.channels).cuda()
        elif self.labels == 'gu':
            model = ComplexNet3D(window=self.window_size, channels=self.channels).cuda()
        elif self.labels == 'vector':
            model = SimpleNet3D(window=self.window_size, channels=self.channels).cuda()

        return model

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def loss_function(self, pred, impl, plan):
        # Example to penalise electrode based on impl-plan-pred
        # penalise more if pred has different direction or overestimates prediction:
        #   wrong direction: diff_p/diff_i same sign but opposite sign to diff_b
        #   underestimate: diff_p and diff_i have opposite signs
        #   overestimate: diff_p/diff_i/diff_b have same sign
        # N = 17
        # plan = torch.tensor([-0.5])
        # impl = torch.tensor([0.5])
        # pred = torch.linspace(-2.0, 2.0, N)    # at 0.25 intervals
        # diff_b = torch.ones(pred.size())*(impl-plan)

        diff_b = impl - plan  # displacement due to bending
        diff_p = pred - plan
        diff_i = pred - impl
        sign_p = torch.sign(diff_p)
        sign_i = torch.sign(diff_i)
        sign_b = torch.sign(diff_b)
        diffSqr_p = diff_p ** 2
        diffSqr_i = diff_i ** 2
        mask_p = torch.sigmoid(diff_p)
        mask_i = torch.sigmoid(diff_i)
        losses_p = 3 * diffSqr_p * (1 - mask_p) + diffSqr_p * mask_p
        losses_i = diffSqr_i * (1 - mask_i) + 2 * diffSqr_i * mask_i

        # but losses_p should only include wrong direction
        # i.e. diff_p/diff_i have same sign but opposite sign to diff_b
        cond_diff_same_sign = (sign_p * sign_i) == 1
        cond_bend_opp_sign = (-sign_p) == sign_b
        losses_p_zero = torch.where(cond_diff_same_sign & cond_bend_opp_sign, losses_p,
                                    torch.tensor([0.0], dtype=torch.float64).to(self.device))
        losses = losses_p_zero + losses_i
        loss = losses.mean()

        return loss

    def train(self):
        # create timestamp dir in checkpoint
        if self.timestamp is None:
            now = datetime.now()
            self.timestamp = now.strftime('%Y%m%d-%H%M')
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.timestamp)
            self.filesystem.create_dir(self.checkpoint_dir)
        else:
            self.checkpoint_dir = os.path.join(self.checkpoint_dir, self.timestamp)

        # save data
        # self.datasets['training'].save(checkpoint_dir=self.checkpoint_dir, fold=self.fold, type='training')
        # self.datasets['validation'].save(checkpoint_dir=self.checkpoint_dir, fold=self.fold, type='validation')
        # self.datasets['testing'].save(checkpoint_dir=self.checkpoint_dir, fold=self.fold, type='testing')

        print('Number of parameters: {}'.format(self.count_parameters()))
        print('Model:\n', self.model)

        # self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.loss_fn = torch.nn.MSELoss()

        # optimizer = optim.RMSprop(self.model.parameters(), lr=1e-3)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.step_decay, gamma=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, cooldown=10, verbose=True)
        print('Optimiser state_dict:')
        for var_name in optimizer.state_dict():
            print('     {}      {}'.format(var_name, optimizer.state_dict()[var_name]))

        end = time.time()

        time_meter = RunningAverageMeter(0.97)
        loss_meter = RunningAverageMeter(0.97)

        # # TODO delete
        # epoch_one_labels = {'case': [], 'elec': [], 'interpolation': [], 'lu': [], 'gu': [], 'vector': []}

        for epoch in range(1, self.niters + 1):
            self.model.train()
            optimizer.zero_grad()

            # scheduler (learning step)
            for param_group in optimizer.param_groups:
                self.learning_rate = param_group['lr']

            # get batches
            train_epoch_loss = []
            train_epoch_loss_mse = []
            train_epoch_loss_x, train_epoch_loss_y, train_epoch_loss_z = [], [], []
            for xd, xw, lu, gu0, gu, vector, impl, impl_next, plan, plan_next, case, name, interpolation in self.generators['training']:
                xd = xd.squeeze(dim=0)
                xw = xw.squeeze(dim=0)
                lu = lu.squeeze(dim=0)
                gu0 = gu0.squeeze(dim=0)
                gu = gu.squeeze(dim=0)
                vector = vector.squeeze(dim=0)
                impl = impl.squeeze(dim=0)
                impl_next = impl_next.squeeze(dim=0)
                plan = plan.squeeze(dim=0)
                plan_next = plan_next.squeeze(dim=0)

                # print('     xw={}', xw)
                # print('     xd={} xw={} lu={} gu={} vector={} impl={} impl_next={} plan_next={}'.format(xd.shape, xw.shape, lu.shape, gu.shape, vector.shape, impl.shape, impl_next.shape, plan_next.shape,))
                # print('     xd={} xw={} lu={}'.format(xd.dtype, xw.dtype, lu.dtype))

                # xw = Variable(xw.view(xw.shape[0], self.channels, xw.shape[1], xw.shape[2], xw.shape[3]))

                xd = xd.to(self.device)
                xw = xw.to(self.device)
                lu = lu.to(self.device)
                gu0 = gu0.to(self.device)
                gu = gu.to(self.device)
                vector = vector.to(self.device)
                impl = impl.to(self.device)
                impl_next = impl_next.to(self.device)
                plan = plan.to(self.device)
                plan_next = plan_next.to(self.device)

                # # TODO delete
                # epoch_one_labels['case'].append(case)
                # epoch_one_labels['elec'].append(name)
                # epoch_one_labels['interpolation'].append(interpolation)
                # epoch_one_labels['lu'].append(lu)
                # epoch_one_labels['gu'].append(gu)
                # epoch_one_labels['vector'].append(vector)

                # pred_lu = self.model(xd.float(), xw.float())
                # loss = self.loss_fn(pred_lu, lu.float())
                # loss_0, loss_1, loss_2 = loss, loss, loss

                # pred_v = self.model(xd.float(), xw.float())
                # loss = self.loss_fn(pred_v, vector.float())
                # loss_0, loss_1, loss_2 = loss, loss, loss

                if self.labels == 'lu':
                    pred_v = self.model(xd.float(), xw.float())
                    loss_mse = self.loss_fn(pred_v, lu.float())
                elif self.labels == 'gu':
                    pred_v = self.model(xd.float(), plan.float(), impl.float(), gu0.float(), xw.float())
                    loss_mse = self.loss_fn(pred_v, gu.float())
                elif self.labels == 'vector':
                    pred_v = self.model(xd.float(), xw.float())
                    loss_mse = self.loss_fn(pred_v, vector.float())

                # pred_next = impl + pred_v
                # plan_next_0, plan_next_1, plan_next_2 = plan_next[0], plan_next[1], plan_next[2]
                # impl_next_0, impl_next_1, impl_next_2 = impl_next[0], impl_next[1], impl_next[2]
                # pred_next_0, pred_next_1, pred_next_2 = pred_next[0], pred_next[1], pred_next[2]
                # loss_0 = self.loss_function(pred_next_0, impl_next_0, plan_next_0)
                # loss_1 = self.loss_function(pred_next_1, impl_next_1, plan_next_1)
                # loss_2 = self.loss_function(pred_next_2, impl_next_2, plan_next_2)
                # loss = loss_0 + loss_1 + loss_2
                loss = loss_mse

                # if case[0]=='T21' and name[0]=='E1i':
                #     print('     PREDICTION: case={} electrode={}\nlabels={}\npred={}'.format(case[0], name[0], lu.float(), pred_v))

                loss.backward()
                optimizer.step()

                time_meter.update(time.time() - end)
                loss_meter.update(loss.item())
                train_epoch_loss.append(loss.item())
                train_epoch_loss_mse.append(loss_mse.item())
                # train_epoch_loss_x.append(loss_0.item())
                # train_epoch_loss_y.append(loss_1.item())
                # train_epoch_loss_z.append(loss_2.item())

            # # TODO delete
            # epoch_pkl_file = os.path.join(self.checkpoint_dir, 'cnn_epoch_one.pkl')
            # pickle_file = open(epoch_pkl_file, "wb")
            # pickle.dump(epoch_one_labels, pickle_file)
            # pickle_file.close()
            # input('break')

            train_epoch_loss = np.mean(train_epoch_loss)
            train_epoch_loss_mse = np.mean(train_epoch_loss_mse)
            # train_epoch_loss_x = np.mean(train_epoch_loss_x)
            # train_epoch_loss_y = np.mean(train_epoch_loss_y)
            # train_epoch_loss_z = np.mean(train_epoch_loss_z)
            # print('Training: Fold {:04d} | Iter {:04d} | Total Loss {:.6f} (x={:.6f}, y={:.6f}, z={:.6f})'.format(self.fold, epoch, train_epoch_loss, train_epoch_loss_x, train_epoch_loss_y, train_epoch_loss_z))
            print('Training: Fold {:04d} | Iter {:04d} | lr {:.6f} | Total Loss {:.6f} (mse={:.6f})'.format(self.fold, epoch, self.learning_rate, train_epoch_loss, train_epoch_loss_mse))
            if epoch == 1:
                self.best_val_loss = train_epoch_loss
                self.best_val_itr = epoch
                self.save_state(itr=epoch)

            if epoch % self.val_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    val_epoch_loss = []
                    val_epoch_loss_mse = []
                    val_epoch_loss_x, val_epoch_loss_y, val_epoch_loss_z = [], [], []
                    for xd, xw, lu, gu0, gu, vector, impl, impl_next, plan, plan_next, case, name, interpolation in self.generators['validation']:
                        xd = xd.squeeze(dim=0)
                        xw = xw.squeeze(dim=0)
                        lu = lu.squeeze(dim=0)
                        gu0 = gu0.squeeze(dim=0)
                        gu = gu.squeeze(dim=0)
                        vector = vector.squeeze(dim=0)
                        impl = impl.squeeze(dim=0)
                        impl_next = impl_next.squeeze(dim=0)
                        plan = plan.squeeze(dim=0)
                        plan_next = plan_next.squeeze(dim=0)

                        # xw = Variable(xw.view(xw.shape[0], self.channels, xw.shape[1], xw.shape[2], xw.shape[3]))

                        xd = xd.to(self.device)
                        xw = xw.to(self.device)
                        lu = lu.to(self.device)
                        gu0 = gu0.to(self.device)
                        gu = gu.to(self.device)
                        vector = vector.to(self.device)
                        impl = impl.to(self.device)
                        impl_next = impl_next.to(self.device)
                        plan = plan.to(self.device)
                        plan_next = plan_next.to(self.device)

                        # pred_lu = self.model(xd.float(), xw.float())
                        # loss = self.loss_fn(pred_lu, lu.float())
                        # loss_0, loss_1, loss_2 = loss, loss, loss

                        # pred_v = self.model(xd.float(), xw.float())
                        # loss = self.loss_fn(pred_v, vector.float())
                        # loss_0, loss_1, loss_2 = loss, loss, loss

                        if self.labels == 'lu':
                            pred_v = self.model(xd.float(), xw.float())
                            loss_mse = self.loss_fn(pred_v, lu.float())
                        elif self.labels == 'gu':
                            pred_v = self.model(xd.float(), plan.float(), impl.float(), gu0.float(), xw.float())
                            loss_mse = self.loss_fn(pred_v, gu.float())
                        elif self.labels == 'vector':
                            pred_v = self.model(xd.float(), xw.float())
                            loss_mse = self.loss_fn(pred_v, vector.float())

                        # pred_next = impl + pred_v
                        # plan_next_0, plan_next_1, plan_next_2 = plan_next[0], plan_next[1], plan_next[2]
                        # impl_next_0, impl_next_1, impl_next_2 = impl_next[0], impl_next[1], impl_next[2]
                        # pred_next_0, pred_next_1, pred_next_2 = pred_next[0], pred_next[1], pred_next[2]
                        # loss_0 = self.loss_function(pred_next_0, impl_next_0, plan_next_0)
                        # loss_1 = self.loss_function(pred_next_1, impl_next_1, plan_next_1)
                        # loss_2 = self.loss_function(pred_next_2, impl_next_2, plan_next_2)
                        # loss = loss_0 + loss_1 + loss_2
                        loss = loss_mse

                        val_epoch_loss.append(loss.item())
                        val_epoch_loss_mse.append(loss_mse.item())
                        # val_epoch_loss_x.append(loss_0.item())
                        # val_epoch_loss_y.append(loss_1.item())
                        # val_epoch_loss_z.append(loss_2.item())

                    # save model if there is improvement
                    val_epoch_loss = np.mean(val_epoch_loss)
                    val_epoch_loss_mse = np.mean(val_epoch_loss_mse)
                    # val_epoch_loss_x = np.mean(val_epoch_loss_x)
                    # val_epoch_loss_y = np.mean(val_epoch_loss_y)
                    # val_epoch_loss_z = np.mean(val_epoch_loss_z)
                    # print('Validation: Fold {:04d} | Iter {:04d} | Total Loss {:.6f} (x={:.6f}, y={:.6f}, z={:.6f})'.format(self.fold, epoch, val_epoch_loss, val_epoch_loss_x, val_epoch_loss_y, val_epoch_loss_z))
                    print('Validation: Fold {:04d} | Iter {:04d} | Total Loss {:.6f} (mse={:.6f})'.format(self.fold, epoch, val_epoch_loss, val_epoch_loss_mse))
                    if val_epoch_loss < self.best_val_loss:
                        self.best_val_loss = val_epoch_loss
                        self.best_val_itr = epoch
                        self.save_state(itr=epoch)
                        print('     [INFO] checkpoint with val_epoch_loss={} saved!'.format(val_epoch_loss))

            # scheduler
            # scheduler.step()    # StepLR
            scheduler.step(train_epoch_loss)  # ReduceLROnPlateau

        end = time.time()

    def test(self):
        # self.model.eval()
        # with torch.no_grad():

        self.test_loss = []
        self.test_loss_mc_mean, self.test_loss_mc_std = [], []
        test_loss_mse = []
        test_loss_x, test_loss_y, test_loss_z = [], [], []
        for xd, xw, lu, gu0, gu, vector, impl, impl_next, plan, plan_next, case, name, interpolation in self.generators['testing']:
            xd = xd.squeeze(dim=0)
            xw = xw.squeeze(dim=0)
            lu = lu.squeeze(dim=0)
            gu0 = gu0.squeeze(dim=0)
            gu = gu.squeeze(dim=0)
            vector = vector.squeeze(dim=0)
            impl = impl.squeeze(dim=0)
            impl_next = impl_next.squeeze(dim=0)
            plan = plan.squeeze(dim=0)
            plan_next = plan_next.squeeze(dim=0)

            # xw = Variable(xw.view(xw.shape[0], self.channels, xw.shape[1], xw.shape[2], xw.shape[3]))

            xd = xd.to(self.device)
            xw = xw.to(self.device)
            lu = lu.to(self.device)
            gu0 = gu0.to(self.device)
            gu = gu.to(self.device)
            vector = vector.to(self.device)
            impl = impl.to(self.device)
            impl_next = impl_next.to(self.device)
            plan = plan.to(self.device)
            plan_next = plan_next.to(self.device)

            # pred_lu = self.model(xd.float(), xw.float())
            # loss = self.loss_fn(pred_lu, lu.float())
            # loss_0, loss_1, loss_2 = loss, loss, loss

            # pred_v = self.model(xd.float(), xw.float())
            # loss = self.loss_fn(pred_v, vector.float())
            # loss_0, loss_1, loss_2 = loss, loss, loss

            # pred_next = impl + pred_v
            # plan_next_0, plan_next_1, plan_next_2 = plan_next[0], plan_next[1], plan_next[2]
            # impl_next_0, impl_next_1, impl_next_2 = impl_next[0], impl_next[1], impl_next[2]
            # pred_next_0, pred_next_1, pred_next_2 = pred_next[0], pred_next[1], pred_next[2]
            # loss_0 = self.loss_function(pred_next_0, impl_next_0, plan_next_0)
            # loss_1 = self.loss_function(pred_next_1, impl_next_1, plan_next_1)
            # loss_2 = self.loss_function(pred_next_2, impl_next_2, plan_next_2)
            # loss = loss_0 + loss_1 + loss_2

            # select label variable
            y = 0
            if self.labels == 'lu':
                y = lu

                ''' standard '''
                self.model.eval()
                pred_v = self.model(xd.float(), xw.float())
                loss_mse = self.loss_fn(pred_v, y.float())
                loss = loss_mse
                self.test_loss.append(loss.item())
                test_loss_mse.append(loss_mse.item())
                # test_loss_x.append(loss_0.item())
                # test_loss_y.append(loss_1.item())
                # test_loss_z.append(loss_2.item())

                ''' MC dropout '''
                self.model.train()
                mcstack = torch.from_numpy(
                    np.zeros(shape=(self.num_forwards_passes, y.size()[0], y.size()[1]), dtype=np.float32))
                for t in range(self.num_forwards_passes):
                    pred_v = self.model(xd.float(), xw.float())  # [100,3]
                    mcstack[t, :, :] = pred_v
                mc_mean = torch.mean(mcstack, 0)  # [100,3]
                mc_std = torch.std(mcstack).item()
                loss = self.loss_fn(mc_mean.to(self.device), y.float())
                self.test_loss_mc_mean.append(loss.item())
                self.test_loss_mc_std.append(mc_std)

            elif self.labels == 'gu':
                y = gu

                ''' standard '''
                self.model.eval()
                pred_v = self.model(xd.float(), plan.float(), impl.float(), gu0.float(), xw.float())
                loss_mse = self.loss_fn(pred_v, y.float())
                loss = loss_mse
                self.test_loss.append(loss.item())
                test_loss_mse.append(loss_mse.item())
                # test_loss_x.append(loss_0.item())
                # test_loss_y.append(loss_1.item())
                # test_loss_z.append(loss_2.item())

                ''' MC dropout '''
                self.model.train()
                mcstack = torch.from_numpy(
                    np.zeros(shape=(self.num_forwards_passes, y.size()[0], y.size()[1]), dtype=np.float32))
                for t in range(self.num_forwards_passes):
                    pred_v = self.model(xd.float(), plan.float(), impl.float(), gu0.float(), xw.float())  # [100,3]
                    mcstack[t, :, :] = pred_v
                mc_mean = torch.mean(mcstack, 0)  # [100,3]
                mc_std = torch.std(mcstack).item()
                loss = self.loss_fn(mc_mean.to(self.device), y.float())
                self.test_loss_mc_mean.append(loss.item())
                self.test_loss_mc_std.append(mc_std)

            elif self.labels == 'vector':
                y = vector

                ''' standard '''
                self.model.eval()
                pred_v = self.model(xd.float(), xw.float())
                loss_mse = self.loss_fn(pred_v, y.float())
                loss = loss_mse
                self.test_loss.append(loss.item())
                test_loss_mse.append(loss_mse.item())
                # test_loss_x.append(loss_0.item())
                # test_loss_y.append(loss_1.item())
                # test_loss_z.append(loss_2.item())

                ''' MC dropout '''
                self.model.train()
                mcstack = torch.from_numpy(
                    np.zeros(shape=(self.num_forwards_passes, y.size()[0], y.size()[1]), dtype=np.float32))
                for t in range(self.num_forwards_passes):
                    pred_v = self.model(xd.float(), xw.float())  # [100,3]
                    mcstack[t, :, :] = pred_v
                mc_mean = torch.mean(mcstack, 0)  # [100,3]
                mc_std = torch.std(mcstack).item()
                loss = self.loss_fn(mc_mean.to(self.device), y.float())
                self.test_loss_mc_mean.append(loss.item())
                self.test_loss_mc_std.append(mc_std)

            # print('label y[{}]'.format(y.size()))

            # ''' standard '''
            # self.model.eval()
            # pred_v = self.model(xd.float(), xw.float())
            # loss_mse = self.loss_fn(pred_v, y.float())
            # loss = loss_mse
            # self.test_loss.append(loss.item())
            # test_loss_mse.append(loss_mse.item())
            # # test_loss_x.append(loss_0.item())
            # # test_loss_y.append(loss_1.item())
            # # test_loss_z.append(loss_2.item())
            #
            # ''' MC dropout '''
            # self.model.train()
            # mcstack = torch.from_numpy(np.zeros(shape=(self.num_forwards_passes, y.size()[0], y.size()[1]), dtype=np.float32))
            # for t in range(self.num_forwards_passes):
            #     pred_v = self.model(xd.float(), xw.float())  # [100,3]
            #     mcstack[t, :, :] = pred_v
            # mc_mean = torch.mean(mcstack, 0)  # [100,3]
            # mc_std = torch.std(mcstack).item()
            # loss = self.loss_fn(mc_mean.to(self.device), y.float())
            # self.test_loss_mc_mean.append(loss.item())
            # self.test_loss_mc_std.append(mc_std)

        self.test_loss = np.mean(self.test_loss)
        self.test_loss_mc_mean = np.mean(self.test_loss_mc_mean)
        self.test_loss_mc_std = np.mean(self.test_loss_mc_std)
        test_loss_mse = np.mean(test_loss_mse)
        # test_loss_x = np.mean(test_loss_x)
        # test_loss_y = np.mean(test_loss_y)
        # test_loss_z = np.mean(test_loss_z)
        # print('Test Loss = {:.6f} (x={:.6f}, y={:.6f}, z={:.6f})'.format(self.test_loss, test_loss_x, test_loss_y, test_loss_z))
        print('Test Loss (standard) = {:.6f} (mse={:.6f})'.format(self.test_loss, test_loss_mse))
        print('Test Loss (mcdropout) = {:.6f} (std={:.6f})'.format(self.test_loss_mc_mean, self.test_loss_mc_std))

    def infer(self):
        # self.model.eval()
        # with torch.no_grad():

        for xd, xw, lu, gu0, gu, vector, impl, impl_next, plan, plan_next, case, name, interpolation in self.generators['testing']:
            xd = xd.squeeze(dim=0)
            xw = xw.squeeze(dim=0)
            lu = lu.squeeze(dim=0)
            gu0 = gu0.squeeze(dim=0)
            gu = gu.squeeze(dim=0)
            vector = vector.squeeze(dim=0)
            impl = impl.squeeze(dim=0)
            impl_next = impl_next.squeeze(dim=0)
            plan = plan.squeeze(dim=0)
            plan_next = plan_next.squeeze(dim=0)

            # xw = Variable(xw.view(xw.shape[0], self.channels, xw.shape[1], xw.shape[2], xw.shape[3]))

            xd = xd.to(self.device)
            xw = xw.to(self.device)
            lu = lu.to(self.device)
            gu0 = gu0.to(self.device)
            gu = gu.to(self.device)
            vector = vector.to(self.device)
            impl = impl.to(self.device)
            impl_next = impl_next.to(self.device)
            plan = plan.to(self.device)
            plan_next = plan_next.to(self.device)

            # pred_lu = self.model(xd.float(), xw.float())
            # loss = self.loss_fn(pred_lu, lu.float())

            # pred_v = self.model(xd.float(), xw.float())
            # pred_next = impl + pred_v

            ''' standard '''
            self.model.eval()
            pred_std = self.model(xd.float(), xw.float())

            ''' MC dropout '''
            self.model.train()
            mcstack = torch.from_numpy(np.zeros(shape=(self.num_forwards_passes, lu.size()[0], lu.size()[1]), dtype=np.float32))
            for t in range(self.num_forwards_passes):
                pred_v = self.model(xd.float(), xw.float())  # [100,3]
                mcstack[t, :, :] = pred_v
            pred_mc_mean = torch.mean(mcstack, 0)  # [100,3]
            pred_mc_std = torch.std(mcstack)

        return pred_std.cpu().detach().numpy(), pred_mc_mean.cpu().detach().numpy(), pred_mc_std.cpu().detach().numpy()

    def compute_loss(self, truth=None, pred=None):
        truth = torch.from_numpy(truth)
        pred = torch.from_numpy(pred)

        truth = truth.to(self.device)
        pred = pred.to(self.device)

        loss = self.loss_fn(pred, truth)

        return loss.cpu().detach().numpy()
