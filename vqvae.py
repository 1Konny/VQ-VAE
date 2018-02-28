import numpy as np
import torch, torchvision, os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

from utils.visdom_utils import VisFunc
from utils.data import data_loader
from utils.model_mnist import MODEL_MNIST
from utils.model_cifar10 import MODEL_CIFAR10
from utils.model_pixelcnn import PIXELCNN

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

np.set_printoptions(precision= 4)
torch.set_printoptions(precision = 4)

class Solver(object):
    def __init__(self, args):
        self.args = args
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.z_dim = args.z_dim
        self.k_dim = args.k_dim
        self.beta = args.beta
        self.env_name = args.env_name
        self.ckpt_dir = os.path.join('checkpoints',args.env_name)
        self.global_iter = 0
        self.dataset = args.dataset
        self.fixed_x_num = args.fixed_x_num
        self.output_dir = os.path.join(args.output_dir,args.env_name)
        self.ckpt_load = args.ckpt_load
        self.ckpt_save = args.ckpt_save

        # Toy Network init
        if self.dataset == 'MNIST':
            self.model = MODEL_MNIST(k_dim=self.k_dim,z_dim=self.z_dim).cuda()
        elif self.dataset == 'CIFAR10':
            self.model = MODEL_CIFAR10(k_dim=self.k_dim,z_dim=self.z_dim).cuda()

        # Visdom Sample Visualization
        self.vf = VisFunc(enval=self.env_name,port=55558)

        # Criterions
        self.MSE_Loss = nn.MSELoss().cuda()

        # Dataset init
        self.train_data, self.train_loader = data_loader(args)
        self.fixed_x = iter(self.train_loader).next()[0][:self.fixed_x_num]

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

        # Resume training
        if self.ckpt_load : self.load_checkpoint()

    def set_mode(self, mode='train'):
        if mode == 'train' :
            self.model.train()
        elif mode == 'eval' :
            self.model.eval()
        else : raise('mode error. It should be either train or eval')

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        if not os.path.exists(self.ckpt_dir) : os.makedirs(self.ckpt_dir)
        file_path = os.path.join(self.ckpt_dir,filename)
        torch.save(state,file_path)
        print("=> saved checkpoint '{}' (iter {})".format(file_path,self.global_iter))

    def load_checkpoint(self):
        filename = 'checkpoint.pth.tar'
        file_path = os.path.join(self.ckpt_dir,filename)
        if os.path.isfile(file_path):
            print("=> loading checkpoint '{}'".format(file_path))
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (iter {})"
                  .format(filename, checkpoint['iter']))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def image_save(self, imgs, name='fixed', **kwargs):
        # required imgs shape : batch_size x channels x width x height
        if not os.path.exists(self.output_dir) : os.makedirs(self.output_dir)
        filename = os.path.join(self.output_dir,name+'_'+str(self.global_iter)+'.jpg')
        torchvision.utils.save_image(imgs,filename,**kwargs)


    def train(self):
        self.set_mode('train')
        for e in range(self.epoch) :
            recon_losses = []
            z_and_sg_embd_losses = []
            sg_z_and_embd_losses = []
            for idx, (images,labels) in enumerate(self.train_loader):
                self.global_iter += 1

                X = Variable(images.cuda(),requires_grad=False)
                X_recon, Z_enc, Z_dec, Z_enc_for_embd = self.model(X)

                recon_loss = self.MSE_Loss(X_recon,X)
                z_and_sg_embd_loss = self.MSE_Loss(Z_enc,Z_dec.detach())
                sg_z_and_embd_loss = self.MSE_Loss(self.model._modules['embd'].weight,
                                                   Z_enc_for_embd.detach())

                total_loss = recon_loss + sg_z_and_embd_loss + self.beta*z_and_sg_embd_loss

                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                Z_enc.backward(self.model.grad_for_encoder)
                self.optimizer.step()

                recon_losses.append(recon_loss.data)
                z_and_sg_embd_losses.append(z_and_sg_embd_loss.data)
                sg_z_and_embd_losses.append(sg_z_and_embd_loss.data)

            # Sample Visualization
            self.vf.imshow_multi(X_recon.data.cpu(),
                                 title='random:{:d}'.format(e+1))
            self.image_save(X_recon.data,name='random')
            self.test()

            # AVG Losses
            recon_losses = torch.cat(recon_losses,0).mean()
            z_and_sg_embd_losses = torch.cat(z_and_sg_embd_losses,0).mean()
            sg_z_and_embd_losses = torch.cat(sg_z_and_embd_losses,0).mean()
            print('[{:02d}/{:d}] recon_loss:{:.2f} z_sg_embd:{:.2f} sg_z_embd:{:.2f}'.format(
                e+1,self.epoch,recon_losses,z_and_sg_embd_losses,sg_z_and_embd_losses))


        print("[*] Training Finished!")


    def test(self):
        self.set_mode('eval')

        X = Variable(self.fixed_x,requires_grad=False).cuda()
        X_recon = self.model(X)[0]
        X_cat = torch.cat([X,X_recon],0)
        self.vf.imshow_multi(X_cat.data.cpu(),
                             nrow=self.fixed_x_num,
                             title='fixed_x_test:'+str(self.global_iter))
        self.image_save(X_cat.data,name='fixed',nrow=self.fixed_x_num)
        if self.ckpt_save :
            self.save_checkpoint({
                'iter':self.global_iter,
                'args': self.args,
                'state_dict': self.model.state_dict(),
                'optimizer' : self.optimizer.state_dict(),
                })

        self.set_mode('train')
