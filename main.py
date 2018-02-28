import argparse
from vqvae import Solver

def main():
    parser = argparse.ArgumentParser(description='toy VQVAE')
    parser.add_argument('--epoch', default = 50, type=int, help='total epoch')
    parser.add_argument('--lr', default = 2e-4, type=float, help='learning rate')
    parser.add_argument('--beta', default = 0.25, type=float, help='beta')
    parser.add_argument('--z_dim', default = 256, type=int, help='latent space dimension')
    parser.add_argument('--k_dim', default = 256, type=int, help='the number of embeddings')
    parser.add_argument('--batch_size', default = 100, type=int, help='batch size')
    parser.add_argument('--fixed_x_num', default = 20, type=int, help='the number of fixed x')
    parser.add_argument('--env_name', default='main', type=str, help='visdom window name')
    parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name')
    parser.add_argument('--data_dir', default='data', type=str, help='data root directory path')
    parser.add_argument('--output_dir', default='output', type=str, help='output directory path')
    parser.add_argument('--ckpt_load', default=False, type=bool, help='resume from checkpoint')
    parser.add_argument('--ckpt_save', default=False, type=bool, help='save checkpoint')
    args = parser.parse_args()

    solver = Solver(args)
    solver.train()

if __name__ == "__main__":
    main()
