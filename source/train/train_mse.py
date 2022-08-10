import os
import math
import numpy as np
import argparse
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch
from torch.autograd import Variable

from Util.data import path_NIC
from Model.Model import Model

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def train(args):
    input_dir = args.input_dir
    test_input_dir = args.test_input_dir

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    ckpt_dir = args.save_model_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logfile = open(args.log_dir + '/log.txt', 'w')

    total_epoch = args.epochs
    patch_size = args.patch_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    init_scale_list = args.scale_list
    lambda_d = args.lambda_d

    epoch = 1
    model = Model(args.wavelet_affine)
    alpha = args.alpha_start

    # 导入数据集
    logfile.write('Load data starting...' + '\n')
    logfile.flush()
    print('Load data starting...')
    # NIC_Dataset
    # train_data = ImageFolder(root=input_dir, 
    train_data = path_NIC(root='/data/ljp105/NIC_Dataset/test/', name='/data/dongcunhui/FVC-wavelet/path.txt',
        transform=transforms.Compose(
        [
            # transforms.Resize(image_size),
            transforms.RandomCrop(patch_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ))

    test_data = ImageFolder(root=test_input_dir, transform=transforms.Compose(
        [
            # transforms.Resize(image_size),
            transforms.RandomCrop(patch_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    ))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    logfile.write('Load all data succeed!' + '\n')
    logfile.flush()
    print('Load all data succeed!')
    max_step = train_loader.__len__()
    max_step = 20000 if 20000 < max_step else max_step
    print("train step:", max_step, end=" ")
    logfile.write("train step:" + str(max_step))
    test_max_step = test_loader.__len__()
    test_max_step = 1000 if 1000 < test_max_step else test_max_step
    print("test step:", test_max_step)
    logfile.write(" test step:" + str(test_max_step) + '\n')

    # 导入预训练的模型参数
    if args.load_model is not None:
        model_dict = model.state_dict()
        checkpoint = torch.load(args.load_model)
        state_dict = checkpoint['state_dict']
        # model.load_state_dict(state_dict)
        part_dict = {"wavelet_transform_trainable"+k[17:]: v for k, v in state_dict.items() if "wavelet_transform" in k}
        model_dict.update(part_dict)
        # part_dict = {k:v for k, v in state_dict.items() if "coding_" in k or "scale_net" in k}
        part_dict = {k:v for k, v in state_dict.items() if "wavelet_transform" not in k}
        model_dict.update(part_dict)
        model.load_state_dict(model_dict)
        # epoch = checkpoint['epoch'] + 1
        # alpha = checkpoint['alpha'] 
        print('Load pre-trained model [' + args.load_model + '] succeed!')
        logfile.write('Load pre-trained model [' + args.load_model + '] succeed!' + '\n')
        logfile.flush()
    else:
        if args.rate_model is not None:
            model_dict = model.state_dict()
            checkpoint = torch.load(args.rate_model)
            part_dict_org = checkpoint['state_dict']
            print("pre train args.rate_model", end=" ")
            logfile.write("pre train args.rate_model" + ' ')
            part_dict = {k: v for k, v in part_dict_org.items() if "coding" in k}
            for name, param in part_dict.items():
                print("pretrain "+name, end=" ")
                logfile.write("pretrain "+name + ' ')
            model_dict.update(part_dict)
            model.load_state_dict(model_dict)
            print('Load pre-trained part model [' +args.rate_model + '] succeed!')
            logfile.write('Load pre-trained part model [' + args.rate_model + '] succeed!' + '\n')
            logfile.flush()
        if args.post_model is not None:
            model_dict = model.state_dict()
            checkpoint = torch.load(args.post_model)
            part_dict_org = checkpoint['state_dict']
            print("pre train args.post_model", end=" ")
            logfile.write("pre train args.post_model" + ' ')
            part_dict = {k: v for k, v in part_dict_org.items() if "post" in k and "coding" not in k}
            for name, param in part_dict.items():
                print("pretrain "+name, end=" ")
                logfile.write("pretrain "+name + ' ')
            model_dict.update(part_dict)
            model.load_state_dict(model_dict)
            print('Load pre-trained part model [' +args.post_model + '] succeed!')
            logfile.write('Load pre-trained part model [' + args.post_model + '] succeed!' + '\n')
            logfile.flush()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    model.train()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    if (args.load_model is None) and (args.post_model is None or args.rate_model is None):
        # 没有预训练模型作为熵编码和后处理模块的初始化。先使用传统的97小波，训练熵编码和后处理模块        
        while True:
            if epoch > 10:
                break

            ori_all = 0.
            post_all = 0.
            bpp_all = 0.
            loss_all = 0.
            scale_all = 0
            for batch_idx, input_img in enumerate(train_loader):
                if batch_idx > max_step - 1:
                    break

                input_img_v = to_variable(input_img[0]) * 255.
                size = input_img_v.size()
                # 因为只训练后处理和熵编码模块不涉及到量化之前的网络，因此下面的量化直接使用round便可
                mse_ori, mse_post, bits, scale, _ = model(input_img_v, alpha=0, train=0, scale_init=init_scale_list[0], wavelet_trainable=0)
                bpp = torch.sum(bits) / size[0] / size[2] / size[3]

                mse_ori = torch.mean(mse_ori)
                psnr_ori = 10. * torch.log10(255. * 255. / mse_ori)
                mse_post = torch.mean(mse_post)
                psnr_post = 10. * torch.log10(255. * 255. / mse_post)
                scale = torch.mean(scale)

                loss = bpp + lambda_d * mse_post
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
                opt.step()

                ori_all += psnr_ori.item()
                post_all += psnr_post.item()
                bpp_all += bpp.item()
                loss_all += loss.item()
                scale_all += scale.item()

                if batch_idx % 10 == 0:
                    logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                                  + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                        psnr_post.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                    logfile.flush()
                    print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                          + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                        psnr_post.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()))

            if epoch % 1 == 0:
                if torch.cuda.device_count() > 1:
                    torch.save({'epoch': epoch, 'state_dict': model.module.state_dict()},
                               ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_post_entropy.pth',
                               _use_new_zipfile_serialization=False)
                else:
                    torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                               ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_post_entropy.pth',
                               _use_new_zipfile_serialization=False)
                logfile.write('ori_mean: ' + str(ori_all / max_step) + '\n')
                logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
                logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
                logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
                logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
                logfile.flush()
                print('ori_mean: ' + str(ori_all / max_step))
                print('post_mean: ' + str(post_all / max_step))
                print('bpp_mean: ' + str(bpp_all / max_step))
                print('loss_mean: ' + str(loss_all / max_step))
                print('scale_mean: ' + str(scale_all / max_step))

            epoch = epoch + 1

    epoch = 10
    # 端到端训练熵编码，后处理部分，可训练的小波
    # 软量化
    alpha = np.array(alpha, dtype=np.float32)
    alpha = torch.from_numpy(alpha)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        alpha = alpha.unsqueeze(0)
        alpha = torch.repeat_interleave(alpha, repeats=torch.cuda.device_count(), dim=0)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    while True:
        if epoch > 20:
            break

        print("alpha:", alpha[0].item())
        logfile.write('alpha: ' + str(alpha[0].item()) + '\n')
        logfile.flush()
        model.train()

        ori_all = 0.
        post_all = 0.
        bpp_all = 0.
        loss_all = 0.
        scale_all = 0.

        for batch_idx, input_img in enumerate(train_loader):

            if batch_idx > max_step - 1:
                break

            input_img_v = to_variable(input_img[0]) * 255.
            size = input_img_v.size()
            mse_ori, mse_post, bits, scale, _ = model(input_img_v, alpha=alpha, train=1, scale_init=init_scale_list[0], wavelet_trainable=1)
            bpp = torch.sum(bits) / size[0] / size[2] / size[3]

            mse_ori = torch.mean(mse_ori)
            psnr_ori = 10. * torch.log10(255. * 255. / mse_ori)
            mse_post = torch.mean(mse_post)
            psnr_post = 10. * torch.log10(255. * 255. / mse_post)
            scale = torch.mean(scale)

            loss = bpp + lambda_d * mse_post
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()

            ori_all += psnr_ori.item()
            post_all += psnr_post.item()
            bpp_all += bpp.item()
            loss_all += loss.item()
            scale_all += scale.item()

            if batch_idx % 10 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                              + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                    psnr_post.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                      + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                    psnr_post.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.module.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_soft.pth',
                           _use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_soft.pth',
                           _use_new_zipfile_serialization=False)
            logfile.write('ori_mean: ' + str(ori_all / max_step) + '\n')
            logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
            logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
            logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
            logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
            logfile.flush()
            print('ori_mean: ' + str(ori_all / max_step))
            print('post_mean: ' + str(post_all / max_step))
            print('bpp_mean: ' + str(bpp_all / max_step))
            print('loss_mean: ' + str(loss_all / max_step))
            print('scale_mean: ' + str(scale_all / max_step))

        epoch = epoch + 1
        if alpha[0] < args.alpha_end:
            alpha += 2.0

    epoch = 20
    # soft then hard  固定小波变换部分，量化使用round即可
    for name, param in model.named_parameters():
        if 'scale_net' in name:
            param.requires_grad = False
        if 'wavelet_transform' in name:
            param.requires_grad = False

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    while True:
        if epoch > 24:
            break

        model.train()

        ori_all = 0.
        post_all = 0.
        bpp_all = 0.
        loss_all = 0.
        scale_all = 0.

        for batch_idx, input_img in enumerate(train_loader):

            if batch_idx > max_step - 1:
                break

            input_img_v = to_variable(input_img[0]) * 255.
            size = input_img_v.size()
            mse_ori, mse_post, bits, scale, _ = model(input_img_v, alpha=0, train=0, scale_init=init_scale_list[random.randint(0, len(init_scale_list)-1)], wavelet_trainable=1)
            bpp = torch.sum(bits)/size[0]/size[2]/size[3]

            mse_ori = torch.mean(mse_ori)
            psnr_ori = 10. * torch.log10(255. * 255. / mse_ori)
            mse_post = torch.mean(mse_post)
            psnr_post = 10. * torch.log10(255. * 255. / mse_post)
            scale = torch.mean(scale)

            loss = bpp + lambda_d * mse_post
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()

            ori_all += psnr_ori.item()
            post_all += psnr_post.item()
            bpp_all += bpp.item()
            loss_all += loss.item()
            scale_all += scale.item()

            if batch_idx % 10 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                                + 'train loss: ' + str(loss.item())+ '/' + str(psnr_ori.item()) + '/' + str(psnr_post.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                        + 'train loss: ' + str(loss.item())+ '/'  + str(psnr_ori.item()) + '/' + str(psnr_post.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'alpha':  alpha[0].item(), 'state_dict': model.module.state_dict()}, ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_hard.pth',_use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch': epoch, 'alpha':  alpha[0].item(), 'state_dict': model.state_dict()}, ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_hard.pth', _use_new_zipfile_serialization=False)
            logfile.write('ori_mean: ' + str(ori_all / max_step) + '\n')
            logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
            logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
            logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
            logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
            logfile.flush()
            print('ori_mean: ' + str(ori_all / max_step))
            print('post_mean: ' + str(post_all / max_step))
            print('bpp_mean: ' + str(bpp_all / max_step))
            print('loss_mean: ' + str(loss_all / max_step))
            print('scale_mean: ' + str(scale_all / max_step))

        epoch = epoch + 1

    logfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("iwave")

    # parameters
    parser.add_argument('--input_dir', type=str, default=r'/data/ljp105/NIC_Dataset/train/')
    parser.add_argument('--test_input_dir', type=str, default=r'/data/ljp105/NIC_Dataset/validation/')
    parser.add_argument('--save_model_dir', type=str, default=r'/model/dongcunhui/FVC-wavelet3/train_loadPosEntropy_0.006/')
    parser.add_argument('--log_dir', type=str, default=r'/output/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lambda_d', type=float, default=0.4)  # 用于训练不同码率的模型（lambda_d越大，训练的模型码率越高）
    parser.add_argument('--scale_list', type=float, nargs="+", default=[8.0])  # 初始的量化步长（跟lambda_d有关，lambda_d越大，scale应设置的越小）。如果不需要训练可变码率，输入的scale只需要为一个数，如果需要可变码率，输入的scale需要时一个列表
    parser.add_argument('--alpha_start', type=float, default=2.0)  # for soft quantize
    parser.add_argument('--alpha_end', type=float, default=12.0)  # for soft quantize
    # parser.add_argument('--load_model', type=str,default="/data/dongcunhui/FVC-wavelet/perceptual/affine_notshare/model/affine_notshare_mse_0.006e10.pth")
    parser.add_argument('--load_model', type=str,default="/data/dongcunhui/NICmodel/1857Models/0.4_mse.pth")
    # parser.add_argument('--rate_model', type=str, default="/model/dongcunhui/FVC-wavelet/entropy/q64.pth")
    # parser.add_argument('--post_model', type=str, default="/model/dongcunhui/FVC-wavelet/post/q64/model_epoch088.pth")
    parser.add_argument('--rate_model', type=str, default=None)
    parser.add_argument('--post_model', type=str, default=None)
    parser.add_argument('--wavelet_affine', type=bool, default=False, help="the type of wavelet: True:affine False:additive")


    args = parser.parse_args()
    print(args)
    train(args)