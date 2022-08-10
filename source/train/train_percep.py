import os
import argparse
import math
import numpy as np
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch
from torch.autograd import Variable

from Model.APNet import APNet
from Model.Model import Model
from Model.gan_post import GANPostProcessing
from Model.VGG import VGGFeatureExtractor

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
    ap_model = APNet()
    netF = VGGFeatureExtractor(feature_layer=34, use_bn=False, use_input_norm=True)
    cri_fea = torch.nn.MSELoss()
    gan_post = GANPostProcessing()

    # 导入数据集
    logfile.write('Load data starting...' + '\n')
    logfile.flush()
    print('Load data starting...')
    # NIC_Dataset
    train_data = ImageFolder(root=input_dir, transform=transforms.Compose(
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
    max_step = 5000 if 5000 < max_step else max_step
    print("train step:", max_step, end=" ")
    logfile.write("train step:" + str(max_step))
    test_max_step = test_loader.__len__()
    test_max_step = 1000 if 1000 < test_max_step else test_max_step
    print("test step:", test_max_step)
    logfile.write(" test step:" + str(test_max_step) + '\n')

    # 导入预训练的mse模型参数
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
        epoch = checkpoint['epoch'] + 1
        alpha = checkpoint['alpha'] 
        print('Load pre-trained model [' + args.load_model + '] succeed!')
        logfile.write('Load pre-trained model [' + args.load_model + '] succeed!' + '\n')
        logfile.flush()
    else:
        print('not find pre-trained mse model!')
        logfile.write('not find pre-trained mse model!' + '\n')
        logfile.flush()
        exit()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model).cuda()
            ap_model = torch.nn.DataParallel(ap_model).cuda()
            netF = torch.nn.DataParallel(netF).cuda()
            gan_post = torch.nn.DataParallel(gan_post).cuda()
        else:
            model = model.cuda()
            ap_model = ap_model.cuda()
            netF = netF.cuda()
            gan_post = gan_post.cuda()

    model.train()
    netF.eval()

    epoch = 0
    # 在loss中加入vgg loss，端到端训练熵编码，后处理部分，可训练的小波。soft quantize
    alpha = args.alpha_start
    alpha = np.array(alpha, dtype=np.float32)
    alpha = torch.from_numpy(alpha)
    if torch.cuda.is_available():
        alpha = alpha.cuda()
        alpha = alpha.unsqueeze(0)
        alpha = torch.repeat_interleave(alpha, repeats=torch.cuda.device_count(), dim=0)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

    while True:
        if epoch > 10:
            break

        print("alpha:", alpha[0].item())
        logfile.write('alpha: ' + str(alpha[0].item()) + '\n')
        logfile.flush()
        model.train()

        ori_all = 0.
        post_all = 0.
        fea_all = 0.
        bpp_all = 0.
        loss_all = 0.
        scale_all = 0.

        for batch_idx, input_img in enumerate(train_loader):

            if batch_idx > max_step - 1:
                break

            input_img_v = to_variable(input_img[0]) * 255.
            size = input_img_v.size()
            mse_ori, mse_post, bits, scale, rgb_post = model(input_img_v, alpha=alpha, train=1, scale_init=init_scale_list[0], wavelet_trainable=1)
            bpp = torch.sum(bits) / size[0] / size[2] / size[3]

            mse_ori = torch.mean(mse_ori)
            psnr_ori = 10. * torch.log10(255. * 255. / mse_ori)
            mse_post = torch.mean(mse_post)
            psnr_post = 10. * torch.log10(255. * 255. / mse_post)
            scale = torch.mean(scale)

            real_fea = netF(input_img_v / 255.0).detach()
            fake_fea = netF(rgb_post / 255.0)
            fea = cri_fea(fake_fea, real_fea)
            fea = torch.mean(fea)

            loss = bpp + lambda_d * (mse_post + 100 * fea)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()

            ori_all += psnr_ori.item()
            post_all += psnr_post.item()
            fea_all += fea.item()
            bpp_all += bpp.item()
            loss_all += loss.item()
            scale_all += scale.item()

            if batch_idx % 10 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                              + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                    psnr_post.item()) + '/' + str(fea.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                      + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                    psnr_post.item()) + '/' + str(fea.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.module.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_percep_vgg_soft.pth',
                           _use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_percep_vgg_soft.pth',
                           _use_new_zipfile_serialization=False)
            logfile.write('ori_mean: ' + str(ori_all / max_step) + '\n')
            logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
            logfile.write('fea_mean: ' + str(fea_all / max_step) + '\n')
            logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
            logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
            logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
            logfile.flush()
            print('ori_mean: ' + str(ori_all / max_step))
            print('post_mean: ' + str(post_all / max_step))
            print('fea_mean: ' + str(fea_all / max_step))
            print('bpp_mean: ' + str(bpp_all / max_step))
            print('loss_mean: ' + str(loss_all / max_step))
            print('scale_mean: ' + str(scale_all / max_step))

        epoch = epoch + 1
        if alpha[0] < args.alpha_end:
            alpha += 2.0

    # 前面是soft quantize，下面为then hard
    for name, param in model.named_parameters():
        if 'scale_net' in name:
            param.requires_grad = False
        if 'wavelet_transform' in name:
            param.requires_grad = False

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    while True:
        if epoch > 18:
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
            mse_ori, mse_post, bits, scale, rgb_post = model(input_img_v, alpha=0, train=0, scale_init=init_scale_list[random.randint(0, len(init_scale_list)-1)], wavelet_trainable=1)
            bpp = torch.sum(bits) / size[0] / size[2] / size[3]

            mse_ori = torch.mean(mse_ori)
            psnr_ori = 10. * torch.log10(255. * 255. / mse_ori)
            mse_post = torch.mean(mse_post)
            psnr_post = 10. * torch.log10(255. * 255. / mse_post)
            scale = torch.mean(scale)

            real_fea = netF(input_img_v / 255.0).detach()
            fake_fea = netF(rgb_post / 255.0)
            fea = cri_fea(fake_fea, real_fea)
            fea = torch.mean(fea)

            loss = bpp + lambda_d * (mse_post + 100 * fea)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()

            ori_all += psnr_ori.item()
            post_all += psnr_post.item()
            fea_all += fea.item()
            bpp_all += bpp.item()
            loss_all += loss.item()
            scale_all += scale.item()

            if batch_idx % 10 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                              + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                    psnr_post.item()) + '/' + str(fea.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                      + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                    psnr_post.item()) + '/' + str(fea.item()) + '/' + str(bpp.item()) + '/' + str(scale.item()))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.module.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_percep_vgg_hard.pth',
                           _use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_percep_vgg_hard.pth',
                           _use_new_zipfile_serialization=False)
            logfile.write('ori_mean: ' + str(ori_all / max_step) + '\n')
            logfile.write('post_mean: ' + str(post_all / max_step) + '\n')
            logfile.write('fea_mean: ' + str(fea_all / max_step) + '\n')
            logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
            logfile.write('loss_mean: ' + str(loss_all / max_step) + '\n')
            logfile.write('scale_mean: ' + str(scale_all / max_step) + '\n')
            logfile.flush()
            print('ori_mean: ' + str(ori_all / max_step))
            print('post_mean: ' + str(post_all / max_step))
            print('fea_mean: ' + str(fea_all / max_step))
            print('bpp_mean: ' + str(bpp_all / max_step))
            print('loss_mean: ' + str(loss_all / max_step))
            print('scale_mean: ' + str(scale_all / max_step))

        epoch = epoch + 1

    epoch = 18
    # 固定小波变换，熵编码，大后处理模块，训练主观后处理，先用mse作为loss来初步训练这个主观后处理网络
    for name, param in model.named_parameters():
        param.requires_grad = False

    opt_G = optim.RMSprop(filter(lambda p: p.requires_grad, gan_post.parameters()), lr=5e-6)
    opt_D = optim.RMSprop(ap_model.parameters(), lr=5e-5)
    L1_loss = torch.nn.L1Loss().cuda()

    while True:
        if epoch >= 20:
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

            opt_G.zero_grad()
            input_img_v = to_variable(input_img[0]) * 255.
            size = input_img_v.size()
            mse_ori, mse_post, bits, scale, rgb_post = model(input_img_v, alpha=0, train=0, scale_init=init_scale_list[random.randint(0, len(init_scale_list)-1)], wavelet_trainable=1, coding=0)
            rgb_gan_post = gan_post(rgb_post)

            input_img_v = input_img_v / 255.0
            rgb_gan_post = rgb_gan_post / 255.0
            l1_loss = torch.mean(((input_img_v - rgb_gan_post).pow(2) + 1e-6).sqrt())
            loss = l1_loss
            loss.backward()
            opt_G.step()

            mse_ori = torch.mean(mse_ori)
            psnr_ori = 10. * torch.log10(255. * 255. / mse_ori)
            mse_post = torch.mean(mse_post)
            psnr_post = 10. * torch.log10(255. * 255. / mse_post)
            scale = torch.mean(scale)

            ori_all += psnr_ori.item()
            post_all += psnr_post.item()
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
                torch.save({'epoch': epoch, 'alpha':  alpha[0].item(), 'state_dict': model.module.state_dict(), "gan_post":gan_post.module.state_dict()}, ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_gan_post_mse.pth',_use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch': epoch, 'alpha':  alpha[0].item(), 'state_dict': model.state_dict(), "gan_post":gan_post.state_dict()}, ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_gan_post_mse.pth', _use_new_zipfile_serialization=False)
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

    epoch = 20
    # 固定小波变换，熵编码，大后处理模块，训练主观后处理
    while True:
        if epoch >= 30:
            break

        model.train()
        ap_model.train()

        ori_all = 0.
        post_all = 0.
        fea_all = 0.
        d_loss2_all = 0
        l1_loss_all = 0
        loss_all = 0.
        scale_all = 0.
        count = 0

        for batch_idx, input_img in enumerate(train_loader):

            if batch_idx > (max_step - 1) // 6 * 6:  # 更新5次判别器，更新一次生成器，所以需要数据是6的倍数
                break

            # train D
            if (batch_idx+1) % 6 != 0:
                opt_D.zero_grad()
                input_img_v = to_variable(input_img[0]) * 255.
                with torch.no_grad():
                    mse_ori, mse_post, bits, scale, rgb_post = model(input_img_v, alpha=0, train=0, scale_init=init_scale_list[random.randint(0, len(init_scale_list)-1)], wavelet_trainable=1, coding=0)
                    rgb_gan_post = gan_post(rgb_post)

                d_loss1 = -torch.mean(ap_model(rgb_gan_post, input_img_v, if_D=True))
                # opt_D.zero_grad()
                d_loss1.backward()
                opt_D.step()

                for p in ap_model.parameters():
                    p.data.clamp_(-0.01, 0.01)
            else:
                # train G
                opt_G.zero_grad()
                input_img_v = to_variable(input_img[0]) * 255.
                size = input_img_v.size()
                mse_ori, mse_post, bits, scale, rgb_post = model(input_img_v, alpha=0, train=0, scale_init=init_scale_list[random.randint(0, len(init_scale_list)-1)], wavelet_trainable=1, coding=0)
                rgb_gan_post = gan_post(rgb_post)
                input_img_v = input_img_v / 255.0
                rgb_gan_post = rgb_gan_post / 255.0
                l1_loss = torch.mean(((input_img_v - rgb_gan_post).pow(2) + 1e-6).sqrt())
                fea_loss = torch.mean(((netF(rgb_gan_post) - netF(input_img_v)).pow(2) + 1e-6).sqrt())
                d_loss2 = torch.mean(ap_model(rgb_gan_post, input_img_v, if_D=False))
                loss = 0.01 * l1_loss + 50 * d_loss2 + fea_loss
                loss.backward()
                opt_G.step()

                mse_ori = torch.mean(mse_ori)
                psnr_ori = 10. * torch.log10(255. * 255. / mse_ori)
                mse_post = torch.mean(mse_post)
                psnr_post = 10. * torch.log10(255. * 255. / mse_post)
                scale = torch.mean(scale)

                ori_all += psnr_ori.item()
                post_all += psnr_post.item()
                loss_all += loss.item()
                fea_all += fea_loss
                d_loss2_all += d_loss2
                l1_loss_all += l1_loss
                scale_all += scale.item()
                count += 1

                if batch_idx % 60 == 0:
                    logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                                  + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                        psnr_post.item()) + '/' + str(l1_loss.item()) + '/' + str(fea_loss.item())
                                  + '/' + str(d_loss2.item())+ '/' + str(scale.item()) + '\n')
                    logfile.flush()
                    print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                        batch_idx) + '/' + str(max_step) + ']   '
                          + 'train loss: ' + str(loss.item()) + '/' + str(psnr_ori.item()) + '/' + str(
                        psnr_post.item()) + '/' + str(l1_loss.item()) + '/' + str(fea_loss.item())
                                  + '/' + str(d_loss2.item())+ '/' + str(scale.item()))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.module.state_dict(),
                            "gan_post": gan_post.module.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_gan_post_percep.pth',
                           _use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch': epoch, 'alpha': alpha[0].item(), 'state_dict': model.state_dict(),
                            "gan_post": gan_post.state_dict()},
                           ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '_gan_post_percep.pth',
                           _use_new_zipfile_serialization=False)
            logfile.write('ori_mean: ' + str(ori_all / count) + '\n')
            logfile.write('post_mean: ' + str(post_all / count) + '\n')
            logfile.write('loss_mean: ' + str(loss_all / count) + '\n')
            logfile.write('fea_loss_mean: ' + str(fea_all / count) + '\n')
            logfile.write('d_loss2_mean: ' + str(d_loss2_all / count) + '\n')
            logfile.write('l1_loss_mean: ' + str(l1_loss_all / count) + '\n')
            logfile.write('scale_mean: ' + str(scale_all / count) + '\n')
            logfile.flush()
            print('ori_mean: ' + str(ori_all / count))
            print('post_mean: ' + str(post_all / count))
            print('loss_mean: ' + str(loss_all / count))
            print('fea_loss_mean: ' + str(fea_all / count))
            print('d_loss2_mean: ' + str(d_loss2_all / count))
            print('l1_loss_mean: ' + str(l1_loss_all / count))
            print('scale_mean: ' + str(scale_all / count))

        epoch = epoch + 1


    logfile.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("iwave")

    # parameters
    parser.add_argument('--input_dir', type=str, default=r'/data/ljp105/NIC_Dataset/train/')
    parser.add_argument('--test_input_dir', type=str, default=r'/data/ljp105/NIC_Dataset/validation/')
    parser.add_argument('--save_model_dir', type=str, default=r'/model/dongcunhui/FVC-wavelet3/train/train_percep_0.002/')
    parser.add_argument('--log_dir', type=str, default=r'/output/')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lambda_d', type=float, default=0.002)  # 用于训练不同码率的模型（lambda_d越大，训练的模型码率越高）
    parser.add_argument('--scale_list', type=float, nargs="+", default=[64.0])  # 初始的量化步长（跟lambda_d有关，lambda_d越大，scale应设置的越小）。如果不需要训练可变码率，输入的scale只需要为一个数，如果需要可变码率，输入的scale需要时一个列表
    parser.add_argument('--alpha_start', type=float, default=2.0)  # for soft quantize
    parser.add_argument('--alpha_end', type=float, default=12.0)  # for soft quantize
    parser.add_argument('--load_model', type=str,default="/data/dongcunhui/FVC-wavelet/perceptual/affine_notshare/model/affine_notshare_mse_0.006e10.pth")  # 使用mse为loss训练出的软量化的模型作为初始化 _soft.pth
    parser.add_argument('--rate_model', type=str, default=None)
    parser.add_argument('--post_model', type=str, default=None)
    parser.add_argument('--wavelet_affine', type=bool, default=True, help="the type of wavelet: True:affine False:additive")


    args = parser.parse_args()
    print(args)
    train(args)

