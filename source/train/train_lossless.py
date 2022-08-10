from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.optim as optim
import argparse
import torch
from torch.autograd import Variable
import os
import math
import numpy as np
import copy
from Model.Model_lossless import Model


# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser(description='.')

# parameters
parser.add_argument('--input_dir', type=str, default=r'/data/ljp105/NIC_Dataset/train/')
parser.add_argument('--test_input_dir', type=str, default=r'/data/ljp105/NIC_Dataset/validation/')
parser.add_argument('--model_dir', type=str, default=r'/output/lossless')
parser.add_argument('--log_dir', type=str, default=r'/output/')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--load_model', type=str, default=None)


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def rgb2yuv(x):
    x = np.array(x, dtype=np.int32)

    r = x[:, :, :, 0:1]
    g = x[:, :, :, 1:2]
    b = x[:, :, :, 2:3]

    yuv = np.zeros_like(x, dtype=np.int32)

    Co = r - b
    tmp = b + np.right_shift(Co, 1)
    Cg = g - tmp
    Y = tmp + np.right_shift(Cg, 1)

    yuv[:, :, :, 0:1] = Y
    yuv[:, :, :, 1:2] = Co
    yuv[:, :, :, 2:3] = Cg

    return yuv

def yuv2rgb(x):
    
    x = np.array(x, dtype=np.int32)

    Y = x[:, :, :, 0:1]
    Co = x[:, :, :, 1:2]
    Cg = x[:, :, :, 2:3]

    

    rgb = np.zeros_like(x, dtype=np.int32)

    tmp = Y - np.right_shift(Cg, 1)
    g = Cg + tmp
    b = tmp - np.right_shift(Co, 1)
    r = b + Co

    
    rgb[:, :, :, 0:1] = r
    rgb[:, :, :, 1:2] = g
    rgb[:, :, :, 2:3] = b

    return rgb

def main():
    args = parser.parse_args()
    input_dir = args.input_dir
    test_input_dir = args.test_input_dir

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    ckpt_dir = args.model_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logfile = open(args.log_dir + '/log.txt', 'w')

    total_epoch = args.epochs
    patch_size = args.patch_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    trainable_set = True
    # trainable_set = False
    model = Model(trainable_set)
    epoch = 1
    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        state_dict = checkpoint['state_dict']
        model_dict = model.state_dict()
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
        print('Load pre-trained model [' + args.load_model + '] succeed!')
        logfile.write('Load pre-trained model [' + args.load_model + '] succeed!' + '\n')
        logfile.flush()

        
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    model.train()

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
    max_step = 20000 if 20000 < max_step else max_step
    print("train step:", max_step, end=" ")
    logfile.write("train step:" + str(max_step))
    test_max_step = test_loader.__len__()
    test_max_step = 1000 if 1000 < test_max_step else test_max_step
    print("test step:", test_max_step)
    logfile.write(" test step:" + str(test_max_step) + '\n')
    

    opt = optim.Adam(model.parameters(), lr=1e-4)

    while True:
        if epoch > total_epoch:
            break

        bpp_all = 0.

        for batch_idx, input_img in enumerate(train_loader):

            if batch_idx > max_step - 1:
                break

            input_img_v = input_img[0]*255.# [0,1]-->[0,255]
            input_img_v = torch.clamp(torch.round(input_img_v), 0., 255.)
            input_img_v = input_img_v.permute(0,2,3,1) # nchw-->nhwc
            input_img_v = input_img_v.data.numpy()
            ori_img = copy.deepcopy(input_img_v) # float

            input_img_v = rgb2yuv(input_img_v) # int32
            input_img_v = input_img_v.astype(np.float32)
            input_img_v = to_variable(torch.from_numpy(input_img_v)).permute(0,3,1,2) # nhwc-->nchw
            size = input_img_v.size()
            input_img_v = input_img_v.view(-1, 1, size[2], size[3])

            bits, recon_img = model(input_img_v)

            bpp = torch.sum(bits)/size[0]/size[2]/size[3]

            recon_img = recon_img.view(-1, 3, size[2], size[3])
            recon_img = recon_img.permute(0,2,3,1).cpu().data.numpy()
            recon_img = yuv2rgb(recon_img).astype(np.float32)
            mse = np.mean((ori_img - recon_img)**2)
            psnr = 10. * np.log10(255. * 255. / mse)

            opt.zero_grad()
            bpp.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, norm_type=2)
            opt.step()

            bpp_all += bpp.item()

            if batch_idx % 10 == 0:
                logfile.write('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                                + 'train loss: ' + str(bpp.item())+ '/'  + str(psnr) + '\n')
                logfile.flush()
                print('Train Epoch: [' + str(epoch) + '/' + str(total_epoch) + ']   ' + 'Step: [' + str(
                    batch_idx) + '/' + str(max_step) + ']   '
                        + 'train loss: ' + str(bpp.item())+ '/'  + str(psnr))

        if epoch % 1 == 0:
            if torch.cuda.device_count() > 1:
                torch.save({'epoch': epoch, 'state_dict': model.module.state_dict()}, ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '.pth',_use_new_zipfile_serialization=False)
            else:
                torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, ckpt_dir + '/model_epoch' + str(epoch).zfill(3) + '.pth', _use_new_zipfile_serialization=False)

            logfile.write('bpp_mean: ' + str(bpp_all / max_step) + '\n')
            logfile.flush()
            print('bpp_mean: ' + str(bpp_all / max_step))

        epoch = epoch + 1

    logfile.close()

if __name__ == "__main__":
    main()
