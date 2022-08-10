import argparse
import torch
from torch.autograd import Variable
import os
import glob as gb
from PIL import Image
import numpy as np
from torch.nn import functional as F
import Quant
import copy
import time

import arithmetic_coding as ac
from utils import img2patch, img2patch_padding, rgb2yuv_lossless, yuv2rgb_lossless, find_min_and_max, model_lambdas, qp_shifts
import Model


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def write_binary(enc, value, bin_num):
    bin_v = '{0:b}'.format(value).zfill(bin_num)
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        enc.write(freqs, int(bin_v[i]))


def enc_lossless(args):

    assert args.isLossless==1

    if not os.path.exists(args.bin_dir):
        os.makedirs(args.bin_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.recon_dir):
        os.makedirs(args.recon_dir)

    logfile = open(args.log_dir + '/enc_log.txt', 'a')

    assert args.model_qp==27
    assert args.qp_shift==0
    init_scale = qp_shifts[args.model_qp][args.qp_shift]
    print(init_scale)
    logfile.write(str(init_scale) + '\n')
    logfile.flush()

    code_block_size = args.code_block_size

    bin_name = args.img_name[0:-4] + '_' + str(args.model_qp) + '_' + str(args.qp_shift)
    bit_out = ac.CountingBitOutputStream(
        bit_out=ac.BitOutputStream(open(args.bin_dir + '/' + bin_name + '.bin', "wb")))
    enc = ac.ArithmeticEncoder(bit_out)
    freqs_resolution = 1e6

    freqs = ac.SimpleFrequencyTable(np.ones([2], dtype=np.int))
    enc.write(freqs, args.isLossless)

    # freqs = ac.SimpleFrequencyTable(np.ones([32], dtype=np.int))
    # enc.write(freqs, args.model_qp)
    #
    # freqs = ac.SimpleFrequencyTable(np.ones([8], dtype=np.int))
    # enc.write(freqs, args.qp_shift)

    trans_steps = 4

    model_path = args.model_path + '/' + str(model_lambdas[args.model_qp]) + '_lossless.pth'
    print(model_path)
    checkpoint = torch.load(model_path)

    all_part_dict = checkpoint['state_dict']

    models_dict = {}

    models_dict['transform'] = Model.Transform_lossless()
    models_dict['entropy_LL'] = Model.CodingLL_lossless()
    models_dict['entropy_HL'] = Model.CodingHL_lossless()
    models_dict['entropy_LH'] = Model.CodingLH_lossless()
    models_dict['entropy_HH'] = Model.CodingHH_lossless()
    # models_dict['post'] = Model.Post()
    # if args.isPostGAN:
    #     freqs = ac.SimpleFrequencyTable(np.ones([2], dtype=np.int))
    #     enc.write(freqs, 1)
    #     assert (args.model_qp > 16.5) # Perceptual models have PostGAN
    #     models_dict['postGAN'] = Model.PostGAN()
    # else:
    #     freqs = ac.SimpleFrequencyTable(np.ones([2], dtype=np.int))
    #     enc.write(freqs, 0)

    models_dict_update = {}
    for key, model in models_dict.items():
        myparams_dict = model.state_dict()
        part_dict = {k: v for k, v in all_part_dict.items() if k in myparams_dict}
        myparams_dict.update(part_dict)
        model.load_state_dict(myparams_dict)
        if torch.cuda.is_available():
            model = model.cuda()
            model.eval()
        models_dict_update[key] = model
    models_dict.update(models_dict_update)

    print('Load pre-trained model succeed!')
    logfile.write('Load pre-trained model succeed!' + '\n')
    logfile.flush()

    img_path = args.input_dir + '/' + args.img_name

    with torch.no_grad():
        start = time.time()

        print(img_path)
        logfile.write(img_path + '\n')
        logfile.flush()

        img = Image.open(img_path)
        img = np.array(img, dtype=np.float32)
        original_img = copy.deepcopy(img)

        img = rgb2yuv_lossless(img).astype(np.float32)
        img = torch.from_numpy(img)
        # img -> [n,c,h,w]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2)
        # original_img = img

        # img -> (%16 == 0)
        size = img.size()
        height = size[2]
        width = size[3]
        # encode height and width, in the range of [0, 2^15=32768]
        write_binary(enc, height, 15)
        write_binary(enc, width, 15)

        pad_h = int(np.ceil(height / 16)) * 16 - height
        pad_w = int(np.ceil(width / 16)) * 16 - width
        paddings = (0, pad_w, 0, pad_h)
        img = F.pad(img, paddings, 'replicate')

        # img -> [3,1,h,w], YUV in batch dim
        # img = rgb2yuv(img)
        img = img.permute(1, 0, 2, 3)

        input_img_v = to_variable(img)
        LL, HL_list, LH_list, HH_list = models_dict['transform'].forward_trans(input_img_v)

        min_v, max_v = find_min_and_max(LL, HL_list, LH_list, HH_list)
        # for all models, the quantized coefficients are in the range of [-6016, 12032]
        # 15 bits to encode this range
        for i in range(3):
            for j in range(13):
                tmp = min_v[i, j] + 6016
                write_binary(enc, tmp, 15)
                tmp = max_v[i, j] + 6016
                write_binary(enc, tmp, 15)
        yuv_low_bound = min_v.min(axis=0)
        yuv_high_bound = max_v.max(axis=0)
        shift_min = min_v - yuv_low_bound
        shift_max = max_v - yuv_low_bound

        subband_h = [(height + pad_h) // 2, (height + pad_h) // 4, (height + pad_h) // 8, (height + pad_h) // 16]
        subband_w = [(width + pad_w) // 2, (width + pad_w) // 4, (width + pad_w) // 8, (width + pad_w) // 16]


        padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_h]
        padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_w]

        coded_coe_num = 0

        # compress LL
        tmp_stride = subband_w[3] + padding_sub_w[3]
        tmp_hor_num = tmp_stride // code_block_size
        paddings = (0, padding_sub_w[3], 0, padding_sub_h[3])
        enc_LL = F.pad(LL, paddings, "constant")
        enc_LL = img2patch(enc_LL, code_block_size, code_block_size, code_block_size)
        paddings = (6, 6, 6, 6)
        enc_LL = F.pad(enc_LL, paddings, "constant")
        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct = copy.deepcopy(enc_LL[:, :, h_i:h_i + 13, w_i:w_i + 13])
                cur_ct[:, :, 13 // 2 + 1:13, :] = 0.
                cur_ct[:, :, 13 // 2, 13 // 2:13] = 0.
                prob = models_dict['entropy_LL'](cur_ct, yuv_low_bound[0], yuv_high_bound[0])

                prob = prob.cpu().data.numpy()
                index = enc_LL[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int)
                # index = index - lower_bound

                prob = prob * freqs_resolution
                prob = prob.astype(np.int64)
                for sample_idx, prob_sample in enumerate(prob):
                    coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size*code_block_size + \
                             h_i * tmp_stride + \
                             ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                             w_i
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:
                        yuv_flag = sample_idx % 3
                        # if True:
                        if shift_min[yuv_flag, 0] < shift_max[yuv_flag, 0]:
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[yuv_flag, 0]:shift_max[yuv_flag, 0] + 1])
                            data = index[sample_idx] - min_v[yuv_flag, 0]
                            assert data >= 0
                            enc.write(freqs, data)
                        coded_coe_num = coded_coe_num + 1
        print('LL encoded...')

        # LL = LL * used_scale

        for i in range(trans_steps):
            j = trans_steps - 1 - i
            tmp_stride = subband_w[j] + padding_sub_w[j]
            tmp_hor_num = tmp_stride // code_block_size
            # compress HL
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(HL_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(LL, paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context = F.pad(context, paddings, "reflect")
            context = img2patch_padding(context, code_block_size+12, code_block_size+12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = copy.deepcopy(enc_oth[:, :, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob = models_dict['entropy_HL'](cur_ct, cur_context,
                                                              yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1], j)

                    prob = prob.cpu().data.numpy()
                    index = enc_oth[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int)
                    # index = index - lower_bound

                    prob = prob * freqs_resolution
                    prob = prob.astype(np.int64)
                    for sample_idx, prob_sample in enumerate(prob):
                        coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size*code_block_size + \
                                 h_i * tmp_stride + \
                                 ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            yuv_flag = sample_idx % 3
                            # if True:
                            if shift_min[yuv_flag, 3 * j + 1] < shift_max[yuv_flag, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[yuv_flag, 3 * j + 1]:shift_max[yuv_flag, 3 * j + 1] + 1])
                                data = index[sample_idx] - min_v[yuv_flag, 3 * j + 1]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1

            print('HL' + str(j) + ' encoded...')

            # HL_list[j] = HL_list[j]*used_scale

            # compress LH
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(LH_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(torch.cat((LL, HL_list[j]), dim=1), paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context = F.pad(context, paddings, "reflect")
            context = img2patch_padding(context, code_block_size+12, code_block_size+12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = copy.deepcopy(enc_oth[:, :, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob = models_dict['entropy_LH'](cur_ct, cur_context,
                                                              yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2], j)

                    prob = prob.cpu().data.numpy()
                    index = enc_oth[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int)
                    # index = index - lower_bound

                    prob = prob * freqs_resolution
                    prob = prob.astype(np.int64)
                    for sample_idx, prob_sample in enumerate(prob):
                        coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size*code_block_size + \
                                 h_i * tmp_stride + \
                                 ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            yuv_flag = sample_idx % 3
                            # if True:
                            if shift_min[yuv_flag, 3 * j + 2] < shift_max[yuv_flag, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[yuv_flag, 3 * j + 2]:shift_max[yuv_flag, 3 * j + 2] + 1])
                                data = index[sample_idx] - min_v[yuv_flag, 3 * j + 2]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1
            print('LH' + str(j) + ' encoded...')

            # LH_list[j] = LH_list[j] * used_scale

            # compress HH
            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            enc_oth = F.pad(HH_list[j], paddings, "constant")
            enc_oth = img2patch(enc_oth, code_block_size, code_block_size, code_block_size)
            paddings = (6, 6, 6, 6)
            enc_oth = F.pad(enc_oth, paddings, "constant")

            paddings = (0, padding_sub_w[j], 0, padding_sub_h[j])
            context = F.pad(torch.cat((LL, HL_list[j], LH_list[j]), dim=1), paddings, "reflect")
            paddings = (6, 6, 6, 6)
            context = F.pad(context, paddings, "reflect")
            context = img2patch_padding(context, code_block_size+12, code_block_size+12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = copy.deepcopy(enc_oth[:, :, h_i:h_i + 13, w_i:w_i + 13])
                    cur_ct[:, :, 13 // 2 + 1:13, :] = 0.
                    cur_ct[:, :, 13 // 2, 13 // 2:13] = 0.
                    cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob = models_dict['entropy_HH'](cur_ct, cur_context,
                                                              yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3], j)

                    prob = prob.cpu().data.numpy()
                    index = enc_oth[:, 0, h_i + 6, w_i + 6].cpu().data.numpy().astype(np.int)
                    # index = index - lower_bound

                    prob = prob * freqs_resolution
                    prob = prob.astype(np.int64)
                    for sample_idx, prob_sample in enumerate(prob):
                        coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size*code_block_size + \
                                 h_i * tmp_stride + \
                                 ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            yuv_flag = sample_idx % 3
                            # if True:
                            if shift_min[yuv_flag, 3 * j + 3] < shift_max[yuv_flag, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[yuv_flag, 3 * j + 3]:shift_max[yuv_flag, 3 * j + 3] + 1])
                                data = index[sample_idx] - min_v[yuv_flag, 3 * j + 3]
                                assert data >= 0
                                enc.write(freqs, data)
                            coded_coe_num = coded_coe_num + 1

            print('HH' + str(j) + ' encoded...')
            # HH_list[j] = HH_list[j] * used_scale

            LL = models_dict['transform'].inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j], j)

        assert (coded_coe_num == (height + pad_h) * (width + pad_w) * 3)

        recon = LL.permute(1, 0, 2, 3)
        recon = recon[:, :, 0:height, 0:width]
        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy()
        recon = yuv2rgb_lossless(recon).astype(np.float32)

        mse = np.mean((recon - original_img) ** 2)
        psnr = (10. * np.log10(255. * 255. / mse))


        recon = np.clip(recon, 0., 255.).astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(args.recon_dir + '/' + bin_name + '.png')

        enc.finish()
        print('encoding finished!')
        logfile.write('encoding finished!' + '\n')
        end = time.time()
        print('Encoding-time: ', end - start)
        logfile.write('Encoding-time: ' + str(end - start) + '\n')

        bit_out.close()
        print('bit_out closed!')
        logfile.write('bit_out closed!' + '\n')

        filesize = bit_out.num_bits / height / width
        print('BPP: ', filesize)
        logfile.write('BPP: ' + str(filesize) + '\n')
        logfile.flush()

        print('PSNR: ', psnr)
        logfile.write('PSNR: ' + str(psnr) + '\n')
        logfile.flush()


    logfile.close()
