import argparse
import torch
from torch.autograd import Variable
import os
import glob as gb
from PIL import Image
import numpy as np
from torch.nn import functional as F
import time

import arithmetic_coding as ac

from utils import img2patch, img2patch_padding, rgb2yuv, yuv2rgb, find_min_and_max, patch2img, model_lambdas, qp_shifts
import Quant
import Model


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def dec_binary(dec, bin_num):
    value = 0
    freqs = ac.SimpleFrequencyTable([1, 1])
    for i in range(bin_num):
        dec_c = dec.read(freqs)
        value = value + (2**(bin_num-1-i))*dec_c
    return value


def dec_lossy(args, bin_name, dec, freqs_resolution, logfile):

    trans_steps = 4
    code_block_size = args.code_block_size

    with torch.no_grad():

        freqs = ac.SimpleFrequencyTable(np.ones([32], dtype=np.int))
        model_qp = dec.read(freqs)

        freqs = ac.SimpleFrequencyTable(np.ones([8], dtype=np.int))
        qp_shift = dec.read(freqs)
        init_scale = qp_shifts[model_qp][qp_shift]
        print(init_scale)
        logfile.write(str(init_scale) + '\n')
        logfile.flush()

        # reload main model
        if model_qp > 12.5:  # 13-->26 Perceptual models
            checkpoint = torch.load(args.model_path + '/' + str(model_lambdas[model_qp]) + '_percep.pth')
        else:  # 0-->12 MSE models
            checkpoint = torch.load(args.model_path + '/' + str(model_lambdas[model_qp]) + '_mse.pth')

        all_part_dict = checkpoint['state_dict']

        models_dict = {}
        if model_lambdas[model_qp] < 0.10 + 0.001:
            models_dict['transform'] = Model.Transform_aiWave(init_scale, isAffine=True)
        else:
            models_dict['transform'] = Model.Transform_aiWave(init_scale, isAffine=False)
        models_dict['entropy_LL'] = Model.CodingLL()
        models_dict['entropy_HL'] = Model.CodingHL()
        models_dict['entropy_LH'] = Model.CodingLH()
        models_dict['entropy_HH'] = Model.CodingHH()
        models_dict['post'] = Model.Post()

        freqs = ac.SimpleFrequencyTable(np.ones([2], dtype=np.int))
        isPostGAN = dec.read(freqs)
        if isPostGAN:
            assert (model_qp > 16.5)  # Perceptual models have PostGAN
            models_dict['postGAN'] = Model.PostGAN()

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

        height = dec_binary(dec, 15)
        width = dec_binary(dec, 15)

        pad_h = int(np.ceil(height / 16)) * 16 - height
        pad_w = int(np.ceil(width / 16)) * 16 - width

        LL = torch.zeros(3, 1, (height + pad_h) // 16, (width + pad_w) // 16).cuda()
        HL_list = []
        LH_list = []
        HH_list = []
        down_scales = [2, 4, 8, 16]
        for i in range(trans_steps):
            HL_list.append(torch.zeros(3, 1, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]).cuda())
            LH_list.append(torch.zeros(3, 1, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]).cuda())
            HH_list.append(torch.zeros(3, 1, (height + pad_h) // down_scales[i],
                                       (width + pad_w) // down_scales[i]).cuda())

        min_v = np.zeros(shape=(3, 13), dtype=np.int)
        max_v = np.zeros(shape=(3, 13), dtype=np.int)
        for i in range(3):
            for j in range(13):
                min_v[i, j] = dec_binary(dec, 15) - 6016
                max_v[i, j] = dec_binary(dec, 15) - 6016
        yuv_low_bound = min_v.min(axis=0)
        yuv_high_bound = max_v.max(axis=0)
        shift_min = min_v - yuv_low_bound
        shift_max = max_v - yuv_low_bound

        subband_h = [(height + pad_h) // 2, (height + pad_h) // 4, (height + pad_h) // 8, (height + pad_h) // 16]
        subband_w = [(width + pad_w) // 2, (width + pad_w) // 4, (width + pad_w) // 8, (width + pad_w) // 16]
        padding_sub_h = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_h]
        padding_sub_w = [(int(np.ceil(tmp / code_block_size)) * code_block_size - tmp) for tmp in subband_w]

        used_scale = models_dict['transform'].used_scale()

        coded_coe_num = 0
        # decompress LL
        tmp_stride = subband_w[3] + padding_sub_w[3]
        tmp_hor_num = tmp_stride // code_block_size
        paddings = (0, padding_sub_w[3], 0, padding_sub_h[3])
        enc_LL = F.pad(LL, paddings, "constant")
        enc_LL = img2patch(enc_LL, code_block_size, code_block_size, code_block_size)
        paddings = (6, 6, 6, 6)
        enc_LL = F.pad(enc_LL, paddings, "constant")
        for h_i in range(code_block_size):
            for w_i in range(code_block_size):
                cur_ct = enc_LL[:, :, h_i:h_i + 13, w_i:w_i + 13]

                prob = models_dict['entropy_LL'](cur_ct, yuv_low_bound[0], yuv_high_bound[0])
                prob = prob.cpu().data.numpy()

                index = []

                prob = prob * freqs_resolution
                prob = prob.astype(np.int64)

                for sample_idx, prob_sample in enumerate(prob):
                    coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                             h_i * tmp_stride + \
                             ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                             w_i
                    if (coe_id % tmp_stride) < subband_w[3] and (coe_id // tmp_stride) < subband_h[3]:
                        yuv_flag = sample_idx % 3
                        if shift_min[yuv_flag, 0] < shift_max[yuv_flag, 0]:
                            freqs = ac.SimpleFrequencyTable(
                                prob_sample[shift_min[yuv_flag, 0]:shift_max[yuv_flag, 0] + 1])
                            dec_c = dec.read(freqs) + min_v[yuv_flag, 0]
                        else:
                            dec_c = min_v[yuv_flag, 0]
                        coded_coe_num = coded_coe_num + 1
                        index.append(dec_c)
                    else:
                        index.append(0)
                enc_LL[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
        LL = enc_LL[:, :, 6:-6, 6:-6]
        LL = patch2img(LL, subband_h[3] + padding_sub_h[3], subband_w[3] + padding_sub_w[3])
        LL = LL[:, :, 0:subband_h[3], 0:subband_w[3]]
        print('LL decoded')

        LL = LL * used_scale

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
            context = img2patch_padding(context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob = models_dict['entropy_HL'](cur_ct, cur_context,
                                                     yuv_low_bound[3 * j + 1], yuv_high_bound[3 * j + 1], j)
                    prob = prob.cpu().data.numpy()

                    index = []

                    prob = prob * freqs_resolution
                    prob = prob.astype(np.int64)

                    for sample_idx, prob_sample in enumerate(prob):
                        coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            yuv_flag = sample_idx % 3
                            if shift_min[yuv_flag, 3 * j + 1] < shift_max[yuv_flag, 3 * j + 1]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[yuv_flag, 3 * j + 1]:shift_max[yuv_flag, 3 * j + 1] + 1])
                                dec_c = dec.read(freqs) + min_v[yuv_flag, 3 * j + 1]
                            else:
                                dec_c = min_v[yuv_flag, 3 * j + 1]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
            HL_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HL_list[j] = patch2img(HL_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HL_list[j] = HL_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('HL' + str(j) + ' decoded')

            HL_list[j] = HL_list[j] * used_scale

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
            context = img2patch_padding(context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob = models_dict['entropy_LH'](cur_ct, cur_context,
                                                     yuv_low_bound[3 * j + 2], yuv_high_bound[3 * j + 2], j)
                    prob = prob.cpu().data.numpy()

                    index = []

                    prob = prob * freqs_resolution
                    prob = prob.astype(np.int64)
                    for sample_idx, prob_sample in enumerate(prob):
                        coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            yuv_flag = sample_idx % 3
                            if shift_min[yuv_flag, 3 * j + 2] < shift_max[yuv_flag, 3 * j + 2]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[yuv_flag, 3 * j + 2]:shift_max[yuv_flag, 3 * j + 2] + 1])
                                dec_c = dec.read(freqs) + min_v[yuv_flag, 3 * j + 2]
                            else:
                                dec_c = min_v[yuv_flag, 3 * j + 2]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
            LH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            LH_list[j] = patch2img(LH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            LH_list[j] = LH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('LH' + str(j) + ' decoded')

            LH_list[j] = LH_list[j] * used_scale

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
            context = img2patch_padding(context, code_block_size + 12, code_block_size + 12, code_block_size, 6)
            for h_i in range(code_block_size):
                for w_i in range(code_block_size):
                    cur_ct = enc_oth[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    cur_context = context[:, :, h_i:h_i + 13, w_i:w_i + 13]
                    prob = models_dict['entropy_HH'](cur_ct, cur_context,
                                                     yuv_low_bound[3 * j + 3], yuv_high_bound[3 * j + 3], j)

                    prob = prob.cpu().data.numpy()
                    index = []

                    prob = prob * freqs_resolution
                    prob = prob.astype(np.int64)
                    for sample_idx, prob_sample in enumerate(prob):
                        coe_id = ((sample_idx // 3) // tmp_hor_num) * tmp_hor_num * code_block_size * code_block_size + \
                                 h_i * tmp_stride + \
                                 ((sample_idx // 3) % tmp_hor_num) * code_block_size + \
                                 w_i
                        if (coe_id % tmp_stride) < subband_w[j] and (coe_id // tmp_stride) < subband_h[j]:
                            yuv_flag = sample_idx % 3
                            if shift_min[yuv_flag, 3 * j + 3] < shift_max[yuv_flag, 3 * j + 3]:
                                freqs = ac.SimpleFrequencyTable(
                                    prob_sample[shift_min[yuv_flag, 3 * j + 3]:shift_max[yuv_flag, 3 * j + 3] + 1])
                                dec_c = dec.read(freqs) + min_v[yuv_flag, 3 * j + 3]
                            else:
                                dec_c = min_v[yuv_flag, 3 * j + 3]
                            coded_coe_num = coded_coe_num + 1
                            index.append(dec_c)
                        else:
                            index.append(0)
                    enc_oth[:, 0, h_i + 6, w_i + 6] = torch.from_numpy(np.array(index).astype(np.float)).cuda()
            HH_list[j] = enc_oth[:, :, 6:-6, 6:-6]
            HH_list[j] = patch2img(HH_list[j], subband_h[j] + padding_sub_h[j], subband_w[j] + padding_sub_w[j])
            HH_list[j] = HH_list[j][:, :, 0:subband_h[j], 0:subband_w[j]]
            print('HH' + str(j) + ' decoded')

            HH_list[j] = HH_list[j] * used_scale

            LL = models_dict['transform'].inverse_trans(LL, HL_list[j], LH_list[j], HH_list[j], j)

        assert (coded_coe_num == (height + pad_h) * (width + pad_w) * 3)

        recon = LL
        recon = yuv2rgb(recon.permute(1, 0, 2, 3))
        recon = recon[:, :, 0:height, 0:width]

        if isPostGAN:

            recon = models_dict['post'](recon)

            if height * width > 1080 * 1920:
                h_list = [0, height//2, height]
                w_list = [0, width//2, width]
                k_ = 2
            else:
                h_list = [0, height]
                w_list = [0, width]
                k_ = 1
            gan_rgb_post = torch.zeros_like(recon)
            for _i in range(k_):
                for _j in range(k_):
                    pad_start_h = max(h_list[_i] - 64, 0) - h_list[_i]
                    pad_end_h = min(h_list[_i + 1] + 64, height) - h_list[_i + 1]
                    pad_start_w = max(w_list[_j] - 64, 0) - w_list[_j]
                    pad_end_w = min(w_list[_j + 1] + 64, width) - w_list[_j + 1]
                    tmp = models_dict['postGAN'](recon[:, :, h_list[_i] + pad_start_h:h_list[_i + 1] + pad_end_h,
                                        w_list[_j] + pad_start_w:w_list[_j + 1] + pad_end_w])
                    gan_rgb_post[:, :, h_list[_i]:h_list[_i + 1], w_list[_j]:w_list[_j + 1]] = tmp[:, :,
                                                                                               -pad_start_h:tmp.size()[
                                                                                                                2] - pad_end_h,
                                                                                               -pad_start_w:tmp.size()[
                                                                                                                3] - pad_end_w]
            recon = gan_rgb_post
        else:

            h_list = [0, height//3, height//3*2, height]
            w_list = [0, width//3, width//3*2, width]
            k_ = 3
            rgb_post = torch.zeros_like(recon)
            for _i in range(k_):
                for _j in range(k_):
                    pad_start_h = max(h_list[_i] - 64, 0) - h_list[_i]
                    pad_end_h = min(h_list[_i + 1] + 64, height) - h_list[_i + 1]
                    pad_start_w = max(w_list[_j] - 64, 0) - w_list[_j]
                    pad_end_w = min(w_list[_j + 1] + 64, width) - w_list[_j + 1]
                    tmp = models_dict['post'](recon[:, :, h_list[_i] + pad_start_h:h_list[_i + 1] + pad_end_h,
                                        w_list[_j] + pad_start_w:w_list[_j + 1] + pad_end_w])
                    rgb_post[:, :, h_list[_i]:h_list[_i + 1], w_list[_j]:w_list[_j + 1]] = tmp[:, :,
                                                                                               -pad_start_h:tmp.size()[
                                                                                                                2] - pad_end_h,
                                                                                               -pad_start_w:tmp.size()[
                                                                                                                3] - pad_end_w]
            recon = rgb_post

        recon = torch.clamp(torch.round(recon), 0., 255.)
        recon = recon[0, :, :, :]
        recon = recon.permute(1, 2, 0)
        recon = recon.cpu().data.numpy().astype(np.uint8)
        img = Image.fromarray(recon, 'RGB')
        img.save(args.recon_dir + '/' + bin_name + '.png')

        logfile.flush()
