
inference instructions
---------------------------

1. To produce the bitstreams, the used GPU is 3090, CPU is Gold 5218, and the version of pytorch is 1.6. It's highly recommended to decode the bitstreams with the above environment.

2. The following is the decode command, which can decode all the bitstreams in the bin_dir one by one.
```
   python your_path/iWaveCodec-LosslessAndLossy/decodeAPP.py --bin_dir --log_dir  --recon_dir  --model_path 
```

+   --bin_dir    bitstream path.
+   --log_dir    output log path.
+   --recon_dir  output reconstructed images path.
+   --model_path model path.

3. Example:
```
   python source/inference//decodeAPP.py --bin_dir /data/dongcunhui/FVC-wavelet/perceptual/1857Submit-final/MSE/splitForTestDecodeTime/bin3 --log_dir /data/dongcunhui/FVC-wavelet/perceptual/1857Submit-final/MSE/splitForTestDecodeTime/bin3/dec_log --recon_dir /data/dongcunhui/FVC-wavelet/perceptual/1857Submit-final/MSE/splitForTestDecodeTime/bin3/dec_recon --model_path /data/dongcunhui/FVC-wavelet/perceptual/1857Models
```
4. The range of model_qp should be set from 0 to 12 if you want to get the results optimized for MSE; from 13 to 26 if you want to get the results optimized for perception. 
    In lossy mode, if the model_qp is no more than 17, the value of isPostGAN should be set as 0, and vice versa.
    The value of model_qp should be set as 27 if you want to get the lossless results.

training instructions
---------------------------    
Using the published model as a pre-training model is recommoned.

Explain of input parameters for training MSE and perception models:

+   --input_dir       the path of training dataset
+   --test_input_dir  the path of test dataset
+   --save_model_dir  the path of models to be saved
+   --log_dir         the path of log
+   --epochs          the value of maximum epoch for training
+   --patch_size      the size of cropped training image
+   --batch_size 
+   --num_workers     the number of threads when reading data
+   --lambda_d        for adjusting the bitrate. The larger lambda_d, the higher the bitrate.
+   --scale_list      it is the initial quantization step, which can be a number or a list, such as --scale_list 8 or --scale_list 8 7 9. When it is a number, the model is not a variable bit rate model, and when it is a list, the model is a variable bit rate model.
    + variable bit rate model：

        + For 13 MSE pre-training models, the corresponding relationship between models and the scale_list is shown as below:\
						0.4_mse.pth:	[8, 7.5, 8.5, 7, 9, 6.5, 9.5],\
						0.25_mse.pth:	[8, 7.5, 8.5, 7, 9, 6.5, 9.5],\
						0.16_mse.pth:	[8, 7.5, 8.5, 7, 9, 6.5, 9.5],\
						0.10_mse.pth:	[16, 15, 17, 14, 18, 13, 19],\
						0.0625_mse.pth:	[16, 15, 17, 14, 18, 13, 19],\
						0.039_mse.pth:	[16, 15, 18, 14, 20, 13, 22],\
						0.024_mse.pth:	[32, 30, 33, 34, 35, 36, 37],\
						0.015_mse.pth:	[32, 31, 34, 30, 36, 29, 38],\
						0.0095_mse.pth:	[32, 30, 36, 28, 40, 26, 44],\
						0.006_mse.pth:	[64, 60, 66, 56, 68, 70, 72],\
						0.0037_mse.pth:	[64, 62, 70, 58, 76, 54, 82],\
						0.002_mse.pth:	[64, 58, 72, 52, 80, 46, 88],\
						0.0012_mse.pth:	[64, 56, 72, 48, 80, 40, 88],
        + For 14 perceptual pre-training models, the corresponding relationship between models and the scale_list is shown as below:\
						0.4_percep.pth:		[8, 7.5, 8.5, 7, 9, 6.5, 9.5],\
						0.25_percep.pth:	[8, 7.5, 8.5, 7, 9, 6.5, 9.5],\
						0.16_percep.pth:	[8, 7.5, 8.5, 7, 9, 6.5, 9.5],\
						0.10_percep.pth: 	[16, 15, 17, 14, 18, 13, 19],\
						0.018_percep.pth:	[16, 15, 17, 14, 18, 13, 19],\
						0.012_percep.pth: 	[16, 15, 18, 14, 20, 13, 22],\
						0.0075_percep.pth:	[32, 30, 33, 34, 35, 36, 37],\
						0.0048_percep.pth: 	[32, 31, 34, 30, 36, 29, 38],\
						0.0032_percep.pth: 	[32, 30, 36, 28, 40, 26, 44],\
						0.002_percep.pth:	[64, 60, 66, 56, 68, 52, 70],\
						0.00145_percep.pth:	[64, 62, 70, 60, 76, 58, 82],\
						0.0008_percep.pth: 	[64, 58, 72, 52, 80, 46, 88],\
						0.00055_percep.pth:	[64, 56, 72, 48, 80, 40, 88],\
						0.00035_percep.pth:	[64, 56, 72, 48, 80, 40, 88],
    + none variable bit rate model：\
  The first value of scale_list in variable bit rate model.
  
+   --alpha_start the starting point of alpha in soft quantization
+   --alpha_end the maximum value of alpha in soft quantization
+   --load_model the path of the pre-training model. Naming rules of pre-training models: for MSE, the optimized model is lambda_d_mse.pth, the model optimized by perception is lambda_d_percep.pth
+   --rate_model the path of pre-training model for entropy coding module 
+   --post_model the path of pre-training model for post-process module 
+   --wavelet_affine whether to use affine wavelet transform\
  (When training the MSE model, the high bit rate model does not use the affine wavelet transform, and the low bit rate model uses the affine wavelet. When training the perceptual model, because the subjective quality of the high bit rate model is very good, the perceptual model does not need to be trained in the high bit rate, and the affine wavelet is used in the training of the low bit rate model)
