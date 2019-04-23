# breastpathq
A solution for SPIE breastpathq challenge

We are based on the intuition that the score we ultimately need is positively correlated with the area of the cancer cell in the total area of the image, so the best way is to get the cancer cell segmentation mask for a given image so that we can get us very accurately. The desired score, but because the supervision information given by the task has only the final score, and the additional data set gives the supervision information of the number of cancer cells, we have proposed the following methods to accomplish this task:

(1) We take the Gaussian of the cancer cell center point coordinates in the cell to obtain coarse segmentation supervision information to train the rough segmentation network.

(2) We use the trained network to get the rough segmentation result of the original data set.

(3) We combine the original data with the features provided by its corresponding rough segmentation results to train the final regression network. We use the attention mechanism to provide auxiliary information from the segmentation results, so that the purpose of the network is clearer.


the frame of our method is :

![image](https://github.com/iceli1007/SPIE2019/tree/master/image/main_framework.png)


Structure description：

Creat_cell_image.py：The images and labels in the cells folder are cut into 256*256 size by sliding window and saved in the cell_image folder.

Unet_creat_mask_train.py:train a segmentation nework

Creat_mask.py:The image added by the train_images file is used as input, and the corresponding mask is output to the train_masks through the model output.

model_train.py:Designed a feature extraction network using mask information as attention and training




