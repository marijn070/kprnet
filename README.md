# [KPRNet: Improving projection-based LiDARsemantic segmentation](https://arxiv.org/pdf/2007.12668.pdf)

![Video](kprnet.gif)

## Installation

Install [apex](https://github.com/NVIDIA/apex) and the packages in requirements.txt

## Experiment 

Download pre-trained [resnext_cityscapes_2p.pth](https://drive.google.com/file/d/1aioKjoxcrfqUtkWQgbo64w8YoLcVAW2Z/view?usp=sharing). The path should be given in `model_dir`.  CityScapes pretraining will be added later.

The result from paper is trained on 8 16GB GPUs (total batch size 24).

To train run:

```bash
python train_kitti.py \
  --semantic-kitti-dir path_to_semantic_kitti \
  --model-dir location_where_your_pretrained_model_is \
  --checkpoint-dir your_output_dir
```

The fully trained model weights can be downloaded [here](https://drive.google.com/file/d/11mUMdFPNT-05lC54Ru_2OwdwqTPV4jrW/view?usp=sharing) .

## Acknowledgments
[KPConv](https://github.com/HuguesTHOMAS/KPConv-PyTorch) 

[RangeNet++](https://github.com/PRBonn/lidar-bonnetal) 

[HRNet](https://github.com/HRNet)


```
@article{kochanov2020kprnet,
  title={KPRNet: Improving projection-based LiDAR semantic segmentation},
  author={Kochanov, Deyvid and Nejadasl, Fatemeh Karimi and Booij, Olaf},
  journal={arXiv preprint arXiv:2007.12668},
  year={2020}
}
```


# Reproduction Documentation

# KPRNet: Improving projection-based LiDAR semantic segmentation
Group 61  - Marijn De Rijk, Jakob Bichler, Christian Reinprecht


## Introduction



The goal of this blog post is to present and describe our implementation to reproduce the Deep Learning paper “KPRNet: Improving projection-based LiDAR semantic segmentation”. 
This is part of the final project in CS4240 Deep Learning (2022/23 Q3) at Delft University of Technology.


## Datasets

#### The KITTI Dataset

The KITTI Dataset was recorded in Karlsruhe, Germany, using a Velodyne HDL-64E rotating 3D laser scanner. It consists of 21 sequences and over 43,000 frames, each of which contains about 120,000 points (Geiger, 2013). 

#### The SemanticKITTI

In their work, Behley et al. annotated KITTI pointwise with labels, covering 25 classes in total (Behley, 2019). This SemanticKITTI was used for training and evaluation of the KPRNet which is presented in the paper we are aiming to reproduce in this assignment (Kochanov, 2020).
With their model, Kochanov et al. achieved a  mean-IoU of  63.1 on the SemanticKITTI dataset, outperforming the previous state-of-the-art model by 3.6 points.

![](https://i.imgur.com/ridNDhG.png)

The image displays one frame of sequence 1 of the SemanticKITTI, in which the datapoints in 3D as well as their colors corresponding to the ground truth labels can be seen. This visualization was created using the  semantic-kitti-api .

![](https://i.imgur.com/tvD4siY.png)

In the above image the number of points that are related to each object are shown. It becomes clear that some objects, such as the road, vegetation and cars are overrepresented compared to objects such as bicyclists and humans.  


#### Ego-motion compensation 

With the following self-created illustration we aim to clarify the problems that arise with the ego-motion compensation for the KITTI dataset, which ITTIK aims to resolve.

![](https://i.imgur.com/wyFaF61.jpg)

Due to the motion of the vehicle, hypothetical Lidar sweeps will form a spiral. From a world-fixed frame of view, this will result in deformed objects in the environment. 


![](https://i.imgur.com/Mx4tXRZ.jpg)

In the Lidar frame of view, points will be evenly spaced and therefore uniformly cover a range image projection (shown on the right).



![](https://i.imgur.com/owKL73Q.jpg)

Ego-motion compensation of the Lidar points will transform the spiral into a circle, fixing the deformations in the fixed-frame world view, but resulting in a non-uniform mapping to the range-image.

Sometimes, multiple Lidar points will be mapped to one pixel, even if their spatial distance may be large. Other pixels might not be mapped to any Lidar point.

This problem is addressed by Olaf Booij's ITTIK dataset (Booij, 2022), as it provides range image coordinates of the Lidar points *before* the ego-motion is compensated.



#### Experimental Dataset: ITTIK

The ITTIK (I jusT wanT my kittI bacK) Dataset is based on the KITTI Dataset. The Velodyne rotation is inversed, resulting in a denser pointcloud than the one provided in the originial KITTI Dataset (Booij, 2022)

![](https://i.imgur.com/RlbcDGZ.png)

In the left image, an excerpt from the KITTI Dataset can be seen. On the right, the same excerpt from the ITTIK Dataset is displayed, in which the Velodyne rotation is reversed, thus containing about 10 percent more data points.


The ITTIK Dataset consists of binary files containing (u,v) coordinates of type uint16. This can be understood as a lookup-table that enables to look up the corresponding (u,v) coordinates for each point in the original pointcloud, since the indices coincide. 
Each point maps to exactly one pixel, but not every pixel is mapped to a point.
The semantic KITTI Dataset contains just as many entries, but of type float32 and has 4 values for every entry (x, y, z, remission).



## Model 
The proposed model combines a 2D semantic segmentation network (Section 2.1), and a 3D point-wise layer. The convolutional network gets as input a LiDAR scan projected to a range image. The resulting 2D CNN features are projected back to their respective 3D points and passed to a 3D point-wise module (Section 2.2), which predicts the final labels. 

The architecture of this KPRnet can be seen in the image below.

![](https://i.imgur.com/3cWKznq.png)









## Results


### 1. Adjusting existing Codebase to work with ITTIK

In order to handle the ITTIK dataset with the pretrained model, we adjusted the following:


1. Created new DataLoader class SemanticITTIK
    - ITTIK coordinates (u,v) are used instead of the original range projection of the pointcloud
    
2. Created new function do_range_projection_ittik
   - Depth information is recovered from the original KITTI dataset since ITTIK does not contain that information
   
3. Adjusted transform_test 
    - ITTIK's horizontal resolution does not match the one of the KITTI dataset for each sweep, so the resizing was changed to be dynamic 

4. Parameterized run_inference.py 
    - Additional imports
    - Added flags to specify which dataset to use from the command line

5. Created bash-script inference.sh
    - Makes running inference with KPRnet easier by allowing to specify which dataset and which split to use  


The changes can be seen at:
https://github.com/marijn070/kprnet


### 2. Results KITTI vs. ITTIK 
Since the ITTIK dataset does not contain the test sequences (yet), we ran both models (original on KITTI and adjusted on ITTIK) on sequence 8 to ensure the comparability of results.


|                 |   KITTI &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|   ITTIK &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;|   Diff: KITTI - ITTIK  &nbsp; &nbsp;|
|-----------------|----------|----------|--------------|
|   mIoU          |   0.631  |   0.606  |  0.025       |
|                 |          |          |              |
|   car:          |   0.946  |   0.949  |  -0.003      |
|   bicycle:      |   0.431  |   0.413  |  0.018       |
|   motorcycle:   |   0.603  |   0.596  |  0.007       |
|   truck:        |   0.759  |   0.597  |  0.162       |
|   other-vehicle:|   0.514  |   0.543  |  -0.029      |
|   person:       |   0.753  |   0.724  |  0.029       |
|   bicyclist:    |   0.811  |   0.792  |  0.019       |
|   motorcyclist: |   0.000  |   0.000  |  0.000       |
|   road:         |   0.956  |   0.943  |  0.013       |
|   parking:      |   0.512  |   0.430  |  0.082       |
|   sidewalk:     |   0.836  |   0.813  |  0.023       |
|   other-ground: |   0.002  |   0.010  |  -0.008      |
|   building:     |   0.896  |   0.881  |  0.015       |
|   fence:        |   0.599  |   0.515  |  0.084       |
|   vegetation:   |   0.884  |   0.886  |  -0.002      |
|   trunk:        |   0.663  |   0.656  |  0.007       |
|   terrain:      |   0.761  |   0.761  |  0.000       |
|   pole:         |   0.633  |   0.583  |  0.050       |
|   traffic-sign: |   0.432  |   0.425  |  0.007       |


As it can be seen in the table, the mean IoU of 60.6% that we achieved with running the model on the ITTIK validation dataset was as expected below the mIoU of the original KITTI dataset with a mIoU of 63.1%. 

The performance on the ITTIK dataset is still remarkable, since the model was neither re-trained nor fine-tuned on that dataset. Nevertheless, it outperforms KITTI in multiple categories of IoUs such as car. We would expect the performance of the model to siginificantly improve, if the model was retrained and fine-tuned on the ITTIK dataset. 



### Visualization of predicted segmentation  KITTI vs. ITTIK


<img src="https://i.imgur.com/NlgScgR.png" width="350"/> <img src="https://i.imgur.com/kDT60f2.png" width="345"/>
In both KITTI (left) and ITTIK (right), the cars are well segmented (IoU of 94.6 and 94.9 respectively). The surroundings are sometimes not that accurately segmented for both datasets. 


<img src="https://i.imgur.com/BZ3Bif6.png" width="250"/> <img src="https://i.imgur.com/pYMoG1p.png" width="333"/>
Traffic signs are not that accurately classified with 43.2 and 42.5 for KITTI and ITTIK respectively. For both models the corresponding pointcloud was segmented with 4 different classes.



## Challenges

One of the challenges results from the compressed SemanticKITTI Dataset containing more than 80GB of Data. Therefore, it was not possible to run and test the KPRNet on Google Colab, as the 107GB storage provided in the Notebook were not sufficient to store the unzipped dataset and the required packages and installations. We therefore first moved the Dataset to a Google One Drive, accessing it from the Colab notebook. As the results could not be stored permanently and the dependencies had to be reinstalled every time, we decided to move our project to a google cloud VM.

As the training of the KPRNet on  8 Tesla V100-SXM2-16GB GPUs and took more than 12 hours, it was not feasible for us to retrain. The according configuration on Google Cloud would cost around 10.100$ per month to deploy.


The binary files posed a challenge in understanding and manipulating the data (especially in the transfer context from KITTI to ITTIK), as the dataset was not explained online. Therefore, we arranged a meeting with Olaf Boiij which clarified the datastructure and enabled us to work with ITTIK.




## Sources

Behley, J., Garbade, M., Milioto, A., Quenzel, J., Behnke, S., Stachniss, C., & Gall, J. (2019). Semantickitti: A dataset for semantic scene understanding of lidar sequences. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 9297-9307).

Geiger, A., Lenz, P., Stiller, C., & Urtasun, R. (2013). Vision meets robotics: The kitti dataset. The International Journal of Robotics Research, 32(11), 1231-1237.

Kochanov, D., Nejadasl, F. K., & Booij, O. (2020). Kprnet: Improving projection-based lidar semantic segmentation. arXiv preprint arXiv:2007.12668.

Booij, O. (2022). ittik. https://github.com/olafbooij/ittik
