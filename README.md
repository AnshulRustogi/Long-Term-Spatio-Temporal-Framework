# Long-term Spatio-temporal Contrastive Learning framework for Skeleton Action Recognition

The repository contains the implementation for Long-term Spatio-temporal Contrastive Learning framework for Skeleton Action Recognition published in IJCNN-2022 [Paper Link to be updated soon](https://www.google.com) 

## Abstract
Recent years have been witnessing significant developments in research in human action recognition based on skeleton data. The graphical representation of the human skeleton, available with the dataset, provides opportunity to apply Graph Convolutional Networks (GCN), to avail efficient analysis of deep spatial-temporal information from the joint and skeleton structure. Most of the current works in skeleton action recognition use the temporal aspect of the video in short-term sequences, ignoring the long-term information present in the evolving skeleton sequence. The proposed long-term Spatio-temporal Contrastive Learning framework for Skeleton Action Recognition uses an encoder-decoder module. The encoder collects deep global-level (long-term) information from the complete action sequence using efficient self-supervision. The proposed encoder combines knowledge from the temporal domain with high-level information of the relative joint and structure movements of the skeleton. The decoder serves two purposes: predicting the human activity and predicting skeleton structure in the future frames. The decoder primarily uses the high-level encodings from the encoder to anticipate the action. For predicting skeleton structure, we extract an even deeper correlation in the Spatio-temporal domain and merge it with the original frame of the video. We apply a contrastive framework in the frame prediction part so that similar actions have similar predicted skeleton structure. The use of the contrastive framework throughout the proposed model helps achieve exemplary performance while employing a self-supervised aspect to the model. We test our model on the NTU-RGB-D-60 dataset and achieve state-of-the-art performance.

## Environment Requirement
- Python 3.9.7
- Pytorch 1.10.0 
- Pyyaml
- Pandas
- Argparse
- Numpy

## Environments

We use the similar input/output interface and system configuration like ST-GCN, where the torchlight module should be set up.
Run
```bash
cd torchlight
python setup.py 
cd ..
```

## Prerequisite

In order to train on our model, you need to preprocess and prepare the data.
The dataset can be downloaded from [Github Link](https://github.com/shahroudy/NTURGB-D) and then be can preprocseed in order to generate the input data as:
```bash
python data_gen/ntu_gen_preprocess.py --data path 'PATH_TO_DATA'
```

## Achknowledgement
Thanks for the framework provided by 'limaosen0/AS-GCN', which is source code of the published work AS-GCN for Skeleton-based Action Recognition in CVPR-2019. The github repo is AS-GCN code. We borrow the framework and interface from the code.

## Citation
**To be added soon**
