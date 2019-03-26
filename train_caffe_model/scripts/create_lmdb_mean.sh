#!/usr/bin/env sh
# Create the imagenet leveldb inputs
# N.B. set the path to the imagenet train + val data dirs

TOOLS=/home/zlin/3rd_party_lib/caffe/build/tools
DATA=/home/zlin/my_code/digit_recognition/model/caffe_training

echo "Creating leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin -resize_width 28 -resize_height 28 -shuffle -gray\
    "" \
    $DATA/train.txt \
    my_train_lmdb

GLOG_logtostderr=1 $TOOLS/convert_imageset.bin -resize_width 28 -resize_height 28 -shuffle -gray\
    "" \
    $DATA/validation.txt \
    my_val_lmdb
    
echo "creating mean..."
sleep 1
$TOOLS/compute_image_mean $DATA/my_train_lmdb \
  $DATA/my_mean.binaryproto

echo "Done."
