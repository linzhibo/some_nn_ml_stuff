#!/usr/bin/env sh

TOOLS=/home/zlin/3rd_party_lib/caffe/build/tools

GLOG_logtostderr=1  $TOOLS//caffe train -solver my_solver.prototxt  -gpu 0 

echo "Done."
