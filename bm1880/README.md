# Caffe to bm1880 workflow
this folder contains files involving caffe to bm1880 workflow

[official link](https://sophon-edge.gitbook.io/project/toolkit/bmnet-compiler)

1. prepare caffe model, raw data and deploy prototxt (in the data layer, it should be python data or lmdb with Phase:test )
2. run the calibration tool
3. run the compiler
4. test the bmodel.

during inference, the data should be correctly converted:
1. if there is a mean file, make sure to substract it to all channel.
2. make sure the data is ranged between [-127, 127]
3. gray image can be pass through directly, colored image should be transposed

succesfully tested on:
* traffic light digit and direction arrow
* road texture
* traffic sign