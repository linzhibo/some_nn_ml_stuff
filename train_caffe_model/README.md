# Train CaffeNet model on custom dataset
Compile/Install caffe
1. run **train_val.py** to generate train.txt and test.txt (set the right path in the script)
    
    Train folder should be organised like Train/cars, Train/bikes etc..
```
+-- train.txt
|   +-- Train/bike/bike.jpg 0
|   +-- Train/car/car.jpg 1
+-- test.txt
|   +-- Test/bike/bike.jpg 0
|   +-- Test/car/car.jpg 1
```
2. run **create_leveldb.sh** to create lmdb data for caffe (set the train and test file correctly)
3. run **start_train.sh**

>Note: Change the *num_output* in *fully-connected layer* for custom number of classes, in both train and val prototxt
