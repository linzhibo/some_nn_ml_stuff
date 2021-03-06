## Generate a loooooooooooot of data with a little set of files and train a model with it

```
cd model
python keras_model.py
python sklearn_model.py
```


* in keras_model.py the **genData()** function allows to generate tons of data from **data/train** and **data/validation** folders, you can put what ever you want in it, make sure to rename properly the folder name. In this example, it trains digital number (unlike the MNIST dataset).
* run **keras_model.py** will generate lot of data and train the model from 0 to 9. and then you can detect digits from the traffic light if your country has it.

![](readme_pics/tl_detect.gif)

*Traffic light detection with Yolov3, then process direction, color and digits with trained model*

![alt text](readme_pics/genData.png "this is image zero")

![alt text](readme_pics/digit_recog_2.png "this is image un")

![alt text](readme_pics/detect_2.png "this is image deux")

# some comparison between keras and sklearn model profile:
![alt text](readme_pics/profile_keras.png "this is image trois")

![alt text](readme_pics/profile_sklearn.png "this is image quatre")

One could find running the keras model is too much **memory consuming**, there are other methods to tune the tf backend, here is an example using scikit-learn.

[hyperParametersTuning.png](readme_pics/hyperParametersTuning.png)

![](readme_pics/lane_boundaries_extract_3.gif)
> deeplab segmentation mask, lane boundaries extraction, ransac to eliminate outliers, cv draw line
