

@profile
def clf():
    import numpy as np
    import caffe
    import cv2 
    img_path = '/home/zlin/my_code/digit_recognition/model/validation/right/right_0_82.png'
    im = cv2.imread(img_path,0)
    caffe.set_mode_cpu()
    model = '/home/zlin/my_code/digit_recognition/model/caffe_training/my_deploy.prototxt'
    weights = '/home/zlin/my_code/digit_recognition/model/caffe_training/my_train_iter_100000.caffemodel'
    net = caffe.Net(model, weights, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('/home/zlin/my_code/digit_recognition/model/caffe_training/my_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    # transformer.set_raw_scale('data', 255.0)
    net.blobs['data'].reshape(1,1,28,28)
    im = cv2.resize(im, (28,28))
    im = im.reshape((28,28,1))
    # im = im/256.0
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    out = net.forward()
    label = ['0','1','2','3','4',
             '5','6','7','8','9',
             'left', 'nd', 'right', 'up']
    print out['prob']
    return label[out['prob'].argmax()]
print clf()
