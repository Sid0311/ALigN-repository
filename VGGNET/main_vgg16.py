########################################################################################
# Quantization of VGG-16 network using ALigN technique                                #
# reference network is taken from below repository                                     #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################


import tensorflow as tf
import gzip
import math
import numpy as np
import collections
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import cv2
import sys
from scipy.spatial import distance
from keras.preprocessing import image
LOGDIR = "/tmp/mnist_tutorial/"
NUM_CHANNELS = 3
IMAGE_SIZE = 224
PIXEL_DEPTH = 255
num_images = 1
num_images_test = 49

#importing quantization functions
from ALigN_repository.Quantization import quant


#for getting classid
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]

    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    #print(("Top1: ", top1, prob[pred[0]]))
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    #print(("Top5: ", top5))
    #return top1
    return top5


def extract_data(data):
  data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
  #print(data)
  #data = data.reshape(IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
  return data

 
#for reading images from dataset
def image_read(i):
    mypath='/media/D/SID/imagenet_val/ILSVRC2012_val_000'+str('{0:05}'.format(i))+'.JPEG'
    #onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ] 
    onlyfiles = mypath
    images2 = np.empty(len(onlyfiles), dtype=object)
    #for n in range(0, 50):
    mypath='/media/D/SID/imagenet_val/ILSVRC2012_val_000'+str('{0:05}'.format(i))+'.JPEG'
    images2= imresize(imread( join(mypath),mode='RGB' ), (224, 224))
    images = images2   
    #for n in range(1, 50):
       #images = np.concatenate((images, images2[n]), 0)
    return images.reshape(1, 224, 224, 3)



#quantization scheme for initial layers
#arguments will come from command line
def quant1(tensor):
    return getattr(quant,sys.argv[1])(tensor)
		
  

#quantization scheme for middle layers
#arguments will come from command line
def quant2(tensor):
	return getattr(quant,sys.argv[2])(tensor)
	

#quantization scheme for farther layers
#arguments will come from command line
def quant3(tensor):
	return getattr(quant,sys.argv[3])(tensor)
    


#vgg-16 network
class vgg16:
    def __init__(self, imgs, conv1_1_er, flag = True, weights=None, sess=None ):
        self.imgs = imgs
        self.conv1_1_er=conv1_1_er
        self.flag=flag
        self.convlayers()
        self.fc_layers() 
        self.probs = tf.nn.softmax(self.fc3l)
        
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            #subtracting the mean from image pixels
            images1 = self.imgs-mean
            images = images1
                        
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.cast(tf.cast(tf.nn.relu(out, name=scope), dtype=tf.float16),dtype=tf.float32)
            print(self.conv1_1)
            self.parameters += [kernel, biases]
	    
        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            #conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            #print(self.flag)
            tmp = tf.cond(self.flag, lambda: True, lambda: False)
            conv1 = tf.cond(tmp, lambda:tf.nn.conv2d(self.conv1_1_er, kernel, [1, 1, 1, 1], padding='SAME'), lambda:tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME'))
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv1), biases)
            #out = tf.nn.bias_add(conv1, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add((conv), biases)
            #out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            #self.fc1 = tf.nn.relu(fc1l)
            self.fc1 = (tf.nn.relu(fc1l))
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            #self.fc2 = tf.nn.relu(fc2l)
            self.fc2 = (tf.nn.relu(fc2l))
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]
    

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        #print(weights)
        keys = sorted(weights.keys())
        #print(sorted(weights.keys()))
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            #weight = np.absolute(weights[k].flatten())
            shape = np.shape(weights[k])
            if(i==0 or i==1):
                sess.run(self.parameters[i].assign(quant1(weights[k])))
            elif(i==2 or i==3):
                sess.run(self.parameters[i].assign(quant2(weights[k])))
            else:
                sess.run(self.parameters[i].assign(quant3(weights[k])))
            

			
#placeholder for image feeding
imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])

init= np.zeros(3211264).reshape(1,224,224,64)
false_flag = False
True_flag = True
total_misclassified = 0
class_num = 1
print(" ")

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        conv1_1_er = tf.placeholder(tf.float32, [1, 224, 224, 64])
        flag = tf.placeholder(tf.bool)
        imgs = tf.placeholder("float", [1, 224, 224, 3])
        vgg = vgg16(imgs, conv1_1_er, flag, 'vgg16_weights.npz', sess)
        with open('val_id_imagenet_new.txt','r') as f:
                for i, line in enumerate(f):
                    images_chunk = image_read(i+1)
                    feed_dict = {imgs: images_chunk, vgg.conv1_1_er: init, vgg.flag: false_flag  }
		    prob = sess.run(vgg.probs, feed_dict=feed_dict)
		            #for top 5
                    if(line.strip() not in (print_prob(prob[0], './synset.txt')[0][0][:9].strip(), print_prob(prob[0], './synset.txt')[1][0][:9].strip(), print_prob(prob[0], './synset.txt')[2][0][:9].strip(), print_prob(prob[0], './synset.txt')[3][0][:9].strip(), print_prob(prob[0], './synset.txt')[4][0][:9].strip())):
                        total_misclassified += 1
                    #for top 1
                    #if(line.strip() not in (print_prob(prob[0], './synset.txt')[0][0][:9].strip())):
                    #    total_misclassified += 1
                    #print(" ")
                    print(total_misclassified) 
                             

