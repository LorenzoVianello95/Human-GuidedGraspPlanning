from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, Reshape
from keras.models import Model
#from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from keras.callbacks import ModelCheckpoint
from keras.backend import shape

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2

from util import load_data


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_grasps"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    #os.makedirs(model_name, exist_ok=True)

    #filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    print(x_test.shape)
    z_mean, _, z= encoder.predict(x_test, batch_size=batch_size)
    print(z_mean.shape)
    x_decoded = decoder.predict(z, batch_size=batch_size)
    x_decoded_mean = decoder.predict(z_mean, batch_size=batch_size)
    print(x_decoded.shape)

    x_gt= np.reshape(x_test, [-1, 100,100,4])
    x_result= np.reshape(x_decoded, [-1, 100,100,4] )
    x_result_mean= np.reshape(x_decoded_mean, [-1, 100,100,4] )

    print(x_gt.shape, x_result.shape)

    fig = plt.figure()
    num_row=3
    num_col=2

    
    plt.subplot(num_col, num_row, 1)
    plt.title("GT ")
    plt.imshow(cv2.cvtColor(x_gt[0,:,:,0:3], cv2.COLOR_BGR2RGB))
    
    plt.subplot(num_col, num_row, 2)
    plt.title("Decoder result")
    plt.imshow(cv2.cvtColor(x_result[0,:,:,0:3], cv2.COLOR_BGR2RGB))

    plt.subplot(num_col, num_row, 3)
    plt.title("Decoder + mean")
    plt.imshow(cv2.cvtColor(x_result_mean[0,:,:,0:3], cv2.COLOR_BGR2RGB))

    plt.subplot(num_col, num_row, 4)
    plt.imshow(x_gt[0,:,:,3], cmap=plt.cm.gray_r, vmin=0.5, vmax=0.8)
    
    plt.subplot(num_col, num_row, 5)
    plt.imshow(x_result[0,:,:,3], cmap=plt.cm.gray_r, vmin=0.5, vmax=0.8)
    
    plt.subplot(num_col, num_row, 6)
    plt.imshow(x_result_mean[0,:,:,3], cmap=plt.cm.gray_r, vmin=0.5, vmax=0.8)


    plt.show()
    #cv2.imshow('image',x_gt[0,:,:,0:3])
    #cv2.waitKey(0)
    #cv2.imshow('image',x_result[0,:,:,0:3])
    #cv2.waitKey(0)
    
    
    """ plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show() 

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()"""

dataset= "/home/lvianell/Desktop/Lorenzo_report/variational_autoencoder/data/GRASPS_IMAGES"

# MNIST dataset
x_train, y_train, x_test, y_test = load_data(dataset,0.8)

image_size = x_train.shape[1]
number_channel = x_train.shape[3]

x_train = x_train.reshape(x_train.shape[0], 100, 100, 4).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 100, 100, 4).astype('float32')


# network parameters
input_shape = (image_size,image_size,number_channel)
#intermediate_dim = 512
batch_size = 128
#latent_dim = 256# 128
epochs = 500

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')

# Encoding network
x = Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(inputs)
x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x)
z_conv = Conv2D(16, (2, 2), activation='relu', padding="same", strides=2)(x)

lat_dim= 13*13*16
z_mid= Reshape((lat_dim,), input_shape=(13,13,16))(z_conv)
z_mid= Dense(lat_dim, activation='relu')(z_mid)
z_mean = Dense(lat_dim//4, name='z_mean')(z_mid)
z_log_var = Dense(lat_dim//4, name='z_log_var')(z_mid)
z = Lambda(sampling, output_shape=(lat_dim//4,), name='z')([z_mean, z_log_var])

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
#z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
#latent_inputs = Input(shape=(13,13,16), name='z_sampling')
latent_inputs = Input(shape=(lat_dim//4,), name='z_sampling')

d= Dense(lat_dim ,activation='relu')(latent_inputs)
r= Reshape((13,13,16), input_shape=(lat_dim,))(d)

x = Conv2D(16, (2, 2), activation='relu', padding="same")(r)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
outputs = Conv2D(4, (3, 3), activation='sigmoid', padding='same')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m",
                        "--mse",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    
    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        print("jjjjjj")
        reconstruction_loss = binary_crossentropy(inputs,
                                                  outputs)
        print("s0",np.shape(reconstruction_loss))
        #kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        #kl_loss = K.sum(kl_loss, axis=-1)  
        #kl_loss *= -0.5     
        #kl_loss= K.mean(kl_loss)                           
        #kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        #print("s1",np.shape(kl_loss)) 

        print(np.shape(inputs[:,:,:,3]), inputs[:,:,:,3], shape(inputs[:,:,:,3]))
        reconstruction_loss_Depth = binary_crossentropy(inputs[:,:,:,3],
                                                  outputs[:,:,:,3])
                                                
        reconstruction_loss_Depth= K.mean(reconstruction_loss_Depth, axis= -1)
        #sub=np.abs(inputs-outputs)*1       #aggiungo un valore ulteriore loss per la depth 
        #add= K.mean(K.mean(sub[:,:,:,3],-1),-1)
        #print(np.shape(add))

    original_dim= 4*100*100
    reconstruction_loss= K.mean(K.mean(reconstruction_loss, axis= -1),axis= -1)
    #reconstruction_loss += reconstruction_loss_Depth/4
    #reconstruction_loss+=add

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    print("loss shape :",reconstruction_loss)

    vae_loss = K.mean(reconstruction_loss + kl_loss)/original_dim
    print(np.shape(vae_loss))
    #vae_loss+= (add)
    
    
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)
    

    if args.weights:
        vae.load_weights(args.weights)
        plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")
    else:
        mc = ModelCheckpoint('logs/weights{epoch:08d}.h5', 
                                     save_weights_only=True, period=5)

        # train the autoencoder
        history_callback =vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None),
                callbacks=[mc])
        vae.save_weights('vae_mlp_grasps.h5')

        loss_history = history_callback.history["loss"]
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

        loss_history = history_callback.history["val_loss"]
        numpy_loss_history = np.array(loss_history)
        np.savetxt("val_loss_history.txt", numpy_loss_history, delimiter=",")

