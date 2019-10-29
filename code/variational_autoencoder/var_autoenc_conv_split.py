from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Conv2D, UpSampling2D, Reshape, Concatenate
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

from util import load_data, extend_data


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
    """
    z_mean, _, z= encoder.predict(x_test, batch_size=batch_size)
    """
    z= encoder.predict(x_test, batch_size=batch_size)
    z_mean=z
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

    print("Difference in the color:", np.mean(np.abs(x_gt[0,:,:,0:3]- x_result[0,:,:,0:3])))
    print("Difference in the color with mean:", np.mean(np.abs(x_gt[0,:,:,0:3]- x_result_mean[0,:,:,0:3])))

    print("Difference in depth:", np.mean(np.abs(x_gt[0,:,:,3]- x_result[0,:,:,3])))
    print("Difference in depth with mean:", np.mean(np.abs(x_gt[0,:,:,3]- x_result_mean[0,:,:,3])))
    
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
    plt.imshow(x_gt[0,:,:,3], cmap=plt.cm.gray_r)#, vmin=0.2, vmax=0.8)
    
    plt.subplot(num_col, num_row, 5)
    plt.imshow(x_result[0,:,:,3], cmap=plt.cm.gray_r)#, vmin=0.2, vmax=0.8)
    
    plt.subplot(num_col, num_row, 6)
    plt.imshow(x_result_mean[0,:,:,3], cmap=plt.cm.gray_r)#, vmin=0.2, vmax=0.8)


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

x_train= extend_data(x_train)       #MANIPOLO LA DEPTH ELIMINANDO VALORI TROPPO VICINI E TROPPO DISTANTI E NORMALIZZO
x_test= extend_data(x_test)

image_size = x_train.shape[1]
number_channel = x_train.shape[3]

x_train = x_train.reshape(x_train.shape[0], 100, 100, 4).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 100, 100, 4).astype('float32')


# network parameters
input_shape = (image_size,image_size,number_channel)
#intermediate_dim = 512
batch_size = 128
#latent_dim = 256# 128
epochs = 10000

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')

#rgb= inputs[:,:,:,0:3]
rgb = Lambda(lambda x: x[:, :,:,0:3])(inputs)
print(rgb)
depth = Lambda(lambda x: x[:, :,:,3:4])(inputs)
print(depth)

FACTOR= 2

# Encoding RGB network _______________________________________________________
x_RGB = Conv2D(6, (3, 3), activation='relu', padding='same', strides=2)(rgb)
x_RGB = Conv2D(12, (3, 3), activation='relu', padding='same', strides=2)(x_RGB)
z_conv_RGB = Conv2D(16, (2, 2), activation='relu', padding="same", strides=2)(x_RGB)
z_conv_RGB = Conv2D(24, (2, 2), activation='relu', padding="same", strides=2)(z_conv_RGB)

lat_dim_RGB= 7*7*24
z_conv_RGB_reshaped= Reshape((lat_dim_RGB,), input_shape=(7,7,24))(z_conv_RGB)
z_conv_RGB_reshaped= Dense(lat_dim_RGB//FACTOR, name= "dense_rgb")(z_conv_RGB_reshaped)

z_mean_rgb= Dense(lat_dim_RGB//FACTOR, name= "z_mean_rgb")(z_conv_RGB_reshaped)
z_var_rgb= Dense(lat_dim_RGB//FACTOR, name= "z_var_rgb")(z_conv_RGB_reshaped)

z_RGB= Lambda(sampling, output_shape=(lat_dim_RGB//FACTOR,), name='z_rgb')([z_mean_rgb, z_var_rgb])


# Encoding DEPTH network______________________________________________________
x_DEPTH = Conv2D(3, (3, 3), activation='relu', padding='same', strides=2)(depth)
x_DEPTH = Conv2D(6, (3, 3), activation='relu', padding='same', strides=2)(x_DEPTH)
z_conv_DEPTH = Conv2D(16, (2, 2), activation='relu', padding="same", strides=2)(x_DEPTH)
z_conv_DEPTH = Conv2D(24, (2, 2), activation='relu', padding="same", strides=2)(z_conv_DEPTH)

lat_dim_DEPTH= 7*7*24
z_conv_DEPTH_reshaped= Reshape((lat_dim_DEPTH,), input_shape=(7,7,24))(z_conv_DEPTH)
z_conv_DEPTH_reshaped = Dense(lat_dim_DEPTH//FACTOR, name= "dense_depth")(z_conv_DEPTH_reshaped)

z_mean_depth= Dense(lat_dim_DEPTH//FACTOR, name= "z_mean_depth")(z_conv_DEPTH_reshaped)
z_var_depth= Dense(lat_dim_DEPTH//FACTOR, name= "z_var_depth")(z_conv_DEPTH_reshaped)

z_DEPTH=Lambda(sampling, output_shape=(lat_dim_DEPTH//FACTOR,), name='z_depth')([z_mean_depth, z_var_depth])



lat_dim= lat_dim_RGB + lat_dim_DEPTH 
"""
FACTOR= 3
z_conv_RGB_reshaped= Dense(lat_dim_RGB//FACTOR, name= "reshaping_RGB")(z_conv_RGB_reshaped)
z_conv_DEPTH_reshaped= Dense(lat_dim_DEPTH//FACTOR, name= "reshaping_DEPTH")(z_conv_DEPTH_reshaped)

"""
z_mid= Concatenate() ([z_RGB, z_DEPTH])

#z_mid= Dense(lat_dim//2, activation='relu')(z_mid)
"""
z_mean = Dense(lat_dim//FACTOR, name='z_mean')(z_mid)
z_log_var = Dense(lat_dim//FACTOR, name='z_log_var')(z_mid)
z = Lambda(sampling, output_shape=(lat_dim//FACTOR,), name='z')([z_mean, z_log_var])
"""
z = z_mid #Dense(lat_dim//FACTOR, name='z_log_var')(z_mid)

latent_space_dimension= lat_dim_RGB//FACTOR+lat_dim_DEPTH//FACTOR
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
#z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
"""
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
"""
encoder = Model(inputs, z, name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
#latent_inputs = Input(shape=(13,13,16), name='z_sampling')
latent_inputs = Input(shape=(latent_space_dimension,), name='z_sampling')

d= Dense(lat_dim ,activation='relu')(latent_inputs)

rgb = Lambda(lambda x: x[:,0:lat_dim_RGB])(d)
print(rgb)
depth = Lambda(lambda x: x[:,lat_dim_RGB:])(d)
print(depth)

#decoder RGB ____________________________________________________________
r= Reshape((7,7,24), input_shape=(lat_dim_RGB,))(rgb)
r = UpSampling2D((2, 2))(r)

x = Conv2D(16, (2, 2), activation='relu')(r)
x = UpSampling2D((2, 2))(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(6, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
outputs_RGB = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

#decoder DEPTH ____________________________________________________________
r= Reshape((7,7,24), input_shape=(lat_dim_DEPTH,))(depth)
r = UpSampling2D((2, 2))(r)

x = Conv2D(16, (2, 2), activation='relu')(r)
x = UpSampling2D((2, 2))(x)
x = Conv2D(6, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
outputs_DEPTH = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

outputs= Concatenate() ([outputs_RGB, outputs_DEPTH])
"""
r= Reshape((7,7,64), input_shape=(lat_dim_RGB,))(d)
r = UpSampling2D((2, 2))(r)
r=Conv2D(32, (3, 3), activation='sigmoid', padding='same')(r)

x = Conv2D(12, (2, 2), activation='relu')(r)
x = UpSampling2D((2, 2))(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(4, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
outputs = Conv2D(4, (3, 3), activation='sigmoid', padding='same')(x)
"""

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
"""
outputs = decoder(encoder(inputs)[2])
"""
outputs = decoder(encoder(inputs))

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
        #print("jjjjjj")
        if True:
            difference= K.square(inputs-outputs)
            reconstruction_loss_DEPTH= difference[:,:,:,3]*100
            reconstruction_loss_RGB=K.mean(difference[:,:,:,0:3], axis=-1) 

            #reconstruction_loss_DEPTH= K.print_tensor(reconstruction_loss_DEPTH, message=" depth tensor diff  is: ")
            #reconstruction_loss_RGB= K.print_tensor(reconstruction_loss_RGB, message=" rgb tensor diff  is: ")
            
            print(np.shape(reconstruction_loss_DEPTH), np.shape(reconstruction_loss_RGB))
            weight_depth= 0.6
            reconstruction_loss= weight_depth*reconstruction_loss_DEPTH+(1-weight_depth)*reconstruction_loss_RGB
            print(np.shape(reconstruction_loss))
            SCALING_FACTOR= 2*4*100*100
            reconstruction_loss= K.mean(K.mean(reconstruction_loss, axis= -1),axis= -1)

            kl_loss_RGB = 1 + z_var_rgb - K.square(z_mean_rgb) - K.exp(z_var_rgb)
            kl_loss_RGB = K.sum(kl_loss_RGB, axis=-1)
            kl_loss_RGB *= -0.5
            #kl_loss_RGB= K.print_tensor(kl_loss_RGB, message=" rgb kl loss  is: ")

            kl_loss_DEPTH = 1 + z_var_depth - K.square(z_mean_depth) - K.exp(z_var_depth)
            kl_loss_DEPTH = K.sum(kl_loss_DEPTH, axis=-1)
            kl_loss_DEPTH *= -0.5
            #kl_loss_DEPTH= K.print_tensor(kl_loss_DEPTH, message=" depth kl loss  is: ")

            vae_loss=  K.mean(reconstruction_loss + kl_loss_DEPTH/100000 + kl_loss_RGB/100000)
        elif False:
            reconstruction_loss = K.mean(K.mean(mse(inputs, outputs), axis=-1),axis=-1)
            #reconstruction_loss= K.print_tensor(reconstruction_loss, message=" rec loss  is: ")
            
            kl_loss_RGB = 1 + z_var_rgb - K.square(z_mean_rgb) - K.exp(z_var_rgb)
            kl_loss_RGB = K.sum(kl_loss_RGB, axis=-1)
            kl_loss_RGB *= -0.5
            #kl_loss_RGB= K.print_tensor(kl_loss_RGB, message=" rgb kl loss  is: ")

            kl_loss_DEPTH = 1 + z_var_depth - K.square(z_mean_depth) - K.exp(z_var_depth)
            kl_loss_DEPTH = K.sum(kl_loss_DEPTH, axis=-1)
            kl_loss_DEPTH *= -0.5
            #kl_loss_DEPTH= K.print_tensor(kl_loss_DEPTH, message=" depth kl loss  is: ")

            vae_loss = K.mean(reconstruction_loss + kl_loss_RGB + kl_loss_DEPTH)
        else:
            reconstruction_loss_RGB = binary_crossentropy(inputs[:,:,:,0:3],
                                                  outputs[:,:,:,0:3])
            reconstruction_loss_DEPTH = binary_crossentropy(inputs[:,:,:,3:4],
                                                  outputs[:,:,:,3:4])
                                                  
            reconstruction_loss_DEPTH= K.print_tensor(reconstruction_loss_DEPTH, message="depth tensor diff  is: ")
            reconstruction_loss_RGB= K.print_tensor(reconstruction_loss_RGB, message="rgb tensor diff  is: ")
                                      
            weight_depth= 0.6
            reconstruction_loss= weight_depth*reconstruction_loss_DEPTH+(1-weight_depth)*reconstruction_loss_RGB

        #print("s0",np.shape(reconstruction_loss))
        #kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        #kl_loss = K.sum(kl_loss, axis=-1)  
        #kl_loss *= -0.5     
        #kl_loss= K.mean(kl_loss)                           
        #kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        #print("s1",np.shape(kl_loss)) 

        #print(np.shape(inputs[:,:,:,3:4]), inputs[:,:,:,3:4], shape(inputs[:,:,:,3:4]))
        #reconstruction_loss_Depth = binary_crossentropy(inputs[:,:,:,3:4],
                                                  #outputs[:,:,:,3:4])
        #reconstruction_loss=     reconstruction_loss_Depth                                    
        #reconstruction_loss_Depth= K.mean(reconstruction_loss_Depth, axis= -1)
        #sub=np.abs(inputs-outputs)*1       #aggiungo un valore ulteriore loss per la depth 
        #add= K.mean(K.mean(sub[:,:,:,3],-1),-1)
        #print(np.shape(add))

    ##original_dim= 4*100*100
    ##reconstruction_loss= K.mean(K.mean(reconstruction_loss, axis= -1),axis= -1)
    #reconstruction_loss += reconstruction_loss_Depth/4
    #reconstruction_loss+=add
    """
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    print("loss shape :",reconstruction_loss)

    vae_loss = K.mean(reconstruction_loss + kl_loss)/original_dim
    """
    ##vae_loss=  K.mean(reconstruction_loss)
    #print(np.shape(vae_loss))
    #vae_loss+= (add)
    

        
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', metrics=[vae_loss])
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
                callbacks=[mc],
                verbose=1)
        vae.save_weights('vae_mlp_grasps.h5')

        loss_history = history_callback.history["loss"]
        numpy_loss_history = np.array(loss_history)
        np.savetxt("loss_history.txt", numpy_loss_history, delimiter=",")

        loss_history = history_callback.history["val_loss"]
        numpy_loss_history = np.array(loss_history)
        np.savetxt("val_loss_history.txt", numpy_loss_history, delimiter=",")

