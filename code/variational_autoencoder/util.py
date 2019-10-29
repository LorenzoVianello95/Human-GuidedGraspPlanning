import numpy as np
import os
from sklearn.preprocessing import normalize


def load_data(folder_path, train_eval_rate):
    #THIS IS DONE FOR THE ENCODER, I CONSIDER ALL EXAMPLES AS GOOD
    x=[]
    y=[] #0 = bad, 1 = good 

    # load good examples 
    pos_examples_path= folder_path+ "/good"
    for file in os.listdir(pos_examples_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(pos_examples_path, file))
            #print("__________________________________")
            #print(data)
            if data.shape[1] == 100:        ## per qualche motivo ci sono dei file danneggiati...
                x.append(data)
                y.append(1)

    print("shape of the positive examples:")
    
    x= np.array(x)
    y= np.array(y)
    input_shape = x.shape
    labels_shape = y.shape
    print(input_shape)
    print(labels_shape)


    # Randomly shuffle data and labels
    idx = np.random.permutation(input_shape[0])
    x,y = x[idx], y[idx]

    # Divide the dataset in train and evaluation sets
    x_train= x[0:int(input_shape[0]*train_eval_rate)]
    y_train= y[0:int(input_shape[0]*train_eval_rate)]

    x_eval= x[int(input_shape[0]*train_eval_rate):]
    y_eval= y[int(input_shape[0]*train_eval_rate):]

    print("shape train dataset")
    print(x_train.shape)
    print(y_train.shape)
    print("shape evaluation dataset")
    print(x_eval.shape)
    print(y_eval.shape)

    return x_train, y_train, x_eval, y_eval


#x, y, _, _= load_data("/home/lvianell/Desktop/Lorenzo_report/variational_autoencoder/data/GRASPS_IMAGES",0.8)

def manipulate_depth(data):
    mu, sigma = 0, 0.001 
    (number_examples, image_size,_,number_channel)= data.shape
    #print(np.max(data[:,:,:,0]), np.min(data[:,:,:,0]))
    #print(np.max(data[:,:,:,1]), np.min(data[:,:,:,1]))
    #print(np.max(data[:,:,:,2]), np.min(data[:,:,:,2]))
    #print(np.max(data[:,:,:,3]), np.min(data[:,:,:,3]))
    data[data[:,:,:,3]> 0.8]=0.8
    data[data[:,:,:,3]< 0.4]=0.4
    #print(np.max(data[:,:,:,0]), np.min(data[:,:,:,0]))
    #print(np.max(data[:,:,:,1]), np.min(data[:,:,:,1]))
    #print(np.max(data[:,:,:,2]), np.min(data[:,:,:,2]))
    #print(np.max(data[:,:,:,3]), np.min(data[:,:,:,3]))
    data[:,:,:,3] = ((data[:,:,:,3] -0.4)/0.4)
    
    #noise = np.random.normal(mu, sigma, [number_examples, image_size, image_size, number_channel])
    #print("noise shape", noise.shape) 

    #new_data= np.concatenate((data, normalize(data+noise)),axis=0) 
    #print("new data shape", new_data.shape)

    return data

def extend_data(data):
    mu, sigma = 0, 0.3 
    (number_examples, image_size,_,number_channel)= data.shape
    noise = np.random.normal(mu, sigma, [number_examples,])
    #noise[:,:,:,3]/=10

    print(np.mean(data[:,:,:,0]))
    print(np.mean(data[:,:,:,1]))
    print(np.mean(data[:,:,:,2]))
    print(np.mean(data[:,:,:,3]))

    f_data =data

    """#change colors
    transpose_data[0:number_examples//3,:,:,0]= data[0:number_examples//3,:,:,1]
    transpose_data[0:number_examples//3,:,:,1]= data[0:number_examples//3,:,:,2]
    transpose_data[0:number_examples//3,:,:,2]= data[0:number_examples//3,:,:,0]

    transpose_data[number_examples//3:2*number_examples//3,:,:,0]= data[number_examples//3:2*number_examples//3,:,:,2]
    transpose_data[number_examples//3:2*number_examples//3,:,:,1]= data[number_examples//3:2*number_examples//3,:,:,1]
    transpose_data[number_examples//3:2*number_examples//3,:,:,2]= data[number_examples//3:2*number_examples//3,:,:,0]

    transpose_data[2*number_examples//3:,:,:,0]= data[2*number_examples//3:,:,:,2]
    transpose_data[2*number_examples//3:,:,:,1]= data[2*number_examples//3:,:,:,0]
    transpose_data[2*number_examples//3:,:,:,2]= data[2*number_examples//3:,:,:,1]"""

    rotate_data= np.fliplr(f_data)

    transpose_data= f_data.transpose(0,2,1,3)
    ext_data= np.concatenate((rotate_data, transpose_data),axis=0) 
    #transpose_data = np.random.permutation(transpose_data)
    new_data= np.random.permutation(np.concatenate((data, ext_data),axis=0) )

    return new_data

def load_GP_data(folder_path, train_eval_rate):

    x=[]
    y=[] #0 = bad, 1 = good 

    # load good examples 
    pos_examples_path= folder_path+ "/good"
    for file in os.listdir(pos_examples_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(pos_examples_path, file))
            #print("__________________________________")
            #print(data)
            x.append(data)
            y.append(1)

    #print("shape of the positive examples:")
    #print(np.array(x).shape)
    #print(np.array(y).shape)

    #load bad examples
    neg_examples_path= folder_path+ "/bad"
    for file in os.listdir(neg_examples_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(neg_examples_path, file))
            #print("__________________________________")
            #print(data)
            x.append(data)
            y.append(0)

    #print("shape of the positive and negative examples:")
    x= np.array(x)
    y= np.array(y)
    input_shape = x.shape
    labels_shape = y.shape
    #print(input_shape)
    #print(labels_shape)


    # Randomly shuffle data and labels
    idx = np.random.permutation(input_shape[0])
    x,y = x[idx], y[idx]

    # Divide the dataset in train and evaluation sets
    x_train= x[0:int(input_shape[0]*train_eval_rate)]
    y_train= y[0:int(input_shape[0]*train_eval_rate)]

    x_eval= x[int(input_shape[0]*train_eval_rate):]
    y_eval= y[int(input_shape[0]*train_eval_rate):]

    """ print("shape train dataset")
    print(x_train.shape)
    print(y_train.shape)
    print("shape evaluation dataset")
    print(x_eval.shape)
    print(y_eval.shape)"""

    return x_train, y_train, x_eval, y_eval


