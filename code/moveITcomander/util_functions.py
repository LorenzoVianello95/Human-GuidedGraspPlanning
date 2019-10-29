import numpy as np
import os


def load_data(folder_path, train_eval_rate):

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

    print("shape of the positive examples:")
    print(np.array(x).shape)
    print(np.array(y).shape)

    #load bad examples
    neg_examples_path= folder_path+ "/bad"
    for file in os.listdir(neg_examples_path):
        if file.endswith(".npy"):
            data = np.load(os.path.join(neg_examples_path, file))
            #print("__________________________________")
            #print(data)
            x.append(data)
            y.append(0)

    print("shape of the positive and negative examples:")
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


#_, _, _, _= load_data("/home/lvianell/Desktop/Lorenzo_report/datasets/grasp_figures")