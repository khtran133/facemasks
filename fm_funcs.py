import cv2, imghdr, matplotlib as mpl, matplotlib.image as mpimg, \
       matplotlib.pyplot as plt, numpy as np, os, pandas as pd, \
       PIL, random, shutil, seaborn as sns, sklearn, sys, \
       tensorflow as tf, tensorflow_addons as tfa, time, yaml

from tensorflow import keras
from keras      import layers
with open('6050_modeling.yml') as f:
    cfg = yaml.safe_load(f)

n_pics = sum([len(os.listdir(j)) for j in \
             [os.path.join(cfg['directory'], i) for 
              i in os.listdir(cfg['directory'])]])

AUTOTUNE = tf.data.AUTOTUNE

f1  = tfa.metrics.F1Score(num_classes = 3, threshold = cfg['thresh'], name = 'f1')
acc = keras.metrics.CategoricalAccuracy(name = 'acc')
auc = keras.metrics.AUC(name = 'auc', multi_label = True, num_labels = 3, num_thresholds = 3)
pre = keras.metrics.Precision(name = 'precision', thresholds = cfg['thresh'])
rec = keras.metrics.Recall(name = 'recall', thresholds = cfg['thresh'])

# using validation split to take a small subset of total data
# resizing to 224x224, since it can work with all models
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = cfg['directory'], 
    labels = 'inferred', 
    label_mode = 'categorical', 
    validation_split = (n_pics - cfg['train_size']) / n_pics, 
    subset = 'training', 
    seed = cfg['rand_seed'], 
    image_size = (cfg['img_size'], cfg['img_size']),
    batch_size = cfg['batch_size']
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    directory = cfg['directory'], 
    labels = 'inferred', 
    label_mode = 'categorical', 
    validation_split = cfg['valid_size'] / n_pics, 
    subset = 'validation', 
    seed = cfg['rand_seed'], 
    image_size = (cfg['img_size'], cfg['img_size']),
    batch_size = cfg['batch_size']
)

## prefetch
train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
valid_ds = valid_ds.cache().prefetch(buffer_size = AUTOTUNE)

## input shape
inp_sh = (cfg['img_size'], cfg['img_size'], 3)

def custom_model(model, model_name = '', train_data = train_ds, valid_data = valid_ds, epochs = 50, \
                 optimizer = 'adam', metric_list = [f1, acc, auc, pre, rec], loss = 'categorical_crossentropy'):
    # Freeze all layers
    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[-20:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

    # Add a flattened sequential model to the end of our pre-trained model
    custom_model = keras.Sequential()
    custom_model.add(model)
    custom_model.add(keras.layers.GlobalAveragePooling2D())
    custom_model.add(keras.layers.Dropout(0.25))
    custom_model.add(keras.layers.Flatten())
    custom_model.add(keras.layers.Dense(1024, \
                                        activation = 'relu'))
    custom_model.add(keras.layers.Dense(len(cfg['classes']), \
                                        activation = 'softmax'))

    ## Add a flattened sequential model to the end of our pre-trained model
    with tf.device('/gpu:0'):
        # if 'label_mode' while setting the image datasets is explicitly set as 'categorical', use 'categorical_crossentropy' loss
        # if 'label_mode' while setting the image datasets is unused, use 'sparse_categorical_crossentrophy'
        custom_model.compile(loss      = loss, \
                             optimizer = optimizer, \
                             metrics   = metric_list)
        checkpoint = keras.callbacks.ModelCheckpoint('{}_custom_model'.format(model_name), monitor = 'val_loss', \
                                                     save_best_only = True, mode = 'min')
        es = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min')
        custom_hist = custom_model.fit(train_data, validation_data = valid_data, epochs = epochs, \
                                       shuffle = True, verbose = True, callbacks = [checkpoint, es]) 
        return custom_model, custom_hist

def plot_model_hist(mhist, model_name, figsize = (12, 8)):
    metrics     = list(mhist.history.keys())[:len(mhist.history.keys()) // 2]
    val_metrics = list(mhist.history.keys())[len(mhist.history.keys()) // 2:]
    
    plt.figure(figsize = figsize)
    for i in range(len(metrics)):
        # print(metrics[i] + ', ' + val_metrics[i])
        plt.plot(mhist.history[metrics[i]], 
                 label = 'Training {}'.format(metrics[i].capitalize()))
        plt.plot(mhist.history[val_metrics[i]], 
                 label = 'Validation {}'.format(metrics[i].capitalize()))
    plt.legend(loc = 'lower right')
    plt.title('Training and Validation Metrics, {}'.format(model_name))
    plt.xlabel('Epoch')
    plt.show()

def plot_model_f1(model_hist, model_name, figsize = (12, 8)):
    f1_score = model_hist.history['f1']
    val_f1   = model_hist.history['val_f1']

    loss     = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    plt.figure(figsize = figsize)
    plt.subplot(2, 1, 1)
    plt.plot(f1_score, label = 'Training F1')
    plt.plot(val_f1, label = 'Validation F1')
    plt.legend(loc = 'lower right')
    plt.ylabel('F1')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation F1, {}'.format(model_name))

    plt.subplot(2, 1, 2)
    plt.plot(loss, label = 'Training Loss')
    plt.plot(val_loss, label = 'Validation Loss')
    plt.legend(loc = 'upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
def model_confusion_matrix(model, validation_data = valid_ds):
    conf_matrix = tf.math.confusion_matrix(tf.argmax(np.concatenate([x[1].numpy() for x in validation_data]), axis = 1), 
                                           tf.argmax(np.around(model.predict(validation_data)),               axis = 1), 
                                           num_classes = 3).numpy()
    return conf_matrix

def confusion_matrix_plot(model, model_name, validation_data = valid_ds, figsize = (10, 8)):
    plt.figure(figsize = figsize)
    cm = model_confusion_matrix(model, validation_data)
    ax = sns.heatmap(cm/np.sum(cm), annot = True, cmap = 'Blues')

    ax.set_title('{} Confusion Matrix\n\n'.format(model_name)); ## set a way to change the title based on model
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(cfg['classes'])
    ax.yaxis.set_ticklabels(cfg['classes'])

    ## Display the visualization of the Confusion Matrix.
    plt.show()
    
def test_model_visualize(model, model_name, show_misclassification = False, \
                         validation_data = valid_ds, n_rows = 3, n_cols = 5):
    results = model.evaluate(validation_data)
    try:
        print(results)
    except:
        pass
    
    preds = np.argmax(model.predict(validation_data), axis = 1)
    count = 0
    
    plt.figure(figsize = (n_cols * 3, n_rows * 3))
    for images, labels in validation_data.take(1):
        if show_misclassification:
            for i in range(n_rows * n_cols):
                if np.argmax(model.predict(images), axis = 1)[i] != np.argmax(labels[i]):
                    count += 1
                    plt.subplot(n_rows, n_cols, count)
                    plt.imshow(images[i].numpy().astype('uint8'))
                    plt.axis('off')
                    plt.title('Predicted: {}\nActual: {}'.format(cfg['classes'][preds[i]], \
                                                                 cfg['classes'][np.argmax(labels[i])]), \
                                                                 fontsize = 9)
            plt.suptitle('Face Mask Misclassifications with {}\n'.format(model_name) ,fontsize = 14)
        else:
            for i in range(n_rows * n_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))
                plt.axis('off')
                plt.title('Predicted: {}\nActual: {}'.format(cfg['classes'][preds[i]], \
                                                             cfg['classes'][np.argmax(labels[i])]), \
                                                             fontsize = 9)
            plt.suptitle('Face Mask Classification with {}\n'.format(model_name) ,fontsize = 14)
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.2, hspace = 0.2)
    return results