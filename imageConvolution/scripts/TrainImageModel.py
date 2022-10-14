import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from CropImage import CropWithPil
from DatasetStructure import DatasetStructureVO
from sklearn.model_selection import train_test_split


class TrainImage():
    def plot_model(self, history, epochs):

        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.gcf().clear()
        plt.figure(figsize = (15, 8))

        plt.subplot(1, 2, 1)
        plt.title('Training and Validation Accuracy')
        plt.plot(epochs_range, accuracy, label = 'Training Accuracy')
        plt.plot(epochs_range, val_accuracy, label = 'Validation Accuracy')
        plt.legend(loc = 'lower right')

        plt.subplot(1, 2, 2)
        plt.title('Training and Validation Loss')
        plt.plot(epochs_range, loss, label = 'Training Loss')
        plt.plot(epochs_range, val_loss, label = 'Validation Loss')
        plt.legend(loc = 'lower right')

        plt.savefig('C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\lixo\\_1_gr_.png')
        #plt.show()

    def plot_dataset_predictions(self, dataset_test, model, d_stru: DatasetStructureVO):
        class_names=d_stru.class_names
        features, labels = dataset_test.as_numpy_iterator().next()

        predictions = model.predict_on_batch(features).flatten()
        predictions = tf.where(predictions < d_stru.limit, 0, 1)

        print('Labels:      %s' % labels)
        print('Predictions: %s' % predictions.numpy())

        plt.gcf().clear()
        plt.figure(figsize = (15, 15))

        for i in range(9):

            plt.subplot(3, 3, i + 1)
            plt.axis('off')

            plt.imshow(features[i].astype('uint8'))
            plt.title(class_names[predictions[i]])
            plt.savefig('C:\\ACMS\\Estudos\\NOTEBOOK\\imageConvolution\\images\\lixo\\_1_am_.png')

    def model_definition(self, d_stru):
        data = []
        label = []
        label_val = 0

        for folder in d_stru.class_names:
            cpath = os.path.join(d_stru.dataset_train, folder)
            for img in os.listdir(cpath):
                image_array = cv2.imread(os.path.join(cpath, img), cv2.IMREAD_COLOR)
                data.append(image_array)
                label.append(label_val)
            label_val = 1

        data = np.asarray(data)
        label = np.asarray(label)
        X_train, X_test, y_train, y_test = train_test_split(data, label, train_size=0.8, random_state=42)

        #train_data=data[0:int(len(data)*0.8)]
        #train_label=label[0:int(len(label)*0.8)]
        #test_data=data[int(len(data)*0.8):len(data)]
        #test_label=label[int(len(label)*0.8):len(label)]
        #X_train, X_tv, y_train, y_tv = train_test_split(train_data, train_label, train_size=0.8, random_state=42)
        #X_test, X_valid, y_test, y_valid=train_test_split(test_data,test_label, train_size=.5, randon_state=20)

        class_num = len(y_test.shape)

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv2D(32, 3, input_shape=(32, 32, 3), activation='relu', padding='same'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(2))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
            
        model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.BatchNormalization())

        model.add(tf.keras.layers.Dense(class_num, activation='softmax'))

        #model.compile(loss='categorical_crossentropy',\
        #    optimizer='adam',\
        #        metrics=['accuracy', 'binary_crossentropy'],\
        #            learning_rate = d_stru.learning_rate)

        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = d_stru.learning_rate),\
                loss = tf.keras.losses.BinaryCrossentropy(),\
                    metrics = ['accuracy', 'binary_crossentropy'])

        print(model.summary())
        seed = 21
        np.random.seed(seed)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=128)

        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

        pd.DataFrame(history.history).plot()
        plt.show()

        return model

    def executar(self, d_stru: DatasetStructureVO):
        
        d_stru.dataset_dir = d_stru.imagens
        
        d_stru.dataset_train_dir = d_stru.dataset_train
        d_stru.dataset_validation_dir = d_stru.dataset_validation

        d_stru.dataset_train_target = d_stru.train_target_dir
        d_stru.dataset_train_target_len = len(os.listdir(d_stru.dataset_train_target))
        d_stru.dataset_validation_target = d_stru.validation_target_dir
        d_stru.dataset_validation_target_len = len(os.listdir(d_stru.dataset_validation_target))

        d_stru.dataset_train_others = d_stru.train_others_dir
        d_stru.dataset_trains_others_len = len(os.listdir(d_stru.dataset_train_others))
        d_stru.dataset_validation_others = d_stru.validation_others_dir
        d_stru.dataset_validation_others_len = len(os.listdir(d_stru.dataset_validation_others))

        print('Train target: %s' % d_stru.dataset_train_target)
        print('Train target length: %s' % d_stru.dataset_train_target_len)
        print('Train Others: %s' % d_stru.dataset_train_others)
        print('Train Others length: %s' % d_stru.dataset_trains_others_len)
        
        print('Validation target: %s' % d_stru.dataset_validation_target)
        print('Validation target length: %s' % d_stru.dataset_validation_target_len)
        print('Validation Others: %s' % d_stru.dataset_validation_others)
        print('Validation Others length: %s' % d_stru.dataset_validation_others_len)

        d_stru.image_width = 160
        d_stru.image_height = 160
        d_stru.image_color_channel = 3
        d_stru.image_color_channel_size = 255
        d_stru.image_size = (d_stru.image_width, d_stru.image_height)
        d_stru.image_shape = d_stru.image_size + (d_stru.image_color_channel,)
        d_stru.batch_size = 32
        d_stru.epochs = 8
        #d_stru.learning_rate = 0.0001
        d_stru.class_names = [d_stru.target_class, d_stru.others_class]
        
        dataset_train = tf.keras.preprocessing.image_dataset_from_directory(
            d_stru.dataset_train_dir,
            image_size = d_stru.image_size,
            batch_size = d_stru.batch_size,
            shuffle = True
        )

        dataset_validation = tf.keras.preprocessing.image_dataset_from_directory(
            d_stru.dataset_validation_dir,
            image_size = (d_stru.image_width, d_stru.image_height),
            batch_size = d_stru.batch_size,
            shuffle = True
        )
        
        dataset_validation_cardinality = tf.data.experimental.cardinality(dataset_validation)
        dataset_validation_batches = dataset_validation_cardinality // 5

        dataset_test = dataset_validation.take(dataset_validation_batches)
        dataset_validation = dataset_validation.skip(dataset_validation_batches)

        print('Validation Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_validation))
        print('Test Dataset Cardinality: %d' % tf.data.experimental.cardinality(dataset_test))

        autotune = tf.data.AUTOTUNE

        dataset_train = dataset_train.prefetch(buffer_size = autotune)
        dataset_validation = dataset_validation.prefetch(buffer_size = autotune)
        dataset_test = dataset_validation.prefetch(buffer_size = autotune)
        rescaling = tf.keras.layers.experimental.preprocessing.\
            Rescaling(1. / (d_stru.image_color_channel_size / 2.), offset = -1, input_shape = d_stru.image_shape)
        model_transfer_learning = tf.keras.applications.MobileNetV2(\
            input_shape = d_stru.image_shape, include_top = False, weights = 'imagenet')
        model_transfer_learning.trainable = False
        model_transfer_learning.summary()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'loss', patience = 3)

        model = tf.keras.models.Sequential([
            rescaling,
            #data_augmentation,
            model_transfer_learning,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation = 'sigmoid')
        ])
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate = d_stru.learning_rate),
            loss = tf.keras.losses.BinaryCrossentropy(),
            metrics = ['accuracy']
        )
        model.summary()
        history = model.fit(
            dataset_train,
            validation_data = dataset_validation,
            epochs = d_stru.epochs,
            callbacks = [
                early_stopping
            ]
        )

        self.plot_model(history=history, epochs=d_stru.epochs)

        dataset_test_loss, dataset_test_accuracy = model.evaluate(dataset_test)

        print('Dataset Test Loss:     %s' % dataset_test_loss)
        print('Dataset Test Accuracy: %s' % dataset_test_accuracy)
        
        self.plot_dataset_predictions(dataset_test=dataset_test,model=model, d_stru=d_stru)
        ## OUTRO MODO
        #model=self.model_definition(d_stru=d_stru)
        ## OUTRO MODO
        model_name=os.path.join(d_stru.modelos,d_stru.target_class)
        model.save(model_name)

        return model

