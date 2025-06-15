
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 08:30:31 2025

@author: itama
"""

#Data Manipulation Modules

#Data Vizualization

#Deep & Machine Learning Modules

#Final model

import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow_hub as hub #is responsible for importing the pretrained models
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator# data augmentation addition- augmentation
from tensorflow.keras.callbacks import EarlyStopping# early stopping addition- augmentation
from collections import Counter
import collections
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB3
from sklearn.utils.class_weight import compute_class_weight# help me decide on the wieghts of the calsses when trying to prevent the overfitting that was caused by the uneven amount of data



def run_model():
    # Setup
    image_size_x = 224
    image_size_y = 224
    batch_size = 32
    sequence_length = 12
    seed = 42
    channels = 3
    epochs = 60
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.0000055)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.000055, momentum=0.9, nesterov= True)
    
    
    """
    #for augmentation

    
    #Used to make augmeneted train data to avoide underfitting
    
    os.makedirs(augmented_train_dir, exist_ok=True)
    os.makedirs(os.path.join(augmented_train_dir, 'down'), exist_ok=True)
    os.makedirs(os.path.join(augmented_train_dir, 'up'), exist_ok=True)
        
    
    datagen = ImageDataGenerator(
        rotation_range= 25,
        channel_shift_range=10,
        width_shift_range =0.05,
        height_shift_range= 0.05,
        shear_range= 0.05,
        zoom_range= 0.05,
        horizontal_flip= False,
        brightness_range =[0.7, 1.1]
        )
    
    
    aug_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(image_size_x, image_size_y),
        batch_size=1,
        class_mode='categorical',
        #save_to_dir=augmented_train_dir,
        save_prefix='aug',
        save_format='png'
        )
    
    
    class_indices = aug_generator.class_indices
    # Reverse the dictionary to map index to class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    num_augmented_images_to_generate = 5000
    print(f"Generating and saving {num_augmented_images_to_generate} augmented images...")
    
    i = 0
    original_image_i = 0
    num_original_images = len(aug_generator.filepaths)
    
    for batch_data in aug_generator:
        images_batch= batch_data[0]
        labels_batch= batch_data[1]
        
        img= images_batch[0]
        label_one_hot = labels_batch[0]
        
        #checks class index and class name
        class_i= np.argmax(label_one_hot)
        class_name= idx_to_class[class_i]
        
        
        original_filepath = aug_generator.filepaths[original_image_i % num_original_images]
        original_filename = os.path.basename(original_filepath).split('.')[0] # Get base name without extension
        
        #makes the file path to the right label
        save_path =os.path.join(augmented_train_dir, class_name)
        
        #Saves the images and ensures vcorrect formatting
        image_to_save= tf.keras.preprocessing.image.array_to_img(img)
        
        #Gives the files names
        augmented_name= f'aug_{original_filename}_{i}.{aug_generator.save_format}'
        full_path= os.path.join(save_path, augmented_name)
        
        image_to_save.save(full_path, format=aug_generator.save_format)
        
        i += 1
        original_image_i += 1
        
        if i >= num_augmented_images_to_generate:
            break
        
        print("Augmentation complete.")
    """
    
    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float16)  # Convert to float16
        return image, label
    
    def augment(image, label):
        image= tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image , 0.9, 1.1)
        image= tf.image.random_saturation(image, 0.9 , 1.1)
        return image, label
    
    # Data loading
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).map(augment).unbatch()  
    
    train_raw_augmented= tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train_aug",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    #combining and shuffling the train dirs for feeding the model and avoiding bias
    train_raw_combined= train_raw.concatenate(train_raw_augmented)
    train_raw = train_raw_combined.shuffle(buffer_size=1000, seed=seed)
    
    def split_images_labels(dataset):
        images = dataset.map(lambda x, y: x)
        labels = dataset.map(lambda x, y: y)
        return images, labels

    
    train_images, train_labels= split_images_labels(train_raw)
    val_images, val_labels= split_images_labels(val_raw)
    test_images, test_labels = split_images_labels(test_raw)
    
    
    """  
        #helped me check whether the data was even for both classes
        down_files = os.listdir(train_dir_label_down)
        up_files = os.listdir(train_dir_label_up)
        
        down_count = len(down_files)
        up_count = len(up_files)
        
        print(f"Number of files in 'down' directory: {down_count}")
        print(f"Number of files in 'up' directory: {up_count}")
        
        if down_count > up_count:
            print("The 'down' directory has more files.")
            elif up_count > down_count:
                print("The 'up' directory has more files.")
                else:
                    print("The 'down' and 'up' directories have the same number of files.")
                    
                    label_list = list(train_labels.as_numpy_iterator())
                    print(Counter(label_list))
                    
                    
                    
                    #helped me calculate the weights for the class weights trick, only needed to do once as I understood
                    # Extract all labels from the dataset
                    label_list = []
                    
                    for _, label in train_raw:
                        label_list.append(int(label.numpy()))
                        
                        label_array = np.array(label_list)
                        
                        # class weight decision
                        class_weights = compute_class_weight(
                            class_weight='balanced',
                            classes=np.unique(label_array),
                            y=label_array
                            )
                        
                        class_weight_dict = dict(enumerate(class_weights))
                        print("Class Weights:", class_weight_dict)
        
                        
                        
                        print(f"Checking image counts in augmented_train_dir:")
                        
                        augmented_down_dir = os.path.join(augmented_train_dir, 'down')
                        augmented_up_dir = os.path.join(augmented_train_dir, 'up')
                        
                        #Ensure directories already exist in the folder
                        if not os.path.exists(augmented_down_dir):
                            print(f"Warning: Directory '{augmented_down_dir}' not found.")
                            augmented_down_count = 0
                            else:
                                augmented_down_files = os.listdir(augmented_down_dir)
                                augmented_down_count = len(augmented_down_files)
                                
                                if not os.path.exists(augmented_up_dir):
                                    print(f"Warning: Directory '{augmented_up_dir}' not found.")
                                    augmented_up_count = 0
                                    else:
                                        augmented_up_files = os.listdir(augmented_up_dir)
                                        augmented_up_count = len(augmented_up_files)
                                        
                                        print(f"  Number of images in '{os.path.basename(augmented_down_dir)}' (augmented): {augmented_down_count}")
                                        print(f"  Number of images in '{os.path.basename(augmented_up_dir)}' (augmented): {augmented_up_count}")
                                        
                                        #cehcks if even
                                        if augmented_down_count > 0 and augmented_down_count == augmented_up_count:
                                            print("  The augmented 'down' and 'up' directories have an even number of images.")
                                            elif augmented_down_count > augmented_up_count:
                                                print("  The augmented 'down' directory has more images.")
                                                elif augmented_up_count > augmented_down_count:
                                                    print("  The augmented 'up' directory has more images.")
                                                    else:
                                                        print("  Counts for augmented directories are not available or are zero.")
                                                        
                                                        
            
        #Checks validation directory
        print(f"\nChecking image counts in validation_dir:")
        
        val_down_dir = os.path.join(val_dir, 'down')
        val_up_dir = os.path.join(val_dir, 'up')
        
        if not os.path.exists(val_down_dir):
            print(f"Warning: Directory '{val_down_dir}' not found.")
            val_down_count = 0
            else:
                val_down_count = len(os.listdir(val_down_dir))
                
                if not os.path.exists(val_up_dir):
                    print(f"Warning: Directory '{val_up_dir}' not found.")
                    val_up_count = 0
                    else:
                        val_up_count = len(os.listdir(val_up_dir))
                        
                        print(f"  Number of images in '{os.path.basename(val_down_dir)}': {val_down_count}")
                        print(f"  Number of images in '{os.path.basename(val_up_dir)}': {val_up_count}")
                        
                        if val_down_count > 0 and val_down_count == val_up_count:
                            print("  The validation 'down' and 'up' directories have an even number of images.")
                            elif val_down_count > val_up_count:
                                print("  The validation 'down' directory has more images.")
                                elif val_up_count > val_down_count:
                                    print("  The validation 'up' directory has more images.")
                                    else:
                                        print("  Counts for validation directories are not available or are zero.")
                                        
                                        #Checks test directory
                                        print(f"\nChecking image counts in test_dir:")
                                        
                                        test_down_dir = os.path.join(test_dir, 'down')
                                        test_up_dir = os.path.join(test_dir, 'up')
                                        
                                        if not os.path.exists(test_down_dir):
                                            print(f"Warning: Directory '{test_down_dir}' not found.")
                                            test_down_count = 0
                                            else:
                                                test_down_count = len(os.listdir(test_down_dir))
                                                
                                                if not os.path.exists(test_up_dir):
                                                    print(f"Warning: Directory '{test_up_dir}' not found.")
                                                    test_up_count = 0
                                                    else:
                                                        test_up_count = len(os.listdir(test_up_dir))
                                                        
                                                        print(f"  Number of images in '{os.path.basename(test_down_dir)}': {test_down_count}")
                                                        print(f"  Number of images in '{os.path.basename(test_up_dir)}': {test_up_count}")
                                                        
                                                        if test_down_count > 0 and test_down_count == test_up_count:
                                                            print("  The test 'down' and 'up' directories have an even number of images.")
                                                            elif test_down_count > test_up_count:
                                                                print("  The test 'down' directory has more images.")
                                                                elif test_up_count > test_down_count:
                                                                    print("  The test 'up' directory has more images.")
                                                                    else:
                                                                        print("  Counts for test directories are not available or are zero.")                                               
                                                                        
                                          
    """
    
    
    class_weight_dict = {0: 1.0, 1: 1.0}

    def create_sequences_tf(images, labels, sequence_length=8):
        image_sequences = images.window(size=sequence_length, shift=1, drop_remainder=True)
        image_sequences = image_sequences.flat_map(lambda window: window.batch(sequence_length))
        
        label_sequences = labels.skip(sequence_length - 1)  # So labels match the last frame of each sequence
        
        dataset = tf.data.Dataset.zip((image_sequences, label_sequences))
        return dataset
    
    
    train_seq = create_sequences_tf(train_images, train_labels, sequence_length)
    val_seq = create_sequences_tf(val_images, val_labels, sequence_length)
    test_seq = create_sequences_tf(test_images, test_labels, sequence_length)


    """
def one_hot_labels(image_seq, label):
    label = tf.one_hot(label, depth=2)
    return image_seq, label
"""


    """
    train_seq = train_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_seq = val_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_seq = test_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    """
    train_seq = train_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_seq = val_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_seq = test_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    #early stopping define addition- augmentation
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    mobilenet_layer = hub.KerasLayer(mobilenet_url, trainable=False)
    
    mobilenet_feature_dim = 1280
    
    def apply_mobilenet_per_frame(x):
        # x: [batch, time, h, w, c]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        features = mobilenet_layer(x_reshaped)  # shape: [batch*time, feature_dim]
        features = tf.reshape(features, (batch_size, time_steps, mobilenet_feature_dim))
        return features
    
    
    """
    resnet50_base = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size_x, image_size_y, channels))
    new_feature_dim = resnet50_base.output_shape[-1]
    print(f"New feature dimension: {new_feature_dim}")
    
    
    def apply_resnet50_per_frame(x):
        # x: [batch, time, h, w, c]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        # Reshape to feed individual frames to the CNN
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        # Apply the pretrained model
        features = resnet50_base(x_reshaped)
        # Global average pooling or flatten to get a flat feature vector per frame
        features = tf.keras.layers.GlobalAveragePooling2D()(features) # or tf.keras.layers.Flatten()(features) depending on preference and model output
        # Reshape back to [batch, time, feature_dim]
        features = tf.reshape(features, (batch_size, time_steps, new_feature_dim))
        return features

    """
    """
    efficientnet_base = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size_x, image_size_y, channels))
    new_feature_dim = efficientnet_base.output_shape[-1]
    print(f"New feature dimension (EfficientNetB5): {new_feature_dim}")
    
    
    def apply_efficientnet_per_frame(x):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        features = efficientnet_base(x_reshaped)
        features = tf.keras.layers.GlobalAveragePooling2D()(features)
        features = tf.reshape(features, (batch_size, time_steps, new_feature_dim))
        return features
    
    """
    """
    efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1" # Using feature vector
    efficientnet_layer = hub.KerasLayer(efficientnet_url, trainable=False) # Keep it frozen for now
    
    # Get the output dimension of the EfficientNetB5 layer
    # You might need to run a small test to get the exact output shape.
    # A typical way is to pass a dummy tensor through the layer.
    dummy_input = tf.random.uniform((1, image_size_x, image_size_y, channels))
    dummy_output = efficientnet_layer(dummy_input)
    efficientnet_feature_dim = dummy_output.shape[-1]
    print(f"EfficientNetB5 feature dimension: {efficientnet_feature_dim}")
    
    
    def apply_efficientnet_per_frame(x):
        # x: [batch, time, h, w, c]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        features = efficientnet_layer(x_reshaped)  # shape: [batch*time, feature_dim]
        features = tf.reshape(features, (batch_size, time_steps, efficientnet_feature_dim))
        return features
    
    """
    
    def display_image_sequence_batch(image_sequence_batch, labels_batch, batch_index=0):
        """
        Displays one image sequence from a batch.
        """
        if batch_index >= image_sequence_batch.shape[0]:
            print(f"Batch index {batch_index} out of bounds for batch size {image_sequence_batch.shape[0]}")
            return
        
        sequence_to_display = image_sequence_batch[batch_index]
        label_to_display = labels_batch[batch_index]
        
        print(f"Displaying sequence from batch index {batch_index}, Label: {label_to_display.numpy()}")
        
        num_frames = sequence_to_display.shape[0]
        fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 2, 2)) # Adjust figure size as needed
        
        for i in range(num_frames):
            image_to_plot = tf.image.convert_image_dtype(sequence_to_display[i], tf.float32)
            axes[i].imshow(image_to_plot.numpy())
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()

    
    for image_seq_batch, label_batch in train_seq.take(1):
        # Call the display function with the first batch
        display_image_sequence_batch(image_seq_batch, label_batch, batch_index=0)
        
        

    #CNN & LSTM model with pretrained model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),
        
        layers.Lambda(
            apply_mobilenet_per_frame,
            output_shape=(sequence_length, mobilenet_feature_dim)
            ),
        
        layers.LSTM(
            256,
            return_sequences=True, 
            activation="relu"
            ),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True,activation="relu")),
        layers.Bidirectional(layers.LSTM(64)),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        ])
    
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),#
        metrics=['accuracy'],
        run_eagerly=False # eager execution is required for some reson when I ran the LSTM model option
        )


    print(model.summary())
    
    # Train model
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=early_stopping,
        class_weight=class_weight_dict
        )

    # Evaluate
    test_loss, test_accuracy= model.evaluate(test_seq)
    print("Test acuracy:", test_accuracy)
    
    
    y_true=[]
    y_pred=[]
    for x_batch , y_batch in test_seq:
        preds=model.predict( x_batch)
        preds_binary =( preds > 0.5).astype(int) 
        y_true.extend(y_batch.numpy().astype(int))
        y_pred.extend(preds_binary.flatten())
        
        #classification Report
        print("Classification Report:")
        print(classification_report(y_true, y_pred, digits=4))
        
        #confusion matrix
        m= confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=m, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        

    #Plot results
    plt.figure(figsize=(12, 6))
    
    #Accuracy Plot
    plt.subplot(1, 2, 1) #1st plot
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    
    #Loss plot
    plt.subplot(1, 2, 2) #2nd plot
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    
    plt.tight_layout() #preventing overstepping
    plt.show()
    
    
    model.save("c:/users/itama/bitcoin prediction/gui/model.keras")


def run_model_gui():
    # Setup
    image_size_x = 224
    image_size_y = 224
    batch_size = 32
    sequence_length = 12
    seed = 42
    channels = 3
    epochs = 60
    optimizer= tf.keras.optimizers.Adam(learning_rate=0.0000055)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=0.000055, momentum=0.9, nesterov= True)
    
    
    
    #for augmentation

    """
    #Used to make augmeneted train data to avoide underfitting
    
    os.makedirs(augmented_train_dir, exist_ok=True)
    os.makedirs(os.path.join(augmented_train_dir, 'down'), exist_ok=True)
    os.makedirs(os.path.join(augmented_train_dir, 'up'), exist_ok=True)
        
    
    datagen = ImageDataGenerator(
        rotation_range= 25,
        channel_shift_range=10,
        width_shift_range =0.05,
        height_shift_range= 0.05,
        shear_range= 0.05,
        zoom_range= 0.05,
        horizontal_flip= False,
        brightness_range =[0.7, 1.1]
        )
    
    
    aug_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(image_size_x, image_size_y),
        batch_size=1,
        class_mode='categorical',
        #save_to_dir=augmented_train_dir,
        save_prefix='aug',
        save_format='png'
        )
    
    
    class_indices = aug_generator.class_indices
    # Reverse the dictionary to map index to class name
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    num_augmented_images_to_generate = 5000
    print(f"Generating and saving {num_augmented_images_to_generate} augmented images...")
    
    i = 0
    original_image_i = 0
    num_original_images = len(aug_generator.filepaths)
    
    for batch_data in aug_generator:
        images_batch= batch_data[0]
        labels_batch= batch_data[1]
        
        img= images_batch[0]
        label_one_hot = labels_batch[0]
        
        #checks class index and class name
        class_i= np.argmax(label_one_hot)
        class_name= idx_to_class[class_i]
        
        
        original_filepath = aug_generator.filepaths[original_image_i % num_original_images]
        original_filename = os.path.basename(original_filepath).split('.')[0] # Get base name without extension
        
        #makes the file path to the right label
        save_path =os.path.join(augmented_train_dir, class_name)
        
        #Saves the images and ensures vcorrect formatting
        image_to_save= tf.keras.preprocessing.image.array_to_img(img)
        
        #Gives the files names
        augmented_name= f'aug_{original_filename}_{i}.{aug_generator.save_format}'
        full_path= os.path.join(save_path, augmented_name)
        
        image_to_save.save(full_path, format=aug_generator.save_format)
        
        i += 1
        original_image_i += 1
        
        if i >= num_augmented_images_to_generate:
            break
        
        print("Augmentation complete.")
    """
    
    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float16)  # Convert to float16
        return image, label
    
    def augment(image, label):
        image= tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image , 0.9, 1.1)
        image= tf.image.random_saturation(image, 0.9 , 1.1)
        return image, label
    
    # Data loading
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).map(augment).unbatch()  
    
    train_raw_augmented= tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train_aug",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    #combining and shuffling the train dirs for feeding the model and avoiding bias
    train_raw_combined= train_raw.concatenate(train_raw_augmented)
    train_raw = train_raw_combined.shuffle(buffer_size=1000, seed=seed)
    
    def split_images_labels(dataset):
        images = dataset.map(lambda x, y: x)
        labels = dataset.map(lambda x, y: y)
        return images, labels

    
    train_images, train_labels= split_images_labels(train_raw)
    val_images, val_labels= split_images_labels(val_raw)
    test_images, test_labels = split_images_labels(test_raw)
    
    
    """  
        #helped me check whether the data was even for both classes
        down_files = os.listdir(train_dir_label_down)
        up_files = os.listdir(train_dir_label_up)
        
        down_count = len(down_files)
        up_count = len(up_files)
        
        print(f"Number of files in 'down' directory: {down_count}")
        print(f"Number of files in 'up' directory: {up_count}")
        
        if down_count > up_count:
            print("The 'down' directory has more files.")
            elif up_count > down_count:
                print("The 'up' directory has more files.")
                else:
                    print("The 'down' and 'up' directories have the same number of files.")
                    
                    label_list = list(train_labels.as_numpy_iterator())
                    print(Counter(label_list))
                    
                    
                    
                    #helped me calculate the weights for the class weights trick, only needed to do once as I understood
                    # Extract all labels from the dataset
                    label_list = []
                    
                    for _, label in train_raw:
                        label_list.append(int(label.numpy()))
                        
                        label_array = np.array(label_list)
                        
                        # class weight decision
                        class_weights = compute_class_weight(
                            class_weight='balanced',
                            classes=np.unique(label_array),
                            y=label_array
                            )
                        
                        class_weight_dict = dict(enumerate(class_weights))
                        print("Class Weights:", class_weight_dict)
        
                        
                        
                        print(f"Checking image counts in augmented_train_dir:")
                        
                        augmented_down_dir = os.path.join(augmented_train_dir, 'down')
                        augmented_up_dir = os.path.join(augmented_train_dir, 'up')
                        
                        #Ensure directories already exist in the folder
                        if not os.path.exists(augmented_down_dir):
                            print(f"Warning: Directory '{augmented_down_dir}' not found.")
                            augmented_down_count = 0
                            else:
                                augmented_down_files = os.listdir(augmented_down_dir)
                                augmented_down_count = len(augmented_down_files)
                                
                                if not os.path.exists(augmented_up_dir):
                                    print(f"Warning: Directory '{augmented_up_dir}' not found.")
                                    augmented_up_count = 0
                                    else:
                                        augmented_up_files = os.listdir(augmented_up_dir)
                                        augmented_up_count = len(augmented_up_files)
                                        
                                        print(f"  Number of images in '{os.path.basename(augmented_down_dir)}' (augmented): {augmented_down_count}")
                                        print(f"  Number of images in '{os.path.basename(augmented_up_dir)}' (augmented): {augmented_up_count}")
                                        
                                        #cehcks if even
                                        if augmented_down_count > 0 and augmented_down_count == augmented_up_count:
                                            print("  The augmented 'down' and 'up' directories have an even number of images.")
                                            elif augmented_down_count > augmented_up_count:
                                                print("  The augmented 'down' directory has more images.")
                                                elif augmented_up_count > augmented_down_count:
                                                    print("  The augmented 'up' directory has more images.")
                                                    else:
                                                        print("  Counts for augmented directories are not available or are zero.")
                                                        
                                                        
            
        #Checks validation directory
        print(f"\nChecking image counts in validation_dir:")
        
        val_down_dir = os.path.join(val_dir, 'down')
        val_up_dir = os.path.join(val_dir, 'up')
        
        if not os.path.exists(val_down_dir):
            print(f"Warning: Directory '{val_down_dir}' not found.")
            val_down_count = 0
            else:
                val_down_count = len(os.listdir(val_down_dir))
                
                if not os.path.exists(val_up_dir):
                    print(f"Warning: Directory '{val_up_dir}' not found.")
                    val_up_count = 0
                    else:
                        val_up_count = len(os.listdir(val_up_dir))
                        
                        print(f"  Number of images in '{os.path.basename(val_down_dir)}': {val_down_count}")
                        print(f"  Number of images in '{os.path.basename(val_up_dir)}': {val_up_count}")
                        
                        if val_down_count > 0 and val_down_count == val_up_count:
                            print("  The validation 'down' and 'up' directories have an even number of images.")
                            elif val_down_count > val_up_count:
                                print("  The validation 'down' directory has more images.")
                                elif val_up_count > val_down_count:
                                    print("  The validation 'up' directory has more images.")
                                    else:
                                        print("  Counts for validation directories are not available or are zero.")
                                        
                                        #Checks test directory
                                        print(f"\nChecking image counts in test_dir:")
                                        
                                        test_down_dir = os.path.join(test_dir, 'down')
                                        test_up_dir = os.path.join(test_dir, 'up')
                                        
                                        if not os.path.exists(test_down_dir):
                                            print(f"Warning: Directory '{test_down_dir}' not found.")
                                            test_down_count = 0
                                            else:
                                                test_down_count = len(os.listdir(test_down_dir))
                                                
                                                if not os.path.exists(test_up_dir):
                                                    print(f"Warning: Directory '{test_up_dir}' not found.")
                                                    test_up_count = 0
                                                    else:
                                                        test_up_count = len(os.listdir(test_up_dir))
                                                        
                                                        print(f"  Number of images in '{os.path.basename(test_down_dir)}': {test_down_count}")
                                                        print(f"  Number of images in '{os.path.basename(test_up_dir)}': {test_up_count}")
                                                        
                                                        if test_down_count > 0 and test_down_count == test_up_count:
                                                            print("  The test 'down' and 'up' directories have an even number of images.")
                                                            elif test_down_count > test_up_count:
                                                                print("  The test 'down' directory has more images.")
                                                                elif test_up_count > test_down_count:
                                                                    print("  The test 'up' directory has more images.")
                                                                    else:
                                                                        print("  Counts for test directories are not available or are zero.")                                               
                                                                        
                                          
    """
    
    
    class_weight_dict = {0: 1.0, 1: 1.0}

    def create_sequences_tf(images, labels, sequence_length=8):
        image_sequences = images.window(size=sequence_length, shift=1, drop_remainder=True)
        image_sequences = image_sequences.flat_map(lambda window: window.batch(sequence_length))
        
        label_sequences = labels.skip(sequence_length - 1)  # So labels match the last frame of each sequence
        
        dataset = tf.data.Dataset.zip((image_sequences, label_sequences))
        return dataset
    
    
    train_seq = create_sequences_tf(train_images, train_labels, sequence_length)
    val_seq = create_sequences_tf(val_images, val_labels, sequence_length)
    test_seq = create_sequences_tf(test_images, test_labels, sequence_length)


    """
def one_hot_labels(image_seq, label):
    label = tf.one_hot(label, depth=2)
    return image_seq, label
"""


    """
    train_seq = train_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_seq = val_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_seq = test_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    """
    train_seq = train_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_seq = val_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_seq = test_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    #early stopping define addition- augmentation
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    mobilenet_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    mobilenet_layer = hub.KerasLayer(mobilenet_url, trainable=False)
    
    mobilenet_feature_dim = 1280
    
    def apply_mobilenet_per_frame(x):
        # x: [batch, time, h, w, c]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        features = mobilenet_layer(x_reshaped)  # shape: [batch*time, feature_dim]
        features = tf.reshape(features, (batch_size, time_steps, mobilenet_feature_dim))
        return features
    
    
    """
    resnet50_base = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size_x, image_size_y, channels))
    new_feature_dim = resnet50_base.output_shape[-1]
    print(f"New feature dimension: {new_feature_dim}")
    
    
    def apply_resnet50_per_frame(x):
        # x: [batch, time, h, w, c]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        # Reshape to feed individual frames to the CNN
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        # Apply the pretrained model
        features = resnet50_base(x_reshaped)
        # Global average pooling or flatten to get a flat feature vector per frame
        features = tf.keras.layers.GlobalAveragePooling2D()(features) # or tf.keras.layers.Flatten()(features) depending on preference and model output
        # Reshape back to [batch, time, feature_dim]
        features = tf.reshape(features, (batch_size, time_steps, new_feature_dim))
        return features

    """
    """
    efficientnet_base = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(image_size_x, image_size_y, channels))
    new_feature_dim = efficientnet_base.output_shape[-1]
    print(f"New feature dimension (EfficientNetB5): {new_feature_dim}")
    
    
    def apply_efficientnet_per_frame(x):
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        features = efficientnet_base(x_reshaped)
        features = tf.keras.layers.GlobalAveragePooling2D()(features)
        features = tf.reshape(features, (batch_size, time_steps, new_feature_dim))
        return features
    
    """
    """
    efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b5/feature-vector/1" # Using feature vector
    efficientnet_layer = hub.KerasLayer(efficientnet_url, trainable=False) # Keep it frozen for now
    
    # Get the output dimension of the EfficientNetB5 layer
    # You might need to run a small test to get the exact output shape.
    # A typical way is to pass a dummy tensor through the layer.
    dummy_input = tf.random.uniform((1, image_size_x, image_size_y, channels))
    dummy_output = efficientnet_layer(dummy_input)
    efficientnet_feature_dim = dummy_output.shape[-1]
    print(f"EfficientNetB5 feature dimension: {efficientnet_feature_dim}")
    
    
    def apply_efficientnet_per_frame(x):
        # x: [batch, time, h, w, c]
        batch_size = tf.shape(x)[0]
        time_steps = tf.shape(x)[1]
        x_reshaped = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
        features = efficientnet_layer(x_reshaped)  # shape: [batch*time, feature_dim]
        features = tf.reshape(features, (batch_size, time_steps, efficientnet_feature_dim))
        return features
    
    """
    
    def display_image_sequence_batch(image_sequence_batch, labels_batch, batch_index=0):
        """
        Displays one image sequence from a batch.
        """
        if batch_index >= image_sequence_batch.shape[0]:
            print(f"Batch index {batch_index} out of bounds for batch size {image_sequence_batch.shape[0]}")
            return
        
        sequence_to_display = image_sequence_batch[batch_index]
        label_to_display = labels_batch[batch_index]
        
        print(f"Displaying sequence from batch index {batch_index}, Label: {label_to_display.numpy()}")
        
        num_frames = sequence_to_display.shape[0]
        fig, axes = plt.subplots(1, num_frames, figsize=(num_frames * 2, 2)) # Adjust figure size as needed
        
        for i in range(num_frames):
            # Convert to float32 or uint8 for matplotlib display
            # Using float32 as it's a common type and preserves range if not scaled to [0, 1]
            # If images are scaled to [0, 1], uint8 is also a good option after scaling by 255
            image_to_plot = tf.image.convert_image_dtype(sequence_to_display[i], tf.float32)
            axes[i].imshow(image_to_plot.numpy())
            axes[i].set_title(f"Frame {i+1}")
            axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()

    # Get one batch from the training sequence dataset
    # To do this, you need to iterate through the dataset
    for image_seq_batch, label_batch in train_seq.take(1):
        # Call the display function with the first batch
        display_image_sequence_batch(image_seq_batch, label_batch, batch_index=0)
        
        

    #CNN & LSTM model with pretrained model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),
        
        layers.Lambda(
            apply_mobilenet_per_frame,
            output_shape=(sequence_length, mobilenet_feature_dim)
            ),
        
        layers.LSTM(
            256,
            return_sequences=True, 
            activation="relu"
            ),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True,activation="relu")),
        layers.Bidirectional(layers.LSTM(64)),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
        ])
    
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),#
        metrics=['accuracy'],
        run_eagerly=False # eager execution is required for some reson when I ran the LSTM model option
        )


    print(model.summary())
    
    # Train model
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=early_stopping,
        class_weight=class_weight_dict
        )

    # Evaluate
    test_loss, test_accuracy= model.evaluate(test_seq)
    
    return test_accuracy, test_loss




"""
import tensorflow as tf
from tensorflow.keras import models, layers 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow_hub as hub #is responsible for importing the pretrained models
import numpy as np
from tensorflow.keras.utils import to_categorical










#CNN +RNN(GRU)- supposed to run faster then LSTM
def run_model():
    # Setup
    image_size_x = 224
    image_size_y = 224
    batch_size = 8
    sequence_length = 8
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    def preprocess(image, label):
       image = tf.image.convert_image_dtype(image, tf.float16)  # Convert to float16
       return image, label
   
    # Loading and unbatching
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
       "candleStick_dataset_input/train",
       image_size=(224, 224),
       batch_size=batch_size,
       seed=seed
       ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()  # UNBATCH here
   
    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
       "candleStick_dataset_input/validation",
       image_size=(224, 224),
       batch_size=batch_size,
       seed=seed
       ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()
   
    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
       "candleStick_dataset_input/test",
       image_size=(224, 224),
       batch_size=batch_size,
       seed=seed
       ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()
    
    
    def create_sequences_tf(images, labels, sequence_length=8):
        image_sequences = images.window(size=sequence_length, shift=1, drop_remainder=True)
        image_sequences = image_sequences.flat_map(lambda window: window.batch(sequence_length))

        label_sequences = labels.skip(sequence_length - 1)  # So labels match the last frame of each sequence

        dataset = tf.data.Dataset.zip((image_sequences, label_sequences))
        return dataset
    
    
    def split_images_labels(dataset):
        images = dataset.map(lambda x, y: x)
        labels = dataset.map(lambda x, y: y)
        return images, labels
    
    train_images, train_labels = split_images_labels(train_raw)
    val_images, val_labels = split_images_labels(val_raw)
    test_images, test_labels = split_images_labels(test_raw)

    train_seq = create_sequences_tf(train_images, train_labels, sequence_length)
    val_seq = create_sequences_tf(val_images, val_labels, sequence_length)
    test_seq = create_sequences_tf(test_images, test_labels, sequence_length)
    
    
    def one_hot_labels(image_seq, label):
        label = tf.one_hot(label, depth=2)
        return image_seq, label

    train_seq = train_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_seq = val_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_seq = test_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    
    
    # Build CNN & GRU model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),
        
        # CNN feature extractor for each frame
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        
        # LSTM for temporal dependencies
        layers.GRU(128, return_sequences=True),
        layers.GRU(64, return_sequences=False),
        
        # Fully connected
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="sigmoid")  # Binary classification
        ])
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
        )

    print(model.summary())


    # Train model
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs
        )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_seq)
    print("Test Accuracy:", test_accuracy)
    
    # Plot results
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

"""
"""
# CNN + RNN(LSTM)
def run_model():
    # Setup
    image_size_x = 224
    image_size_y = 224
    batch_size = 8
    sequence_length = 8
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    def preprocess(image, label):
       image = tf.image.convert_image_dtype(image, tf.float16)  # Convert to float16
       return image, label
   
    # Loading and unbatching
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
       "candleStick_dataset_input/train",
       image_size=(224, 224),
       batch_size=batch_size,
       seed=seed
       ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()  # UNBATCH here
   
    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
       "candleStick_dataset_input/validation",
       image_size=(224, 224),
       batch_size=batch_size,
       seed=seed
       ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()
   
    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
       "candleStick_dataset_input/test",
       image_size=(224, 224),
       batch_size=batch_size,
       seed=seed
       ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()
    
    
    def create_sequences_tf(images, labels, sequence_length=8):
        image_sequences = images.window(size=sequence_length, shift=1, drop_remainder=True)
        image_sequences = image_sequences.flat_map(lambda window: window.batch(sequence_length))

        label_sequences = labels.skip(sequence_length - 1)  # So labels match the last frame of each sequence

        dataset = tf.data.Dataset.zip((image_sequences, label_sequences))
        return dataset
    
    
    def split_images_labels(dataset):
        images = dataset.map(lambda x, y: x)
        labels = dataset.map(lambda x, y: y)
        return images, labels
    
    train_images, train_labels = split_images_labels(train_raw)
    val_images, val_labels = split_images_labels(val_raw)
    test_images, test_labels = split_images_labels(test_raw)

    train_seq = create_sequences_tf(train_images, train_labels, sequence_length)
    val_seq = create_sequences_tf(val_images, val_labels, sequence_length)
    test_seq = create_sequences_tf(test_images, test_labels, sequence_length)
    
    
    def one_hot_labels(image_seq, label):
        label = tf.one_hot(label, depth=2)
        return image_seq, label

    train_seq = train_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_seq = val_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_seq = test_seq.map(one_hot_labels).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    
    
    # Build CNN & LSTM model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),
        
        # CNN feature extractor for each frame
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        
        # LSTM for temporal dependencies
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64, return_sequences=False),
        
        # Fully connected
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="sigmoid")  # Binary classification
        ])
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
        )

    print(model.summary())


    # Train model
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs
        )
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_seq)
    print("Test Accuracy:", test_accuracy)
    
    # Plot results
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    
""" 
"""
    # Setup
    image_size_x = 224
    image_size_y = 224
    batch_size = 8
    sequence_length = 8
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    # Load dataset and preprocess
    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float16)  # Convert to float16
        return image, label

    # Loading and unbatching
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
    ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()  # UNBATCH here

    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
    ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
    ).map(lambda x, y: (x/255, y)).map(preprocess).unbatch()

    # Sequence creation function
    def create_sequences_tf(dataset, sequence_length=8):
        ds = dataset.window(size=sequence_length + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda window: window.batch(sequence_length + 1))
        ds = ds.map(lambda window: (window[:-1], window[-1]))  # (sequence of images, label of last image)
        return ds

    # Create sequences
    train_seq = create_sequences_tf(train_raw, sequence_length)
    val_seq = create_sequences_tf(val_raw, sequence_length)
    test_seq = create_sequences_tf(test_raw, sequence_length)

    # One-hot encode labels
    def one_hot_labels(image_seq, label):
        label = tf.one_hot(label, depth=2)
        return image_seq, label

    train_seq = train_seq.map(one_hot_labels)
    val_seq = val_seq.map(one_hot_labels)
    test_seq = test_seq.map(one_hot_labels)

    # Final batching
    train_seq = train_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_seq = val_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_seq = test_seq.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build CNN + LSTM Model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),

        # CNN feature extractor per time step
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Dropout(0.2)),

        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Dropout(0.2)),

        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation="relu")),
        layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        layers.TimeDistributed(layers.Dropout(0.2)),

        layers.TimeDistributed(layers.GlobalAveragePooling2D()),

        # LSTM layers
        layers.LSTM(128, return_sequences=True),
        layers.LSTM(64, return_sequences=False),

        # Fully Connected layers
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(2, activation="sigmoid")
    ])

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.summary()

    # Train
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs
    )

    # Evaluate
    test_loss, test_accuracy = model.evaluate(test_seq)
    print("Test Accuracy:", test_accuracy)

    # Plot results
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


"""
"""
#CNN+LSTM
def run_model():
    
  
    #Error: End of sequence
    image_size_x = 224
    image_size_y = 224
    batch_size = 8
    sequence_length = 8  # Number of images per sequence
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)# changed to adam and lowered the learning rate 

    #Load dataset without nympy array conversion to reduce memory usage

    def preprocess(image, label):
        image = tf.image.convert_image_dtype(image, tf.float16)  # Convert to float16
        return image, label

    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x,y: (x/255,y)).map(preprocess)

    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x,y: (x/255,y)).map(preprocess)# divided by 255

    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(224, 224),
        batch_size=batch_size,
        seed=seed
        ).map(lambda x,y: (x/255,y)).map(preprocess)

    #Function to convert data to sequences because of the LTSM model, changed it beacuse there was aproblem with the dataset shape
    def create_sequences(dataset, sequence_length=4):
        images, labels = [], []
    
        all_images = []
        all_labels = []
    
        for batch_images, batch_labels in dataset:
            all_images.extend(batch_images.numpy())  # Convert tensors to NumPy
            all_labels.extend(batch_labels.numpy())

        all_images = np.array(all_images, dtype='float16')
        all_labels = np.array(all_labels)

        for i in range(len(all_images) - sequence_length-1):
            images.append(all_images[i:i + sequence_length])  # Sequence of N images
            labels.append(all_labels[i + sequence_length])  # Label of last image in sequence

        return tf.convert_to_tensor(images, dtype='float16'), tf.convert_to_tensor(labels, dtype=tf.int32)  # Ensure correct dtype
    
    # Convert to sequences
    x_train, y_train = create_sequences(train_raw, sequence_length)
    x_val, y_val = create_sequences(val_raw, sequence_length)
    x_test, y_test = create_sequences(test_raw, sequence_length)
    
    #Convert labels to one-hot encoding to prevent shape problems, also uses tf functions to prevent excess memorty usage
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    
    print("y_train sample:", y_train[:10])  # See the first 10 labels
    print("y_train shape:", y_train.shape)  # Check the dimensions
    print("y_train dtype:", y_train.dtype)  # Check the data type
    print("Unique labels in y_train:", np.unique(y_train))  # See the possible class labels

    unique, counts = np.unique(np.argmax(y_train, axis=1), return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples")

    
    # Build CNN & LSTM model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),

        # CNN feature exxtractor per time sequence
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),#duplicated layer
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),#dupliacted layer
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),#duplicated layer
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),

        # Flatten feature maps per time step
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),

        # LSTM for Temporal Processing
        layers.LSTM(128, return_sequences=True),  # more processing units 
        layers.LSTM(64, return_sequences=False),

        # Fully Connected Layers
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),#reduced dropouts to prevenet exceesive neuron deactivation
        layers.Dense(2, activation="sigmoid")#sigmoid is better for binary classification than softmax that was there before
    ])

    # Compiling
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    print(model.summary())

    # Training
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val,y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    # Evaluation
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print("Test Accuracy:", test_accuracy)

    # Ploting the training results
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

"""
"""
    #ValueError: Exception encountered when calling Sequential.call().    Invalid input shape for input Tensor("data:0", shape=(3,), dtype=float32). Expected shape (None, 4, 224, 224, 3), but input has incompatible shape (3,)
    #Test Accuracy: 0.49760764837265015
    # Variables
    image_size_x = 224
    image_size_y = 224
    batch_size = 3
    sequence_length = 4  # Number of images per sequence
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    # Load raw dataset
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed
    )
    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed
    )
    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed
    )
    

    # Function to convert data to sequences because of the LTSM model
    def create_sequences(dataset, sequence_length=5):
        images, labels = [], []

        for batch in dataset:
            batch_images, batch_labels = batch
            for i in range(len(batch_images) - sequence_length):
                images.append(batch_images[i:i + sequence_length])  # Take N consecutive images
                labels.append(batch_labels[i + sequence_length])    # Label of last image in sequence

        return np.array(images, dtype='float16'), np.array(labels, dtype='float16')#added label dtype to reserve memory

    # Convert to sequences
    X_train, y_train = create_sequences(train_raw, sequence_length)
    X_val, y_val = create_sequences(val_raw, sequence_length)
    X_test, y_test = create_sequences(test_raw, sequence_length)
    
    print("y_train sample:", y_train[:10])  # See the first 10 labels
    print("y_train shape:", y_train.shape)  # Check the dimensions
    print("y_train dtype:", y_train.dtype)  # Check the data type
    print("Unique labels in y_train:", np.unique(y_train))  # See the possible class labels

    
    
    # Optimize memory usage by converting data type to float16
    X_train = np.array(X_train, dtype='float16')
    X_val = np.array(X_val, dtype='float16')
    X_test = np.array(X_test, dtype='float16')

    # Convert labels to one-hot encoding to prevent shape issues
    y_train = to_categorical(y_train, num_classes=2)
    y_val = to_categorical(y_val, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    
    
    
    # Build CNN & LSTM model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),

        # CNN feature exxtractor per time sequence
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),#duplicated layer
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),#dupliacted layer
        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),
        
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),#duplicated layer
        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="relu")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.2)),

        # Flatten feature maps per time step
        layers.TimeDistributed(layers.Flatten()),

        # LSTM for Temporal Processing
        layers.LSTM(128, return_sequences=True),  # more processing units 
        layers.LSTM(64, return_sequences=False),

        # Fully Connected Layers
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),#reduced dropouts to prevenet exceesive neuron deactivation
        layers.Dense(2, activation="sigmoid")#sigmoid is better for binary classification than softmax that was there before
    ])

    # Compiling
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    print(model.summary())

    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    # Evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

    # Ploting the training results
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

"""  
"""
    #has preprocessing problems with the ViT model

    # Variables
    image_size = 224
    batch_size = 8  # Reduced for better memory management
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Load Dataset (Binary classification: "up" and "down")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="binary"
        )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="binary"
        )
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        seed=seed,
        label_mode="binary"
        )
    
    # Load ViT model (feature extractor)
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    vit_model = hub.KerasLayer(vit_url, trainable=False)
    
    # Define input layer
    inputs = layers.Input(shape=(image_size, image_size, channels))
    
    # Normalize the image according to the ViT's model preprocessing requirements
    x = layers.Rescaling(1./127.5, offset=-1)(inputs)
    
    # Pass through ViT model (fixing the input issue)
    x = vit_model(x)  # ViT outputs a feature vector
    
    # Apply global average pooling to handle tensor shape properly
    x = layers.GlobalAveragePooling1D()(x)  
    
    # Add custom classification head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Binary classification
    
    # Build model using Functional API
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=['accuracy']
        )
    
    # Print model summary
    print(model.summary())
    
    # Improve training speed with prefetching
    Autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(Autotune)
    val_dataset = val_dataset.cache().prefetch(Autotune)
    test_dataset = test_dataset.cache().prefetch(Autotune)
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
        )
    
    # Evaluate on test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
"""
  
"""
    #ViT model- missing preprocessing for ViT
    # Variables
    image_size_x = 224
    image_size_y = 224
    batch_size = 8  # Reduced for better memory management
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # Load Dataset (Binary classification: "up" and "down")
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed,
        label_mode="binary"  # Binary classification (0 or 1)
        )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed,
        label_mode="binary"
        )
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed,
        label_mode="binary"
        )
    
    # Load ViT feature extractor (correct URL)
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    vit_model = hub.KerasLayer(vit_url, trainable=False)
    
    # Define input layer
    inputs = layers.Input(shape=(image_size_x, image_size_y, channels))
    
    # Normalize the image
    x = layers.Rescaling(1./255)(inputs)
    
    # Pass through ViT feature extractor
    x = vit_model(x)
    
    # Add custom classification head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # Binary classification
    
    # Build model using Functional API
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),  # Binary classification loss
        metrics=['accuracy']
        )
    
    # Print model summary
    print(model.summary())
    
    # Improve training speed with prefetching
    Autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(Autotune)
    val_dataset = val_dataset.cache().prefetch(Autotune)
    test_dataset = test_dataset.cache().prefetch(Autotune)
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
        )
    
    # Evaluate on test dataset
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    
    """
""" 
    #ViT Model- 3. normal neural network structure did not work
 
    #Variables
    image_size_x =224
    image_size_y = 224
    batch_size = 8 #changed from 32 for segmenting the data to smaller chuncks so my computer could handle it
    seed = 42
    channels= 3
    epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


    #Load Dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size = (image_size_x,image_size_y),
        batch_size = batch_size,
        seed = seed,
        label_mode="categorical"
        )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size = ( image_size_x,image_size_y),
        batch_size = batch_size,
        seed = seed,
        label_mode="categorical"
        )
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size = ( image_size_x,image_size_y),
        batch_size = batch_size,
        seed = seed,
        label_mode="categorical"
        )
    
    vit_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
    base_model = hub.KerasLayer(vit_url, trainable=False)
    
    #Neural Network-ViT
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(image_size_x, image_size_y, channels)),  # Normalization of pixel values
        base_model,  # Vision Transformer
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")  # Output layer for 3 classes
        ])
    

    #Compiling
    print(model.summary())
    
    model.compile(
        optimizer= optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),  # Adjusted for categorical labels
        metrics = ['accuracy']
        )
    
    #Improving Train Speed
    Autotune = tf.data.AUTOTUNE
    
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(Autotune)
    val_dataset = val_dataset.cache().prefetch(Autotune)
    test_dataset = test_dataset.cache().prefetch(Autotune)
    
    
    #model training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
        )
    
    #Evaluation
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.4f}")  # Formats to 4 decimal places
    
    #plotting
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
        
        
        
        
     """   
"""
    # Variables
    image_size_x = 224
    image_size_y = 224
    batch_size = 5
    sequence_length = 4  # Number of images per sequence
    seed = 42
    channels = 3
    epochs = 10
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

    # Load raw dataset
    train_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/train",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed
    )
    val_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/validation",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed
    )
    test_raw = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_dataset_input/test",
        image_size=(image_size_x, image_size_y),
        batch_size=batch_size,
        seed=seed
    )
    

    # Function to convert data to sequences because of the LTSM model
    def create_sequences(dataset, sequence_length=5):
        images, labels = [], []

        for batch in dataset:
            batch_images, batch_labels = batch
            for i in range(len(batch_images) - sequence_length):
                images.append(batch_images[i:i + sequence_length])  # Take N consecutive images
                labels.append(batch_labels[i + sequence_length])    # Label of last image in sequence

        return np.array(images, dtype='float16'), np.array(labels)

    # Convert to sequences
    X_train, y_train = create_sequences(train_raw, sequence_length)
    X_val, y_val = create_sequences(val_raw, sequence_length)
    X_test, y_test = create_sequences(test_raw, sequence_length)
    
    print("y_train sample:", y_train[:10])  # See the first 10 labels
    print("y_train shape:", y_train.shape)  # Check the dimensions
    print("y_train dtype:", y_train.dtype)  # Check the data type
    print("Unique labels in y_train:", np.unique(y_train))  # See the possible class labels

    
    
    # Optimize memory usage by converting data type to float16
    X_train = np.array(X_train, dtype='float16')
    X_val = np.array(X_val, dtype='float16')
    X_test = np.array(X_test, dtype='float16')


    # Build CNN & LSTM model
    model = models.Sequential([
        layers.Input(shape=(sequence_length, image_size_x, image_size_y, channels)),

        # CNN Feature Extractor per time step(timedistributed means it considers the sequences and not only asingle image)
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation="softmax")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.3)),

        layers.TimeDistributed(layers.Conv2D(64, (3,3), activation="softmax")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.3)),

        layers.TimeDistributed(layers.Conv2D(128, (3,3), activation="softmax")),
        layers.TimeDistributed(layers.BatchNormalization()),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Dropout(0.3)),

        # Flatten feature maps per time step
        layers.TimeDistributed(layers.Flatten()),

        # LSTM for Temporal Processing
        layers.LSTM(64, return_sequences=False),

        # Fully Connected Layers
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(3, activation="softmax")
    ])

    # Compiling
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    print(model.summary())

    # Training
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs
    )

    # Evaluation
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

    # Ploting the training results
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

"""
"""
#ViT Model
 
#Variables
image_size_x =432
image_size_y = 316
batch_size = 8 #changed from 32 for segmenting the data to smaller chuncks so my computer could handle it
seed = 42
channels= 3
epochs = 10
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


#Load Dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/train",
    image_size = (image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
    )
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/validation",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/test",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)

vit_url = "https://tfhub.dev/google/vit-base-patch16-224/1"
base_model = hub.KerasLayer(vit_url, trainable=False)

#Neural Network-ViT
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(image_size_x, image_size_y, channels)),  # Normalization of pixel values
    base_model,  # Vision Transformer
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(3, activation="softmax")  # Output layer for 3 classes
])


#Compiling
model.build()
print(model.summary())

model.compile(
    optimizer= optimizer,
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
    )

#Improving Train Speed
Autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(Autotune)
val_dataset = val_dataset.cache().prefetch(Autotune)
test_dataset = test_dataset.cache().prefetch(Autotune)


#model training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

#Evaluation
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy: "+ test_accuracy)

#plotting
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""
"""
3.changed the neural network structure
#Variables
image_size_x =432
image_size_y = 316
batch_size = 8 #changed from 32 for segmenting the data to smaller chuncks so my computer could handle it
seed = 42
channels= 3
epochs = 10
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)


#Load Dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/train",
    image_size = (image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
    )
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/validation",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/test",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)

#Neural Network
model = models.Sequential([
    layers.Input(shape=(image_size_x,image_size_y,channels)),
    #chatgpt recommendation
    # First Conv Block
    layers.Conv2D(32, (3,3), activation="relu"),
    layers.BatchNormalization(),  # Normalize feature maps
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),  # Prevents overfitting

    # Second Conv Block
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    # Third Conv Block
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),
    layers.Dropout(0.3),

    # Fourth Conv Block
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    # Fully Connected Layer
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),  # Larger dropout before final layer
    layers.Dense(3, activation="softmax")  # Output Layer
    ])

#Compiling
model.build()
print(model.summary())

model.compile(
    optimizer= optimizer,
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
    )

#Improving Train Speed
#I changed the Improving Train Speed by calling the functions in one line; Suffeling Only Train, overlaps data preprocessing with model execution

Autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(Autotune)
val_dataset = val_dataset.cache().prefetch(Autotune)
test_dataset = test_dataset.cache().prefetch(Autotune)


#model training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

#Evaluation
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy: "+ test_accuracy)

#plotting
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
"""
"""
#2.improved the running time of the model

#Variables
image_size_x =432
image_size_y = 316
batch_size = 8 #changed from 32 for segmenting the data to smaller chuncks so my computer could handle it
seed = 42
channels= 3
epochs = 10
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)


#Load Dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/train",
    image_size = (image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
    )
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/validation",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/test",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)

#Neural Network
model = models.Sequential([
    #I changed this layer to two seperate ones of input and conv2d because the kernel warned me about it: layers.Conv2D(32,(3,3), activation="relu", input_shape=(image_size_x,image_size_y,channels)),
    layers.Input(shape=(image_size_x,image_size_y,channels)),
    layers.Conv2D(32,(3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu" ),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
    ])

#Compiling
model.build()
print(model.summary())

model.compile(
    optimizer= optimizer,
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
    )

#Improving Train Speed
#I changed the Improving Train Speed by calling the functions in one line; Suffeling Only Train, overlaps data preprocessing with model execution

Autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(Autotune)
val_dataset = val_dataset.cache().prefetch(Autotune)
test_dataset = test_dataset.cache().prefetch(Autotune)


#model training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

#Evaluation
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy: "+ test_accuracy)

#plotting
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

"""
"""
1.Had running issues, I guess that my computer couldn't handle it

#Variables
image_size_x =432
image_size_y = 316
batch_size = 
seed = 42
channels= 3
epochs = 10

#Load Dataset
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/train",
    image_size = (image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
    )
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/validation",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "candleStick_charts/test",
    image_size = ( image_size_x,image_size_y),
    batch_size = batch_size,
    seed = seed
)

#Neural Network
model = models.Sequential([
    layers.Conv2D(32,(3,3), activation="relu", input_shape=(image_size_x,image_size_y,channels)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu" ),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(3, activation="softmax")
    ])

#Compiling
model.build()
print(model.summary())

model.compile(
    optimizer='adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics = ['accuracy']
    )

#Improving Train Speed

Autotune = tf.data.AUTOTUNE

train_dataset = train_dataset.cache()
val_dataset = val_dataset.cache()
test_dataset = test_dataset.cache()

#Suffeling Only Train
train_dataset = train_dataset.shuffle(1000)

#overlaps data preprocessing with model execution
train_dataset = train_dataset.prefetch(Autotune)
val_dataset = val_dataset.prefetch(Autotune)
test_dataset = test_dataset.prefetch(Autotune)

#model training
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

#Evaluation
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Accuracy: "+ test_accuracy)

#plotting
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

"""
"""
function form

def run_model(image_size_x,image_size_y,channels,batch_size,seed,epochs):
    #Variables
    image_size_x = image_size_x #432
    image_size_y = image_size_y #316
    batch_size = batch_size #32
    seed = seed #42
    channels= channels #3
    epochs = epochs #10

    #Load Dataset
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_charts/train",
        image_size = (image_size_x,image_size_y),
        batch_size = batch_size,
        seed = seed
        )
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_charts/validation",
        image_size = ( image_size_x,image_size_y),
        batch_size = batch_size,
        seed = seed
    )
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "candleStick_charts/test",
        image_size = ( image_size_x,image_size_y),
        batch_size = batch_size,
        seed = seed
    )

    #Neural Network
    model = models.Sequential([
        layers.Conv2D(32,(3,3), activation="relu", input_shape=(image_size_x,image_size_y,channels)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3), activation="relu" ),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64,(3,3), activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(3, activation="softmax")
        ])

    #Compiling
    model.build()
    print(model.summary())

    model.compile(
        optimizer='adam',
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
        )

    #Improving Train Speed

    Autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.cache()
    val_dataset = val_dataset.cache()
    test_dataset = test_dataset.cache()

    #Suffeling Only Train
    train_dataset = train_dataset.shuffle(1000)

    #overlaps data preprocessing with model execution
    train_dataset = train_dataset.prefetch(Autotune)
    val_dataset = val_dataset.prefetch(Autotune)
    test_dataset = test_dataset.prefetch(Autotune)

    #model training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    #Evaluation
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print("Test Accuracy: "+ test_accuracy)
    
    #plotting
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

"""
"""
# print image function 
for image_batch, label_batch in dataset.take(1):
    plt.imshow(image_batch[0].numpy().astype("uint8"))
    plt.axis('off')  # Optional: Removes axis ticks for a cleaner image
    plt.show()
"""