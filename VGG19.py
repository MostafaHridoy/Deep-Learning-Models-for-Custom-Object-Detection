import tensorflow
from tensorflow.keras.layers import BatchNormalization, Dropout

input_shape = (224, 224, 3)
def create_vgg19():
  inputs=Input(input_shape)

  #Block 1
  x= Conv2D(64,(3,3), activation='relu',padding='same')(inputs)#1
  x= Conv2D(64,(3,3), activation='relu', padding='same')(x)#2
  x= MaxPooling2D((2,2), strides=(2,2))(x)

  #Block 2
  x= Conv2D(128, (3,3), activation='relu', padding='same')(x)#3
  x= BatchNormalization()(x)
  x= Conv2D(128, (3,3), activation='relu', padding='same')(x)#4
  x= MaxPooling2D((2,2), strides=(2,2))(x)

  #Block 3
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)#5
  x= Conv2D(256,(3,3), activation='relu', padding='same')(x)#6
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)#7
  x= Dropout(0.3)(x)
  x= Conv2D(256, (3,3), activation='relu', padding='same')(x)#8
  x= MaxPooling2D((2,2), strides=(2,2))(x)

  #Block 4
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#9
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#10
  x= BatchNormalization()(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#11
  x= Dropout(0.2)(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#12
  x= MaxPooling2D((2,2), strides=(2,2))(x)

  #Block 5
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#13
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#14
  x= Dropout(0.1)(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#15
  x= BatchNormalization()(x)
  x= Conv2D(512, (3,3), activation='relu', padding='same')(x)#16
  x= MaxPooling2D((2,2), strides=(2,2))(x)

  #Block 6
  x= Flatten()(x)#17
  x= Dense(4096, activation='relu')(x)#18
  x= Dense(4096, activation='relu')(x)#19

  outputs= Dense(2, activation='softmax')(x)

  model= Model(inputs, outputs)

  return model


vgg19= create_vgg19()

vgg19.summary()

vgg19.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

import os
import glob
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


dataset_dir = "/content/vision_transformer"

class_names = ["Pothole", "speedbreaker"]


images = []
labels = []


for split in ["train", "test", "valid"]:
    split_dir = os.path.join(dataset_dir, split)

    
    for class_name in class_names:
        class_dir = os.path.join(split_dir, class_name)

        
        image_files = glob.glob(os.path.join(class_dir, "*.jpg"))  

        
        for image_file in image_files:
            img = load_img(image_file, target_size=(224, 224))  
            img = img_to_array(img)
            img = img / 255.0 
            images.append(img)

            label = class_names.index(class_name)
            labels.append(label)

images = np.array(images)
labels = np.array(labels)


num_classes = len(class_names)
one_hot_labels = to_categorical(labels, num_classes=num_classes)

X_train, X_test, y_train, y_test = train_test_split(images, one_hot_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


vgg19.fit(X_train, y_train, epochs=5, batch_size=32)

test_loss, test_acc = vgg19.evaluate(X_test, y_test)

print(f"Test accuracy: {test_acc}")

vgg19.save('VGG19.h5')

import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

image_path = "/content/vision_transformer/test/speedbreaker/speedbreaker401.jpg" 
img = load_img(image_path, target_size=(224, 224)) 
img_array = img_to_array(img)
img_array = img_array / 255.0  
img_array = np.expand_dims(img_array, axis=0) 


predictions = vgg16_model.predict(img_array)


class_names = ["Pothole", "Speedbreaker"]

predicted_class_index = np.argmax(predictions)
predicted_class = class_names[predicted_class_index]
confidence_score = predictions[0][predicted_class_index]
plt.figure(figsize=(3, 5))  
plt.subplot(2, 1, 1)  
plt.imshow(img)
plt.axis('off') 
plt.subplot(2, 1, 2)  
text = f"Predicted class: {predicted_class}\nConfidence: {confidence_score:.2f}"
plt.text(0.5, 0.5, text, fontsize=12, ha='center', va='center')
plt.axis('off')  

plt.tight_layout()  
plt.show()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
y_pred = vgg16_model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test[:, 0], y_pred[:, 0])
roc_auc = auc(fpr, tpr)

# Plotting ROC curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')

# Creating a confusion matrix
y_true = np.argmax(y_test, axis=1)
y_pred_class = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_true, y_pred_class)

# Displaying the confusion matrix
disp = ConfusionMatrixDisplay(confusion_mtx, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.tight_layout()
plt.show()

