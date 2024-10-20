import cv2
from skimage import feature
import matplotlib.pyplot as plt
import random
import numpy as np


#Readin an image and converting it into gray scale
image= cv2.imread("soft1.jpeg")
gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#Findind texture using local binary pattern                     
texture= feature.local_binary_pattern(gray, P=8, R=1)

#creating a list to store images with their class labels
classes= []

#using loops to take images(1 loop for each class) and storing the images in the list
#Class 1: Soft Texture
for i in range(1,4):
   soft=cv2.imread(f"soft{i}.jpeg")
   classes.append((soft, 'Soft Texture'))

#Class2: Rough Texture
for i in range(1,4):
   rough=cv2.imread(f"rough{i}.jpeg")
   classes.append((rough, 'Rough Texture'))  

#Class 3: 3D Images
for i in range(1,4):
   threeD=cv2.imread(f"3d{1}.jpeg")   
   classes.append((threeD, "3D Image")) 

#a function to take random image, converting the image into grayscale and finding its textue   
random_image, label=random.choice(classes)
gray_texture= cv2.cvtColor(random_image, cv2.COLOR_BGR2GRAY)
random_texture= feature.local_binary_pattern(gray_texture, P=8, R=1)

#Finding variance of selected and random image
#selected image
image_array = np.array(image)
mean1 = np.mean(image_array)
variance1 = np.mean((image_array - mean1) ** 2)

#Random image
random_array = np.array(random_image)
mean2 = np.mean(random_array)
variance2 = np.mean((random_array - mean2) ** 2)

# Defining the label for the selected image
main_label = 'Soft Texture' 

# Check if the labels are the same
if label == main_label:
    print("The images belong to the same class.")
else:
    print("The images belong to different classes.")

#checking if the variance is same or not
if variance1==variance2:
   print("Same Image")
else:
   print("Different Images")

#Displaying the images side by side
plt.subplot(1,2,1)
plt.imshow(random_texture, cmap='gray')
plt.title(f"Random Image\nClass:{label}\nVariance:{variance2}")

plt.subplot(1,2,2)
plt.title(f"Selected Image\nClass: Soft Texture\nVariance:{variance1}")
plt.imshow(texture, cmap='gray')

plt.show()







