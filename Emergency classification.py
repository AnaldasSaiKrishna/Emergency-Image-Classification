#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 


# In[2]:


os.getcwd()


# In[3]:


os.chdir('E:\Workings\emergency_classification')


# In[4]:


import numpy as np


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#from skimage.io import imread, imshow


# In[8]:


image = plt.imread("E:/Workings/emergency_classification/images/1.JPG")


# In[9]:


image


# In[10]:


plt.imshow(image)


# In[11]:


from skimage.exposure import adjust_gamma
plt.imshow(adjust_gamma(image,0.5))


# In[12]:


from glob import glob


# In[13]:


images= glob("E:/Workings/emergency_classification/images/*.JPG")


# In[14]:


images


# In[15]:


rng= np.random.RandomState()


# In[16]:


rng.choice(images)


# In[17]:


img_rng=rng.choice(images)
img=plt.imread(img_rng)
plt.imshow(img)


# In[18]:


import pandas as pd


# In[19]:


data=pd.read_csv('data.csv')


# In[20]:


data.head()


# In[21]:


data['emergency_or_not'].value_counts()


# In[22]:


data['emergency_or_not'].value_counts()/len(data['emergency_or_not'])*100


# In[24]:


# randomly select any row from the data 

row_index= rng.choice(data.index)


# pick the name of the image and combine the path 

img_name=data.iloc[row_index]['image_names']

#read and plot the image 

img= plt.imread('E:/Workings/emergency_classification/images/'+img_name)
plt.imshow(img)

#pick out the class from the target for the corresponding image 

target= data.iloc[row_index]['emergency_or_not']

#print the class of the vehicle
if target==1:
    print( 'this is an emergency vehicle')
else:
    print ('It is not Emergency vehicle')


# In[26]:


from keras.layers import Dense
from keras.models import Sequential
from keras.preprocessing import image

from sklearn.model_selection import train_test_split


# In[27]:


# seed : It initializes the pseudorandom number generator. You should call it before generating the random number. 
#If you use the same seed to initialize

seed=42
rng=np.random.RandomState(seed)


# In[28]:


data=pd.read_csv('data.csv')


# In[31]:


x=[]
for img_name in data.image_names:
    img= plt.imread('E:/Workings/emergency_classification/images/'+img_name)
    x.append(img)
    
x=np.array(x)

y=data.emergency_or_not.values


# In[32]:


x=x.reshape(2352, 224*224*3) #changing the shape to make sure the data should have in single, SO NN might take it as input


# In[35]:


x=x/x.max() # bring the range to {0,1}


# In[34]:


x_train,x_valid,y_train,y_valid = train_test_split(x,y,test_size=0.3,random_state=seed)


# In[36]:


model=Sequential()

model.add(Dense(100,input_dim=224*224*3,activation='sigmoid'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[37]:


model.summary()


# In[38]:


#train model on training data 
model.fit(x_train,y_train,epochs=10,batch_size=128,validation_data=(x_valid,y_valid))


# In[39]:


#get predictions 

predictions = model.predict_classes(x_valid)[:,0]
predictions_probabilities = model.predict(x_valid)[:,0]


# In[41]:


# pull out the original images from the data which corresponds to validation data 
_,valid_vehicles,_, valid_y=train_test_split(data.image_names.values,y,test_size=0.3,random_state=seed)


# In[43]:



#get a random index to plot image randomly

index= rng.choice(range(len(valid_vehicles)))

index


# In[45]:


#get the corresponding image name and prob

img_name=valid_vehicles[index]
img_name

prob=(predictions_probabilities*100).astype(int)[index]


# In[46]:


#read the image

img=plt.imread("E:/Workings/emergency_classification/images/"+img_name)


# In[47]:


#print the prob and actual class

print(prob,'% sure that it is emergency')

print('whereas actual class is',valid_y[index])


# In[48]:


#plot the image

plt.imshow(img)


# In[49]:


incorrect_indices= np.where(predictions !=y_valid)[0]


# In[50]:


len(incorrect_indices),predictions.shape


# In[51]:


img=plt.imread("E:/Workings/emergency_classification/images/"+img_name)


# In[52]:


print(prob,'% sure that it is emergency')

print('whereas actual class is',valid_y[index])

