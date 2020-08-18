#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import cv2


# In[2]:


import os
import numpy as np
import keras
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split

from keras.models import load_model 



# In[3]:


from keras.layers import Dense,BatchNormalization,Dropout,Conv2D,Flatten,MaxPool2D,Input,Activation,ZeroPadding2D
from keras import Model


# In[49]:


DIRECTORY=r"C:\Users\vishal\3D Objects\Masks"
CATEGORIES=["without mask","with mask"]
data=[]


# In[50]:


for category in CATEGORIES:
  label=CATEGORIES.index(category)
  path=os.path.join(DIRECTORY,category)
  m=os.listdir(path)
  for img in m:
    arr=os.path.join(path,img)
    img_arr=cv2.imread(arr)
    img_arr=cv2.resize(img_arr,(64,64))
    data.append([img_arr,label])
    

    




# In[51]:


random.shuffle(data)

plt.imshow(data[0][0])
plt.show()

plt.imshow(data[1200][0])
plt.show()


# In[52]:


X=[]
Y=[]
for i,j in data:
  X.append(i)
  Y.append(j)

X=np.array(X)
Y=np.array(Y)




print(X.shape,Y.shape)
X=X/255


# In[53]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1)
print(X_train.shape,X_test.shape)


# In[54]:


def happy_model(input_shape):
  X_input=Input(input_shape)
  X=ZeroPadding2D((3,3))(X_input)
  X=Conv2D(32,(7,7),strides=(1,1),name="conv1")(X)
  X=BatchNormalization(axis=3,name="bn1")(X)
  X=Activation("relu")(X)
  X=MaxPool2D((2,2),name="maxpool1")(X)
  X=Flatten()(X)
  X=Dense(1,activation="sigmoid",name="dense1")(X)
  model=Model(inputs=X_input,outputs=X)
  
  return model


Happy_model=happy_model(X_train.shape[1:])

                       
                                                


# In[70]:


Happy_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

history=Happy_model.fit(x=X_train,y=Y_train,epochs=10,batch_size=64,validation_split=0.2)


# In[71]:


Happy_model.save("my_model")


# In[72]:


plt.plot(history.history["loss"],label="training_loss")
plt.plot(history.history["val_loss"],label="validation_loss")
plt.legend()
plt.show()


# In[73]:


plt.plot(history.history["accuracy"],label="training_accuracy")
plt.plot(history.history["val_accuracy"],label="validation_accuracy")
plt.legend()
plt.show()


# In[92]:


print(Happy_model.evaluate(X_test,Y_test))


# In[75]:


model=load_model("my_model")


# In[94]:


a=r"C:\Users\vishal\3D Objects\Masks\with mask\10-with-mask.jpg"
img=cv2.imread(a)
img=cv2.resize(img,(64,64))
cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


X=np.array(img)
X=X.reshape(1,64,64,3)

print(X.shape)


out=model.predict(X)
a=output[0][0]
print(a)
if a>0.5:
    print("with mask")
else:
    print("without mask")


# In[19]:





# In[ ]:





# In[ ]:





# In[88]:



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)
b=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter("mask.avi",b,20.0,(640,480))

while True:
    ret, img = cap.read()
    out.write(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img=img[y:y+h,x:x+w]
        resized=cv2.resize(face_img,(64,64))
        normalize=resized/255.0
        reshape=normalize.reshape(1,64,64,3)
        output=model.predict(reshape)
        label=output[0][0]
#         print(label)
        if label>0.5:
            a=1
        else:
            a=0
#         print(CATEGORIES[a])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.rectangle(img,(x,y-40),(x+w,y),(25,0,0),-1)
        cv2.putText(img,CATEGORIES[a],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    cv2.imshow("img", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
out.release()
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




