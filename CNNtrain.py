import tensorflow as tf
from Load_images import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu,True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus),"Physical GPUs", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


#Custom loss function
def custom_loss(y_rot):
  def loss(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true-y_pred)) + tf.multiply(tf.reduce_mean(tf.square(y_pred-y_rot)),
           tf.constant(0.005))
  return loss

#Custom Training loop
def trainCNN(model,ds_train,epochs,custom_loss,optimizer,steps):
  data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])
  for epoch in range(1,epochs+1):
    history = []
    print("Epoch: ",epoch)
    for _ in range(steps):
      x_batch,y_batch = next(ds_train)
      with tf.GradientTape() as tape:
        y_rot_means = np.empty(y_batch.shape)
        for i in range(len(x_batch)):
          rot1  = np.expand_dims(tf.keras.preprocessing.image.random_rotation(x_batch[i],180,row_axis=0,col_axis=1,channel_axis=2),axis=0).astype('float32')
          rot2  = np.expand_dims(tf.keras.preprocessing.image.random_rotation(x_batch[i],155,row_axis=0,col_axis=1,channel_axis=2),axis=0).astype('float32')
          rot3 = np.expand_dims(tf.keras.preprocessing.image.random_rotation(x_batch[i], 130, row_axis=0, col_axis=1, channel_axis=2),axis=0).astype('float32')
          rot4 = np.expand_dims(tf.keras.preprocessing.image.random_rotation(x_batch[i], 105, row_axis=0, col_axis=1, channel_axis=2),axis=0).astype('float32')
          rot5 = np.expand_dims(tf.keras.preprocessing.image.random_rotation(x_batch[i], 80, row_axis=0, col_axis=1, channel_axis=2),axis=0).astype('float32')
          rot6 = np.expand_dims(tf.keras.preprocessing.image.random_rotation(x_batch[i], 55, row_axis=0, col_axis=1, channel_axis=2),axis=0).astype('float32')
          rot7 = np.expand_dims(
            tf.keras.preprocessing.image.random_rotation(x_batch[i], 30, row_axis=0, col_axis=1, channel_axis=2),
            axis=0).astype('float32')
          rot8 =np.expand_dims(
            tf.keras.preprocessing.image.random_rotation(x_batch[i], 15, row_axis=0, col_axis=1, channel_axis=2),
            axis=0).astype('float32')
          rot = np.vstack((rot1,rot2,rot3,rot4,rot5,rot6,rot7,rot8)).astype('float32')
          y_rot = np.mean(model(rot,training=True),axis=0)
          y_rot_means[i] = y_rot.astype('float32')
        y_pred = model(x_batch,training = True)
        loss_fn = custom_loss(y_rot=y_rot_means)
        loss = loss_fn(y_batch,y_pred)
      gradients = tape.gradient(loss,model.trainable_weights)
      optimizer.apply_gradients(zip(gradients,model.trainable_weights))
    print('loss: ',loss.numpy())



f = open('ResultsTrain-TestB.txt', 'w+')

#Establish the same dimensions for every image
HEIGHT = []
WIDTH = []
for name in os.listdir('Dataset/'):
  for fname in os.listdir('Dataset/'+name):
    if 'disp' in fname and '.png' in fname:
      struct = cv2.imread('Dataset/'+name+'/'+fname)
      HEIGHT.append(struct.shape[0])
      WIDTH.append(struct.shape[1])
HEIGHT = int(np.mean(HEIGHT).round())
WIDTH = int(np.mean(WIDTH).round())
print('height: ', HEIGHT,'\n Width: ',WIDTH)


# Model - Best model found
input = tf.keras.Input(shape=(460,660, 1))
l = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(input)
l = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(l)
l = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(l)
l = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(l)
l = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(l)
l = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same')(l)
modelT = tf.keras.Model(input,l)
modelT.trainable = True
optimizer = tf.keras.optimizers.Adam()



# load the original test images
masks = [rgb2gray(cv2.resize(cv2.imread('Dataset/adi/mask.png'),(WIDTH,HEIGHT))),rgb2gray(cv2.resize(cv2.imread('Dataset/Motorcycle/mask_50.png'),(WIDTH,HEIGHT)))]
lista=[]
lista_n=[]
i = 0
for fname in os.listdir('natural_images/test/'):
  struct = cv2.imread('natural_images/test/'+fname)
  struct = rgb2gray(struct)
  struct = cv2.resize(struct,(WIDTH,HEIGHT),interpolation = cv2.INTER_AREA)
  structx = np.where(masks[0].round()!=255,0,struct) if i%2 == 0 else np.where(masks[1].round()<=120,0,struct)
  struct = struct.astype('float32')/255
  structx = structx.astype('float32')/255
  lista.append(struct)
  lista_n.append(structx)
  i+=1
lista = np.expand_dims(np.array(lista),axis=-1)
lista_n= np.expand_dims(np.array(lista_n),axis=-1)
f.write("Dimensiones X,Y:"+str(lista_n.shape)+str(lista.shape))



# Train-Test
generatorT = generate_Data('natural_images/train/', batch_size=2, image_size=(WIDTH, HEIGHT))
history = trainCNN(modelT,generatorT,50,custom_loss,optimizer,int(4859 / 2))




f.write("Epochs MSE: " + str(history))
# test with same dataset
y_predict = modelT.predict(lista_n)

f.write("\nResultados Test dataset prueba\nPSNR: " + str(psnr(y_predict, lista)) + "\n MSE: " + str(
    np.mean(np.square(lista - y_predict))))
f.write("\nRango de valores de los pixeles:"+str(lista.max())+ "-"+ str(lista.min()))
# Test DATASET 16 imagenes

f.close()
modelT.save('modelR.h5')