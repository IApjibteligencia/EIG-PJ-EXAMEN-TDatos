
# EIG-PJ-EXAMEN-TDatos
Examen Tratamiento Datos Patricio Jimenez

Paso 1 Primero instalamos algunas librerias necesarias y otras de prueba 
pip install pandas 
pip install vboxapi 
pip install scikit-learn
pip install pupyterlab
pip install numpy
pip install matplotlib
pip install opencv-python

Paso 2 Cargamos las imagenes del directorio descargado 

Paso 3 mostramos las imagenes en modo grises para que el entrenamiento sea con menos resolucion y funcione mejor 

Paso 4 CREACION Y ENTRENAMIENTO MODELO CNN

Paso 5 Realiza el entrenamiento del modelo utilizando el conjunto de datos

Paso 6 Grabar modelo en disco
modelo.save("c:\\EIG\\Ejercicios\\CarneDataset\\modelo_CNN.h5")

Paso 7 #evaluar el modelo del conjunto train
modelo.evaluate(train_ds1, return_dict=True)
evaluar imagenes test

Paso 8 #evaluar el modelo del conjunto test
modelo.evaluate(test_ds, return_dict=True)

Paso 8 Evaluar con una imagen diferente

#evaluacion de una imagen prueba
image_path = 'c:\\EIG\\Ejercicios\\CarneDataset\\test\\Gato.png'
image = tf.keras.preprocessing.image.load_img(image_path).resize((300, 300))
input_arr = tf.keras.preprocessing.image.img_to_array(image)
input_arr = np.array([input_arr])
predictions = modelo.predict(input_arr)

Paso 9 Matriz de confusión y métricas de desempeño

  precision    recall  f1-score   support

           0     0.0000    0.0000    0.0000         1
           1     0.9444    0.3542    0.5152        48
           2     0.7788    0.8351    0.8060        97
           3     0.5915    0.9333    0.7241        45
           4     0.9581    0.9455    0.9518       459
           5     1.0000    0.7895    0.8824        19
           6     0.7634    0.8772    0.8163       114
           7     0.7778    0.5185    0.6222        27

    accuracy                         0.8679       810
   macro avg     0.7268    0.6567    0.6647       810
weighted avg     0.8818    0.8679    0.8629       810
 
 
