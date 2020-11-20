from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.utils import plot_model

# Dense Layer
model=Sequential()
model.add(Dense(16,input_shape=(8,)))
model.summary()

#plot_model(model,to_file='model_plot.png',show_shapes=True,show_layer_names=True)

print(model.get_config())
print(model.get_weights())

# Dropout layer
model=Sequential()
model.add(Dense(16,input_shape=(8,)))
model.add(Dropout(.2)) # Reduce 20% input at the time of model training
model.add(Dense(10))
model.summary()
