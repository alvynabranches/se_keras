import os
import pickle
import numpy as np
from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Dropout, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda
from keras import backend as K
from keras.optimizers import Adam
from keras.losses import MeanSquaredError, kullback_leibler_divergence, mean_squared_error
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class AutoEncoder:
    def __init__(
            self, 
            input_shape: list | tuple,
            conv_filters: list | tuple,
            conv_kernels: list | tuple,
            conv_strides: list | tuple,
            conv_dropout: list | tuple,
            latent_space_dim: int
        ):
        self._input_shape = input_shape
        self._conv_filters = conv_filters
        self._conv_kernels = conv_kernels
        self._conv_strides = conv_strides
        self._conv_dropout = conv_dropout
        self._latent_space_dim = latent_space_dim
        
        self._encoder = None
        self._model_input = None
        self._decoder = None
        self._model = None
        
        self._shape_before_bottleneck = None
        self._num_conv_layers = len(conv_filters)
        
        self._build()
        
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
        
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self._encoder = Model(encoder_input, bottleneck, name="encoder")
        
    def _add_encoder_input(self):
        return Input(shape=self._input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x
    
    def _add_conv_layer(self, layer_index, x):
        conv_layer = Conv2D(
            filters=self._conv_filters[layer_index],
            kernel_size=self._conv_kernels[layer_index],
            strides=self._conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_index+1}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_index+1}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_index+1}")(x)
        x = Dropout(self._conv_dropout[layer_index], name=f"encoder_dropout_{layer_index+1}")(x)
        return x
    
    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten(name="encoder_flatten")(x)
        x = Dense(self._latent_space_dim, name="encoder_output")(x)
        return x
    
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self._decoder = Model(decoder_input, decoder_output,  name="decoder")
        
    def _add_decoder_input(self):
        return Input(shape=self._latent_space_dim, name="decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck)
        return Dense(num_neurons, name="decoder_dense")(decoder_input)
    
    def _add_reshape_layer(self, dense_layer):
        return Reshape(self._shape_before_bottleneck, name="decoder_reshape")(dense_layer)
    
    def _add_conv_transpose_layers(self, x):
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
    
    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self._conv_filters[layer_index],
            kernel_size=self._conv_kernels[layer_index],
            strides=self._conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer{layer_num+1}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num+1}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num+1}")(x)
        x = Dropout(self._conv_dropout[layer_index], name=f"decoder_dropout_{layer_num+1}")(x)
        return x
    
    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1,
            kernel_size=self._conv_kernels[0],
            strides=self._conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer{self._num_conv_layers+1}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name=f"decoder_output")(x)
        return output_layer
    
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self._decoder(self._encoder(model_input))
        self._model = Model(model_input, model_output, name="autoencoder")
    
    def summary(self):
        self._encoder.summary()
        self._decoder.summary()
        self._model.summary()
        
    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self._model.compile(optimizer=optimizer, loss=mse_loss)
        
    def train(self, x_train, y_train, batch_size, num_epochs):
        self._model.fit(x_train, y_train, batch_size, num_epochs, validation_split=0.1)
        
    def save(self, save_folder="."):
        self._create_folder_if_does_not_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
        
    def _create_folder_if_does_not_exist(self, folder):
        if not os.path.isdir(folder):
            os.makedirs(folder)
    
    def _save_parameters(self, save_folder):
        parameters = [
            self._input_shape,
            self._conv_filters,
            self._conv_kernels,
            self._conv_strides,
            self._conv_dropout,
            self._latent_space_dim,
        ]
        
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
            f.close()
            
    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self._model.save_weights(save_path)
        
    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
            f.close()
        autoencoder = AutoEncoder(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder
        
    def load_weights(self, weights_path):
        self._model.load_weights(weights_path)
        

class VariationalAutoEncoder(AutoEncoder):
    def __init__(
            self,
            input_shape: list | tuple,
            conv_filters: list | tuple,
            conv_kernels: list | tuple,
            conv_strides: list | tuple,
            conv_dropout: list | tuple,
            latent_space_dim: int
        ):
        self._mu = None
        self._log_variance = None
        self.reconstruction_loss_weight = 1_000_000
        super().__init__(input_shape, conv_filters, conv_kernels, conv_strides, conv_dropout, latent_space_dim)
    
    def _add_bottleneck(self, x):
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten(name="encoder_flatten")(x)
        self._mu = Dense(self._latent_space_dim, name="encoder_mu")(x)
        self._log_variance = Dense(self._latent_space_dim, name="log_variance")(x)
        
        def sample_point_from_normal_distribution(args):
            mu, log_variance = args
            epsilion = K.random_normal(shape=K.shape(self._mu), mean=0., stddev=1.)
            return mu + K.exp(log_variance / 2) * epsilion
        
        x = Lambda(sample_point_from_normal_distribution, name="encoder_output")([self._mu, self._log_variance])
        return x

    # def _calculate_reconstruction_loss(self, y_target, y_predicted):
    #     return K.mean(K.square(y_target - y_predicted), axis=[1, 2, 3])
    
    # def _calculate_kl_loss(self, y_target, y_predicted):
    #     return -0.5 * K.sum(1 + self._log_variance - K.square(self._mu) - K.exp(self._log_variance), axis=1)
    
    # def _calculate_combine_loss(self, y_target, y_predicted):
    #     reconstruction_loss = self._calculate_reconstruction_loss(y_target, y_predicted)
    #     kl_loss = self._calculate_kl_loss(y_target, y_predicted)
    #     return self.reconstruction_loss_weight * reconstruction_loss + kl_loss
    
    def _calculate_combine_loss(self, y_true, y_pred):
        reconstruction_loss = mean_squared_error(y_true, y_pred)
        kl_loss = kullback_leibler_divergence(y_true, y_pred)
        return self.reconstruction_loss_weight * reconstruction_loss + kl_loss
    
    def compile(self, learning_rate=0.001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self._model.compile(optimizer=optimizer, loss=self._calculate_combine_loss, metrics=[mse_loss, kullback_leibler_divergence])
    
    
if __name__ == "__main__":
    autoencoder = VariationalAutoEncoder(
        input_shape=(28, 28, 1),
        conv_filters=(  32,   64,   64,   64),
        conv_kernels=(   3,    3,    3,    3),
        conv_strides=(   1,    2,    2,    1),
        conv_dropout=(0.30, 0.30, 0.30, 0.30),
        latent_space_dim=2
    )
    autoencoder.summary()