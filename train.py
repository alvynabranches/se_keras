import os
import numpy as np
from tqdm import tqdm
from model import AutoEncoder, VariationalAutoEncoder

SPECTOGRAM_PATH = os.path.join(os.path.dirname(__file__), "data", "spectograms")
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

# def load_mnist():
#     (x_train, _), (x_test, _) = mnist.load_data()
#     x_train = x_train.astype("float32") / 255
#     x_train = x_train.reshape(x_train.shape + (1,))
#     x_test = x_test.astype("float32") / 255
#     x_test = x_test.reshape(x_test.shape + (1,))
    
#     return x_train, x_test

# def train(x_train, learning_rate, batch_size, epochs):
#     model = VariationalAutoEncoder(
#         input_shape=(28, 28, 1),
#         conv_filters=( 32,  64,  64,  64),
#         conv_kernels=(  3,   3,   3,   3),
#         conv_strides=(  1,   2,   2,   1),
#         conv_dropout=(0.3, 0.3, 0.3, 0.3),
#         latent_space_dim=2
#     )
#     model.compile(learning_rate)
#     model.train(x_train, x_train, batch_size, epochs)
#     return model

def load_data(spectogram_path, ):
    x_noisy, x_clean = [], []
    for root, _, files in os.walk(spectogram_path):
        clean_files = [file for file in files if file.endswith("clean.npy")]
        noisy_files = [file for file in files if file.endswith("noisy.npy")]
        clean_files.sort()
        noisy_files.sort()
        for clean_file, noisy_file in zip(clean_files, noisy_files):
            clean_file = os.path.join(root, clean_file)
            noisy_file = os.path.join(root, noisy_file)
            x_clean.append(np.load(clean_file))
            x_noisy.append(np.load(noisy_file))
            break
    
    x_clean = np.array(x_clean)[..., np.newaxis]
    x_noisy = np.array(x_noisy)[..., np.newaxis]
            
    return x_noisy, x_clean

class LazyData:
    def __init__(self, spectogram_path, batch_size) -> None:
        self._noisy_files = [os.path.join(spectogram_path, file) for file in os.listdir(spectogram_path) if file.endswith("noisy.npy")]
        self._clean_files = [os.path.join(spectogram_path, file) for file in os.listdir(spectogram_path) if file.endswith("clean.npy")]
        self._batch_size = batch_size
        
    def __len__(self):
        return (len(self._noisy_files) // self._batch_size)
    
    def __getitem__(self, key):
        noisy_spectogram, clean_spectogram = [], []
        for i in range(key*self._batch_size, (key+1)*self._batch_size):
            clean_spectogram.append(np.load(self._clean_files[i])[..., np.newaxis])
            noisy_spectogram.append(np.load(self._noisy_files[i])[..., np.newaxis])
        return np.array(noisy_spectogram), np.array(clean_spectogram)

def load_lazy_data(spectogram_path, batch_size):
    noisy_files = [os.path.join(spectogram_path, file) for file in os.listdir(spectogram_path) if file.endswith("noisy.npy")]
    clean_files = [os.path.join(spectogram_path, file) for file in os.listdir(spectogram_path) if file.endswith("clean.npy")]
    clean_files.sort()
    noisy_files.sort()
    for batch_start in range(0, len(clean_files), batch_size):
        batch_end = batch_start + batch_size
        noisy_spectogram, clean_spectogram = [], []
        for i in range(batch_start, batch_end):
            clean_spectogram.append(np.load(clean_files[i])[..., np.newaxis])
            noisy_spectogram.append(np.load(noisy_files[i])[..., np.newaxis])
        yield np.array(noisy_spectogram), np.array(clean_spectogram)
        
def train(data, learning_rate, batch_size, epochs):
    model = VariationalAutoEncoder(
        input_shape=(256, 1723, 1),
        conv_filters=( 32,  64,  64,  64),
        conv_kernels=(  3,   3,   3,   3),
        conv_strides=(  1,   2,   2,   1),
        conv_dropout=(0.3, 0.3, 0.3, 0.3),
        latent_space_dim=2
    )
    model.compile(learning_rate)
    for i in tqdm(range(len(data)), total=len(data)):
        x_noisy, x_clean = data[i]
        model.train(x_noisy, x_clean, batch_size, epochs)
    return model


if __name__ == "__main__":
    # noisy_files, clean_files = load_lazy_data(SPECTOGRAM_PATH)
    # for batch in load_lazy_data(SPECTOGRAM_PATH, BATCH_SIZE):
    #     noisy_spectogram, clean_spectogram = batch
    #     print(noisy_spectogram.shape, clean_spectogram.shape)
    # autoencoder = train(x_train[:500], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # autoencoder.save("vae")
    # model = VariationalAutoEncoder.load("model")
    # model.summary()
    data = LazyData(SPECTOGRAM_PATH, BATCH_SIZE)
    train(data, LEARNING_RATE, BATCH_SIZE, EPOCHS)