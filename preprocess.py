import os
import pickle
import librosa
import numpy as np
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings("ignore")

class Loader:
    def __init__(self, sample_rate: int, duration: float, mono: bool) -> None:
        self.sample_rate = sample_rate
        self.duration = duration
        self._mono = mono
        
    def load(self, file_path: str) -> np.ndarray:
        return librosa.load(
            file_path, sr=self.sample_rate, duration=self.duration, mono=self._mono
        )[0]
    
class Padder:
    def __init__(self, mode: str="constant"):
        self._mode = mode
        
    def left_pad(self, array: np.ndarray, num_missing_items: int) -> np.ndarray:
        return np.pad(array, (num_missing_items, 0), mode=self._mode)

    def right_pad(self, array: np.ndarray, num_missing_items: int) -> np.ndarray:
        return np.pad(array, (0, num_missing_items), mode=self._mode)
    
    def both_side_pad(self, array: np.ndarray, num_missing_items: int) -> np.ndarray:
        left_pad_items = int(num_missing_items // 2)
        right_pad_items = num_missing_items - left_pad_items
        
        padded_array = self.left_pad(array, left_pad_items)
        padded_array = self.right_pad(padded_array, right_pad_items)
        return padded_array
    
class LogSpectogramExtractor:
    def __init__(self, frame_size: int, hop_length: int) -> None:
        self._frame_size = frame_size
        self._hop_length = hop_length
        
    def __call__(self, signal) -> np.ndarray:
        stft = librosa.stft(signal, n_fft=self._frame_size, hop_length=self._hop_length)[:-1]
        spectogram = np.abs(stft)
        log_spectogram = librosa.amplitude_to_db(spectogram)
        return log_spectogram
    
class MinMaxNormalizer:
    def __init__(self, min_val: int|float, max_val: int|float) -> None:
        self._min = min_val
        self._max = max_val
        
    def normalize(self, array: np.ndarray) -> np.ndarray:
        norm_array = (array - array.min()) / (array.max() - array.min())
        norm_array = norm_array * (self._max - self._min) + self._min
        return norm_array
        
    def denormalize(self, norm_array: np.ndarray, original_min: int|float, original_max: int|float) -> np.ndarray:
        array = (norm_array - self._min) / (self._max - self._min)
        array = array * (original_max - original_min) + original_min
        return array
    
class Saver:
    def __init__(self, feature_save_dir, min_max_values_save_dir) -> None:
        self._feature_save_dir = feature_save_dir
        self._min_max_values_save_dir = min_max_values_save_dir
    
    def save_feature(self, clean_norm_feature: np.ndarray, noisy_norm_feature: np.ndarray, file_path: str) -> str:
        save_path = self._generate_save_path(file_path)
        np.save(save_path.format("clean"), clean_norm_feature)
        np.save(save_path.format("noisy"), noisy_norm_feature)
        return save_path
    
    def _generate_save_path(self, file_path) -> str:
        file_name = os.path.split(file_path)[1]
        return os.path.join(self._feature_save_dir, file_name + "_{}.npy")
    
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self._min_max_values_save_dir, "min_max_values.pkl")
        self._save(min_max_values, save_path)
    
    @staticmethod
    def _save(data, save_path) -> bool:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
class PreprocessingPipeline:
    def __init__(self, audio_files_dir: str) -> None:
        self._loader: Loader = None
        self.padder: Padder = None
        self.extrator: LogSpectogramExtractor = None
        self.normalizer: MinMaxNormalizer = None
        self.saver: Saver = None
        
        self._audio_files_dir = audio_files_dir
        self._clean_files = []
        self._noisy_files = []
        self._min_max_values = {}
        self._num_expected_samples = None
        
    @property
    def loader(self) -> Loader:
        return self._loader
    
    @loader.setter
    def loader(self, loader: Loader) -> None:
        self._loader = loader
        self._num_expected_samples = int(loader.sample_rate * loader.duration)
        
    def process(self) -> None:
        clean_files_dir_1 = os.path.join(self._audio_files_dir, "clean_trainset_28spk_wav")
        noisy_files_dir_1 = os.path.join(self._audio_files_dir, "noisy_trainset_28spk_wav")
        clean_files_dir_2 = os.path.join(self._audio_files_dir, "clean_trainset_56spk_wav")
        noisy_files_dir_2 = os.path.join(self._audio_files_dir, "noisy_trainset_56spk_wav")
        files_dir_1 = [os.path.join(file) for file in os.listdir(clean_files_dir_1) if file.endswith(".wav")]
        files_dir_2 = [os.path.join(file) for file in os.listdir(clean_files_dir_2) if file.endswith(".wav")]
        for i in tqdm(range(len(files_dir_1)), total=len(files_dir_1)):
            if not os.path.isfile(os.path.join(clean_files_dir_1, files_dir_1[i])):
                raise ValueError("CLEAN:", files_dir_1[i])
            if not os.path.isfile(os.path.join(noisy_files_dir_1, files_dir_1[i])):
                raise ValueError("NOISY:", files_dir_1[i])
            self._clean_files.append(os.path.join(clean_files_dir_1, files_dir_1[i]))
            self._noisy_files.append(os.path.join(noisy_files_dir_1, files_dir_1[i]))
        for i in tqdm(range(len(files_dir_2)), total=len(files_dir_2)):
            if not os.path.isfile(os.path.join(clean_files_dir_2, files_dir_2[i])):
                raise ValueError("CLEAN:", files_dir_2[i])
            if not os.path.isfile(os.path.join(noisy_files_dir_2, files_dir_2[i])):
                raise ValueError("NOISY:", files_dir_2[i])
            self._clean_files.append(os.path.join(clean_files_dir_2, files_dir_2[i]))
            self._noisy_files.append(os.path.join(noisy_files_dir_2, files_dir_2[i]))
            self.saver.save_min_max_values(self._min_max_values)
            
        for i in tqdm(range(len(self._clean_files))):
            self.process_file(i)
            
    def process_file(self, index) -> None:
        clean_signal = self.loader.load(self._clean_files[index])
        noisy_signal = self.loader.load(self._noisy_files[index])
        if self._is_padding_necessary(clean_signal):
            clean_signal = self._apply_padding(clean_signal)
        if self._is_padding_necessary(noisy_signal):
            noisy_signal = self._apply_padding(noisy_signal)
        clean_feature = self.extrator(clean_signal)
        noisy_feature = self.extrator(noisy_signal)
        clean_norm_feature = self.normalizer.normalize(clean_feature)
        noisy_norm_feature = self.normalizer.normalize(noisy_feature)
        save_path = self.saver.save_feature(clean_norm_feature, noisy_norm_feature, self._clean_files[index])
        self._store_min_max_value(save_path, clean_feature.min(), clean_feature.max(), noisy_feature.min(), noisy_feature.max())
        
    def _is_padding_necessary(self, signal: np.ndarray) -> bool:
        return len(signal) < self._num_expected_samples
    
    def _apply_padding(self, signal: np.ndarray) -> np.ndarray:
        num_missing_samples = self._num_expected_samples - len(signal)
        return self.padder.right_pad(signal, num_missing_samples)
    
    def _store_min_max_value(self, save_path, clean_min_val, clean_max_val, noisy_min_val, noisy_max_val):
        self._min_max_values[save_path] = {
            "clean_min": clean_min_val,
            "clean_max": clean_max_val,
            "noisy_min": noisy_min_val,
            "noisy_max": noisy_max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 20
    SAMPLE_RATE = 22050
    MONO = True
    
    SPECTOGRAM_SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "spectograms")
    MIN_MAX_VALUES_SAVE_DIR = os.path.join(os.path.dirname(__file__), "data", "min_max_values")
    FILES_DIR = os.path.join(os.path.dirname(__file__), "data", "audio")
    
    pipleline = PreprocessingPipeline("/media/alvynabranches/Data/se/data")
    pipleline.loader = Loader(SAMPLE_RATE, DURATION, MONO)
    pipleline.padder = Padder()
    pipleline.extrator = LogSpectogramExtractor(FRAME_SIZE, HOP_LENGTH)
    pipleline.normalizer = MinMaxNormalizer(0, 1)
    pipleline.saver = Saver(SPECTOGRAM_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)
    pipleline.process()