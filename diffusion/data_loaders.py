import os
import random
import re
import numpy as np
import librosa
import torch
import torchaudio
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from ddsp.vocoder import F0_Extractor, Volume_Extractor
from diffusion.vocoder import Vocoder

def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True
    ):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args, whole_audio=False):
    # initialize f0 extractor
    f0_extractor = F0_Extractor(
                        args.data.f0_extractor, 
                        args.data.sampling_rate, 
                        args.data.block_size, 
                        args.data.f0_min, 
                        args.data.f0_max)
    
    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size)
    
    # initialize mel extractor
    mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device = "cpu")

    data_train = ModifiedAudioDataset(
        args.data.train_path,
        f0_extractor=f0_extractor,
        volume_extractor=volume_extractor,
        mel_extractor=mel_extractor,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True)
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
    )
    
    data_valid = ModifiedAudioDataset(
        args.data.valid_path,
        f0_extractor=f0_extractor,
        volume_extractor=volume_extractor,
        mel_extractor=mel_extractor,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk)
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return loader_train, loader_valid 


class AudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
    ):
        super().__init__()

        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            name = os.path.splitext(name_ext)[0]
            path_audio = os.path.join(self.path_root, 'audio', name_ext)
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)
            
            path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
            f0 = np.load(path_f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)
                
            path_volume = os.path.join(self.path_root, 'volume', name_ext) + '.npy'
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)
            
            path_augvol = os.path.join(self.path_root, 'aug_vol', name_ext) + '.npy'
            aug_vol = np.load(path_augvol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)
                        
            if n_spk is not None and n_spk > 1:
                dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
                if spk_id < 1 or spk_id > n_spk:
                    raise ValueError(' [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk ')
            else:
                spk_id = 1
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)

            if load_all_data:
                '''
                audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                audio = torch.from_numpy(audio).to(device)
                '''
                path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
                mel = np.load(path_mel)
                mel = torch.from_numpy(mel).to(device)
                
                path_augmel = os.path.join(self.path_root, 'aug_mel', name_ext) + '.npy'
                aug_mel = np.load(path_augmel)
                aug_mel = torch.from_numpy(aug_mel).to(device)
                
                path_units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
                units = np.load(path_units)
                units = torch.from_numpy(units).to(device)
                
                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()
                    
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
            else:
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
           

    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))
            
        # get item
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug
        '''
        audio = data_buffer.get('audio')
        if audio is None:
            path_audio = os.path.join(self.path_root, 'audio', name) + '.wav'
            audio, sr = librosa.load(
                    path_audio, 
                    sr = self.sample_rate, 
                    offset = start_frame * frame_resolution,
                    duration = waveform_sec)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            # clip audio into N seconds
            audio = audio[ : audio.shape[-1] // self.hop_size * self.hop_size]       
            audio = torch.from_numpy(audio).float()
        else:
            audio = audio[start_frame * self.hop_size : (start_frame + units_frame_len) * self.hop_size]
        '''
        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = os.path.join(self.path_root, mel_key, name_ext) + '.npy'
            mel = np.load(mel)
            mel = mel[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load units
        units = data_buffer.get('units')
        if units is None:
            units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            units = np.load(units)
            units = units[start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units).float() 
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        
        return dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext)

    def __len__(self):
        return len(self.paths)

class ModifiedAudioDataset(Dataset):
    def __init__(
        self,
        path_root,
        waveform_sec,
        hop_size,
        sample_rate,
        mel_extractor,
        f0_extractor,
        volume_extractor,
        load_all_data=True,
        whole_audio=False,
        extensions=['wav'],
        n_spk=1,
        device='cpu',
        fp16=False,
        use_aug=False,
        use_pitch_aug=False,
    ):
        super().__init__()

        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.mel_extractor = mel_extractor
        self.f0_extractor = f0_extractor
        self.volume_extractor = volume_extractor
        self.paths = traverse_dir(
            path_root,
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )

        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer = {}
        self.use_pitch_aug = use_pitch_aug

        self.speaker_dict = {}

        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, name_ext)
            duration = librosa.get_duration(filename = path_audio, sr = self.sample_rate)

            speaker_name = os.path.basename(os.path.dirname(path_audio))
            if speaker_name not in self.speaker_dict:
                # if is the first speaker, then set spk_id to 0
                if len(self.speaker_dict) == 0:
                    spk_id = torch.LongTensor(np.array([0])).to(device)
                else:
                    spk_id = torch.LongTensor(np.array([len(self.speaker_dict)])).to(device)

                self.speaker_dict[speaker_name] = spk_id

            else:
                spk_id = self.speaker_dict[speaker_name]

            self.data_buffer[name_ext] = {
                "spk_id": spk_id,
                "duration": duration
            }

        assert len(self.speaker_dict) == n_spk, "n_spk is not equal to the number of speakers in the dataset"

        # print uniques spk_ids
        self.spk_ids = []
        for name_ext in self.data_buffer.keys():
            self.spk_ids.append(self.data_buffer[name_ext]["spk_id"].squeeze(0).item())
        self.spk_ids = list(set(self.spk_ids))
        print(" > spk_ids:", self.spk_ids)


    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        # check duration. if too short, then skip
        if data_buffer["duration"] < (self.waveform_sec + 0.1):
            return self.__getitem__( (file_idx + 1) % len(self.paths))

        # get item
        return self.get_data(name_ext, data_buffer, os.path.join(self.path_root, name_ext))

    def get_data(self, name_ext, data_buffer, filepath):
        name = os.path.splitext(name_ext)[0]
        duration = data_buffer["duration"]
        waveform_sec = duration if self.whole_audio else self.waveform_sec

        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        aug_flag = random.choice([True, False]) and self.use_aug

        audio, sr = torchaudio.load(filepath)

        # tranform to mono if needed
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if sr != self.sample_rate:
            audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
        if sr < self.sample_rate:
            raise ValueError("Sample rate of audio is lower than the target sample rate")

        start = int(idx_from * self.sample_rate)
        end = int((idx_from + waveform_sec) * self.sample_rate)

        # Trim audio
        audio = audio[:, start:end]

        audio_numpy = audio.squeeze().to("cpu").numpy()

        # load volume
        volume = self.volume_extractor.extract(audio_numpy)

        # Load mel
        mel = self.mel_extractor.extract(audio, self.sample_rate).squeeze().to("cpu")
        max_amp = float(torch.max(torch.abs(audio))) + 1e-5
        max_shift = min(1, np.log10(1/max_amp))
        log10_vol_shift = random.uniform(-1, max_shift)
        if self.use_pitch_aug:
            keyshift = random.uniform(-5, 5)
        else:
            keyshift = 0

        aug_mel_t = self.mel_extractor.extract(audio * (10 ** log10_vol_shift), self.sample_rate, keyshift = keyshift)
        aug_mel = aug_mel_t.squeeze().to('cpu')
        aug_vol = self.volume_extractor.extract(audio_numpy * (10 ** log10_vol_shift))

        volume_frames = aug_vol if aug_flag else volume
        mel = aug_mel if aug_flag else mel

        # load f0
        f0 = self.f0_extractor.extract(audio_numpy, uv_interp = False)

        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

        aug_shift = 0
        if aug_flag:
            # aug_shift = self.pitch_aug_dict[name_ext]
            aug_shift = 0
        f0_frames = 2 ** (aug_shift / 12) * f0

        # load spk_id
        spk_id = data_buffer.get('spk_id')

        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()

        return dict(mel=mel, f0=f0_frames, volume=volume_frames, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext,
                    audio=audio)

    def __len__(self):
        return len(self.paths)