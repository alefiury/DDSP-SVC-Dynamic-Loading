from torch.utils.data import DataLoader

from logger import utils
from diffusion.vocoder import Vocoder
from ddsp.vocoder import F0_Extractor, Volume_Extractor
from diffusion.data_loaders import ModifiedAudioDataset


def main():
    config_path = "configs/diffusion-new.yaml"
    base_dir = "/home/alefiury/Downloads/audios_test"
    device = "cpu"

    args = utils.load_config(config_path)

    mel_extractor = Vocoder(args.vocoder.type, args.vocoder.ckpt, device = device)

    f0_extractor = F0_Extractor(
        args.data.f0_extractor,
        args.data.sampling_rate,
        args.data.block_size,
        args.data.f0_min,
        args.data.f0_max
    )

    # initialize volume extractor
    volume_extractor = Volume_Extractor(args.data.block_size)

    data_train = ModifiedAudioDataset(
        base_dir,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        load_all_data=args.train.cache_all_data,
        whole_audio=False,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True,
        mel_extractor=mel_extractor,
        f0_extractor=f0_extractor,
        volume_extractor=volume_extractor,
    )

    data_loader = DataLoader(
        data_train,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    for data in data_loader:
        print(data)
        break

if __name__ == '__main__':
    main()