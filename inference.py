import pathlib

import torch
from torchvision import transforms, utils
from PIL import Image

from face_align import FaceAligner
from model import build_model


CELEBRITY_LABELS = {'female': 0, 'male': 1}
ANIMAL_LABELS = {"cat": 0, "dog": 1, "wild": 2}

CHECKPOINTS_ROOT_DIR = pathlib.Path("checkpoints/")


class Predictor(torch.nn.Module):
    def __init__(self, image_size=256, style_dim=64, latent_dim=16, entity="celebrity",
                 checkpoint_dir=CHECKPOINTS_ROOT_DIR, checkpoint_file="100000_nets_ema.ckpt"):
        
        super(Predictor, self).__init__()
        
        if entity == "celebrity":
            self.labels = CELEBRITY_LABELS
            sub_folder = "celeba_hq"
            self.w_hpf = 1
        elif entity == "animal":
            self.labels = ANIMAL_LABELS
            sub_folder = "afhq"
            self.w_hpf = 0
        else:
            raise KeyError("'entity' parameter must be 'celebrity' or 'animal'")

        self.image_size = image_size
        self.wing_path = checkpoint_dir / "wing.ckpt"
        self.lm_path = checkpoint_dir / "celeba_lm_mean.npz"

        self.transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        self.checkpoint_path = pathlib.Path(checkpoint_dir) / sub_folder / checkpoint_file
        
        self.nets_ema = build_model(img_size=image_size, 
                                    style_dim=style_dim, 
                                    w_hpf=self.w_hpf,
                                    latent_dim=latent_dim,
                                    num_domains=len(self.labels),
                                    wing_path=self.wing_path)

        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def create_interpolation(
            self,
            ref_label: str,
            src_image: Image,
            ref_image: Image
    ):

        result_image = pathlib.Path("images/res.jpg")

        if self.labels == CELEBRITY_LABELS:
            aligned_src, aligned_ref = self._align(src_image=src_image, ref_image=ref_image)
        else:
            aligned_src = self.transform(src_image).unsqueeze(0)  # batch_size = 1
            aligned_ref = self.transform(ref_image).unsqueeze(0)  # batch_size = 1

        ref_target = torch.tensor([self.labels[ref_label]])
        self._load_checkpoint(self.checkpoint_path, self.device, **self.nets_ema)
        self._translate_using_reference(aligned_src.to(self.device),
                                        aligned_ref.to(self.device),
                                        ref_target.to(self.device),
                                        result_image)

    def _align(self, src_image: Image, ref_image: Image):
        aligner = FaceAligner(self.wing_path, self.lm_path, self.image_size)
        aligned_images = []
        for image in src_image, ref_image:
            x = self.transform(image).unsqueeze(0)
            x_aligned = aligner.align(x)
            aligned_images.append(x_aligned)
        return aligned_images

    @torch.no_grad()
    def _translate_using_reference(self, x_src, x_ref, y_ref, filename):
        masks = self.nets_ema.fan.get_heatmap(x_src) if self.w_hpf > 0 else None
        s_ref = self.nets_ema.style_encoder(x_ref, y_ref)
        x_fake = self.nets_ema.generator(x_src, s_ref, masks=masks)
        self._save_image(x_fake, 1, filename)
        del s_ref

    @staticmethod
    def _load_checkpoint(checkpoint_path, device, **nets_ema):
        module_dict = torch.load(checkpoint_path, map_location=device)
        for name, module in nets_ema.items():
            module.load_state_dict(module_dict[name])

    @staticmethod
    def _denormalize(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def _save_image(self, x, ncol, filename):
        x = self._denormalize(x)
        filename.parent.mkdir(exist_ok=True)
        utils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
