import os
from cog import BasePredictor, Input, Path
import sys
sys.path.append('/content/ReNoise-Inversion-hf')
os.chdir('/content/ReNoise-Inversion-hf')

from PIL import Image
import torch
from src.eunms import Model_Type, Scheduler_Type, Gradient_Averaging_Type, Epsilon_Update_Type
from src.enums_utils import model_type_to_size, get_pipes
from src.config import RunConfig
from main import run as run_model

def main_pipeline(
        input_image: str,
        src_prompt: str,
        tgt_prompt: str,
        edit_cfg: float,
        number_of_renoising_iterations: int,
        inersion_strength: float,
        avg_gradients: bool,
        first_step_range_start: int,
        first_step_range_end: int,
        rest_step_range_start: int,
        rest_step_range_end: int,
        lambda_ac: float,
        lambda_kl: float,
        noise_correction: bool,
        model_type: None, scheduler_type: None, image_size: None, pipe_inversion: None, pipe_inference: None, cache_size: None, prev_configs: None,
        prev_inv_latents: None, prev_images: None, prev_noises: None):

        update_epsilon_type = Epsilon_Update_Type.OPTIMIZE if noise_correction else Epsilon_Update_Type.NONE
        avg_gradients_type = Gradient_Averaging_Type.ON_END if avg_gradients else Gradient_Averaging_Type.NONE

        first_step_range = (first_step_range_start, first_step_range_end)
        rest_step_range = (rest_step_range_start, rest_step_range_end)

        config = RunConfig(model_type = model_type,
                    num_inference_steps = 4,
                    num_inversion_steps = 4, 
                    guidance_scale = 0.0,
                    max_num_aprox_steps_first_step = first_step_range_end+1,
                    num_aprox_steps = number_of_renoising_iterations,
                    inversion_max_step = inersion_strength,
                    gradient_averaging_type = avg_gradients_type,
                    gradient_averaging_first_step_range = first_step_range,
                    gradient_averaging_step_range = rest_step_range,
                    scheduler_type = scheduler_type,
                    num_reg_steps = 4,
                    num_ac_rolls = 5,
                    lambda_ac = lambda_ac,
                    lambda_kl = lambda_kl,
                    update_epsilon_type = update_epsilon_type,
                    do_reconstruction = True)
        config.prompt = src_prompt

        inv_latent = None
        noise_list = None
        for i in range(cache_size):
            if prev_configs[i] is not None and prev_configs[i] == config and prev_images[i] == input_image:
                print(f"Using cache for config #{i}")
                inv_latent = prev_inv_latents[i]
                noise_list = prev_noises[i]
                prev_configs.pop(i)
                prev_inv_latents.pop(i)
                prev_images.pop(i)
                prev_noises.pop(i)
                break

        original_image = Image.open(input_image).convert("RGB").resize(image_size)

        res_image, inv_latent, noise, all_latents = run_model(original_image,
                                    config,
                                    latents=inv_latent,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    edit_prompt=tgt_prompt,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg)

        prev_configs.append(config)
        prev_inv_latents.append(inv_latent)
        prev_images.append(input_image)
        prev_noises.append(noise)
        
        if len(prev_configs) > cache_size:
            print("Popping cache")
            prev_configs.pop(0)
            prev_inv_latents.pop(0)
            prev_images.pop(0)
            prev_noises.pop(0)
        return res_image

class Predictor(BasePredictor):
    def setup(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = Model_Type.SDXL_Turbo
        self.scheduler_type = Scheduler_Type.EULER
        self.image_size = model_type_to_size(Model_Type.SDXL_Turbo)
        self.pipe_inversion, self.pipe_inference = get_pipes(self.model_type, self.scheduler_type, device=self.device)
        self.cache_size = 10
        self.prev_configs = [None for i in range(self.cache_size)]
        self.prev_inv_latents = [None for i in range(self.cache_size)]
        self.prev_images = [None for i in range(self.cache_size)]
        self.prev_noises = [None for i in range(self.cache_size)]
    def predict(
        self,
        Input_Image: Path = Input(description="Input Image"),
        Source_Prompt: str = Input(default=""),
        Target_Prompt: str = Input(default=""),
        Denoise_Classifier_Free_Guidence_Scale: float = Input(default=1.0),
        Number_of_ReNoise_Iterations: int = Input(default=1.0),
        Inversion_Strength: float = Input(default=1.0),
        Preform_Estimation_Averaging: bool = Input(default=True),
        First_Estimation_in_Average_T_L: int = Input(default=0),
        Last_Estimation_in_Average_T_L: int = Input(default=5),
        First_Estimation_in_Average_T_G: int = Input(default=8),
        Last_Estimation_in_Average_T_G: int = Input(default=10),
        Labmda_AC: int = Input(default=20),
        Labmda_Patch_KL: float = Input(default=0.065),
        Preform_Noise_Correction: bool = Input(default=True),
    ) -> Path:
        output_image = main_pipeline(Input_Image,
                                    Source_Prompt,
                                    Target_Prompt,
                                    Denoise_Classifier_Free_Guidence_Scale,
                                    Number_of_ReNoise_Iterations,
                                    Inversion_Strength,
                                    Preform_Estimation_Averaging,
                                    First_Estimation_in_Average_T_L,
                                    Last_Estimation_in_Average_T_L,
                                    First_Estimation_in_Average_T_G,
                                    Last_Estimation_in_Average_T_G,
                                    Labmda_AC,
                                    Labmda_Patch_KL,
                                    Preform_Noise_Correction,
                                    self.model_type, self.scheduler_type, self.image_size, self.pipe_inversion, self.pipe_inference, self.cache_size, self.prev_configs,
                                    self.prev_inv_latents, self.prev_images, self.prev_noises)
        return Path(output_image)

import os
from cog import BasePredictor, Input, Path
from pyngrok import ngrok, conf

class Predictor(BasePredictor):
    def setup(self) -> None:
        directory = "/content"
        if not os.path.exists(directory):
            os.mkdir(directory)
    def predict(
        self,
        token: str = Input()
    ) -> str:
        conf.get_default().auth_token = token
        public_url = ngrok.connect(7860).public_url
        print(public_url)
        os.system(f"jupyter notebook --allow-root --port 7860 --ip 0.0.0.0 --NotebookApp.token '' --no-browse --notebook-dir /content")
        return public_url