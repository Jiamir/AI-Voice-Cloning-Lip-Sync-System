# ============================================================
# Tortoise + Wav2Lip Voice Cloning & Lip Sync Pipeline
# Description: This pipeline performs AI voice cloning using
# Tortoise TTS and lip-syncs a face video using Wav2Lip.
# ============================================================

# -----------------------------------------------
# STEP 1: Install dependencies and clone repos
# -----------------------------------------------
!pip3 install -U scipy
!git clone https://github.com/neonbjb/tortoise-tts
!pip install -r tortoise-tts/requirements.txt
!pip install -e tortoise-tts/
!pip3 install transformers==4.26.1 einops==0.5.0 rotary_embedding_torch==0.1.5 unidecode==1.3.5
!python3 tortoise-tts/setup.py install

# OPTIONAL: Uninstall if previously installed incorrectly
!pip uninstall -y TorToiSe

# -----------------------------------------------
# STEP 2: Import libraries
# -----------------------------------------------
import os
import torch
import torchaudio
import IPython
from time import time

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
from tortoise.utils.text import split_and_recombine_text

# -----------------------------------------------
# STEP 3: Upload and prepare custom voice
# -----------------------------------------------
from google.colab import files

CUSTOM_VOICE_NAME = "angelina"
custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
os.makedirs(custom_voice_folder, exist_ok=True)

for i, file_data in enumerate(files.upload().values()):
    with open(os.path.join(custom_voice_folder, f'{i}.wav'), 'wb') as f:
        f.write(file_data)

# -----------------------------------------------
# STEP 4: Generate TTS output using Tortoise
# -----------------------------------------------
tts = TextToSpeech()

# Prepare input text
input_text = (
    "The ongoing tensions between Pakistan and India are deeply concerning. "
    "As someone who has witnessed the consequences of conflict around the world, "
    "I urge both nations to prioritize dialogue and diplomacy over hostility."
)

# Save and load text
with open("text.txt", "w", encoding="utf-8") as f:
    f.write(input_text)

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# Split if long
texts = split_and_recombine_text(text)

# Load custom voice reference
ref_audio_path = "AngelinaJolie.wav"  # Upload beforehand
voice_samples = [load_audio(ref_audio_path, 22050)]
conditioning_latents = tts.get_conditioning_latents(voice_samples)

# Generate audio
os.makedirs("results/longform", exist_ok=True)
all_parts = []
for j, part in enumerate(texts):
    gen = tts.tts_with_preset(part, voice_samples=voice_samples,
                               conditioning_latents=conditioning_latents,
                               preset="fast", k=1)
    gen = gen.squeeze(0).cpu()
    torchaudio.save(f"results/longform/{j}.wav", gen, 24000)
    all_parts.append(gen)

# Merge output audio
final_audio = torch.cat(all_parts, dim=-1)
final_audio_path = "results/longform/AngelinaJolie_TTS.wav"
torchaudio.save(final_audio_path, final_audio, 24000)

# Play output
IPython.display.Audio(final_audio_path)

# -----------------------------------------------
# STEP 5: Clone and set up Wav2Lip
# -----------------------------------------------
!git clone https://github.com/justinjohn0306/Wav2Lip
%cd Wav2Lip/Wav2Lip/

!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth' -O 'checkpoints/wav2lip.pth'
!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth' -O 'checkpoints/wav2lip_gan.pth'
!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/resnet50.pth' -O 'checkpoints/resnet50.pth'
!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/mobilenet.pth' -O 'checkpoints/mobilenet.pth'

!pip install batch-face==1.5.0
!pip install git+https://github.com/elliottzheng/batch-face.git@master
!pip install ffmpeg-python mediapipe==0.10.18

# -----------------------------------------------
# STEP 6: Run inference with lip-sync
# -----------------------------------------------
!python inference.py \
  --checkpoint_path checkpoints/wav2lip.pth \
  --face /content/Angelina.mp4 \
  --audio /content/tortoise-tts/results/longform/AngelinaJolie_TTS.wav

# -----------------------------------------------
# STEP 7: Download result
# -----------------------------------------------
from google.colab import files
files.download('results/result_output.mp4')
