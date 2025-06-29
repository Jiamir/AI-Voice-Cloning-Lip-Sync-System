# ================================
# Tortoise TTS – Voice Cloning Pipeline
# ================================

# 🔧 Setup
!pip3 install -U scipy
!git clone https://github.com/neonbjb/tortoise-tts
%cd tortoise-tts/
!pip install -r requirements.txt
!pip install -e .
!pip3 install transformers==4.26.1 einops==0.5.0 rotary_embedding_torch==0.1.5 unidecode==1.3.5
!python3 setup.py install
!pip uninstall -y TorToiSe

# 📦 Imports
import os
import torch
import torchaudio
import IPython
from time import time
from google.colab import files

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio
from tortoise.utils.text import split_and_recombine_text

# 🎙️ Load custom voice samples
CUSTOM_VOICE_NAME = "angelina"
custom_voice_folder = f"tortoise/voices/{CUSTOM_VOICE_NAME}"
os.makedirs(custom_voice_folder, exist_ok=True)

for i, file_data in enumerate(files.upload().values()):
    with open(os.path.join(custom_voice_folder, f'{i}.wav'), 'wb') as f:
        f.write(file_data)

# 📝 Prepare input text
input_text = (
    "The ongoing tensions between Pakistan and India are deeply concerning. "
    "As someone who has witnessed the consequences of conflict around the world, "
    "I urge both nations to prioritize dialogue and diplomacy over hostility."
)
with open("text.txt", "w", encoding="utf-8") as f:
    f.write(input_text)

with open("text.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

texts = split_and_recombine_text(text)

# 🧠 Generate speech
tts = TextToSpeech()
ref_audio_path = "AngelinaJolie.wav"
voice_samples = [load_audio(ref_audio_path, 22050)]
conditioning_latents = tts.get_conditioning_latents(voice_samples)

outpath = "results/longform"
os.makedirs(outpath, exist_ok=True)
all_parts = []

for j, part in enumerate(texts):
    gen = tts.tts_with_preset(part, voice_samples=voice_samples,
                               conditioning_latents=conditioning_latents,
                               preset="fast", k=1)
    gen = gen.squeeze(0).cpu()
    torchaudio.save(os.path.join(outpath, f'{j}.wav'), gen, 24000)
    all_parts.append(gen)

# 🔊 Merge audio
full_audio = torch.cat(all_parts, dim=-1)
final_path = os.path.join(outpath, "AngelinaJolie_TTS.wav")
torchaudio.save(final_path, full_audio, 24000)

# ▶️ Play the final audio
IPython.display.Audio(final_path)
