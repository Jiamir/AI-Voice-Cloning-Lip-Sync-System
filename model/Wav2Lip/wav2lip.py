# ================================
# Wav2Lip – Lip Sync Pipeline
# ================================

# 🔧 Setup
!git clone https://github.com/justinjohn0306/Wav2Lip
%cd /content/Wav2Lip/Wav2Lip/

!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip.pth' -O 'checkpoints/wav2lip.pth'
!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/wav2lip_gan.pth' -O 'checkpoints/wav2lip_gan.pth'
!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/resnet50.pth' -O 'checkpoints/resnet50.pth'
!wget 'https://github.com/justinjohn0306/Wav2Lip/releases/download/models/mobilenet.pth' -O 'checkpoints/mobilenet.pth'

!pip install batch-face==1.5.0
!pip install git+https://github.com/elliottzheng/batch-face.git@master
!pip install ffmpeg-python mediapipe==0.10.18
a = !pip install https://raw.githubusercontent.com/AwaleSajil/ghc/master/ghc-1.0-py3-none-any.whl

# 🎬 Run Inference
!python inference.py \
  --checkpoint_path checkpoints/wav2lip.pth \
  --face /content/Angelina.mp4 \
  --audio /content/tortoise-tts/results/longform/AngelinaJolie_TTS.wav

# 📥 Download the result
from google.colab import files
files.download('/content/Wav2Lip/Wav2Lip/results/result_output.mp4')
