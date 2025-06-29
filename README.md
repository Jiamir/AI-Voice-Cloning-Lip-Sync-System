# AI Voice Cloning & Lip Sync System
This project implements an AI-based voice cloning and lip-sync system using the Tortoise Text-to-Speech (TTS) model combined with the Wav2Lip model. Given input text and a reference voice sample, the system generates high-quality synthesized speech in the cloned voice, then produces a lip-synced talking-head video by generating a face video to match the generated audio. The pipeline is designed to run efficiently in Google Colab, enabling easy experimentation and customization. <br>

--------

### 🔊 Tortoise TTS
Purpose: High-quality voice synthesis with voice cloning.
Pipeline:
- Input: Text + reference WAV (e.g., AngelinaJolie.wav)
- Text splitting: Large input text is split into manageable chunks.
- Voice synthesis: Generates speech in the reference voice.
- Output: Synthesized audio WAV file. <br>

### 🗣️ Wav2Lip
Purpose: Lip-syncing a face video to the generated audio.
Pipeline:
- Input: Face video (with clear lips) + synthesized audio (from Tortoise)
- Model: Pre-trained Wav2Lip model
- Output: Talking-head video where lip movement matches the audio.

### **Team Members**
Developed by [Jaweria Amir](https://github.com/Jiamir), [Farwa Attaria](https://github.com/farwaattaria), [Mah Jabeen](https://github.com/mahjabeenhere)
