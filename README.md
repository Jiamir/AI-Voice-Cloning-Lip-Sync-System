# AI-Voice-Cloning-Lip-Sync-System
AI-based voice cloning and lip-sync pipeline using the Tortoise Text-to-Speech (TTS) model combined with the Wav2Lip model. 
<br>
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
