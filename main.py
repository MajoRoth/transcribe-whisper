import torchaudio as torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
# from datasets import load_dataset

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="hebrew", task="transcribe")

# load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
# sample = ds[0]["audio"]
wav, sr = torchaudio.load("/Users/amitroth/PycharmProjects/vall-e/data/reference/saspeech/amit.wav")
print("WAV")
print(wav)
input_features = processor(wav, sampling_rate=sr, return_tensors="pt").input_features

print("input_features")
print(input_features)

# generate token ids
predicted_ids = model.generate(input_features)

print("predicted_ids")
print(predicted_ids)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

print("transcription")
print(transcription)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

print("transcription")
print(transcription)
