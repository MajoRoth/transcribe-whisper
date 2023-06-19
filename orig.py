import whisper

model = whisper.load_model("large-v2")

# load audio and pad/trim it to fit 30 seconds
result = model.transcribe("/cs/labs/adiyoss/amitroth/vall-e/data/reference/saspeech/amit.wav")
print(result)

# print the recognized text
print(result.text)