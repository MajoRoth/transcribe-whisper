import whisper

model = whisper.load_model("large-v2")

# load audio and pad/trim it to fit 30 seconds
try:
    print("Hebrew")
    result = model.transcribe("/cs/labs/adiyoss/amitroth/vall-e/data/reference/saspeech/amit.wav", language='Hebrew')
    print(result)

    # print the recognized text
    print(result.text)

except Exception as e:
    print(e)

# load audio and pad/trim it to fit 30 seconds
try:
    print("hebrew")
    result = model.transcribe("/cs/labs/adiyoss/amitroth/vall-e/data/reference/saspeech/amit.wav", language='hebrew')
    print(result)

    # print the recognized text
    print(result.text)

except Exception as e:
    print(e)