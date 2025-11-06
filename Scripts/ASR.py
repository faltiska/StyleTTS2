import whisper
model = whisper.load_model("tiny")
result = model.transcribe("../outputs/Obama1.wav")
print(result["text"])
