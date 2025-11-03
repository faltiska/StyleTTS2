# Enum adversarial_audio_generation
# 0 = generate and save target
# 1 = generate and save ground truth
# 2 = generate ground truth and load target
# 3 = load ground truth and load target

from functions import StyleTTS2_Helper
import soundfile as sf
import torch

import whisper

adversarial_audio_generation = 2

interpolation_percentage = 0.8

text = ''' No way this really works. '''

import os,sys

# Add project root (one level up) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# Change working directory to project root
os.chdir(os.path.dirname(os.path.dirname(__file__)))

def main():
    pipe = StyleTTS2_Helper()
    pipe.load_models()  # builds self.model and loads self.params
    pipe.load_checkpoints()  # puts params into self.model
    pipe.sample_diffusion()  # builds self.sampler
    wav = pipe.inference(text, adversarial_audio_generation, interpolation_percentage, noise=torch.randn(1,1,256).to(pipe.device))

    if adversarial_audio_generation == 0:
        name = "target.wav"
    elif adversarial_audio_generation == 1:
        name = "ground_truth.wav"
    else:
        name = "ground_truth_interpolated.wav"

    name = "outputs/" + name

    sf.write(name, wav, samplerate=24000)

    model = whisper.load_model("tiny")
    result = model.transcribe(name)
    print(result["text"])

    # let's inspect segments
    for seg in result["segments"]:
        print(
            f"[{seg['start']:.2f} -> {seg['end']:.2f}] "
            f"text='{seg['text']}' "
            f"avg_logprob={seg['avg_logprob']:.3f} "
            f"no_speech_prob={seg['no_speech_prob']:.3f}"
        )


if __name__ == "__main__":
    main()