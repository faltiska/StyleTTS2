# Enum adversarial_audio_generation
# 0 = generate and save target
# 1 = generate and save ground truth
# 2 = generate ground truth and load target
# 3 = load ground truth and load target
from Demo.functions import InferenceResult, length_to_mask
from functions import StyleTTS2_Helper
import soundfile as sf
import torch
import torch.nn.functional as F

import os,sys

# Add project root (one level up) to sys.path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# Change working directory to project root
os.chdir(os.path.dirname(os.path.dirname(__file__)))

def analyzeAudio(name):
    model = whisper.load_model("tiny")
    result = model.transcribe(audio=name)
    print(result["text"])

    # let's inspect segments
    for seg in result["segments"]:
        print(
            f"[{seg['start']:.2f} -> {seg['end']:.2f}] "
            f"text='{seg['text']}' "
            f"avg_logprob={seg['avg_logprob']:.3f} "
            f"no_speech_prob={seg['no_speech_prob']:.3f}"
        )

def generateAudio(pipe, name, text):

    inferenceResult = pipe.inference(text, noise=torch.randn(1,1,256).to(pipe.device))
    inferenceResult.save(name)
    audio = pipe.synthesizeSpeech(inferenceResult)
    sf.write("outputs/audio/" + name + ".wav", audio, samplerate=24000)
    return inferenceResult

def interpolateAllLatents(ground_truth, target, interpolation_percentage):

    interpolation_result = {}

    for name in ground_truth.__dataclass_fields__:

        latent_ground_truth = getattr(ground_truth, name)
        latent_target = getattr(target, name)

        if name != "h_text":
            interpolation_result[name] = latent_ground_truth
            continue

        print("Starting interpolation for " + name)
        if latent_ground_truth.shape != latent_target.shape:
            print(f"Shape mismatch with ground_truth={latent_ground_truth.shape}, target={latent_target.shape}")

            if (latent_ground_truth.dim() < 3) and (latent_target.dim() < 3):
                latent_ground_truth = latent_ground_truth.unsqueeze(1)
                latent_target = latent_target.unsqueeze(1)

            latent_target = F.interpolate(
                input=latent_target,
                size=latent_ground_truth.shape[-1],
                mode="linear",
                align_corners=False
            ).squeeze(0)
        interpolation_result[name] = latent_ground_truth * (1 - interpolation_percentage) + latent_target * interpolation_percentage

    return InferenceResult(**interpolation_result)

def interpolateLatent(ground_truth: InferenceResult, target: InferenceResult, interpolation_percentage: int, latent: str):

    latent_ground_truth = getattr(ground_truth, latent)
    latent_target = getattr(target, latent)

    print("Starting interpolation for " + latent)
    if latent_ground_truth.shape != latent_target.shape:
        print(f"Shape mismatch with ground_truth={latent_ground_truth.shape}, target={latent_target.shape}")

        if (latent_ground_truth.dim() < 3) and (latent_target.dim() < 3):
            latent_ground_truth = latent_ground_truth.unsqueeze(1)
            latent_target = latent_target.unsqueeze(1)

        latent_target = F.interpolate(
            input=latent_target,
            size=latent_ground_truth.shape[-1],
            mode="linear",
            align_corners=False
        ).squeeze(0)

    return latent_ground_truth * (1 - interpolation_percentage) + latent_target * interpolation_percentage

def main():

    embedding_scale = 5
    diffusion_steps = 1

    interpolation_percentage = 0.8 # How much of Target to be used, small interpolation_percentage means more of ground_truth (Minimization)

    name_gt = "ground_truth"
    text_gt = "Munich is one of the best cities to live in."

    name_target = "target"
    text_target = "No way this really works."

    pipe = StyleTTS2_Helper()

    noise = torch.randn(1, 1, 256).to(pipe.device)

    pipe.load_models()  # builds self.model and loads self.params
    pipe.load_checkpoints()  # puts params into self.model
    pipe.sample_diffusion()  # builds self.sampler

    inferenceResult = pipe.inference(text_gt, noise)

    audio = pipe.synthesizeSpeech(inferenceResult)

    sf.write("outputs/audio/ground_truth.wav", audio, samplerate=24000)

    """
    inferenceResult_interpolated = interpolateAllLatents(inferenceResult_groundTruth, inferenceResult_target, interpolation_percentage)
    inferenceResult_interpolated.save("interpolated")
    audio = pipe.synthesizeSpeech(inferenceResult_interpolated)
    sf.write("outputs/audio/interpolated.wav", audio, samplerate=24000)
    """

    #analyzeAudio(audio)


if __name__ == "__main__":
    main()