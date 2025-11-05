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

    diffusion_steps = 5
    embedding_scale = 1

    functions = True

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


    # Ground Truth
    tokens = pipe.preprocessText(text_target)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)  # Number of phoneme / Length of tokens, shape[-1] = last element in list/array
        text_mask = length_to_mask(input_lengths).to(tokens.device)  # Creates a bitmask based on number of phonemes

        h_text = pipe.model.text_encoder(tokens, input_lengths, text_mask)  # Creates acoustic text encoder (phoneme -> feature vectors)
        h_bert = pipe.model.bert(tokens, attention_mask=(~text_mask).int())
        bert_encoder = pipe.model.bert_encoder(h_bert).transpose(-1, -2)  # Creates prosodic text encoder (phoneme -> feature vectors)

        ## Function Call
        style_vector_acoustic, style_vector_prosodic = pipe.computeStyleVector(noise, h_bert, embedding_scale, diffusion_steps)

        # AdaIN, Adding information of style vector to phoneme
        bert_encoder_with_style = pipe.model.predictor.text_encoder(bert_encoder, style_vector_acoustic, input_lengths, text_mask)

        ## Function Call
        a_pred = pipe.predictDuration(bert_encoder_with_style, input_lengths)

        # Multiply alignment matrix with h_text
        h_aligned = h_text @ a_pred.unsqueeze(0).to(pipe.device)  # (B, D_text, T_frames)

        # encode prosody
        bert_encoder_with_style_per_frame = (bert_encoder_with_style.transpose(-1, -2) @ a_pred.unsqueeze(0).to(pipe.device))  # Multiply per-phoneme embedding (bert_encoder_with_style) with frame-per-phoneme matrix -> per-frame text embedding
        f0_pred, n_pred = pipe.model.predictor.F0Ntrain(bert_encoder_with_style_per_frame, style_vector_acoustic)

    inferenceResult = InferenceResult(
        h_text=h_text,
        h_aligned=h_aligned,
        f0_pred=f0_pred,
        a_pred=a_pred,
        n_pred=n_pred,
        style_vector_prosodic=style_vector_prosodic,
    )

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