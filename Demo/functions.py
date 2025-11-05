import os
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"  # <-- adjust if different

import torch # Deep Learning Framework
torch.manual_seed(0) # Fixes starting point of random seed for torch
torch.backends.cudnn.benchmark = False # Fix convolution algorithm
torch.backends.cudnn.deterministic = True # Only use deterministic algorithms

import soundfile as sf
from nltk.tokenize import word_tokenize # Tokenizers divide strings into lists of substrings
import time # Used for timing operations
import yaml

import torch.nn.functional as F
import whisper

from dataclasses import dataclass

from models import *
from utils import *
from text_utils import TextCleaner

import phonemizer

from Utils.PLBERT.util import load_plbert

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

def length_to_mask(lengths):
    mask = torch.arange(lengths.max())  # Creates a Vector [0,1,2,3,...,x], where x = biggest value in lengths
    mask = mask.unsqueeze(0)  # Creates a Matrix [1,x] from Vector [x]
    mask = mask.expand(lengths.shape[0],
                       -1)  # Expands the matrix from [1,x] to [y,x], where y = number of elements in lengths
    mask = mask.type_as(lengths)  # Assign mask the same type as lengths
    mask = torch.gt(mask + 1, lengths.unsqueeze(
        1))  # gt = greater than, compares each value from lengths to a row of values in mask; unsqueeze = splits vector lengths into vectors of size 1
    return mask  # returns a mask of shape (batch_size, max_length) where mask[i, j] = 1 if j < lengths[i] and mask[i, j] = 0 otherwise.

@dataclass
class InferenceResult:
    h_text: torch.Tensor
    h_aligned: torch.Tensor
    f0_pred: torch.Tensor
    a_pred: torch.Tensor
    n_pred: torch.Tensor
    style_vector_prosodic: torch.Tensor

    def save(self, folder: str):

        os.makedirs("outputs/latent/"+folder, exist_ok=True)

        # Iterate through all fields of the dataclass
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                path = os.path.join("outputs/latent/"+folder, f"{name}.pt")
                torch.save(value, path)
                print(f"✅ Saved {name} -> {path}")
            else:
                print(f"⚠️ Skipping {name} (not a tensor)")

class StyleTTS2_Helper:
    def __init__(self):

        # Splits words into phonemes (symbols that represent how words are pronounced)
        self.model = None
        self.params = None
        self.sampler = None

        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us',
            preserve_punctuation=True,  # Keeps Punctuation such as , . ? !
            with_stress=True  # Adds stress marks to vowels
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.textcleaner = TextCleaner()  # Lowercasing & trimming, expanding numbers & symbols, handling punctuation, phoneme conversion, tokenization

    def load_models(self, yml_path="Models/LJSpeech/config.yml"):
        config = yaml.safe_load(open(yml_path))  # YAML File with model settings and pretrained checkpoints (ASR, F0, PL-BERT)

        # load pretrained ASR (Automatic Speech Recognition) model
        ASR_config = config.get('ASR_config', False)  # YAML config that describes the model’s structure
        ASR_path = config.get('ASR_path', False)  # Checkpoint File
        text_aligner = load_ASR_models(ASR_path, ASR_config)  # Load PyTorch model

        # load pretrained F0 model (Extracts Pitch Features from Audio, How Pitch Changes over time)
        F0_path = config.get('F0_path', False)  # YAML config that describes the model’s structure
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model (encodes input text with prosodic cues)
        BERT_path = config.get('PLBERT_dir', False)  # YAML config that describes the model’s structure
        plbert = load_plbert(BERT_path)

        self.model = build_model(
            recursive_munch(config['model_params']),  # Allows attribute-style access to keys of model_params,
            text_aligner,  # Automatic Speech Recognition model
            pitch_extractor,  # F0 model
            plbert  # BERT model
        )

        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
        self.params = params_whole['net']

    def load_checkpoints(self):
        for key in self.model:
            if key in self.params:
                try:
                    self.model[key].load_state_dict(self.params[key])
                except:
                    from collections import OrderedDict
                    state_dict = self.params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [self.model[key].eval() for key in self.model]

    def sample_diffusion(self):
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

    # Turns text to tensor with token ID
    def preprocessText(self, text):
        # 1. Preprocessing Text
        text = text.strip()  # Removes whitespaces from beginning and end of string
        text = text.replace('"', '')  # removes " to prevent unpredictable behavior

        # 2. Text -> Phoneme
        phonemes = self.global_phonemizer.phonemize([text])  # text -> list of phoneme
        phonemes = word_tokenize(phonemes[0])  # Split into individual tokens
        phonemes = ' '.join(phonemes)  # Join tokens together, split by a empty space

        # 3. Phoneme -> Token ID
        tokens = self.textcleaner(phonemes)  # Look up numeric ID per phoneme
        tokens.insert(0, 0)  # Insert leading 0 to mark start

        # 4. Token ID -> PyTorch Tensor
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)  # Converts numeric ID to PyTorch Tensor

        return tokens

    def predictDuration(self, h_bert_with_style, input_lengths):

        # Duration Predictor, frames per phoneme
        d_pred, _ = self.model.predictor.lstm(h_bert_with_style)  # Model temporal dependencies between phonemes, LSTM = RNN
        d_pred = self.model.predictor.duration_proj(d_pred)  # Predict how long each phoneme lasts
        d_pred = torch.sigmoid(d_pred).sum(axis=-1)  # Sum of duration prediction -> Result: Prediction of frame duration
        d_pred = torch.round(d_pred.squeeze()).clamp(min=1)  # Convert duration prediction into integers, add clamp to ensure that each phoneme has at least one frame
        d_pred[-1] += 5  # Makes last phoneme last 5 frames longer, to ensure it not being cut off too fast

        # Creates predicted alignment matrix between text (phonemes) and audio frames
        a_pred = torch.zeros(input_lengths, int(d_pred.sum().data))  # Initializes a matrix with sizes: [# of Phonemes (input_lengths)] x [Sum of total predicted frames]
        current_frame = 0
        for i in range(a_pred.size(0)):  # Iterates over phoneme
            a_pred[i, current_frame:current_frame + int(d_pred[i].data)] = 1  # Changes for row-i (the i-th phoneme) all the values from current_frame to current_frame + int(d_pred[i].data) to 1
            current_frame += int(d_pred[i].data)  # Move current_frame to new first start

        return a_pred

    def computeStyleVector(self, noise, h_bert, embedding_scale, diffusion_steps):

        style_vector = self.sampler(
            noise,
            embedding=h_bert[0].unsqueeze(0),
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps
        ).squeeze(0)

        # Split Style Vector
        style_vector_acoustic = style_vector[:, 128:]  # Right Half = Acoustic Style Vector
        style_vector_prosodic = style_vector[:, :128]  # Left Half = Prosodic Style Vector

        return style_vector_acoustic, style_vector_prosodic

    def inference(self, text, noise=torch.randn(1,1,256), diffusion_steps=5, embedding_scale=1):

        # Ground Truth
        tokens = self.preprocessText(text)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)  # Number of phoneme / Length of tokens, shape[-1] = last element in list/array
            text_mask = length_to_mask(input_lengths).to(tokens.device)  # Creates a bitmask based on number of phonemes

            h_text = self.model.text_encoder(tokens, input_lengths, text_mask)  # Creates acoustic text encoder (phoneme -> feature vectors)
            h_bert = self.model.bert_encoder(self.model.bert(tokens, attention_mask=(~text_mask).int())).transpose(-1, -2)  # Creates prosodic text encoder (phoneme -> feature vectors)

            ## Function Call
            style_vector_acoustic, style_vector_prosodic = self.computeStyleVector(noise, h_bert, embedding_scale, diffusion_steps)

            # AdaIN, Adding information of style vector to phoneme
            h_bert_with_style = self.model.predictor.text_encoder(h_bert, style_vector_acoustic, input_lengths, text_mask)

            ## Function Call
            a_pred = self.predictDuration(h_bert_with_style, input_lengths)

            # Multiply alignment matrix with h_text
            h_aligned = h_text @ a_pred.unsqueeze(0).to(self.device)  # (B, D_text, T_frames)

            # encode prosody
            bert_encoder_with_style_per_frame = (h_bert_with_style.transpose(-1, -2) @ a_pred.unsqueeze(0).to(self.device))  # Multiply per-phoneme embedding (h_bert_with_style) with frame-per-phoneme matrix -> per-frame text embedding
            f0_pred, n_pred = self.model.predictor.F0Ntrain(bert_encoder_with_style_per_frame, style_vector_acoustic)

        return InferenceResult(
            h_text=h_text,
            h_aligned=h_aligned,
            f0_pred=f0_pred,
            a_pred=a_pred,
            n_pred=n_pred,
            style_vector_prosodic=style_vector_prosodic,
        )

    @torch.no_grad()
    def synthesizeSpeech(self, inferenceResult):

        with torch.no_grad():
            out = self.model.decoder(
                inferenceResult.h_aligned,
                inferenceResult.f0_pred,
                inferenceResult.n_pred,
                inferenceResult.style_vector_prosodic.squeeze().unsqueeze(0)
            )

        return out.squeeze().cpu().numpy()