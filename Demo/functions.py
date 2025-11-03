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

from models import *
from utils import *
from text_utils import TextCleaner

import phonemizer

from Utils.PLBERT.util import load_plbert

from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule



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

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max())  # Creates a Vector [0,1,2,3,...,x], where x = biggest value in lengths
        mask = mask.unsqueeze(0)  # Creates a Matrix [1,x] from Vector [x]
        mask = mask.expand(lengths.shape[0],
                           -1)  # Expands the matrix from [1,x] to [y,x], where y = number of elements in lengths
        mask = mask.type_as(lengths)  # Assign mask the same type as lengths
        mask = torch.gt(mask + 1, lengths.unsqueeze(
            1))  # gt = greater than, compares each value from lengths to a row of values in mask; unsqueeze = splits vector lengths into vectors of size 1
        return mask  # returns a mask of shape (batch_size, max_length) where mask[i, j] = 1 if j < lengths[i] and mask[i, j] = 0 otherwise.

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
                print('%s loaded' % key)
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

    def inference(self, text, adversarial_audio_generation, interpolation_percentage, noise=torch.randn(1,1,256), diffusion_steps=5, embedding_scale=1):
        text = text.strip()  # Removes whitespaces from beginning and end of string
        text = text.replace('"', '')  # removes " to prevent unpredictable behavior

        phonemes = self.global_phonemizer.phonemize([text])  # text -> list of phoneme
        print("1. List of phonemes: ", phonemes)
        phonemes = word_tokenize(phonemes[0])  # Split into individual tokens
        print("2. String of phonemes: ", phonemes)
        phonemes = ' '.join(phonemes)  # Join tokens together, split by a empty space
        print("3. Final string of phonemes: ", phonemes)

        tokens = self.textcleaner(phonemes)  # Look up numeric ID per phoneme
        print("4. ID of phonemes: ", tokens)
        tokens.insert(0, 0)  # Insert leading 0 to mark start
        print("5. ID with leading 0: ", tokens)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)  # Converts numeric ID to PyTorch Tensor
        print("6. Pytorch Tensor Dimension: ", tokens.shape)

        with torch.no_grad():  # No training, so no gradient-descent / backpropagation
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(
                tokens.device)  # Number of phoneme / Length of tokens, shape[-1] = last element in list/array
            text_mask = length_to_mask(input_lengths).to(tokens.device)  # Creates a bitmask based on number of phonemes
            print("Text Mask: ", text_mask)

            h_text = self.model.text_encoder(tokens, input_lengths,
                                        text_mask)  # Creates text encoder (phoneme -> feature vectors)
            h_bert = self.model.bert(tokens, attention_mask=(~text_mask).int())
            bert_encoder = self.model.bert_encoder(h_bert).transpose(-1, -2)

            print("Features Vectors")
            for i in range(input_lengths):  # loop over phonemes/time steps
                feature_vec = h_text[0, :, i]  # shape [D]
                short_vec = feature_vec[:10]  # first 10 elements
                print(f"{i}. {short_vec}")

            if adversarial_audio_generation == 0:
                torch.save(h_text, "latents/h_text_target.pt")

            elif adversarial_audio_generation == 1:
                torch.save(h_text, "latents/h_text_ground_truth.pt")

            elif adversarial_audio_generation == 2:
                torch.save(h_text, "latents/h_text_ground_truth.pt")
                h_text_target = torch.load("latents/h_text_target.pt")

                print("h_text shape:", h_text.shape)
                print("h_text_target shape:", h_text_target.shape)

                h_text_target = F.interpolate(h_text_target, size=h_text.size(-1), mode='linear', align_corners=False)

                h_text = (1 - interpolation_percentage) * h_text + interpolation_percentage * h_text_target
                torch.save(h_text, "latents/h_text_interpolated.pt")

            elif adversarial_audio_generation == 3:
                h_text = torch.load("latents/h_text_ground_truth.pt")
                h_text_target = torch.load("latents/h_text_target.pt")

                print("t_en shape:", h_text.shape)
                print("t_en_target shape:", h_text_target.shape)

                h_text_target = F.interpolate(h_text_target, size=h_text.size(-1), mode='linear', align_corners=False)

                h_text = (1 - interpolation_percentage) * h_text + interpolation_percentage * h_text_target
                torch.save(h_text, "latents/h_text_interpolated.pt")

            s_pred = self.sampler(
                noise,
                embedding=h_bert[0].unsqueeze(0),
                embedding_scale=embedding_scale,
                num_steps=diffusion_steps
            ).squeeze(0)

            # Split Style Vector
            style_vector_acoustic = s_pred[:, 128:]  # Right Half = Acoustic Style Vector
            style_vector_prosodic = s_pred[:, :128]  # Left Half = Prosodic Style Vector

            # AdaIN, Adding information of style vector to phoneme
            bert_encoder_with_style = self.model.predictor.text_encoder(bert_encoder, style_vector_acoustic, input_lengths,
                                                                   text_mask)

            x, _ = self.model.predictor.lstm(
                bert_encoder_with_style)  # Model temporal dependencies between phonemes, LSTM = RNN
            duration = self.model.predictor.duration_proj(x)  # Predict how long each phoneme lasts
            duration = torch.sigmoid(duration).sum(axis=-1)  # Sum of duration prediction -> Result: Prediction of frame duration
            d_pred = torch.round(duration.squeeze()).clamp(
                min=1)  # Convert duration prediction into integers, add clamp to ensure that each phoneme has at least one frame

            d_pred[-1] += 5  # Makes last phoneme last 5 frames longer, to ensure it not being cut off too fast

            # Creates predicted alignment matrix between text (phonemes) and audio frames
            a_pred = torch.zeros(input_lengths,
                                 int(d_pred.sum().data))  # Initializes a matrix with sizes: [# of Phonemes (input_lengths)] x [Sum of total predicted frames]
            c_frame = 0
            for i in range(a_pred.size(0)):  # Iterates over phoneme
                a_pred[i, c_frame:c_frame + int(d_pred[
                                                    i].data)] = 1  # Changes for row-i (the i-th phoneme) all the values from c_frame to c_frame + int(d_pred[i].data) to 1
                c_frame += int(d_pred[i].data)  # Move c_frame to new first start

            # Multiply alignment matrix with h_text
            h_aligned = h_text @ a_pred.unsqueeze(0).to(self.device)  # (B, D_text, T_frames)

            # encode prosody
            en = (bert_encoder_with_style.transpose(-1, -2) @ a_pred.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, style_vector_acoustic)

            out = self.model.decoder(
                h_aligned,
                F0_pred,
                N_pred,
                style_vector_prosodic.squeeze().unsqueeze(0)
            )

        return out.squeeze().cpu().numpy()
