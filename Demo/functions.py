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

        ps = self.global_phonemizer.phonemize([text])  # text -> list of phoneme
        print("1. List of phonemes: ", ps)
        ps = word_tokenize(ps[0])  # Split into individual tokens
        print("2. String of phonemes: ", ps)
        ps = ' '.join(ps)  # Join tokens together, split by a empty space
        print("3. Final string of phonemes: ", ps)

        tokens = self.textcleaner(ps)  # Look up numeric ID per phoneme
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

            t_en = self.model.text_encoder(tokens, input_lengths,
                                      text_mask)  # Creates text encoder (phoneme -> feature vectors)

            print("Features Vectors")
            for i in range(input_lengths):  # loop over phonemes/time steps
                feature_vec = t_en[0, :, i]  # shape [D]
                short_vec = feature_vec[:10]  # first 10 elements
                print(f"{i}. {short_vec}")

            if adversarial_audio_generation == 0:
                torch.save(t_en, "latents/h_text_target.pt")

            elif adversarial_audio_generation == 1:
                torch.save(t_en, "latents/h_text_ground_truth.pt")

            elif adversarial_audio_generation == 2:
                torch.save(t_en, "latents/h_text_ground_truth.pt")
                t_en_target = torch.load("latents/h_text_target.pt")

                print("t_en shape:", t_en.shape)
                print("t_en_target shape:", t_en_target.shape)

                t_en_target = F.interpolate(t_en_target, size=t_en.size(-1), mode='linear', align_corners=False)

                t_en = (1 - interpolation_percentage) * t_en + interpolation_percentage * t_en_target
                torch.save(t_en, "latents/h_text_interpolated.npy")

            elif adversarial_audio_generation == 3:
                t_en = torch.load("latents/h_text_ground_truth.pt")
                t_en_target = torch.load("latents/h_text_target.pt")

                print("t_en shape:", t_en.shape)
                print("t_en_target shape:", t_en_target.shape)

                t_en_target = F.interpolate(t_en_target, size=t_en.size(-1), mode='linear', align_corners=False)

                t_en = (1 - interpolation_percentage) * t_en + interpolation_percentage * t_en_target
                torch.save(t_en, "latents/h_text_interpolated.pt")

            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self.sampler(noise,
                             embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                             embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            h_aligned = t_en @ pred_aln_trg.unsqueeze(0).to(self.device)  # (B, D_text, T_frames)

            jitter_strength = 0.3  # try 0.1–0.5
            pred_dur = torch.round(pred_dur.float() + torch.randn_like(pred_dur.float()) * jitter_strength).clamp(
                min=1).long()

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)

            out = self.model.decoder(
                h_aligned,
                F0_pred,
                N_pred,
                ref.squeeze().unsqueeze(0)
            )

        return out.squeeze().cpu().numpy()
