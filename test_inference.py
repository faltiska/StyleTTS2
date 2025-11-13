#!/usr/bin/env python3
"""
StyleTTS2 Text-to-Speech Script
Simple text to speech conversion using StyleTTS2
"""

import argparse
import os
import sys
import time

import phonemizer
import soundfile as sf
from nltk.tokenize import word_tokenize

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# Set environment variable for phonemizer
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def length_to_mask(lengths):
    mask = torch.arange(lengths.max())
    mask = mask.unsqueeze(0)
    mask = mask.expand(lengths.shape[0], -1)
    mask = mask.type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

class StyleTTS2_Helper:
    def __init__(self):
        self.model = None
        self.params = None
        self.sampler = None
        
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us',
            preserve_punctuation=True,
            with_stress=True
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.textcleaner = TextCleaner()

    def load_models(self, yml_path="Models/LJSpeech/config.yml", checkpoint_path="Models/LJSpeech/epoch_1st_00004.pth"):
        config = yaml.safe_load(open(yml_path))
        
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        text_aligner = load_ASR_models(ASR_path, ASR_config)
        
        F0_path = config.get('F0_path', False)
        pitch_extractor = load_F0_models(F0_path)
        
        BERT_path = config.get('PLBERT_dir', False)
        plbert = load_plbert(BERT_path)
        
        self.model = build_model(
            recursive_munch(config['model_params']),
            text_aligner,
            pitch_extractor,
            plbert
        )
        
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]
        
        params_whole = torch.load(checkpoint_path, map_location='cpu')
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
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

    def sample_diffusion(self):
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )

    def inference(self, text, noise, diffusion_steps=5, embedding_scale=1):
        tokens = self.preprocessText(text)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)
            
            h_text = self.model.text_encoder(tokens, input_lengths, text_mask)
            h_bert = self.model.bert(tokens, attention_mask=(~text_mask).int())
            bert_encoder = self.model.bert_encoder(h_bert).transpose(-1, -2)
            
            style_vector_acoustic, style_vector_prosodic = self.computeStyleVector(noise, h_bert, embedding_scale, diffusion_steps)
            
            bert_encoder_with_style = self.model.predictor.text_encoder(bert_encoder, style_vector_acoustic, input_lengths, text_mask)
            
            a_pred = self.predictDuration(bert_encoder_with_style, input_lengths)
            
            h_aligned = h_text @ a_pred.unsqueeze(0).to(self.device)
            
            bert_encoder_with_style_per_frame = (bert_encoder_with_style.transpose(-1, -2) @ a_pred.unsqueeze(0).to(self.device))
            f0_pred, n_pred = self.model.predictor.F0Ntrain(bert_encoder_with_style_per_frame, style_vector_acoustic)
            
            out = self.model.decoder(
                h_aligned,
                f0_pred,
                n_pred,
                style_vector_prosodic.squeeze().unsqueeze(0)
            )
        
        return out.squeeze().cpu().numpy()

    def preprocessText(self, text):
        text = text.strip()
        text = text.replace('"', '')
        
        phonemes = self.global_phonemizer.phonemize([text])
        phonemes = word_tokenize(phonemes[0])
        phonemes = ' '.join(phonemes)
        
        tokens = self.textcleaner(phonemes)
        tokens.insert(0, 0)
        
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        return tokens

    def predictDuration(self, bert_encoder_with_style, input_lengths):
        d_pred, _ = self.model.predictor.lstm(bert_encoder_with_style)
        d_pred = self.model.predictor.duration_proj(d_pred)
        d_pred = torch.sigmoid(d_pred).sum(axis=-1)
        d_pred = torch.round(d_pred.squeeze()).clamp(min=1)
        d_pred[-1] += 5
        
        a_pred = torch.zeros(input_lengths, int(d_pred.sum().data))
        current_frame = 0
        for i in range(a_pred.size(0)):
            a_pred[i, current_frame:current_frame + int(d_pred[i].data)] = 1
            current_frame += int(d_pred[i].data)
        
        return a_pred

    def computeStyleVector(self, noise, h_bert, embedding_scale, diffusion_steps):
        style_vector = self.sampler(
            noise,
            embedding=h_bert[0].unsqueeze(0),
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps
        ).squeeze(0)
        
        style_vector_acoustic = style_vector[:, 128:]
        style_vector_prosodic = style_vector[:, :128]
        
        return style_vector_acoustic, style_vector_prosodic

def main():
    parser = argparse.ArgumentParser(description='StyleTTS2 Text-to-Speech')
    parser.add_argument('text', help='Text to convert to speech')
    parser.add_argument('--output', default='output.wav', help='Output audio file')
    parser.add_argument('--checkpoint', default='Models/LJSpeech/epoch_1st_00002.pth', help='Checkpoint file to load')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    print("Loading StyleTTS2 models...")
    pipe = StyleTTS2_Helper()
    print(f"Using device: {pipe.device}")
    pipe.load_models(checkpoint_path=args.checkpoint)
    pipe.load_checkpoints()
    pipe.sample_diffusion()
    print("Models loaded successfully!")
    
    print(f"Converting text to speech...")
    noise = torch.randn(1, 1, 256).to(pipe.device)
    
    start_time = time.time()
    audio = pipe.inference(args.text, noise)
    inference_time = time.time() - start_time
    
    audio_duration = len(audio) / 24000
    rtf = inference_time / audio_duration
    
    sf.write(args.output, audio, samplerate=24000)
    print(f"{audio_duration:.3f}s audio saved to: {args.output}, {inference_time=:.3f}s, {rtf=:.3f}x")

if __name__ == "__main__":
    main()