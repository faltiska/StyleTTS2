import os
import phonemizer
from text_utils import TextCleaner

# Set environment variable for phonemizer
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"

corpus_dir = "e:/Programare/Corpus/Small"
output_dir = "Data-Small"
dict_path = "Data/word_index_dict.txt"
os.makedirs(output_dir, exist_ok=True)

text_cleaner = TextCleaner(dict_path)

# Initialize phonemizer backend
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us',
    preserve_punctuation=True,
    with_stress=True,
    language_switch='remove-flags'
)

def text_to_phonemes(text):
    phonemes = global_phonemizer.phonemize([text], strip=True)
    phonemes = phonemes[0]
    return phonemes

def process_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.split('|', 1)
                filename = parts[0]
                text = parts[1].strip()
                
                phonemes = text_to_phonemes(text)
                indices = text_cleaner(phonemes)
                indices_str = ' '.join(map(str, indices))
                
                f_out.write(f"{filename}.wav|{indices_str}|0\n")
    print(f"Created {output_file}")

process_csv(
    os.path.join(corpus_dir, "train.csv"),
    os.path.join(output_dir, "train_list.txt")
)

process_csv(
    os.path.join(corpus_dir, "validate.csv"),
    os.path.join(output_dir, "val_list.txt")
)

process_csv(
    os.path.join(corpus_dir, "OOD.csv"),
    os.path.join(output_dir, "OOD_texts.txt")
)

print(f"Done! Files saved to {output_dir}")
