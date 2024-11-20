import random
import spacy
import torch
import nltk

from googletrans import Translator
from transformers import pipeline
from nltk.corpus import wordnet
from parrot import Parrot
from random import choice

parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=torch.cuda.is_available())
paraphraser = pipeline("text2text-generation", model="t5-small", device=0 if torch.cuda.is_available() else -1)
nlp = spacy.load("en_core_web_sm")
nltk.download("wordnet")
nltk.download('omw-1.4')

translator = Translator()


def back_translation_two_languages(text: str, languages: list = ["ja", "ar", "zh-CN", "sw", "ko"]) -> str:
    """
    Back translates the text through two randomly chosen intermediate languages.
    
    Args:
        text (str): The input text to be paraphrased.
        languages (list): List of intermediate language codes.
    
    Returns:
        str: The back-translated text.
    """
    try:
        # Pick two distinct languages
        lang1, lang2 = random.sample(languages, 2)
        
        # Step 1: Translate to the first intermediate language
        translated_1 = translator.translate(text, src='en', dest=lang1).text
        
        # Step 2: Translate from the first to the second intermediate language
        translated_2 = translator.translate(translated_1, src=lang1, dest=lang2).text
        
        # Step 3: Translate back to English
        back_translated = translator.translate(translated_2, src=lang2, dest='en').text
        
        return back_translated
    except Exception as e:
        print(f"Error during back translation: {e}")
        return text

def paraphrase_text(text: str) -> str:
    if parrot:
        paraphrases = parrot.augment(input_phrase=text, max_return_phrases=5, use_gpu=True, do_diverse = True, adequacy_threshold = 0.7)
        if paraphrases:
            return paraphrases[0][0]
    return text


def synonym_replacement(text: str) -> str:
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name()
            new_words.append(synonym)
        else:
            new_words.append(word)
    return " ".join(new_words)


def synonym_insertion(text: str) -> str:
    # Tokenize and process the input text
    doc = nlp(text)
    words = text.split()
    
    # Select a random token that is a noun, adjective, verb, or adverb
    candidates = [token for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']]
    
    if not candidates:
        return text  # Return the original text if no suitable word is found
    
    selected_token = choice(candidates)
    synonyms = wordnet.synsets(selected_token.text)
    
    # Find a suitable synonym that is not the same as the original word
    synonym_candidates = [
        lemma.name().replace('_', ' ')
        for synset in synonyms for lemma in synset.lemmas()
        if lemma.name().lower() != selected_token.text.lower()
    ]
    
    if not synonym_candidates:
        return text  # Return the original text if no synonyms are found
    
    synonym = choice(synonym_candidates)
    
    # Insert the synonym near the original word
    position = selected_token.i
    words.insert(position + 1, synonym)
    
    return " ".join(words)


def apply_augmentations(text: str, augmentations: list = [paraphrase_text, synonym_replacement, synonym_insertion]) -> str:
    """
    Applies a list of augmentation functions to the input text and returns a set of unique results.
    
    Args:
        text (str): The input text to be augmented.
        augmentations (list): A list of augmentation functions to apply.
        
    Returns:
        set: A set of unique augmented texts.
    """
    for augmentation in augmentations:
        try:
            text = augmentation(text)
        except Exception as e:
            print(f"Augmentation {augmentation.__name__} failed: {e}")
    return text
