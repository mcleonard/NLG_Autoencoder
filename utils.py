from collections import Counter
import random

import numpy as np

punc_tokens = {'!': ' <EXCLAIM> ',
 '.': ' <PERIOD> ',
 '?': ' <QMARK> ',
 ',': ' <COMMA> ',
 '(': ' <LPAREN> ',
 ')': ' <RPAREN> ',
 '"': ' <QUOTE> ',
 ';': ' <SEMICOLON> ',
 '\n': ' <RETURN> ',
 '\t': ' <TAB> ',
 '~': ' <TILDE> ',
 '-': ' <HYPHEN> ',
 '\'': ' <APOST> ',
 ':': ' <COLON> '
}


def replace_punctuation(dataset):
    return [''.join([punc_tokens.get(char, char) for char in seq]) for seq in dataset]


def extract_ngrams(sequence, n=2):
    """ Extract n-grams from a sequence """
    ngrams = list(zip(*[sequence[ii:] for ii in range(n)]))

    return ngrams


def corrupt(dataset, p_drop=0.6):
    """ Corrupt sequences in a dataset by randomly dropping words """
    values, counts = np.unique(np.concatenate(dataset), return_counts=True)
    to_drop = set(values[counts > 100])

    out_seq = [[each for each in seq if np.random.rand() > p_drop*int(each in to_drop)] for seq in dataset]

    return out_seq


def shuffle(original_seq, corrupted):
    """ Shuffle elements in a corrupted sequence while keeping bigrams
        appearing in original sequence.
    """

    if not corrupted:
        return corrupted

    # Need to swap words around now but keep bigrams
    # Get bigrams for original sequence
    seq_grams = extract_ngrams(original_seq)
    # Copy this
    cor = corrupted.copy()

    # Here I need to collect the tokens into n-grams that show up in the
    # original sequence. That way when I shuffle, 2-grams, 3-grams, etc
    # will stay together during the randomization.
    to_shuffle = [[cor.pop(0)]]
    while cor:
        if len(cor) == 1:
            to_shuffle.append([cor.pop()])
        elif (to_shuffle[-1][-1], cor[0]) not in seq_grams:
            to_shuffle.append([cor.pop(0)])
        else:
            to_shuffle[-1].append(cor.pop(0))

    random.shuffle(to_shuffle)
    flattened = [elem for lst in to_shuffle for elem in lst]
    return flattened


def get_tokens(dataset):
    # Tokenize our dataset
    corpus = " ".join(dataset)
    vocab_counter = Counter(corpus.split())
    vocab = vocab_counter.keys()
    total_words = sum(vocab_counter.values())

    vocab_freqs = {word: count/total_words for word, count in vocab_counter.items()}
    vocab_sorted = sorted(vocab, key=vocab_freqs.get, reverse=True)

    # Starting at 3 here to reserve special tokens
    vocab_to_int = dict(zip(vocab_sorted, range(3, len(vocab)+3)))
    
    vocab_to_int["<SOS>"] = 0 # Start of sentence
    vocab_to_int["<EOS>"] = 1 # End of sentence
    vocab_to_int["<UNK>"] = 2 # Unknown word
    
    int_to_vocab = {val: key for key, val in vocab_to_int.items()}
    
    return vocab_to_int, int_to_vocab