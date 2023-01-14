"""
Prepare vocabulary and initial word vectors.
"""
import pickle
import argparse
from collections import Counter
import pandas as pd


class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<unk>']):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)  # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', default='../data', help='TACRED directory.')
    parser.add_argument('--vocab_dir', default='../data', help='Output vocab directory.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # input files
    train_file = args.data_dir + '/mosi_train_df_pos.csv'
    val_file = args.data_dir + '/mosi_val_df_pos.csv'
    test_file = args.data_dir + '/mosi_test_df_pos.csv'

    # output files

    # pos_tag
    vocab_pos_file = args.vocab_dir + '/vocab_pos.vocab'

    # load files
    print("loading files...")
    train_pos = load_tokens(train_file)
    val_pos = load_tokens(val_file)
    test_pos = load_tokens(test_file)




    # counters
    pos_counter = Counter(train_pos + val_pos + test_pos)

    # build vocab
    print("building vocab...")
    pos_vocab = VocabHelp(pos_counter, specials=['<pad>', '<unk>'])
    print("pos_vocab: {}".format(len(pos_vocab)))

    print("dumping to files...")
    pos_vocab.save_vocab(vocab_pos_file)
    print("all done.")


def load_tokens(filename):
    df = pd.read_csv(filename)
    pos = []
    for i in range(df.shape[0]):
        a = df['text_pos'][i]
        pos.extend(a.split())
    return pos



if __name__ == '__main__':
    main()