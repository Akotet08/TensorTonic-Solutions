class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        uniq = set()
        for string in texts:
            for sub in string.split(" "):
                uniq.add(sub.lower())

        lst = sorted(list(uniq))

        for idx, word in enumerate(lst):
            self.id_to_word[idx + 4] = word

        for k in self.id_to_word.keys():
            w =  self.id_to_word[k]
            self.word_to_id[w] = k

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        if text == "" : return []
        enc = [self.word_to_id[w.lower()] if w.lower() in self.word_to_id else 1 for w in text.split(" ")]

        return enc

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        dec = " ".join([self.id_to_word[id] if id in self.id_to_word else "<UNK>" for id in ids])

        return dec