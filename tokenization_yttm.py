from transformers.tokenization_utils import PreTrainedTokenizer
import youtokentome as yttm
from collections import OrderedDict
from typing import List, Union
import os


VOCAB_FILES_NAMES = {"vocab_file": "encoder.model"}


class YTTMTransformersTokenizer(PreTrainedTokenizer):
    """
    YTTMTransformersTokenizer BPE tokenizer. Peculiarities:

    - Byte-level Byte-Pair-Encoding
    - Requires a space to start the input string => the encoding methods should be called with the
      ``add_prefix_space`` flag set to ``True``.
      Otherwise, this tokenizer ``encode`` and ``decode`` method will not conserve
      the absence of a space at the beginning of a string:

    ::

        tokenizer.decode(tokenizer.encode("Hello", add_special_tokens=False))

    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the methods. Users
    should refer to the superclass for more information regarding methods.

    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        unk_token (:obj:`string`, `optional`, defaults to <UNK>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (:obj:`string`, `optional`, defaults to `<BOS>`):
            The beginning of sequence token.
        eos_token (:obj:`string`, `optional`, defaults to `<EOS>`):
            The end of sequence token.
        pad_token (:obj:`string`, `optional`, defaults to `<PAD>`):
            The padding of sequence token.
        model_max_length: (`Optional`) int: the maximum length in number of tokens for the inputs to the transformer
            model. When the tokenizer is loaded with `from_pretrained`,
            this will be set to the value stored for the associated.
    """

    def __init__(
            self,
            vocab_file,
            unk_token="<UNK>",
            bos_token="<BOS>",
            eos_token="<EOS>",
            pad_token="<PAD>",
            model_max_length=512,
            **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            **kwargs
        )
        # no default special tokens - you can update this value if you add special tokens
        self.max_len_single_sentence = model_max_length - 2
        # no default special tokens - you can update this value if you add special tokens
        self.max_len_sentences_pair = model_max_length - 2
        vocab_file = str(vocab_file)

        if os.path.isfile(vocab_file):
            self.bpe = yttm.BPE(vocab_file, n_threads=-1)
        elif os.path.isfile(os.path.join(vocab_file, VOCAB_FILES_NAMES["vocab_file"])):
            vocab_file = os.path.join(vocab_file, VOCAB_FILES_NAMES["vocab_file"])
            self.bpe = yttm.BPE(vocab_file, n_threads=-1)
        else:
            raise OSError("vocab_file should be a path to model file or dir with encoder.model")
        self.vocab_file = vocab_file
        self.decoder = self.bpe
        self.cache = {}
        self.vocab = OrderedDict([(key, val) for val, key in enumerate(self.bpe.vocab())])
        self.encoder = self.vocab
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self) -> int:
        return self.bpe.vocab_size()

    def get_vocab(self):
        return dict(self.vocab)

    def save_vocabulary(self, save_directory: str):
        pass

    def _tokenize(self, text: str, **kwargs):
        """Converts a string in a sequence of tokens (string), using the tokenizer.
        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies (BPE).
        """
        return self.bpe.encode([text], output_type=yttm.OutputType.SUBWORD)[0]

    def tokenize(self, text: Union[List[str], str], add_special_tokens=True, **kwargs):
        if isinstance(text, list):
            return list(map(
                lambda x: self.tokenize(x, add_special_tokens=add_special_tokens, **kwargs),
                text
            ))
        res = self._tokenize(text)
        if add_special_tokens:
            res = [self.bos_token] + res + [self.eos_token]
        return res

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens: List[str]):
        """Converts a sequence of tokens (string) in a single string. """
        return self.bpe.decode(list(map(self._convert_token_to_id, tokens)))[0]

    @classmethod
    def from_pretrained(cls, **kwargs):
        """Load from file. Actually only call __init__"""
        return cls(**kwargs)
