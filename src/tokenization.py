from functools import lru_cache
from typing import Dict, List, Tuple, Set
import regex
import json


def bpe_train(dataset: str, num_merges: int) -> Dict[int, str]:
    pass


# def read_corpus(filename: str) -> Dict[int, str]:
#     with open(filename, "r", encoding="utf-8") as f:
#         data = f.read().encode("utf-8")
#

# read_corpus("./data/taylor_swift_wiki.txt")

encoding_map: Dict[str, int] = {}

with open("weights/gpt2_encoder.json", "rb") as f:
    for k, v in json.loads(f.read()).items():
        encoding_map[k] = v

with open("weights/gpt2_vocab.bpe", "rb") as f:
    merges: Dict[Tuple[str, str], int] = {tuple(l[:-1].decode("utf-8").split(' ')): i + 256 for i, l in
                                          enumerate(f.readlines()[1:])}


def get_stats(unicode_str: List[str]) -> Dict[Tuple[str, str], int]:
    occurrences: Dict[Tuple[str, str], int] = {}
    for i in range(len(unicode_str) - 1):
        pair = (unicode_str[i], unicode_str[i + 1])
        occurrences[pair] = occurrences.get(pair, 0) + 1
    return occurrences


class GPT2Tokenizer:
    def __init__(self):
        self.merges = merges
        self.encoding_map = encoding_map
        self.decoding_map = {v: k for k, v in encoding_map.items()}
        self.byte_encoding = self.bytes_to_unicode()
        self.byte_decoding = {v: k for k, v in self.byte_encoding.items()}

    # Taken from https://github.com/openai/gpt-2/blob/master/src/encoder.py (MIT LICENSE)
    @lru_cache()
    def bytes_to_unicode(self):
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
            range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))

    def encode(self, str_to_encode: str) -> List[int]:
        pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        gpt2_regex = regex.compile(pat_str)
        final_list: List[str] = []
        out_count = 0
        in_count = 0
        for sub_string in gpt2_regex.findall(str_to_encode):
            in_count += 1
            current_str: List[str] = [self.byte_encoding[b] for b in sub_string.encode("utf-8")]
            not_found_strs: Set[Tuple[str, str]] = set()
            while True:
                in_count += 1
                stats = get_stats(current_str)
                large_number = 999999999
                merge_options = [(self.merges.get((pair[0], pair[1]), large_number), pair)
                                 for pair in stats.keys()
                                 if pair not in not_found_strs]
                if len(merge_options) == 0:
                    final_list += current_str
                    break
                encoding_num, pair = min(merge_options)
                if encoding_num == large_number:
                    not_found_strs.add(pair)
                    continue
                new_str: List[str] = []
                i = 0
                while i < len(current_str):
                    if (i + 1) < len(current_str) and (current_str[i], current_str[i + 1]) == pair:
                        new_str.append(pair[0] + pair[1])
                        i += 2
                    else:
                        new_str.append(current_str[i])
                        i += 1
                if len(current_str) == len(new_str):
                    not_found_strs.add(pair)
                current_str = new_str
        print(f"Out {out_count} in {in_count}")
        return [encoding_map[w] for w in final_list]

    def decode(self, encoded: List[int]) -> str:
        unicode_str = "".join(self.decoding_map[i] for i in encoded)
        decoded_utf8 = b"".join([bytes([self.byte_decoding[c]]) for c in unicode_str])
        return decoded_utf8.decode("utf-8")

def _test():
    gpt2_tokenizer = GPT2Tokenizer()
    test_str = "hello world こんにちは, 今日は \n\n don't say but do... !"
    result = gpt2_tokenizer.encode(test_str)
    assert result == [31373, 995, 23294, 241, 22174, 28618, 2515, 94, 31676, 11, 220, 20015, 232, 33768, 98, 31676, 220,
                      628, 836, 470, 910, 475, 466, 986, 5145]
    assert(gpt2_tokenizer.decode(result) == test_str)

# _test()