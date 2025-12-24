import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
import json
from pathlib import Path

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[.,!?;:/()\-]")

def simple_tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    pad_token: str = "<pad>"
    unk_token: str = "<unk>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @property
    def pad_id(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def unk_id(self) -> int:
        return self.stoi[self.unk_token]

    @property
    def bos_id(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.stoi[self.eos_token]

    def encode(self, tokens: List[str], max_len: int) -> List[int]:
        ids = [self.bos_id] + [self.stoi.get(t, self.unk_id) for t in tokens] + [self.eos_id]
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
            ids[-1] = self.eos_id
        return ids

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str) -> "Vocab":
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        stoi = {t: i for i, t in enumerate(itos)}
        return Vocab(stoi=stoi, itos=itos)

def build_vocab(texts: Iterable[str], min_freq: int = 2, max_size: int = 40000) -> Vocab:
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    itos = specials[:]
    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        if token in specials:
            continue
        itos.append(token)
        if len(itos) >= max_size:
            break
    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)
