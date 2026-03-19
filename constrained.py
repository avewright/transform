"""Constrained decoding for legal chess moves.

Instead of generating freely and hoping the model produces a legal move,
we constrain the token-level logits at each decoding step so that only
token continuations consistent with at least one legal UCI move are
possible. This guarantees the final output is always a legal move.

Approach:
  1. Before generation, compute all legal moves in UCI notation for the position.
  2. Build a trie (prefix tree) of legal move strings.
  3. At each decoding step, mask logits to only allow tokens whose text
     continues a valid prefix in the trie.
  4. The model can only produce complete legal moves.

This is a LogitsProcessor compatible with HuggingFace's generate().
"""

from __future__ import annotations

from typing import Optional

import chess
import torch
from transformers import LogitsProcessor, LogitsProcessorList


class TrieNode:
    """Simple trie node for string prefixes."""
    __slots__ = ("children", "is_terminal")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_terminal: bool = False


class MoveTrie:
    """Trie of legal UCI move strings.

    Supports prefix lookup: given what's been generated so far,
    returns which characters can legally come next.
    """

    def __init__(self, moves: list[str]):
        self.root = TrieNode()
        for move in moves:
            self._insert(move)

    def _insert(self, move: str):
        node = self.root
        for ch in move:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_terminal = True

    def get_valid_next_chars(self, prefix: str) -> set[str]:
        """Given a prefix, return the set of valid next characters."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return set()
            node = node.children[ch]
        return set(node.children.keys())

    def is_complete_move(self, s: str) -> bool:
        """Check if s is a complete legal move in the trie."""
        node = self.root
        for ch in s:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_terminal

    def has_prefix(self, prefix: str) -> bool:
        """Check if any move starts with this prefix."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True


def build_move_trie(board: chess.Board) -> MoveTrie:
    """Build a trie of all legal UCI moves for the current position."""
    moves = [move.uci() for move in board.legal_moves]
    return MoveTrie(moves)


def build_token_text_map(tokenizer) -> dict[int, str]:
    """Build a reusable token_id → decoded_text map for the given tokenizer.

    This is expensive (~150K decode calls for Qwen) so should be done once
    and passed to every LegalMoveLogitsProcessor instance.
    """
    token_texts: dict[int, str] = {}
    vocab = tokenizer.get_vocab()
    for token_str, token_id in vocab.items():
        try:
            decoded = tokenizer.decode([token_id], skip_special_tokens=True)
            decoded_clean = decoded.replace(" ", "").lower()
            if decoded_clean:
                token_texts[token_id] = decoded_clean
        except Exception:
            continue
    return token_texts


class LegalMoveLogitsProcessor(LogitsProcessor):
    """HuggingFace LogitsProcessor that constrains generation to legal chess moves.

    At each token step, it:
      1. Decodes the tokens generated so far (after the prompt) to get the current text.
      2. Looks up valid next characters in the move trie.
      3. For each token in the vocabulary, checks if it starts with a valid next character.
      4. Masks all invalid tokens to -inf.

    This guarantees the generated text will be a legal UCI move.
    """

    def __init__(
        self,
        trie: MoveTrie,
        tokenizer,
        prompt_length: int,
        eos_token_id: int | None = None,
        token_texts: dict[int, str] | None = None,
    ):
        self.trie = trie
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id

        # Use pre-built map if provided, otherwise build (slow path)
        self._token_texts = token_texts if token_texts is not None else build_token_text_map(tokenizer)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for batch_idx in range(batch_size):
            # Decode what's been generated so far (after prompt)
            generated_ids = input_ids[batch_idx, self.prompt_length:]
            generated_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).replace(" ", "").lower()

            # Check if we already have a complete move
            if self.trie.is_complete_move(generated_text):
                # Force EOS — the move is done
                mask = torch.full_like(scores[batch_idx], float("-inf"))
                if self.eos_token_id is not None:
                    mask[self.eos_token_id] = 0.0
                scores[batch_idx] = scores[batch_idx] + mask
                continue

            # Find valid next characters from the trie
            valid_chars = self.trie.get_valid_next_chars(generated_text)
            if not valid_chars:
                # No valid continuation — shouldn't happen if trie is correct.
                # Allow EOS to end gracefully.
                if self.eos_token_id is not None:
                    mask = torch.full_like(scores[batch_idx], float("-inf"))
                    mask[self.eos_token_id] = 0.0
                    scores[batch_idx] = scores[batch_idx] + mask
                continue

            # Build mask: allow only tokens whose decoded text would continue
            # a valid prefix in the trie
            mask = torch.full_like(scores[batch_idx], float("-inf"))

            for token_id, token_text in self._token_texts.items():
                # Check if appending this token's text would still be a valid prefix
                candidate = generated_text + token_text
                # Check character by character against the trie
                if self._is_valid_continuation(generated_text, token_text):
                    mask[token_id] = 0.0

            # Also allow EOS if current text is a complete move
            if self.trie.is_complete_move(generated_text) and self.eos_token_id is not None:
                mask[self.eos_token_id] = 0.0

            scores[batch_idx] = scores[batch_idx] + mask

        return scores

    def _is_valid_continuation(self, prefix: str, new_text: str) -> bool:
        """Check if appending new_text to prefix stays on a valid trie path."""
        combined = prefix + new_text
        # The combined string must be a prefix of at least one legal move
        # Check character by character
        node = self.trie.root
        for ch in combined:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True


def make_legal_move_processor(
    board: chess.Board,
    tokenizer,
    prompt_length: int,
    token_texts: dict[int, str] | None = None,
) -> LogitsProcessorList:
    """Create a LogitsProcessorList for constrained legal move generation.

    Pass a pre-built token_texts map (from build_token_text_map) for speed.
    """
    trie = build_move_trie(board)
    eos_id = tokenizer.eos_token_id

    processor = LegalMoveLogitsProcessor(
        trie=trie,
        tokenizer=tokenizer,
        prompt_length=prompt_length,
        eos_token_id=eos_id,
        token_texts=token_texts,
    )
    return LogitsProcessorList([processor])
