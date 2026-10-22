"""Tiny random models for the CPU tests: a Qwen3 LLM, a word-level tokenizer and a Higgs codec."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, PreTrainedTokenizerFast

from omnivoice_tts.models.omnivoice import OmniVoice, OmniVoiceConfig
from omnivoice_tts.utils.duration import RuleDurationEstimator

TINY_LLM = dict(model_type="qwen3", hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=2, head_dim=16, vocab_size=1100, max_position_embeddings=512, rope_theta=1e6,
                rms_norm_eps=1e-6, tie_word_embeddings=True)  # vocab >= 1025: the text embedding also sees audio ids
TINY_CODEC = dict(
    acoustic_model_config=dict(encoder_hidden_size=8, decoder_hidden_size=64, hidden_size=8,
                               downsampling_ratios=[8, 5, 4, 2, 3], upsampling_ratios=[8, 5, 4, 2, 3]),
    semantic_model_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                               conv_dim=[16] * 7, num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2))
SPECIALS = ["<|endoftext|>", "<|im_end|>", "<|denoise|>", "<|lang_start|>", "<|lang_end|>", "<|instruct_start|>",
            "<|instruct_end|>", "<|text_start|>", "<|text_end|>"]


def tiny_tokenizer():
    vocab = {t: i for i, t in enumerate(SPECIALS + ["[UNK]", "es", "None", "hola", "happy", "[laughter]"])}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="<|endoftext|>",
                                   eos_token="<|im_end|>", additional_special_tokens=SPECIALS[2:])


def tiny_codec():
    torch.manual_seed(0)
    codec = HiggsAudioV2TokenizerModel(HiggsAudioV2TokenizerConfig(**TINY_CODEC))
    for q in codec.quantizer.quantizers:  # fresh codebooks are all zero, so decode would not depend on the codes
        q.codebook.embed.normal_()
    return codec.eval()


def tiny_model(dtype=torch.float32):
    torch.manual_seed(0)
    m = OmniVoice(OmniVoiceConfig(llm_config=TINY_LLM))
    for p in m.parameters():  # before the codec is attached, so it stays fp32; buffers such as inv_freq stay fp32
        p.data = p.data.to(dtype)
    m.text_tokenizer, m.audio_tokenizer, m.duration_estimator = tiny_tokenizer(), tiny_codec(), RuleDurationEstimator()
    m.sampling_rate = m.audio_tokenizer.config.sample_rate
    return m


def cfg_batch():
    """A 2-row CFG batch shaped like _generate_iterative's: the conditional row sees all S positions, the
    unconditional row only its first T, plus a diagonal over the padding."""
    torch.manual_seed(1)
    S, T = 40, 20
    input_ids = torch.randint(0, 300, (2, 8, S))
    input_ids[:, :, S - T:] = torch.randint(0, 1025, (2, 8, T))
    audio_mask = torch.zeros(2, S, dtype=torch.bool)
    audio_mask[:, S - T:] = True
    attention_mask = torch.zeros(2, 1, S, S, dtype=torch.bool)
    attention_mask[0] = True
    attention_mask[1, :, :T, :T] = True
    attention_mask[1, :, torch.arange(T, S), torch.arange(T, S)] = True
    return input_ids, audio_mask, attention_mask
