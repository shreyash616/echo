import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch

try:
    import laion_clap
    HAS_LAION = True
except ImportError:
    HAS_LAION = False

try:
    from msclap import CLAP as MSCLAP
    from transformers import GPT2Tokenizer
    if not hasattr(GPT2Tokenizer, "encode_plus"):
        GPT2Tokenizer.encode_plus = lambda self, *args, **kwargs: self(*args, **kwargs)
    HAS_MSCLAP = True
except ImportError:
    HAS_MSCLAP = False


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_preprocess_audio(audio_path: str, target_sr: int) -> np.ndarray:
    """Decodes audio via ffmpeg to mono float32 at target_sr, -23 LUFS, peak-limited 0.99, max 30s."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        audio_path,
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = proc.communicate()
    if proc.returncode != 0 or not stdout:
        raise RuntimeError(f"FFmpeg failed to decode {audio_path}")

    audio = np.frombuffer(stdout, dtype=np.float32).copy()
    max_len = target_sr * 30
    if len(audio) > max_len:
        audio = audio[:max_len]

    meter = pyln.Meter(target_sr)
    try:
        loudness = meter.integrated_loudness(audio)
        if np.isfinite(loudness) and loudness > -70.0:
            audio = pyln.normalize.loudness(audio, loudness, -23.0)
    except Exception:
        pass

    peak = np.max(np.abs(audio))
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    return audio


def window_audio(audio: np.ndarray, sr: int, window_sec: float) -> list[np.ndarray]:
    """Splits into consecutive non-overlapping windows of window_sec. Drops final window shorter than half window_sec."""
    win_len = int(sr * window_sec)
    half_win = win_len // 2
    windows = []
    idx = 0
    while idx + win_len <= len(audio):
        windows.append(audio[idx : idx + win_len])
        idx += win_len
    rem = len(audio) - idx
    if rem >= half_win:
        pad = np.zeros(win_len - rem, dtype=np.float32)
        windows.append(np.concatenate([audio[idx:], pad]))
    if not windows:
        windows.append(np.pad(audio, (0, max(0, win_len - len(audio)))))
    return windows


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_backbone(name: str, model_wrapper, previews: list[dict], anchor_prompts: dict[str, str]) -> dict:
    prompt_keys = list(anchor_prompts.keys())
    prompt_texts = [anchor_prompts[k] for k in prompt_keys]

    # 1. Embed text prompts
    set_seed(0)
    text_embs = model_wrapper.embed_text(prompt_texts)  # (N, D), L2-normalised
    N = len(prompt_keys)
    pairwise_cosines = []
    for i in range(N):
        for j in range(i + 1, N):
            pairwise_cosines.append(cosine_sim(text_embs[i], text_embs[j]))
    max_text_cosine = max(pairwise_cosines) if pairwise_cosines else 1.0

    # 2. Embed 7 previews
    track_embs = []
    for p in previews:
        local_path = p["local_path"]
        set_seed(0)
        emb = model_wrapper.embed_audio(local_path)
        track_embs.append(emb)

    # 3. Match anchors for target tracks (0 to 6)
    townes_scores = {prompt_keys[i]: cosine_sim(track_embs[4], text_embs[i]) for i in range(N)}
    townes_top = sorted(townes_scores.items(), key=lambda x: x[1], reverse=True)[0][0]

    boyfriend_scores = {prompt_keys[i]: cosine_sim(track_embs[5], text_embs[i]) for i in range(N)}
    boyfriend_top = sorted(boyfriend_scores.items(), key=lambda x: x[1], reverse=True)[0][0]

    simple_man_scores = {prompt_keys[i]: cosine_sim(track_embs[0], text_embs[i]) for i in range(N)}
    simple_man_top = sorted(simple_man_scores.items(), key=lambda x: x[1], reverse=True)[0][0]

    # Audio-audio cosines
    cos_sm_rooster = cosine_sim(track_embs[0], track_embs[2])
    cos_sm_peaches = cosine_sim(track_embs[0], track_embs[6])
    cos_sm_start = cosine_sim(track_embs[0], track_embs[1])
    cos_sm_levee = cosine_sim(track_embs[0], track_embs[3])

    cos_townes_rooster = cosine_sim(track_embs[4], track_embs[2])
    cos_townes_boyfriend = cosine_sim(track_embs[4], track_embs[5])

    cos_boyfriend_peaches = cosine_sim(track_embs[5], track_embs[6])

    # Checks (a) to (g)
    # (a) text-text pairwise cosine max < 0.95
    pass_a = bool(max_text_cosine < 0.95)

    # (b) townes top anchor is "sparse folk" (Sparse and intimate or Folk story)
    pass_b = bool("Sparse" in townes_top or "Folk" in townes_top)

    # (c) target 5 top anchor is "trap bounce" (Lean-back trap bounce)
    pass_c = bool("trap bounce" in townes_scores and boyfriend_top == "Lean-back trap bounce" or "trap" in boyfriend_top.lower() or "bounce" in boyfriend_top.lower())

    # (d) simple_man top anchor is "slow heavy" (Slow and heavy, builds or Slow and heavy, grinding)
    pass_d = bool("Slow and heavy" in simple_man_top)

    # (e) cos(track_sm, track_rooster) > cos(track_sm, track_peaches)
    pass_e = bool(cos_sm_rooster > cos_sm_peaches)

    # (f) cos(track_townes, track_rooster) > cos(track_townes, track_boyfriend)
    pass_f = bool(cos_townes_rooster > cos_townes_boyfriend)

    # (g) cos(track_sm, track_rooster) > cos(track_sm, track_start)
    pass_g = bool(cos_sm_rooster > cos_sm_start)

    checks = {
        "(a) text_pairwise_max_lt_0.95": {"passed": pass_a, "value": max_text_cosine},
        "(b) townes_top_anchor_sparse_folk": {"passed": pass_b, "top_anchor": townes_top},
        "(c) boyfriend_top_anchor_trap_bounce": {"passed": pass_c, "top_anchor": boyfriend_top},
        "(d) simple_man_top_anchor_slow_heavy": {"passed": pass_d, "top_anchor": simple_man_top},
        "(e) cos(track_sm, track_rooster) > cos(track_sm, track_peaches)": {
            "passed": pass_e,
            "cos_rooster": cos_sm_rooster,
            "cos_peaches": cos_sm_peaches,
        },
        "(f) cos(track_townes, track_rooster) > cos(track_townes, track_boyfriend)": {
            "passed": pass_f,
            "cos_rooster": cos_townes_rooster,
            "cos_boyfriend": cos_townes_boyfriend,
        },
        "(g) cos(track_sm, track_rooster) > cos(track_sm, track_start)": {
            "passed": pass_g,
            "cos_rooster": cos_sm_rooster,
            "cos_start_me_up": cos_sm_start,
        },
    }
    total_passed = sum(1 for c in checks.values() if c["passed"])

    return {
        "name": name,
        "total_passed": total_passed,
        "out_of": 7,
        "checks": checks,
        "audio_pairwise": {
            "cos(track_sm, track_rooster)": cos_sm_rooster,
            "cos(track_sm, track_levee)": cos_sm_levee,
            "cos(track_sm, track_start)": cos_sm_start,
            "cos(track_sm, track_peaches)": cos_sm_peaches,
            "cos(track_boyfriend, track_peaches)": cos_boyfriend_peaches,
            "cos(track_townes, track_rooster)": cos_townes_rooster,
            "cos(track_townes, track_boyfriend)": cos_townes_boyfriend,
        },
    }


class LaionClapWrapper:
    def __init__(self, ckpt_path: str):
        self.sr = 48000
        self.duration = 10.0
        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt(ckpt_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # model.load_ckpt leaves it on cpu or handles it; if cuda available:
        if self.device == "cuda":
            self.model = self.model.to(self.device)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        # get_text_embedding returns numpy array
        embs = self.model.get_text_embedding(texts, use_tensor=False)
        embs = np.array(embs, dtype=np.float32)
        # L2-normalise
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    def embed_audio(self, audio_path: str) -> np.ndarray:
        audio = load_and_preprocess_audio(audio_path, self.sr)
        windows = window_audio(audio, self.sr, self.duration)
        win_embs = []
        for w in windows:
            # get_audio_embedding_from_data expects (batch, samples)
            emb = self.model.get_audio_embedding_from_data(x=w[None, :], use_tensor=False)
            emb = np.array(emb, dtype=np.float32).flatten()
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            win_embs.append(emb)
        avg_emb = np.mean(win_embs, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm
        return avg_emb


class MsClapWrapper:
    def __init__(self, model_fp: str | None = None):
        self.model = MSCLAP(model_fp=model_fp, version="2023", use_cuda=torch.cuda.is_available())
        self.sr = getattr(self.model.args, "sampling_rate", 44100)
        self.duration = getattr(self.model.args, "duration", 7.0)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        embs = self.model.get_text_embeddings(texts)
        if isinstance(embs, torch.Tensor):
            embs = embs.detach().cpu().numpy()
        embs = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embs / norms

    def embed_audio(self, audio_path: str) -> np.ndarray:
        audio = load_and_preprocess_audio(audio_path, self.sr)
        windows = window_audio(audio, self.sr, self.duration)
        win_embs = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, w in enumerate(windows):
                tmp_wav = os.path.join(tmpdir, f"win_{i}.wav")
                sf.write(tmp_wav, w, self.sr)
                emb = self.model.get_audio_embeddings([tmp_wav], resample=False)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                emb = np.array(emb, dtype=np.float32).flatten()
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                win_embs.append(emb)
        avg_emb = np.mean(win_embs, axis=0)
        norm = np.linalg.norm(avg_emb)
        if norm > 0:
            avg_emb = avg_emb / norm
        return avg_emb


def main():
    previews_file = Path("tests/backbone_previews.json")
    if not previews_file.exists():
        raise FileNotFoundError("tests/backbone_previews.json not found. Run fetch_backbone_previews.py first.")

    with open(previews_file, "r", encoding="utf-8") as f:
        previews = json.load(f)

    anchors_file = Path("ml/inference/anchor_prompts.json")
    with open(anchors_file, "r", encoding="utf-8") as f:
        anchor_prompts = json.load(f)

    results = {}

    # Check LAION CLAP
    ckpt_path = Path.home() / ".cache" / "clap" / "music_audioset_epoch_15_esc_90.14.pt"
    if HAS_LAION and ckpt_path.exists():
        print("Evaluating LAION CLAP (official loader)...")
        laion_wrapper = LaionClapWrapper(str(ckpt_path))
        results["laion_clap_official"] = evaluate_backbone(
            "laion_clap_official", laion_wrapper, previews, anchor_prompts
        )
        print(f"LAION passed {results['laion_clap_official']['total_passed']}/7 checks")
    else:
        print(f"Skipping LAION CLAP (HAS_LAION={HAS_LAION}, ckpt_exists={ckpt_path.exists()})")

    # Check MSCLAP
    if HAS_MSCLAP:
        print("Evaluating MSCLAP (2023)...")
        ms_ckpt = Path.home() / ".cache" / "clap" / "CLAP_weights_2023.pth"
        ms_fp = str(ms_ckpt) if ms_ckpt.exists() else None
        msclap_wrapper = MsClapWrapper(model_fp=ms_fp)
        results["msclap_2023"] = evaluate_backbone(
            "msclap_2023", msclap_wrapper, previews, anchor_prompts
        )
        print(f"MSCLAP passed {results['msclap_2023']['total_passed']}/7 checks")
    else:
        print("Skipping MSCLAP (not installed)")

    # Decision rule: backbone with more passes, tie goes to LAION
    laion_passes = results.get("laion_clap_official", {}).get("total_passed", -1)
    msclap_passes = results.get("msclap_2023", {}).get("total_passed", -1)

    if laion_passes >= msclap_passes and laion_passes > 0:
        winner = "laion_clap_official"
    elif msclap_passes > laion_passes:
        winner = "msclap_2023"
    else:
        winner = "none"

    summary = {
        "results": results,
        "winner": winner,
        "decision_rule": "Backbone with more passes among (a)-(g); tie goes to LAION",
    }

    out_file = Path("ml/experiments/backbone_check_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote backbone check results to {out_file}")
    print(f"Winner: {winner}")


if __name__ == "__main__":
    main()
