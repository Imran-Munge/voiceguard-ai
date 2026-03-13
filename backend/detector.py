"""
VoiceGuardAI v4.2 — Detection Engine
Verdict logic:
  - AI_FRAUD  : fraud keywords found in transcript (Whisper)
  - AI_VOICE  : physics/ML layers detect synthetic voice (no keywords needed)
  - SUSPICIOUS: borderline — 2-3 physics layers fail but not enough for AI_VOICE
  - HUMAN     : clean — fewer than 2 physics layers fail, no keywords
"""

import numpy as np
import torch
import librosa
import scipy.signal as signal
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging, warnings, re, time
warnings.filterwarnings("ignore")

logger = logging.getLogger("voiceguard.detector")

# ─────────────────────────────────────────────────────────────
# THRESHOLDS  — Option 1+2 weighted system
#
#  Old system: raw layer count (threshold=4) — missed compressed AI voices
#              and false-flagged humans on noisy speakerphone.
#
#  New system: weighted score + wav2vec special rules
#
#  W_AI_THRESHOLD         = 5.5   weighted score ≥ 5.5 AND ≥1 high-conf AI
#                                 layer → AI_VOICE
#  W_SUSPICIOUS_THRESHOLD = 2.0   weighted score ≥ 2.0 → SUSPICIOUS
#  HIGH_CONF_AI_LAYERS            wav2vec, tts_artifacts, prosodic_naturalness
#                                 These 3 are the most discriminative AI signals.
#
#  5 decision rules (evaluated in order):
#    Rule 1: wscore ≥ 5.5  AND  ≥1 high-conf layer  → AI_VOICE
#    Rule 2: wav2vec fails  AND  any other layer also fails  → AI_VOICE
#    Rule 3: 2+ high-conf layers fail (even without wav2vec)  → AI_VOICE
#    Rule 4: wav2vec fails alone  → SUSPICIOUS  (strong solo signal)
#    Rule 5: wscore ≥ 2.0  → SUSPICIOUS
#    Rule 6: otherwise  → HUMAN
# ─────────────────────────────────────────────────────────────
W_AI_THRESHOLD         = 5.5   # weighted score for AI_VOICE (with high-conf guard)
W_SUSPICIOUS_THRESHOLD = 2.0   # weighted score for SUSPICIOUS
HIGH_CONF_AI_LAYERS    = {"wav2vec_combined", "tts_frequency_artifacts", "prosodic_naturalness"}

# ─────────────────────────────────────────────────────────────
# LAYER 1: FRAUD KEYWORDS  (only trigger on transcript words)
# ─────────────────────────────────────────────────────────────
FRAUD_KEYWORDS = [
    # Financial fraud
    "money", "send money", "urgent payment", "bank transfer",
    "transaction", "wire transfer", "western union",
    "loan", "investment", "crypto", "bitcoin", "refund",
    "lottery", "gift card", "google play card", "amazon card",
    # Card / account data
    "otp", "credit card", "debit card", "password", "pin number",
    "cvv", "expiry", "account verification", "kyc", "kyc update",
    "verify account", "aadhaar", "pan card",
    # Payment apps
    "upi", "paytm", "phonepe", "gpay", "wallet", "transfer now",
    # Parcel / customs scams
    "gift parcel", "customs parcel", "tax payment",
    # Threat / urgency scams
    "kidnapped", "ransom", "legal notice", "police case",
    "arrest warrant", "fraud department", "suspicious activity",
    "emergency payment",
    # Prize scams
    "act fast", "limited time", "claim prize", "you won",
    "selected", "lucky winner",
    # Hindi / Hinglish fraud terms
    "paisa", "paise", "rupees", "rupaya", "khata", "bank account",
    "sim band", "court", "case", "arrest", "jail",
    "income tax", "emi", "insurance", "policy",
    "prize money", "lottery ticket", "foreign parcel",
    "custom duty", "delivery agent", "account block",
    "account suspend", "account freeze", "aadhar", "aadhar card",
    "borrow money", "send rupees", "send paisa",
]

def detect_fraud_keywords(transcript: str) -> Tuple[bool, List[str]]:
    """
    HARD OVERRIDE keyword detection.
    If ANY keyword is found → returns (True, matched_list).
    This result CANNOT be overridden by physics, ML, or LLM layers.
    """
    if not transcript or not transcript.strip():
        return False, []
    text = transcript.lower().strip()
    found = []
    for kw in FRAUD_KEYWORDS:
        # Exact substring match (most common)
        if kw in text:
            if kw not in found:
                found.append(kw)
            continue
        # Fuzzy: allow 1-char gap for keywords > 5 chars
        if len(kw) > 5:
            try:
                pattern = r'\b' + r'.{0,1}'.join(re.escape(ch) for ch in kw) + r'\b'
                if re.search(pattern, text) and kw not in found:
                    found.append(kw)
            except re.error:
                pass
    if found:
        logger.info(f"[KEYWORD OVERRIDE] Found: {found} in: '{text[:80]}'")
    return len(found) > 0, list(set(found))


# ─────────────────────────────────────────────────────────────
# DETECTION RESULT
# ─────────────────────────────────────────────────────────────
@dataclass
class DetectionResult:
    verdict:           str              # HUMAN | SUSPICIOUS | AI_VOICE | AI_FRAUD
    is_ai:             bool
    is_fraud:          bool
    is_suspicious:     bool
    confidence:        float
    ai_probability:    float
    human_probability: float
    failure_count:     int
    failures:          List[str]
    layer_scores:      dict
    latency_ms:        float
    fraud_keywords:    List[str] = field(default_factory=list)
    transcript:        str       = ""
    details:           dict      = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# WAV2VEC2
# ─────────────────────────────────────────────────────────────
_wav2vec_model     = None
_wav2vec_processor = None

def _load_wav2vec():
    global _wav2vec_model, _wav2vec_processor
    if _wav2vec_model is None:
        from transformers import Wav2Vec2Processor, Wav2Vec2Model
        logger.info("Loading Wav2Vec2...")
        _wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        _wav2vec_model     = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        _wav2vec_model.eval()
        logger.info("Wav2Vec2 loaded ✓")
    return _wav2vec_model, _wav2vec_processor


# ─────────────────────────────────────────────────────────────
# PRE-PROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_audio(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    audio = audio.astype(np.float32)
    mx = np.max(np.abs(audio))
    if mx > 1e-6: audio = audio / mx
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    b, a = signal.butter(4, 80.0 / (sr / 2), btype='high')
    audio = signal.filtfilt(b, a, audio)
    mx = np.max(np.abs(audio))
    if mx > 1e-6: audio = audio / mx
    return audio, sr


# ─────────────────────────────────────────────────────────────
# PHYSICS LAYERS
# ─────────────────────────────────────────────────────────────
def layer_spectral_flux(audio, sr):
    try:
        spec = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
        flux = np.sqrt(np.sum(np.diff(spec, axis=1)**2, axis=0))
        if len(flux) < 10: return False, 1.0, "spectral_flux"
        cv    = np.std(flux) / (np.mean(flux) + 1e-10)
        burst = np.sum(flux > np.mean(flux)*2.0) / len(flux)
        return (cv < 0.5 and burst < 0.15), float(cv*0.6+burst*0.4), "spectral_flux"
    except: return False, 1.0, "spectral_flux"

def layer_gaussianity(audio, sr):
    try:
        fl = int(0.025*sr); hop = int(0.010*sr)
        frames = librosa.util.frame(audio, frame_length=fl, hop_length=hop)
        kvals, pvals = [], []
        for i in range(0, frames.shape[1], 4):
            f = frames[:, i]
            if np.std(f) < 1e-6: continue
            kvals.append(stats.kurtosis(f))
            if len(f) >= 20:
                _, p = stats.shapiro(f[:50]); pvals.append(p)
        mk = np.mean(kvals) if kvals else 3.0
        mp = np.mean(pvals) if pvals else 0.5
        return (mp > 0.08 and abs(mk) < 0.8), float(mp), "gaussianity"
    except: return False, 0.5, "gaussianity"

def layer_formant_stability(audio, sr):
    try:
        fl = int(0.025*sr); hop = int(0.010*sr)
        frames = librosa.util.frame(audio, frame_length=fl, hop_length=hop)
        f1s, f2s = [], []
        for i in range(frames.shape[1]):
            frame = frames[:,i] * np.hamming(fl)
            if np.max(np.abs(frame)) < 0.01: continue
            try:
                a = librosa.lpc(frame, order=12)
                roots = np.roots(a)
                roots = roots[(np.abs(roots)<1.0) & (np.imag(roots)>0)]
                if len(roots)==0: continue
                freqs = np.sort(np.angle(roots)*sr/(2*np.pi))
                freqs = freqs[freqs>100]
                if len(freqs)>=2:
                    f1s.append(freqs[0]); f2s.append(freqs[1])
            except: continue
        if len(f1s)<5: return False, 10.0, "formant_stability"
        stab = (np.std(np.diff(f1s))+np.std(np.diff(f2s)))/2.0
        return (stab < 5.0), float(stab), "formant_stability"
    except: return False, 10.0, "formant_stability"

def layer_harmonicity(audio, sr):
    try:
        _, f0v = librosa.piptrack(y=audio, sr=sr, n_fft=2048)
        pitches = f0v[f0v>80]
        if len(pitches)<10: return False, 1.5, "harmonicity"
        f0m = np.median(pitches)
        if f0m<50: return False, 1.5, "harmonicity"
        spec  = np.abs(librosa.stft(audio, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        ratios = []
        for fi in range(0, spec.shape[1], 5):
            fs = spec[:,fi]
            h1 = np.argmin(np.abs(freqs-f0m))
            h2 = np.argmin(np.abs(freqs-2*f0m))
            if h2<len(fs) and fs[h2]>1e-8:
                ratios.append(fs[h1]/(fs[h2]+1e-10))
        mr = np.mean(ratios) if ratios else 1.5
        return (mr < 0.8), float(mr), "harmonicity"
    except: return False, 1.5, "harmonicity"

def layer_pitch_jitter(audio, sr):
    try:
        f0 = librosa.yin(audio, fmin=60, fmax=500, sr=sr,
                         frame_length=2048, hop_length=256)
        fv = f0[(f0>60)&(f0<500)]
        if len(fv)<15: return False, 0.05, "pitch_jitter"
        periods = 1.0/(fv+1e-10)
        jitter  = np.mean(np.abs(np.diff(periods)))/(np.mean(periods)+1e-10)
        return (jitter < 0.008), float(jitter), "pitch_jitter"
    except: return False, 0.05, "pitch_jitter"

def layer_spectrogram_turbulence(audio, sr):
    try:
        spec  = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        dspec = np.diff(spec, axis=1)
        dfreq = np.diff(spec, axis=0)
        tv    = float(np.mean(np.abs(dspec)))
        fv    = float(np.mean(np.abs(dfreq)))
        return (tv < 0.004 and fv < 0.003), float(tv*100+fv*100), "spectrogram_turbulence"
    except: return False, 1.0, "spectrogram_turbulence"

def layer_tts_frequency_artifacts(audio, sr):
    try:
        spec  = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        hi    = spec[freqs>4000,:]
        estd  = float(np.mean(np.std(hi, axis=1)))
        zcr   = librosa.feature.zero_crossing_rate(audio)[0]
        zcv   = float(np.std(zcr)/(np.mean(zcr)+1e-10))
        burst = float(np.sum(np.std(spec,axis=1) > np.mean(np.std(spec,axis=1))*2)/spec.shape[0])
        failed = estd<2.5 and zcv<0.35 and burst<0.08
        return failed, float(estd*0.3+zcv*20+burst*10), "tts_frequency_artifacts"
    except: return False, 3.0, "tts_frequency_artifacts"

def layer_prosodic_naturalness(audio, sr):
    try:
        f0  = librosa.yin(audio, fmin=60, fmax=500, sr=sr,
                          frame_length=2048, hop_length=512)
        fv  = f0[f0>60]
        if len(fv)<10: return False, 1.0, "prosodic_naturalness"
        f0r = (np.percentile(fv,90)-np.percentile(fv,10))/(np.mean(fv)+1e-10)
        f0v = np.std(np.diff(fv))
        rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(rms, height=np.mean(rms)*0.7, distance=3)
        ri = np.std(np.diff(peaks))/(np.mean(np.diff(peaks))+1e-10) if len(peaks)>=4 else 0.3
        failed = f0r<0.15 and f0v<5.0 and ri<0.2
        return failed, float(f0r*2+f0v/10+ri), "prosodic_naturalness"
    except: return False, 1.0, "prosodic_naturalness"

def layer_codec_artifact_pattern(audio, sr):
    try:
        spec  = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        edges = [300,800,1600,3200,6400]
        be, bv = [], []
        for i in range(len(edges)-1):
            mask = (freqs>=edges[i])&(freqs<edges[i+1])
            if not np.any(mask): continue
            be.append(np.mean(spec[mask,:]))
            bt = np.mean(spec[mask,:], axis=0)
            bv.append(np.std(bt)/(np.mean(bt)+1e-10))
        if not be: return False, 1.0, "codec_artifact_pattern"
        ben = np.array(be); ben /= (np.sum(ben)+1e-10)
        ent = -np.sum(ben*np.log(ben+1e-10))
        mv  = np.mean(bv)
        return (ent<1.2 and mv<0.4), float(ent*0.5+mv*2.0), "codec_artifact_pattern"
    except: return False, 1.0, "codec_artifact_pattern"

def _extract_wav2vec_features(audio, sr):
    try:
        model, processor = _load_wav2vec()
        if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio[:5*16000]
        with torch.no_grad():
            inputs  = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            hidden  = outputs.last_hidden_state.squeeze(0).numpy()
        return hidden
    except Exception as e:
        logger.warning(f"Wav2Vec2 failed: {e}")
        return None

def layer_wav2vec_combined(hidden):
    try:
        fr   = np.ptp(hidden, axis=0)+1e-10
        norm = (hidden-hidden.min(axis=0))/fr+1e-10
        norm /= norm.sum(axis=0)
        entropy = float(np.mean(-np.sum(norm*np.log(norm+1e-10), axis=0)))
        tv      = float(np.mean(np.var(np.diff(hidden, axis=0), axis=1)))
        norms   = np.linalg.norm(hidden, axis=1)
        z       = abs(float(np.mean(norms))-12.5)/3.2
        cov     = np.cov(hidden.T[:50,:])
        ev      = np.sort(np.linalg.eigvalsh(cov))[::-1]
        top5    = float(np.sum(ev[:5]/(np.sum(ev)+1e-10))) if len(ev)>=5 else 0.5
        failed  = (entropy<1.8 and tv<0.025) or (z>3.2 or top5>0.78)
        return failed, float(entropy*0.3+tv*10+(1/(z+0.1))*0.2), "wav2vec_combined"
    except: return False, 1.0, "wav2vec_combined"


# ─────────────────────────────────────────────────────────────
# MAIN DETECTOR
# ─────────────────────────────────────────────────────────────
class VoiceGuardDetector:
    LAYER_WEIGHTS = {
        "spectral_flux":            1.5,
        "gaussianity":              0.8,
        "formant_stability":        1.2,
        "harmonicity":              1.0,
        "pitch_jitter":             1.2,
        "spectrogram_turbulence":   1.0,
        "tts_frequency_artifacts":  2.5,
        "prosodic_naturalness":     2.5,
        "codec_artifact_pattern":   2.0,
        "wav2vec_combined":         3.0,
    }

    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml
        if use_ml:
            try: _load_wav2vec()
            except Exception as e:
                logger.warning(f"Wav2Vec2 unavailable: {e}")
                self.use_ml = False

    def analyze(self, audio: np.ndarray, sr: int = 16000,
                transcript: str = "") -> DetectionResult:
        t0 = time.perf_counter()

        # ════════════════════════════════════════════════════
        # LAYER 1: KEYWORD HARD OVERRIDE
        # If ANY fraud keyword is detected → INSTANT AI_FRAUD.
        # NO physics, NO ML, NO LLM can change this verdict.
        # ════════════════════════════════════════════════════
        fraud_found, fraud_kws = detect_fraud_keywords(transcript)
        if fraud_found:
            ms = round((time.perf_counter()-t0)*1000, 1)
            logger.info(f"[KEYWORD HARD OVERRIDE] → AI_FRAUD | keywords={fraud_kws}")
            return DetectionResult(
                verdict="AI_FRAUD", is_ai=True, is_fraud=True, is_suspicious=False,
                confidence=1.0, ai_probability=100.0, human_probability=0.0,
                failure_count=0, failures=[], layer_scores={},
                latency_ms=ms, fraud_keywords=fraud_kws, transcript=transcript,
                details={"trigger": "KEYWORD_HARD_OVERRIDE", "keywords": fraud_kws}
            )

        # ── Layers 2–3: Physics + ML ─────────────────────────
        audio, sr = preprocess_audio(audio, sr)
        failures, layer_scores = [], {}

        for name, fn in [
            ("spectral_flux",           lambda: layer_spectral_flux(audio, sr)),
            ("gaussianity",             lambda: layer_gaussianity(audio, sr)),
            ("formant_stability",       lambda: layer_formant_stability(audio, sr)),
            ("harmonicity",             lambda: layer_harmonicity(audio, sr)),
            ("pitch_jitter",            lambda: layer_pitch_jitter(audio, sr)),
            ("spectrogram_turbulence",  lambda: layer_spectrogram_turbulence(audio, sr)),
            ("tts_frequency_artifacts", lambda: layer_tts_frequency_artifacts(audio, sr)),
            ("prosodic_naturalness",    lambda: layer_prosodic_naturalness(audio, sr)),
            ("codec_artifact_pattern",  lambda: layer_codec_artifact_pattern(audio, sr)),
        ]:
            try:
                failed, score, _ = fn()
                layer_scores[name] = score
                if failed: failures.append(name)
            except Exception as e:
                logger.debug(f"Layer {name}: {e}")
                layer_scores[name] = 0.5

        # ML layer
        if self.use_ml:
            hidden = _extract_wav2vec_features(audio, sr)
            if hidden is not None:
                try:
                    failed, score, _ = layer_wav2vec_combined(hidden)
                    layer_scores["wav2vec_combined"] = score
                    if failed: failures.append("wav2vec_combined")
                except: layer_scores["wav2vec_combined"] = 0.5
            else:
                layer_scores["wav2vec_combined"] = 0.5
        else:
            layer_scores["wav2vec_combined"] = 0.5

        # ── Verdict — 5-rule weighted system ─────────────────
        fc               = len(failures)
        wfail            = sum(self.LAYER_WEIGHTS.get(f, 1.0) for f in failures)
        wmax             = sum(self.LAYER_WEIGHTS.values())
        wav2vec_failed   = "wav2vec_combined" in failures
        high_conf_count  = sum(1 for f in failures if f in HIGH_CONF_AI_LAYERS)
        has_high_conf    = high_conf_count > 0

        ai_prob  = round(min(100.0, (wfail / wmax) * 100 * 2.0), 1)
        hum_prob = round(max(0.0, 100.0 - ai_prob), 1)
        conf     = min(1.0, wfail / (wmax * 0.35))

        # ── Option 1+2: Weighted decision rules ──────────────
        # Rule 1: High weighted score + at least 1 high-conf AI layer
        if wfail >= W_AI_THRESHOLD and has_high_conf:
            verdict, is_ai, is_suspicious = "AI_VOICE", True, False
            rule_hit = "Rule1:HighScore+HiConf"

        # Rule 2 (Option 2): wav2vec fails + any other layer → AI_VOICE
        # wav2vec alone failing on a real human call is extremely rare
        elif wav2vec_failed and fc >= 2:
            verdict, is_ai, is_suspicious = "AI_VOICE", True, False
            rule_hit = "Rule2:wav2vec+other"

        # Rule 3 (Option 2): 2+ high-conf AI layers fail (no wav2vec needed)
        # e.g. TTS artifacts + prosodic naturalness both failing = strong AI signal
        elif high_conf_count >= 2:
            verdict, is_ai, is_suspicious = "AI_VOICE", True, False
            rule_hit = "Rule3:2+HiConf"

        # Rule 4 (Option 2): wav2vec alone → SUSPICIOUS minimum
        # Single ML failure is notable but not conclusive
        elif wav2vec_failed:
            verdict, is_ai, is_suspicious = "SUSPICIOUS", False, True
            rule_hit = "Rule4:wav2vec_solo"

        # Rule 5: weighted score above suspicious threshold
        elif wfail >= W_SUSPICIOUS_THRESHOLD:
            verdict, is_ai, is_suspicious = "SUSPICIOUS", False, True
            rule_hit = "Rule5:WeightedSusp"

        # Rule 6: clean audio
        else:
            verdict, is_ai, is_suspicious = "HUMAN", False, False
            rule_hit = "Rule6:Clean"

        logger.debug(
            f"[Verdict] {rule_hit} | wfail={wfail:.1f}/{wmax:.1f} "
            f"fc={fc} hc={high_conf_count} wav2vec={wav2vec_failed} → {verdict}"
        )

        ms = round((time.perf_counter()-t0)*1000, 1)
        return DetectionResult(
            verdict=verdict, is_ai=is_ai, is_fraud=False,
            is_suspicious=is_suspicious,
            confidence=float(conf),
            ai_probability=ai_prob, human_probability=hum_prob,
            failure_count=fc, failures=failures,
            layer_scores=layer_scores, latency_ms=ms,
            fraud_keywords=[], transcript=transcript,
            details={
                "ml_available":       self.use_ml,
                "audio_duration_sec": round(len(audio)/sr, 2),
                "total_layers":       10,
                "weighted_score":     round(wfail, 2),
                "weighted_max":       round(wmax, 2),
                "high_conf_count":    high_conf_count,
                "verdict_rule":       rule_hit,
            }
        )


_detector_instance: Optional[VoiceGuardDetector] = None

def get_detector(use_ml: bool = True) -> VoiceGuardDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = VoiceGuardDetector(use_ml=use_ml)
    return _detector_instance