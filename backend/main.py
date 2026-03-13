"""
PhonoForensics v4.2 — Complete Backend
Verdict:  AI_FRAUD (red) | AI_VOICE (red) | SUSPICIOUS (yellow) | HUMAN (green)
Reports:  Excel auto-append for AI_FRAUD, AI_VOICE, SUSPICIOUS — never for HUMAN
Whisper:  Transcribes every chunk → keyword detection
"""
import asyncio, base64, io, json, logging, os, tempfile, time
from typing import Optional
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa, uvicorn, httpx, openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from detector import get_detector, DetectionResult

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
TARGET_SR    = 16000
REPORTS_FILE = Path("reports.xlsx")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "YOUR_OPENROUTER_KEY_HERE")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL   = "google/gemini-flash-1.5"
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("voiceguard.server")

# ─────────────────────────────────────────────────────────────
# WHISPER
# ─────────────────────────────────────────────────────────────
_whisper_model = None

def get_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return None if _whisper_model == "unavailable" else _whisper_model
    try:
        import whisper
        logger.info(f"Loading Whisper '{WHISPER_MODEL_SIZE}'...")
        _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        logger.info(f"✓ Whisper '{WHISPER_MODEL_SIZE}' ready")
    except Exception as e:
        logger.warning(f"Whisper unavailable: {e}  — install: pip install openai-whisper")
        _whisper_model = "unavailable"
    return None if _whisper_model == "unavailable" else _whisper_model

def transcribe_audio(audio: np.ndarray, sr: int) -> str:
    model = get_whisper()
    if model is None:
        return ""
    tmp_path = None
    try:
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        audio = audio.astype(np.float32)
        mx = np.max(np.abs(audio))
        if mx > 1e-6: audio = audio / mx

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, audio, 16000)

        result = model.transcribe(
            tmp_path,
            language=None,       # auto-detect Hindi / English / Hinglish
            task="transcribe",
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,
        )
        transcript = result.get("text", "").strip()
        logger.info(f"[Whisper] '{transcript[:100]}'")
        return transcript
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        return ""
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except: pass


# ─────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="PhonoForensics", version="4.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    logger.info("PhonoForensics v4.2 starting...")
    use_ml = os.environ.get("VOICEGUARD_ML", "1") == "1"
    loop   = asyncio.get_event_loop()
    await asyncio.gather(
        loop.run_in_executor(None, lambda: get_detector(use_ml=use_ml)),
        loop.run_in_executor(None, get_whisper),
    )
    _init_excel()
    logger.info("✓ Ready")


# ─────────────────────────────────────────────────────────────
# AUDIO DECODE
# ─────────────────────────────────────────────────────────────
def decode_audio(data: bytes, fmt: str = "") -> Optional[np.ndarray]:
    """Try every decode method. 3gp/m4a/amr from Android needs ffmpeg temp file."""
    for fn in [
        lambda: _sf_decode(data),
        lambda: _librosa_decode(data),
        lambda: _ffmpeg_decode(data, fmt),
        lambda: np.frombuffer(data, dtype=np.float32).copy() if len(data)%4==0 else None,
        lambda: (np.frombuffer(data, dtype=np.int16).astype(np.float32)/32768.0).copy() if len(data)%2==0 else None,
    ]:
        try:
            r = fn()
            if r is not None and len(r) > 500:
                return r.astype(np.float32)
        except: pass
    return None

def _sf_decode(data):
    buf = io.BytesIO(data)
    a, sr = sf.read(buf, dtype="float32")
    if a.ndim > 1: a = a.mean(axis=1)
    if sr != TARGET_SR: a = librosa.resample(a, orig_sr=sr, target_sr=TARGET_SR)
    return a

def _librosa_decode(data):
    buf = io.BytesIO(data)
    a, _ = librosa.load(buf, sr=TARGET_SR, mono=True)
    return a

def _ffmpeg_decode(data: bytes, fmt: str = "") -> Optional[np.ndarray]:
    """ffmpeg handles 3gp/m4a/amr/aac that soundfile and librosa cannot decode."""
    import subprocess, shutil
    if not shutil.which("ffmpeg"):
        return None
    ext = ".3gp" if fmt in ("3gp", "live", "", "live_call") else f".{fmt}"
    tmp_in = tmp_out = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            tmp_in = f.name
        tmp_out = tmp_in + ".wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in,
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", tmp_out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
        )
        if result.returncode != 0:
            logger.debug(f"ffmpeg returned {result.returncode} for fmt={fmt}")
            return None
        a, sr = sf.read(tmp_out, dtype="float32")
        if a.ndim > 1: a = a.mean(axis=1)
        if sr != TARGET_SR: a = librosa.resample(a, orig_sr=sr, target_sr=TARGET_SR)
        logger.info(f"ffmpeg decoded {fmt} → {len(a)/TARGET_SR:.1f}s audio")
        return a
    except Exception as e:
        logger.debug(f"ffmpeg decode error: {e}")
        return None
    finally:
        for p in [tmp_in, tmp_out]:
            if p:
                try: os.unlink(p)
                except: pass


# ─────────────────────────────────────────────────────────────
# EXCEL — auto-append, never overwrite
# ─────────────────────────────────────────────────────────────
EXCEL_HEADERS = [
    "#", "Timestamp", "Caller Name", "Phone Number", "File Name", "Verdict",
    "Confidence %", "AI Probability %", "Human Probability %",
    "Layers Failed", "Transcript", "Fraud Keywords",
    "Source", "IP Address", "District", "City", "Region", "ZIP", "Country", "ISP",
    "AI Reasoning",
]
EXCEL_COL_WIDTHS = [4, 20, 22, 18, 28, 14, 13, 14, 15, 12, 45, 30, 12, 16, 16, 14, 16, 10, 12, 22, 45]

def _init_excel():
    if REPORTS_FILE.exists():
        return
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "VoiceGuard Reports"
    hdr_fill = PatternFill(start_color="0B0C1A", end_color="0B0C1A", fill_type="solid")
    for c, h in enumerate(EXCEL_HEADERS, 1):
        cell = ws.cell(1, c, h)
        cell.fill = hdr_fill
        cell.font = Font(color="6C63FF", bold=True, size=10)
        cell.alignment = Alignment(horizontal="center", vertical="center")
    for i, w in enumerate(EXCEL_COL_WIDTHS, 1):
        if i <= 26:
            ws.column_dimensions[chr(64+i)].width = w
    ws.row_dimensions[1].height = 20
    wb.save(REPORTS_FILE)
    logger.info(f"✓ Excel report file created: {REPORTS_FILE}")

def _append_report(d: dict):
    """
    Append one row to Excel. Only called for AI_FRAUD, AI_VOICE, SUSPICIOUS.
    Never called for HUMAN.
    Handles file-locked gracefully — saves to backup name if needed.
    """
    # Colour coding per verdict
    verdict = d.get("verdict", "HUMAN")
    if verdict == "AI_FRAUD":
        clr = "FF1744"   # red
    elif verdict == "AI_VOICE":
        clr = "FF1744"   # red
    elif verdict == "SUSPICIOUS":
        clr = "FFD600"   # yellow
    else:
        return  # HUMAN — never save

    try:
        if not REPORTS_FILE.exists():
            _init_excel()

        wb = openpyxl.load_workbook(REPORTS_FILE)
        ws = wb.active
        row = ws.max_row + 1

        conf_pct = round(d.get("confidence", 0) * 100, 1)
        vals = [
            row - 1,                                              # #
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),        # Timestamp
            d.get("caller_name", ""),                            # Caller Name
            d.get("phone_number", ""),                           # Phone Number
            d.get("file_name", "live"),                          # File Name
            verdict,                                              # Verdict
            conf_pct,                                            # Confidence %
            d.get("ai_probability", 0),                         # AI Prob %
            d.get("human_probability", 100),                    # Human Prob %
            d.get("failure_count", 0),                          # Layers Failed
            d.get("transcript", ""),                            # Transcript
            ", ".join(d.get("fraud_keywords", [])),             # Fraud Keywords
            d.get("source", "live"),                            # Source
            d.get("ip", ""),                                    # IP
            d.get("district", ""),                              # District
            d.get("city", ""),                                  # City
            d.get("region", ""),                                # Region
            d.get("zip", ""),                                   # ZIP
            d.get("country", ""),                               # Country
            d.get("isp", ""),                                   # ISP
            d.get("ai_reasoning", ""),                          # AI Reasoning
        ]

        fill       = PatternFill(start_color=clr, end_color=clr, fill_type="solid")
        verdict_col = 4   # column D = Verdict

        for c, val in enumerate(vals, 1):
            cell = ws.cell(row, c, val)
            cell.alignment = Alignment(vertical="center", wrap_text=(c in (9, 10, 16)))
            if c == verdict_col:
                cell.fill = fill
                cell.font = Font(bold=True, color="000000", size=10)

        ws.row_dimensions[row].height = 18

        # Try saving — handle locked file
        try:
            wb.save(REPORTS_FILE)
            logger.info(f"✓ Report saved row {row-1}: {verdict}")
        except PermissionError:
            # File is open in Excel — save to backup
            backup = REPORTS_FILE.with_name(f"reports_backup_{int(time.time())}.xlsx")
            wb.save(backup)
            logger.warning(f"Excel locked — saved to {backup}")

    except Exception as e:
        logger.error(f"Excel append error: {e}")


# ─────────────────────────────────────────────────────────────
# IP LOOKUP
# ─────────────────────────────────────────────────────────────
async def get_ip_info(ip: str) -> dict:
    is_local = (not ip or ip in ("127.0.0.1", "localhost") or
                ip.startswith(("192.168.", "10.", "172.")))
    if is_local:
        try:
            async with httpx.AsyncClient(timeout=4.0) as cl:
                pub = await cl.get("https://api.ipify.org?format=json")
                ip  = pub.json().get("ip", ip)
            logger.info(f"[IP] Local caller — resolved public IP: {ip}")
        except:
            return {"ip": ip, "city": "Local", "region": "Private",
                    "country": "N/A", "isp": "Local Network",
                    "location_string": "Local Network",
                    "district": "", "zip": ""}
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(
                f"http://ip-api.com/json/{ip}"
                "?fields=status,city,district,regionName,country,isp,org,query,zip,as"
            )
            d = r.json()
        if d.get("status") == "success":
            city     = d.get("city", "")
            district = d.get("district", "")   # e.g. "Raigad", "Thane"
            region   = d.get("regionName", "")
            country  = d.get("country", "")
            zipcode  = d.get("zip", "")
            isp      = d.get("isp") or d.get("org", "?")

            # Build detailed location string: District > City, Region (ZIP)
            parts = []
            if district and district != city:
                parts.append(district)
            if city:
                parts.append(city)
            if region and region not in parts:
                parts.append(region)
            if country:
                parts.append(country)
            loc_str = ", ".join(parts)
            if zipcode:
                loc_str += f" - {zipcode}"

            return {
                "ip":              d.get("query", ip),
                "city":            city,
                "district":        district,
                "region":          region,
                "country":         country,
                "zip":             zipcode,
                "isp":             isp,
                "location_string": loc_str,
            }
    except: pass
    return {"ip": ip, "city": "?", "district": "", "region": "?",
            "country": "?", "zip": "", "isp": "?", "location_string": "Unavailable"}


# ─────────────────────────────────────────────────────────────
# LAYER 4: LLM (OpenRouter / Gemini)
# ─────────────────────────────────────────────────────────────
async def llm_analyze(audio: np.ndarray, sr: int, layer_scores: dict,
                       failures: list, phys_verdict: str, transcript: str = "") -> dict:
    NO = {"ai_verdict": phys_verdict, "ai_confidence": 0,
          "ai_reasoning": "OpenRouter key not configured", "ai_available": False}
    if OPENROUTER_API_KEY == "YOUR_OPENROUTER_KEY_HERE":
        return NO
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt, matplotlib.gridspec as gs

        fig = plt.figure(figsize=(9, 5), facecolor="#0B0C1A")
        g   = gs.GridSpec(2, 1, hspace=0.5, figure=fig)

        ax1 = fig.add_subplot(g[0])
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr,
            x_axis="time", y_axis="mel", ax=ax1, cmap="magma")
        ax1.set_title(
            f"Mel Spectrogram | Physics:{phys_verdict} | Fails:{len(failures)}/10",
            color="#C5C3FF", fontsize=9)

        ax2 = fig.add_subplot(g[1])
        try:
            f0  = librosa.yin(audio, fmin=60, fmax=500, sr=sr)
            t   = np.linspace(0, len(audio)/sr, len(f0))
            ax2.plot(t, np.where(f0>0, f0, np.nan), color="#6C63FF", lw=1.2, label="Pitch F0")
            rms = librosa.feature.rms(y=audio)[0]
            tr  = np.linspace(0, len(audio)/sr, len(rms))
            ax2.plot(tr, rms*500, color="#00E676", lw=1.0, alpha=0.7, label="Energy")
            ax2.legend(facecolor="#12132A", labelcolor="#C5C3FF", fontsize=7)
        except: pass
        ax2.set_title("Pitch & Energy", color="#C5C3FF", fontsize=9)
        ax2.set_facecolor("#0B0C1A")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=80, bbox_inches="tight", facecolor="#0B0C1A")
        plt.close(fig); buf.seek(0)
        img = base64.b64encode(buf.read()).decode()

        scores_str = "\n".join(f"  {k}: {round(v,3)}" for k,v in layer_scores.items())
        transcript_line = f'\nTranscript: "{transcript}"' if transcript else ""
        prompt = f"""You are an expert AI voice fraud detection system.
Analyze the spectrogram and pitch/energy charts for AI voice indicators.{transcript_line}

Physics results:
- Verdict: {phys_verdict}
- Failed layers ({len(failures)}/10): {', '.join(failures) if failures else 'None'}
- Layer scores:\n{scores_str}

AI voice indicators:
- Spectrogram: too-clean bands, missing noise, symmetric formants
- Pitch: unnaturally smooth, no micro-jitter, robotic contour
- Energy: monotonous amplitude, no breath pauses

Respond ONLY in exact JSON (no markdown):
{{"verdict":"AI_VOICE","confidence":87,"reasoning":"Clear TTS pattern"}}
verdict must be exactly: AI_VOICE or HUMAN"""

        async with httpx.AsyncClient(timeout=20.0) as c:
            resp = await c.post(OPENROUTER_URL,
                headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}",
                         "Content-Type": "application/json",
                         "HTTP-Referer": "https://voiceguardai.app",
                         "X-Title": "PhonoForensics"},
                json={"model": OPENROUTER_MODEL,
                      "messages": [{"role": "user", "content": [
                          {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
                          {"type": "text", "text": prompt}
                      ]}],
                      "max_tokens": 150, "temperature": 0.05})
        d = resp.json()
        c = d["choices"][0]["message"]["content"].strip()
        if "```" in c: c = c.split("```")[1].replace("json","").strip()
        if "{" in c:   c = c[c.index("{"):c.rindex("}")+1]
        r = json.loads(c)
        v = r.get("verdict", "HUMAN")
        if v not in ("AI_VOICE", "HUMAN"):
            v = "AI_VOICE" if "AI" in v else "HUMAN"
        return {"ai_verdict": v, "ai_confidence": int(r.get("confidence", 50)),
                "ai_reasoning": r.get("reasoning", ""), "ai_available": True}

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {"ai_verdict": phys_verdict, "ai_confidence": 0,
                "ai_reasoning": f"LLM error: {str(e)[:60]}", "ai_available": False}


# ─────────────────────────────────────────────────────────────
# VERDICT FUSION
# ─────────────────────────────────────────────────────────────
def fuse_verdict(phys: str, ai: dict) -> str:
    """
    Combine physics verdict with LLM verdict.
    AI_FRAUD is a HARD OVERRIDE — LLM can NEVER downgrade it.
    """
    # ── HARD RULE: AI_FRAUD is always final — nothing can override it ──
    if phys == "AI_FRAUD":
        return "AI_FRAUD"

    if not ai.get("ai_available"):
        return phys

    av = ai["ai_verdict"]
    ac = ai.get("ai_confidence", 0)

    if phys == "SUSPICIOUS":
        return "AI_VOICE" if av == "AI_VOICE" and ac >= 75 else "SUSPICIOUS"

    # phys is AI_VOICE or HUMAN
    if ac >= 70: return av
    if ac >= 40:
        return "HUMAN" if av == "HUMAN" and phys == "HUMAN" else \
               "AI_VOICE" if "AI" in str(av) or "AI" in str(phys) else "HUMAN"
    return phys

def result_to_dict(r: DetectionResult) -> dict:
    return {
        "verdict":           r.verdict,
        "is_ai":             r.is_ai,
        "is_fraud":          r.is_fraud,
        "is_suspicious":     getattr(r, "is_suspicious", False),
        "confidence":        round(r.confidence, 4),
        "ai_probability":    r.ai_probability,
        "human_probability": r.human_probability,
        "failure_count":     r.failure_count,
        "failures":          r.failures,
        "layer_scores":      {k: round(float(v), 6) for k, v in r.layer_scores.items()},
        "latency_ms":        round(r.latency_ms, 1),
        "fraud_keywords":    r.fraud_keywords,
        "transcript":        r.transcript,
        "details":           r.details,
    }


# ─────────────────────────────────────────────────────────────
# FULL 4-LAYER PIPELINE
# ─────────────────────────────────────────────────────────────
async def full_pipeline(audio: np.ndarray, phone: str = "", caller_name: str = "",
                         source: str = "live", caller_ip: str = "",
                         file_name: str = "live") -> dict:
    loop = asyncio.get_event_loop()
    det  = get_detector()

    # Layer 1: Whisper transcription
    transcript = await loop.run_in_executor(
        None, lambda: transcribe_audio(audio, TARGET_SR)
    )

    # Layers 1+2+3: keyword + physics + ML
    result = await loop.run_in_executor(
        None, lambda: det.analyze(audio, sr=TARGET_SR, transcript=transcript)
    )
    payload = result_to_dict(result)
    payload["physics_verdict"] = result.verdict
    payload["transcript"]      = transcript
    payload["file_name"]       = file_name
    payload["phone_number"]    = phone          # ← now tracked through full pipeline
    payload["caller_name"]     = caller_name    # ← caller name from user input

    ip_info = await get_ip_info(caller_ip)
    payload["ip_info"] = ip_info
    payload["source"]  = source

    # Fraud keywords — instant response, no LLM needed
    if result.is_fraud:
        payload["ai_analysis"] = {
            "ai_available":  False,
            "ai_verdict":    "AI_FRAUD",
            "ai_confidence": 100,
            "ai_reasoning":  f"Fraud keywords in speech: {', '.join(result.fraud_keywords)}",
        }
        # Auto-save to Excel
        _append_report({
            **payload,
            "ip":           ip_info.get("ip",""),
            "district":     ip_info.get("district",""),
            "city":         ip_info.get("city",""),
            "region":       ip_info.get("region",""),
            "zip":          ip_info.get("zip",""),
            "country":      ip_info.get("country",""),
            "isp":          ip_info.get("isp",""),
            "phone_number": phone,
            "caller_name":  caller_name,
            "ai_reasoning": payload["ai_analysis"]["ai_reasoning"],
            "file_name":    file_name,
        })
        logger.info(f"[FRAUD] Keywords:{result.fraud_keywords} transcript:'{transcript[:60]}'")
        return payload

    # Layer 4: LLM
    ai = await llm_analyze(audio, TARGET_SR, result.layer_scores,
                            result.failures, result.verdict, transcript)
    payload["ai_analysis"] = ai

    # Fuse physics + LLM
    final = fuse_verdict(result.verdict, ai)

    # ════════════════════════════════════════════════════════
    # FINAL KEYWORD SAFETY NET — runs AFTER all 4 layers
    # If the transcript contains fraud keywords, override
    # everything regardless of what physics/ML/LLM said.
    # ════════════════════════════════════════════════════════
    if transcript:
        from detector import detect_fraud_keywords
        kw_found, kw_list = detect_fraud_keywords(transcript)
        if kw_found:
            logger.warning(f"[PIPELINE KEYWORD NET] Overriding '{final}' → AI_FRAUD | {kw_list}")
            final                     = "AI_FRAUD"
            payload["verdict"]        = "AI_FRAUD"
            payload["is_ai"]          = True
            payload["is_fraud"]       = True
            payload["ai_probability"] = 100.0
            payload["human_probability"] = 0.0
            payload["fraud_keywords"] = kw_list
            payload["ai_analysis"]    = {
                "ai_available":  True,
                "ai_verdict":    "AI_FRAUD",
                "ai_confidence": 100,
                "ai_reasoning":  f"KEYWORD OVERRIDE: {', '.join(kw_list)}",
            }
            payload["ip_info"] = ip_info
            payload["source"]  = source
            _append_report({
                **payload,
                "ip":           ip_info.get("ip",""),
                "district":     ip_info.get("district",""),
                "city":         ip_info.get("city",""),
                "region":       ip_info.get("region",""),
                "zip":          ip_info.get("zip",""),
                "country":      ip_info.get("country",""),
                "isp":          ip_info.get("isp",""),
                "phone_number": phone,
                "caller_name":  caller_name,
                "ai_reasoning": f"KEYWORD OVERRIDE: {', '.join(kw_list)}",
                "file_name":    file_name,
            })
            return payload

    payload["verdict"] = final
    payload["is_ai"]   = final not in ("HUMAN", "SUSPICIOUS")

    # Auto-save to Excel for AI_VOICE and SUSPICIOUS (not HUMAN)
    if final in ("AI_VOICE", "AI_FRAUD", "SUSPICIOUS"):
        _append_report({
            **payload,
            "ip":           ip_info.get("ip",""),
            "district":     ip_info.get("district",""),
            "city":         ip_info.get("city",""),
            "region":       ip_info.get("region",""),
            "zip":          ip_info.get("zip",""),
            "country":      ip_info.get("country",""),
            "isp":          ip_info.get("isp",""),
            "phone_number": phone,
            "caller_name":  caller_name,
            "ai_reasoning": ai.get("ai_reasoning",""),
            "file_name":    file_name,
        })

    logger.info(
        f"[{source}] transcript:'{transcript[:40]}' "
        f"physics:{result.verdict} LLM:{ai.get('ai_verdict','?')}({ai.get('ai_confidence',0)}%) "
        f"FINAL:{final} fails:{result.failure_count}/10 {result.latency_ms}ms"
    )
    return payload


# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class AudioRequest(BaseModel):
    audio_b64:    str
    format:       str = "wav"
    phone_number: str = "Unknown"
    caller_name:  str = ""        # caller's name (optional, entered by user)
    caller_ip:    str = ""
    source:       str = "live"
    transcript:   str = ""
    file_name:    str = "live"

class ReportRequest(BaseModel):
    phone_number:      str   = "Unknown"
    caller_name:       str   = ""         # caller's name (optional)
    verdict:           str   = "HUMAN"
    ai_probability:    float = 0.0
    human_probability: float = 100.0
    confidence:        float = 0.0
    failure_count:     int   = 0
    failures:          list  = []
    caller_ip:         str   = ""
    source:            str   = "live"
    fraud_keywords:    list  = []
    ai_reasoning:      str   = ""
    transcript:        str   = ""
    file_name:         str   = "live"


# ─────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────
@app.post("/analyze/audio")
async def analyze_audio(req: AudioRequest, request: Request):
    try:
        raw = base64.b64decode(req.audio_b64)
        logger.info(f"[DECODE] fmt={req.format} raw_bytes={len(raw)} b64_len={len(req.audio_b64)}")
        logger.info(f"[DECODE] first_bytes={raw[:12].hex()}")

        # Try each method individually and log result
        audio = None

        try:
            audio = _sf_decode(raw)
            if audio is not None and len(audio) > 100:
                logger.info(f"[DECODE] soundfile OK → {len(audio)} samples")
            else:
                logger.info(f"[DECODE] soundfile failed or too short")
                audio = None
        except Exception as e:
            logger.info(f"[DECODE] soundfile error: {e}")

        if audio is None:
            try:
                audio = _librosa_decode(raw)
                if audio is not None and len(audio) > 100:
                    logger.info(f"[DECODE] librosa OK → {len(audio)} samples")
                else:
                    logger.info(f"[DECODE] librosa failed or too short")
                    audio = None
            except Exception as e:
                logger.info(f"[DECODE] librosa error: {e}")

        if audio is None:
            try:
                audio = _ffmpeg_decode(raw, req.format)
                if audio is not None and len(audio) > 100:
                    logger.info(f"[DECODE] ffmpeg OK → {len(audio)} samples")
                else:
                    logger.info(f"[DECODE] ffmpeg failed or too short")
                    audio = None
            except Exception as e:
                logger.info(f"[DECODE] ffmpeg error: {e}")

        if audio is None:
            logger.error(f"[DECODE] ALL methods failed for fmt={req.format} bytes={len(raw)}")
            return JSONResponse({
                "error": f"Could not decode audio. fmt={req.format} bytes={len(raw)} header={raw[:12].hex()}"
            }, status_code=400)

        ip    = req.caller_ip or request.client.host
        fname = req.file_name if req.file_name != "live" else (
                req.source if req.source != "live" else "live_call")
        payload = await full_pipeline(audio, req.phone_number, req.caller_name,
                                      req.source, ip, fname)
        return JSONResponse(payload)
    except Exception as e:
        logger.error(f"analyze error: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/analyze/base64")
async def analyze_b64(req: AudioRequest, request: Request):
    return await analyze_audio(req, request)

@app.get("/ip/lookup")
async def ip_lookup(ip: str):
    return JSONResponse(await get_ip_info(ip))

@app.post("/report/call")
async def report_call(req: ReportRequest):
    """Manual report save from frontend (for non-auto cases or re-saves)."""
    try:
        # Only save non-HUMAN
        if req.verdict == "HUMAN":
            return JSONResponse({"success": True, "skipped": True,
                                 "reason": "HUMAN verdicts are not reported"})
        ip = await get_ip_info(req.caller_ip) if req.caller_ip else {}
        _append_report({
            "verdict":           req.verdict,
            "confidence":        req.confidence,
            "ai_probability":    req.ai_probability,
            "human_probability": req.human_probability,
            "failure_count":     req.failure_count,
            "transcript":        req.transcript,
            "fraud_keywords":    req.fraud_keywords,
            "source":            req.source,
            "file_name":         req.file_name,
            "phone_number":      req.phone_number,
            "caller_name":       req.caller_name,
            "ip":                req.caller_ip,
            "district":          ip.get("district",""),
            "city":              ip.get("city",""),
            "region":            ip.get("region",""),
            "zip":               ip.get("zip",""),
            "country":           ip.get("country",""),
            "isp":               ip.get("isp",""),
            "ai_reasoning":      req.ai_reasoning,
        })
        return JSONResponse({"success": True,
                             "location": ip.get("location_string","Unknown")})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/report/download")
async def download_report():
    if not REPORTS_FILE.exists():
        _init_excel()
    return FileResponse(
        str(REPORTS_FILE),
        filename="voiceguard_reports.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

@app.get("/report/list")
async def list_reports():
    try:
        if not REPORTS_FILE.exists():
            return JSONResponse({"reports": [], "total": 0})
        wb   = openpyxl.load_workbook(REPORTS_FILE)
        ws   = wb.active
        hdrs = [ws.cell(1, c).value for c in range(1, ws.max_column+1)]
        rows = [
            {hdrs[c-1]: ws.cell(r, c).value for c in range(1, ws.max_column+1)}
            for r in range(2, ws.max_row+1)
        ]
        # Filter: only AI_VOICE, AI_FRAUD, SUSPICIOUS
        filtered = [
            row for row in rows
            if row.get("Verdict") in ("AI_VOICE", "AI_FRAUD", "SUSPICIOUS")
        ]
        return JSONResponse({"reports": filtered, "total": len(filtered)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/health")
async def health():
    whisper_ok = _whisper_model is not None and _whisper_model != "unavailable"
    return {
        "status":        "ok",
        "version":       "4.2.0",
        "whisper":       WHISPER_MODEL_SIZE if whisper_ok else "unavailable",
        "whisper_ready": whisper_ok,
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000,
                reload=False, workers=1, log_level="info")