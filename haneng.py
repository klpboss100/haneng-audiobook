"""
한영소리 (KOEN Audio)
─────────────────────────────────────────
실행: python -m streamlit run haneng.py
─────────────────────────────────────────
한국어 소설 → 문학적 영어 번역(원고 전체) → 한국어/영어 각각 화자(M/W) 태그 → 한국어/영어 MP3 동시 제작
"""

import streamlit as st
import re, io, time, json, os, random
import lameenc

# 환경 자동 감지: Streamlit Cloud = /home/appuser
IS_CLOUD = os.environ.get('HOME', '') == '/home/appuser'
from google import genai
from google.genai import types

# ═══════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════
SAMPLE_RATE     = 24000
MAX_CHUNK_CHARS = 900   # TTS 1회 호출당 최대 글자수 (길면 잡음/에코 위험 ↑ - 구글 TTS 알려진 한계)
SEED_BASE       = 7     # 목소리 톤이 매 호출마다 튀는 것을 완화 (best-effort 결정성)
CONFIG_FILE     = "config_haneng.json"

MALE_VOICES_KO   = ["Charon", "Fenrir"]
FEMALE_VOICES_KO = ["Kore", "Aoede"]
MALE_VOICES_EN   = ["Charon", "Orus"]
FEMALE_VOICES_EN = ["Aoede", "Leda"]

VOICE_DESC = {
    "Charon": "깊고 성숙한 목소리",
    "Fenrir": "강하고 힘있는 목소리",
    "Kore":   "감성적이고 따뜻한 목소리",
    "Aoede":  "서사적·지적인 목소리",
    "Orus":   "중성적이고 안정적인 목소리",
    "Leda":   "따뜻하고 친근한 목소리",
}


# ═══════════════════════════════════════════
# 설정 저장/로드 (로컬 전용 — 웹에서는 저장 안 함)
# ═══════════════════════════════════════════
def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            return json.load(open(CONFIG_FILE, encoding="utf-8"))
        except Exception:
            pass
    return {"api_key": "", "project_name": ""}


def save_config(data: dict):
    json.dump(data, open(CONFIG_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════
# 화자(M/W) 태그 변환 — 감정 태그 없음 (한국어/영어 원고 모두에 사용)
# ═══════════════════════════════════════════
TAG_PROMPT = """당신은 소설 원고에 화자 태그를 추가하는 전문가입니다. 원고는 한국어 또는 영어일 수 있으며,
원고의 언어를 그대로 유지한 채(번역하지 말고) 태그만 추가하세요.

## 절대 규칙
- 원고 텍스트를 절대 수정·번역·요약·생략하지 마세요
- 원고 내용 전체를 빠짐없이 원래 언어 그대로 출력하세요
- 태그만 각 줄 앞에 추가하세요

## 태그 형식
[화자] 텍스트   (예: [M] 그는 조용히 웃었다. / [M] He smiled quietly.)

## 화자 종류 (이 두 가지만 사용, 감정 태그는 절대 사용 금지)
- [M] : 내레이션(지문·묘사) + 모든 남자 대화
- [W] : 모든 여자 대화

## 처리 방법
1. 챕터 제목 → [M]
2. 내레이션·지문·묘사 → [M]
3. 대화문("...") → 앞뒤 문맥(대명사, 호칭, 서술 등)으로 화자 성별 판단 (여자=[W], 남자/중성/불명=[M])
4. 내레이션과 대화가 섞인 문단 → 반드시 줄 단위로 분리

지금 바로 아래 원고 전체에 화자 태그를 추가하세요 (언어는 원문 그대로 유지):

{manuscript}"""


def convert_tags(api_key: str, manuscript: str, model: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=TAG_PROMPT.format(manuscript=manuscript)
    )
    text = response.text
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    return text


def parse_tagged_script(text: str):
    lines = []
    tag_pattern = re.compile(r'^\[([MW])\]\s*(.+)$')
    for raw_line in text.split('\n'):
        stripped = raw_line.strip()
        if not stripped:
            continue
        m = tag_pattern.match(stripped)
        if m:
            speaker, content = m.groups()
            lines.append({'speaker': speaker, 'text': content.strip()})
    return lines


def group_into_segments(lines):
    """연속된 같은 화자 줄을 하나의 세그먼트로 묶음"""
    if not lines:
        return []
    segments = []
    cur_spk = lines[0]['speaker']
    cur_lines = [lines[0]]
    for line in lines[1:]:
        if line['speaker'] == cur_spk:
            cur_lines.append(line)
        else:
            segments.append({'speaker': cur_spk, 'lines': cur_lines})
            cur_spk = line['speaker']
            cur_lines = [line]
    segments.append({'speaker': cur_spk, 'lines': cur_lines})
    return segments


def get_voice_for_speaker(spk: str, voices: dict) -> str:
    return voices.get(spk, voices.get('M'))


def merge_segments_by_voice(segs, voices):
    """같은 목소리로 연결되는 연속 세그먼트 병합 → API 호출 감소"""
    if not segs:
        return []
    merged = []
    for seg in segs:
        voice = get_voice_for_speaker(seg['speaker'], voices)
        if merged and get_voice_for_speaker(merged[-1]['speaker'], voices) == voice:
            merged[-1]['lines'].extend(seg['lines'])
        else:
            merged.append({'speaker': seg['speaker'], 'lines': list(seg['lines'])})
    return merged


def chunk_segment_lines(lines, max_chars: int):
    chunks, current, current_len = [], [], 0
    for line in lines:
        size = len(line['text']) + 10
        if current_len + size > max_chars and current:
            chunks.append(current)
            current, current_len = [line], size
        else:
            current.append(line)
            current_len += size
    if current:
        chunks.append(current)
    return chunks


def build_speaker_script(lines) -> str:
    return "\n".join(l['text'] for l in lines)


# ═══════════════════════════════════════════
# 번역 (원고 전체를 한 번에 — 문맥이 끊기지 않아 번역 품질이 좋음)
# ═══════════════════════════════════════════
TRANSLATE_PROMPT = """당신은 한국 문학을 영어로 옮기는 전문 문학 번역가입니다.
아래 한국어 소설 원고를 자연스럽고 문학적인 영어(literary English prose)로 번역하세요.

## 번역 원칙
- 직역이 아닌 의역으로, 원작의 정서와 분위기를 살릴 것
- 지문·묘사는 소설체 영어로, 대화문은 자연스러운 영어 회화체로
- 인명·지명 등 고유명사는 로마자 표기를 유지
- 원문의 문단 구분을 최대한 그대로 유지
- 번역문 외의 설명, 마크다운, 안내 문구는 절대 출력하지 마세요

번역할 원고:
{manuscript}

지금 위 원고 전체를 영어로 번역하세요. 번역문만 출력하세요."""


def translate_to_english(api_key: str, manuscript: str, model: str) -> str:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=TRANSLATE_PROMPT.format(manuscript=manuscript)
    )
    text = response.text.strip()
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    return text


# ═══════════════════════════════════════════
# TTS
# ═══════════════════════════════════════════
def generate_silence(seconds: float) -> bytes:
    return bytes(int(SAMPLE_RATE * seconds) * 2)


def to_pcm_bytes(data):
    """TTS 응답의 오디오 데이터를 항상 순수 bytes로 변환.
    google-genai 버전/환경에 따라 str(base64)나 bytearray로 올 수 있어 방어적으로 처리."""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        import base64
        return base64.b64decode(data)
    raise TypeError(f"예상치 못한 TTS 오디오 데이터 타입: {type(data)}")


def call_tts(client, text, voice_name, tts_model, seed=None, retry=3, status=None):
    rate_limit_retries = 0
    config_kwargs = dict(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )
    )
    if seed is not None:
        config_kwargs["seed"] = seed
    for attempt in range(retry):
        try:
            response = client.models.generate_content(
                model=tts_model,
                contents=text,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            if (response.candidates and
                response.candidates[0].content and
                response.candidates[0].content.parts and
                response.candidates[0].content.parts[0].inline_data):
                return to_pcm_bytes(response.candidates[0].content.parts[0].inline_data.data)
            return generate_silence(0.5)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()
            is_server_err = ("500" in msg or "503" in msg or "INTERNAL" in msg
                              or "UNAVAILABLE" in msg or "DEADLINE_EXCEEDED" in msg)
            if (is_rate_limit or is_server_err) and rate_limit_retries < 10:
                rate_limit_retries += 1
                wait_s = 60 if is_rate_limit else 15
                reason = "API 분당 요청 제한에 걸림" if is_rate_limit else "구글 서버 일시 오류"
                if status is not None:
                    status.markdown(f"⏳ {reason}. {wait_s}초 대기 후 재시도 ({rate_limit_retries}/10)...")
                time.sleep(wait_s)
                continue
            if attempt < retry - 1:
                time.sleep(3)
            else:
                raise e


def merge_to_mp3(pcm_list) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_data = encoder.encode(b"".join(pcm_list))
    return mp3_data + encoder.flush()


def pcm_duration_seconds(pcm_list) -> float:
    total_bytes = sum(len(p) for p in pcm_list)
    return total_bytes / 2 / SAMPLE_RATE


def format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}시간 {m}분 {s}초"
    if m:
        return f"{m}분 {s}초"
    return f"{s}초"


def generate_mp3_for_tagged_text(api_key, tagged_text, voices, tts_model, max_chunk_chars,
                                  label, status, progress):
    """화자 태그가 붙은 텍스트를 화자별 목소리로 청크 생성 후 MP3로 합침"""
    lines = parse_tagged_script(tagged_text)
    segs_raw = group_into_segments(lines)
    segs = merge_segments_by_voice(segs_raw, voices)
    total_chunks = sum(len(chunk_segment_lines(s['lines'], max_chunk_chars)) for s in segs)
    if total_chunks == 0:
        raise ValueError("화자 태그가 붙은 줄을 찾지 못했습니다. 태그 변환을 먼저 실행하세요.")

    client = genai.Client(api_key=api_key)
    pcm_list = []
    chunk_idx = 0
    for seg in segs:
        voice = get_voice_for_speaker(seg['speaker'], voices)
        chunks = chunk_segment_lines(seg['lines'], max_chunk_chars)
        for chunk in chunks:
            chars = sum(len(l['text']) for l in chunk)
            status.markdown(
                f"🎙️ {label} [{seg['speaker']}] ({voice}) — {chunk_idx+1}/{total_chunks} ({chars}자)"
            )
            progress.progress(chunk_idx / total_chunks)
            script = build_speaker_script(chunk)
            seed = SEED_BASE + chunk_idx
            pcm = call_tts(client, script, voice, tts_model, seed=seed, status=status)
            pcm_list.append(pcm)
            chunk_idx += 1
    progress.progress(1.0)
    mp3 = merge_to_mp3(pcm_list)
    duration = pcm_duration_seconds(pcm_list)
    return mp3, duration


# ═══════════════════════════════════════════
# 페이지 설정 & CSS
# ═══════════════════════════════════════════
st.set_page_config(page_title="한영소리 · KOEN Audio", page_icon="📓", layout="wide")

NAVY = "#0f3460"

if st.session_state.pop('_pending_reset', False):
    for _k in ['translated_text', 'tagged_ko', 'tagged_en', 'ko_audio', 'en_audio', 'ko_seconds', 'en_seconds']:
        st.session_state.pop(_k, None)
    st.session_state['manuscript']    = ""
    st.session_state['chapter_name']  = ""
    st.session_state['project_name_input'] = ""

if '_pending_chapter_name' in st.session_state:
    st.session_state['chapter_name'] = st.session_state.pop('_pending_chapter_name')

st.markdown(f"""
<style>
h1 a, h2 a, h3 a {{ display: none !important; }}
[data-testid="stHeaderActionElements"] {{ display: none !important; }}

.step-box {{
    background:#eef2fb;
    border:2px solid {NAVY};
    border-left:6px solid {NAVY};
    border-radius:8px;
    padding:10px 16px;
    margin:20px 0 8px 0;
}}
[data-testid="stSidebar"] {{ background:var(--secondary-background-color); }}
[data-testid="stSidebar"] input[type="text"],
[data-testid="stSidebar"] input[type="password"] {{
    border:1.5px solid {NAVY} !important;
    border-radius:6px !important;
}}
[data-testid="stSidebar"] input[type="text"]:focus,
[data-testid="stSidebar"] input[type="password"]:focus {{
    border:2px solid {NAVY} !important;
    box-shadow:0 0 0 3px rgba(15,52,96,0.2) !important;
}}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
    border:1.5px solid {NAVY} !important;
    border-radius:6px !important;
}}
</style>
""", unsafe_allow_html=True)


def step_header(num, title, subtitle=""):
    sub = f"<small style='color:#666'> — {subtitle}</small>" if subtitle else ""
    return f"<div class='step-box'><b>{num}. {title}</b>{sub}</div>"


# ══════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════
with st.sidebar:
    cfg = load_config()

    st.markdown(f"""
    <div style='background:{NAVY};border-radius:8px;padding:10px 12px;
                margin-bottom:10px;text-align:center'>
        <div style='color:white;font-size:15px;font-weight:800'>⚙️ 필수 설정</div>
        <div style='color:#c9d6e8;font-size:11px;margin-top:3px'>아래 설정 후 원고를 입력하세요</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='background:{NAVY};border-radius:8px 8px 0 0;padding:8px 12px;margin-top:6px'><span style='color:white;font-size:13px;font-weight:800'>🔑 Gemini API Key</span></div><div style='border:2px solid {NAVY};border-top:none;border-radius:0 0 8px 8px;padding:8px 10px;margin-bottom:6px'>", unsafe_allow_html=True)
    if IS_CLOUD:
        api_key = st.text_input("", value="", type="password",
                                 placeholder="AIzaSy...", label_visibility="collapsed",
                                 key="api_key_input",
                                 help="Google AI Studio 무료 발급\nhttps://aistudio.google.com/apikey\n매번 새로 입력해야 합니다 (서버에 저장되지 않음)")
    else:
        api_key = st.text_input("", value=cfg.get("api_key", ""), type="password",
                                 placeholder="AIzaSy...", label_visibility="collapsed",
                                 key="api_key_input",
                                 help="Google AI Studio 무료 발급\nhttps://aistudio.google.com/apikey")
        if api_key != cfg.get("api_key", ""):
            cfg["api_key"] = api_key
            save_config(cfg)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background:{NAVY};border-radius:8px 8px 0 0;padding:8px 12px;margin-top:6px'><span style='color:white;font-size:13px;font-weight:800'>📁 프로젝트명</span></div><div style='border:2px solid {NAVY};border-top:none;border-radius:0 0 8px 8px;padding:8px 10px;margin-bottom:6px'>", unsafe_allow_html=True)
    proj_default = "" if IS_CLOUD else cfg.get("project_name", "")
    project_name = st.text_input("", value=proj_default, placeholder="예: 봄의시작",
                                  label_visibility="collapsed", key="project_name_input")
    if not IS_CLOUD and project_name != cfg.get("project_name", ""):
        cfg["project_name"] = project_name
        save_config(cfg)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background:{NAVY};border-radius:8px 8px 0 0;padding:8px 12px;margin-top:6px'><span style='color:white;font-size:13px;font-weight:800'>🎙️ 한국어 성우</span></div><div style='border:2px solid {NAVY};border-top:none;border-radius:0 0 8px 8px;padding:8px 10px;margin-bottom:6px'>", unsafe_allow_html=True)
    ko_saved = cfg.get("voices_ko", {"M": "Charon", "W": "Kore"})
    col_km, col_kw = st.columns(2)
    with col_km:
        st.caption("🔵 남성(M)")
        ko_m_def = ko_saved.get("M", "Charon")
        ko_m_voice = st.selectbox("", MALE_VOICES_KO,
                                   index=MALE_VOICES_KO.index(ko_m_def) if ko_m_def in MALE_VOICES_KO else 0,
                                   key="ko_m_voice", label_visibility="collapsed",
                                   help="\n".join(f"{v}: {VOICE_DESC.get(v,'')}" for v in MALE_VOICES_KO))
    with col_kw:
        st.caption("🔴 여성(W)")
        ko_w_def = ko_saved.get("W", "Kore")
        ko_w_voice = st.selectbox("", FEMALE_VOICES_KO,
                                   index=FEMALE_VOICES_KO.index(ko_w_def) if ko_w_def in FEMALE_VOICES_KO else 0,
                                   key="ko_w_voice", label_visibility="collapsed",
                                   help="\n".join(f"{v}: {VOICE_DESC.get(v,'')}" for v in FEMALE_VOICES_KO))
    ko_voices = {"M": ko_m_voice, "W": ko_w_voice}
    if not IS_CLOUD and ko_voices != ko_saved:
        cfg["voices_ko"] = ko_voices
        save_config(cfg)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background:{NAVY};border-radius:8px 8px 0 0;padding:8px 12px;margin-top:6px'><span style='color:white;font-size:13px;font-weight:800'>🎙️ 영어 성우</span></div><div style='border:2px solid {NAVY};border-top:none;border-radius:0 0 8px 8px;padding:8px 10px;margin-bottom:6px'>", unsafe_allow_html=True)
    en_saved = cfg.get("voices_en", {"M": "Charon", "W": "Aoede"})
    col_em, col_ew = st.columns(2)
    with col_em:
        st.caption("🔵 남성(M)")
        en_m_def = en_saved.get("M", "Charon")
        en_m_voice = st.selectbox("", MALE_VOICES_EN,
                                   index=MALE_VOICES_EN.index(en_m_def) if en_m_def in MALE_VOICES_EN else 0,
                                   key="en_m_voice", label_visibility="collapsed",
                                   help="\n".join(f"{v}: {VOICE_DESC.get(v,'')}" for v in MALE_VOICES_EN))
    with col_ew:
        st.caption("🔴 여성(W)")
        en_w_def = en_saved.get("W", "Aoede")
        en_w_voice = st.selectbox("", FEMALE_VOICES_EN,
                                   index=FEMALE_VOICES_EN.index(en_w_def) if en_w_def in FEMALE_VOICES_EN else 0,
                                   key="en_w_voice", label_visibility="collapsed",
                                   help="\n".join(f"{v}: {VOICE_DESC.get(v,'')}" for v in FEMALE_VOICES_EN))
    en_voices = {"M": en_m_voice, "W": en_w_voice}
    if not IS_CLOUD and en_voices != en_saved:
        cfg["voices_en"] = en_voices
        save_config(cfg)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='background:{NAVY};border-radius:8px 8px 0 0;padding:8px 12px;margin-top:6px'><span style='color:white;font-size:13px;font-weight:800'>⚙️ 모델 설정</span></div><div style='border:2px solid {NAVY};border-top:none;border-radius:0 0 8px 8px;padding:8px 10px;margin-bottom:6px'>", unsafe_allow_html=True)
    tag_model = st.selectbox("🏷️ 화자 태그 변환", ["gemini-2.5-flash", "gemini-2.5-pro"],
                              index=0, key="tag_model",
                              help="Flash: 빠르고 충분한 품질 (추천)\nPro: 더 정교한 화자 판단")
    translate_model = st.selectbox("🌍 번역 모델", ["gemini-2.5-pro", "gemini-2.5-flash"],
                                    index=0, key="translate_model",
                                    help="Pro: 문학적 품질 우선 (추천)\nFlash: 빠른 번역")
    tts_model = st.selectbox("🔊 TTS 오디오", ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"],
                              index=0, key="tts_model",
                              help="Flash: 빠름·저비용 (테스트용)\nPro: 고품질 (최종 제작용)")
    max_chunk_chars = st.slider("🔊 TTS 청크 최대 글자수", 300, 4000, MAX_CHUNK_CHARS, 100,
                                 key="max_chunk_chars",
                                 help="길수록 뒷부분에 잡음·에코가 생길 위험이 커집니다.\n추천: 900자 이하")
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════
# 메인 헤더
# ══════════════════════════════════════════
col_title, col_reset = st.columns([5, 1])
with col_title:
    st.markdown(f"""
    <div style='margin-bottom:6px;display:grid;grid-template-columns:64px 1fr;align-items:center;gap:14px;max-width:520px'>
      <div style='width:64px;height:64px;background:white;border-radius:12px;
                  display:flex;align-items:center;justify-content:center;
                  box-shadow:0 2px 10px rgba(15,52,96,0.3)'>
        <div style='width:32px;height:20px;border:3px solid {NAVY};border-radius:3px;position:relative'>
          <div style='position:absolute;bottom:-8px;left:-7px;width:44px;height:4px;background:{NAVY};border-radius:2px'></div>
        </div>
      </div>
      <div>
        <div style='line-height:1.1'>
          <span style='font-size:30px;font-weight:800;color:{NAVY}'>한영소리</span>
          <span style='font-size:16px;font-weight:600;color:#6b7280;margin-left:8px'>KOEN Audio</span>
        </div>
        <div style='font-size:13px;color:{NAVY};font-weight:500;margin-top:4px'>
          한국어 소설을 세계로
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    proj_display = f"**{project_name}**" if project_name else "*(프로젝트명 없음)*"
    st.caption(f"프로젝트: {proj_display}  |  KO M={ko_m_voice}/W={ko_w_voice}  |  EN M={en_m_voice}/W={en_w_voice}")
with col_reset:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 새로 시작", use_container_width=True):
        st.session_state['_pending_reset'] = True
        st.rerun()

st.divider()

chapter_name = st.text_input("챕터명 (파일명용)", value="",
                              placeholder="예: chapter_01",
                              key="chapter_name", help="저장 파일명에 사용됩니다")

# ══════════════════════════════════════════
# STEP 1: 한국어 원고 입력
# ══════════════════════════════════════════
st.markdown(step_header("1", "한국어 원고 입력"), unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📂 파일 가져오기 (TXT, DOCX, PDF)",
    type=["txt", "docx", "pdf"], key="file_uploader",
    help="TXT, DOCX, PDF 파일을 직접 불러올 수 있습니다"
)
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.txt'):
            file_text = uploaded_file.read().decode('utf-8', errors='ignore')
        elif uploaded_file.name.endswith('.docx'):
            from docx import Document as DocxDoc
            doc = DocxDoc(io.BytesIO(uploaded_file.read()))
            file_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        elif uploaded_file.name.endswith('.pdf'):
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(uploaded_file.read()))
            file_text = "\n".join(p.extract_text() or "" for p in reader.pages)
        st.session_state['manuscript'] = file_text
        if st.session_state.get('_last_uploaded_name') != uploaded_file.name:
            st.session_state['_last_uploaded_name'] = uploaded_file.name
            st.session_state['_pending_chapter_name'] = os.path.splitext(uploaded_file.name)[0]
            st.rerun()
        st.success(f"✅ {uploaded_file.name} 불러오기 완료 ({len(file_text):,}자)")
    except Exception as e:
        st.error(f"❌ 파일 읽기 오류: {e}. pip install python-docx pypdf 를 실행해 주세요.")

manuscript = st.text_area("", height=250,
    placeholder="여기에 한국어 원고를 붙여넣거나 위에서 파일을 불러오세요...",
    label_visibility="collapsed", key="manuscript")

char_count = len(manuscript) if manuscript else 0
st.markdown(f"<p style='font-size:16px;font-weight:600;color:{NAVY};margin:4px 0'>글자 수: {char_count:,}자</p>",
            unsafe_allow_html=True)

if not api_key:
    st.info("👋 왼쪽 사이드바에서 Gemini API Key를 먼저 입력하세요. 무료 발급: "
            "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)")

has_text = bool(manuscript and manuscript.strip())


# ══════════════════════════════════════════
# STEP 2: 문학적 영어 번역 (원고 전체를 한 번에)
# ══════════════════════════════════════════
st.markdown(step_header("2", "문학적 영어 번역", "원고 전체를 한 번에 번역 — 문맥이 끊기지 않아 품질이 좋음"),
            unsafe_allow_html=True)

if st.button("🌍 영어로 번역", type="primary" if has_text else "secondary",
             disabled=not (api_key and has_text), use_container_width=True):
    with st.status("🌍 문학적 영어로 번역 중...", expanded=True) as status:
        st.write("Gemini가 소설체 영어로 번역하고 있습니다. (30초~1분 소요)")
        try:
            translated = translate_to_english(api_key, manuscript, translate_model)
            st.session_state['translated_text'] = translated
            for _k in ['tagged_ko', 'tagged_en', 'ko_audio', 'en_audio']:
                st.session_state.pop(_k, None)
            status.update(label="✅ 번역 완료", state="complete")
        except Exception as e:
            status.update(label="❌ 오류 발생", state="error")
            st.error(f"❌ {e}")

if 'translated_text' in st.session_state:
    edited_translation = st.text_area("영어 번역 결과 (직접 수정 가능)",
        value=st.session_state['translated_text'], height=250, key="translated_text_edit")
    st.session_state['translated_text'] = edited_translation
    st.markdown(f"<p style='font-size:14px;color:{NAVY};margin:4px 0'>번역 글자 수: {len(edited_translation):,}자</p>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════
# STEP 3: 화자 태그 변환 (한국어 원고 + 영어 번역본 각각 독립적으로)
# ══════════════════════════════════════════
st.markdown(step_header("3", "화자 태그 변환", "한국어·영어 각각 독립적으로 남/여 대사 태그 — 감정 태그 없음"),
            unsafe_allow_html=True)
st.caption(
    "두 언어의 태그는 서로 위치가 안 맞아도 상관없습니다 — 한국어 오디오와 영어 오디오는 "
    "각각 따로 재생되는 파일이라, 언어 안에서만 화자가 일관되면 됩니다."
)

has_translation = bool(st.session_state.get('translated_text', '').strip())
col_tag_ko, col_tag_en = st.columns(2)
with col_tag_ko:
    if st.button("🏷️ 한국어 태그 변환", type="primary" if has_text else "secondary",
                 disabled=not (api_key and has_text), use_container_width=True, key="tag_ko_btn"):
        with st.status("🏷️ 한국어 화자 분석 중...", expanded=True) as status:
            try:
                tagged = convert_tags(api_key, manuscript, tag_model)
                st.session_state['tagged_ko'] = tagged
                st.session_state.pop('ko_audio', None)
                status.update(label="✅ 완료", state="complete")
            except Exception as e:
                status.update(label="❌ 오류 발생", state="error")
                st.error(f"❌ {e}")
with col_tag_en:
    if st.button("🏷️ 영어 태그 변환", type="primary" if has_translation else "secondary",
                 disabled=not (api_key and has_translation), use_container_width=True, key="tag_en_btn"):
        with st.status("🏷️ 영어 화자 분석 중...", expanded=True) as status:
            try:
                tagged = convert_tags(api_key, st.session_state['translated_text'], tag_model)
                st.session_state['tagged_en'] = tagged
                st.session_state.pop('en_audio', None)
                status.update(label="✅ 완료", state="complete")
            except Exception as e:
                status.update(label="❌ 오류 발생", state="error")
                st.error(f"❌ {e}")

col_show_ko, col_show_en = st.columns(2)
with col_show_ko:
    if 'tagged_ko' in st.session_state:
        edited_ko = st.text_area("한국어 화자 태그 (직접 수정 가능)",
            value=st.session_state['tagged_ko'], height=250, key="tagged_ko_edit")
        st.session_state['tagged_ko'] = edited_ko
        ko_lines = parse_tagged_script(edited_ko)
        if ko_lines:
            sc = {}
            for l in ko_lines:
                sc[l['speaker']] = sc.get(l['speaker'], 0) + 1
            cols = st.columns(len(sc) if sc else 1)
            for i, (spk, cnt) in enumerate(sc.items()):
                cols[i].metric(spk, cnt)
with col_show_en:
    if 'tagged_en' in st.session_state:
        edited_en = st.text_area("영어 화자 태그 (직접 수정 가능)",
            value=st.session_state['tagged_en'], height=250, key="tagged_en_edit")
        st.session_state['tagged_en'] = edited_en
        en_lines = parse_tagged_script(edited_en)
        if en_lines:
            sc = {}
            for l in en_lines:
                sc[l['speaker']] = sc.get(l['speaker'], 0) + 1
            cols = st.columns(len(sc) if sc else 1)
            for i, (spk, cnt) in enumerate(sc.items()):
                cols[i].metric(spk, cnt)


# ══════════════════════════════════════════
# STEP 4: 오디오 생성
# ══════════════════════════════════════════
st.markdown(step_header("4", "한국어 · 영어 MP3 오디오 생성"), unsafe_allow_html=True)

has_tagged_ko = bool(st.session_state.get('tagged_ko', '').strip())
has_tagged_en = bool(st.session_state.get('tagged_en', '').strip())

col_ko, col_en = st.columns(2)

with col_ko:
    st.markdown(f"<b style='color:{NAVY}'>🇰🇷 한국어 MP3</b>", unsafe_allow_html=True)
    if st.button("🎙️ 한국어 MP3 생성", type="primary",
                 disabled=not (api_key and has_tagged_ko), use_container_width=True, key="gen_ko"):
        status = st.empty()
        progress = st.progress(0)
        try:
            mp3, duration = generate_mp3_for_tagged_text(
                api_key, st.session_state['tagged_ko'], ko_voices, tts_model, max_chunk_chars,
                "한국어", status, progress
            )
            st.session_state['ko_audio'] = mp3
            st.session_state['ko_seconds'] = duration
            status.markdown(f"🎧 완료! (길이 {format_duration(duration)})")
        except Exception as e:
            st.error(f"❌ {e}")

    if 'ko_audio' in st.session_state:
        ko_mp3 = st.session_state['ko_audio']
        ko_fname = f"{project_name}_{chapter_name}_KO.mp3"
        st.success(f"✅ {len(ko_mp3)/1024/1024:.1f} MB  |  🎵 {format_duration(st.session_state.get('ko_seconds', 0))}")
        st.audio(ko_mp3, format="audio/mp3")
        st.download_button(f"⬇️ {ko_fname} 저장", data=ko_mp3, file_name=ko_fname,
                            mime="audio/mpeg", use_container_width=True, key="dl_ko")

with col_en:
    st.markdown(f"<b style='color:{NAVY}'>🇺🇸 영어 MP3</b>", unsafe_allow_html=True)
    if st.button("🎙️ 영어 MP3 생성", type="primary",
                 disabled=not (api_key and has_tagged_en), use_container_width=True, key="gen_en"):
        status = st.empty()
        progress = st.progress(0)
        try:
            mp3, duration = generate_mp3_for_tagged_text(
                api_key, st.session_state['tagged_en'], en_voices, tts_model, max_chunk_chars,
                "English", status, progress
            )
            st.session_state['en_audio'] = mp3
            st.session_state['en_seconds'] = duration
            status.markdown(f"🎧 Done! (length {format_duration(duration)})")
        except Exception as e:
            st.error(f"❌ {e}")

    if 'en_audio' in st.session_state:
        en_mp3 = st.session_state['en_audio']
        en_fname = f"{project_name}_{chapter_name}_EN.mp3"
        st.success(f"✅ {len(en_mp3)/1024/1024:.1f} MB  |  🎵 {format_duration(st.session_state.get('en_seconds', 0))}")
        st.audio(en_mp3, format="audio/mp3")
        st.download_button(f"⬇️ {en_fname} 저장", data=en_mp3, file_name=en_fname,
                            mime="audio/mpeg", use_container_width=True, key="dl_en")
