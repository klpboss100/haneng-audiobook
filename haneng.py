"""
한영소리 (KOEN Audio)
─────────────────────────────────────────
실행: python -m streamlit run haneng.py
─────────────────────────────────────────
한국어 소설 → 문학적 영어 번역(원고 전체) → 영어 품질검사 → 영어 화자(M/W) 태그 → 영어 MP3 제작
(한국어 오디오 제작은 audiobook-maker 앱에서 별도로 진행합니다)
"""

import streamlit as st
import re, io, time, json, os, random, pickle
import lameenc

# 환경 자동 감지: Streamlit Cloud = /home/appuser
IS_CLOUD = os.environ.get('HOME', '') == '/home/appuser'
from google import genai
from google.genai import types

REQUEST_TIMEOUT_MS = 120_000  # 요청 하나가 응답 없이 무한정 매달리는 것 방지 (2분)

def make_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key,
                         http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))

# ═══════════════════════════════════════════
# 상수
# ═══════════════════════════════════════════
SAMPLE_RATE     = 24000
MAX_CHUNK_CHARS = 900   # TTS 1회 호출당 최대 글자수 (길면 잡음/에코 위험 ↑ - 구글 TTS 알려진 한계)
SEED_BASE       = 7     # 목소리 톤이 매 호출마다 튀는 것을 완화 (best-effort 결정성)
CONFIG_FILE     = "config_haneng.json"

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
# 영어 번역 원고 품질 검사
# ═══════════════════════════════════════════
ANALYSIS_PROMPT_EN = """You are a professional literary editor reviewing an English translation of a Korean novel.

## Review criteria
1. Awkward phrasing: unnatural "translated-sounding" expressions, awkward word choice, broken sentence flow
2. AI-writing patterns: clichéd AI phrasing, overly formulaic sentences, repetitive structures
3. Grammar/spelling: spelling mistakes, grammar errors, punctuation issues

Respond ONLY in the following JSON format (no markdown, no extra text):
{{
  "issues": [
    {{
      "original": "the exact text found in the manuscript",
      "suggestion": "suggested fix",
      "type": "어색함",
      "reason": "reason, written in Korean"
    }}
  ],
  "summary": "overall summary, written in Korean, 2-3 lines"
}}

type must be exactly one of "어색함", "AI패턴", "맞춤법".
If there are no issues, return an empty array [].

Manuscript:
{manuscript}"""


def analyze_manuscript_en(api_key: str, manuscript: str, model: str) -> dict:
    """영어 번역 원고 품질 분석"""
    import json as _json
    client = make_client(api_key)
    response = client.models.generate_content(
        model=model,
        contents=ANALYSIS_PROMPT_EN.format(manuscript=manuscript)
    )
    text = response.text.strip()
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return _json.loads(text)
    except Exception:
        return {"issues": [], "summary": "분석 결과를 파싱할 수 없습니다."}


# ═══════════════════════════════════════════
# 화자(M/W) 태그 변환 — 감정 태그 없음
# ═══════════════════════════════════════════
TAG_PROMPT = """당신은 소설 원고에 화자 태그를 추가하는 전문가입니다. 원고의 언어를 그대로 유지한 채
(번역하지 말고) 태그만 추가하세요.

## 절대 규칙
- 원고 텍스트를 절대 수정·번역·요약·생략하지 마세요
- 원고 내용 전체를 빠짐없이 원래 언어 그대로 출력하세요
- 태그만 각 줄 앞에 추가하세요

## 태그 형식
[화자] 텍스트   (예: [M] He smiled quietly.)

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
    client = make_client(api_key)
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
    client = make_client(api_key)
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


def call_tts(client, text, voice_name, tts_model, seed=None, retry=5, status=None):
    """항상 bytes를 반환하거나 예외를 던짐 — 절대 None을 반환하지 않음.
    구글 TTS는 500 INTERNAL 같은 일시적 서버 오류가 몇 분씩 이어지는 경우가 있어
    (알려진 불안정성), 재시도 한도를 넉넉하게 잡아 어지간한 일시 장애는 사람이
    다시 누르지 않아도 자동으로 버텨내도록 함."""
    MAX_SERVER_RETRIES = 20  # 서버 오류·rate limit 재시도 한도 (최악의 경우 최대 약 20분 대기)
    rate_limit_retries = 0
    other_retries = 0
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
    while True:
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
            if (is_rate_limit or is_server_err) and rate_limit_retries < MAX_SERVER_RETRIES:
                rate_limit_retries += 1
                wait_s = 60 if is_rate_limit else 20
                reason = "API 분당 요청 제한에 걸림" if is_rate_limit else "구글 서버 일시 오류"
                if status is not None:
                    status.markdown(f"⏳ {reason}. {wait_s}초 대기 후 재시도 ({rate_limit_retries}/{MAX_SERVER_RETRIES})...")
                time.sleep(wait_s)
                continue
            other_retries += 1
            if other_retries < retry:
                if status is not None:
                    status.markdown(f"⏳ 알 수 없는 오류로 재시도 중... ({other_retries}/{retry}) — {msg[:80]}")
                time.sleep(10)
                continue
            raise e


def merge_to_mp3(pcm_list) -> bytes:
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(2)
    mp3_data = encoder.encode(b"".join(pcm_list))
    # lameenc는 bytearray를 반환함 — st.audio는 bytearray를 받아들이지 않아
    # "Invalid binary data format" 오류가 나므로 항상 순수 bytes로 변환
    return bytes(mp3_data) + bytes(encoder.flush())


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


def plan_tts_chunks(tagged_text, voices, max_chunk_chars):
    """화자 태그 텍스트를 화자별 목소리 기준으로 병합·청크 분할한 세그먼트 목록과 총 청크 수 계산"""
    lines = parse_tagged_script(tagged_text)
    segs_raw = group_into_segments(lines)
    segs = merge_segments_by_voice(segs_raw, voices)
    total_chunks = sum(len(chunk_segment_lines(s['lines'], max_chunk_chars)) for s in segs)
    return segs, total_chunks


# ═══════════════════════════════════════════
# 오디오 생성 진행상황 저장/재개
# ─────────────────────────────────────────
# 예전 방식은 청크 하나를 만들 때마다 "지금까지 만든 오디오 전체"를 통째로
# 다시 pickle로 저장했음 → 챕터가 길어질수록 저장 시간이 계속 늘어나서
# 뒤로 갈수록 화면이 멈춘 것처럼 느려지는 원인이 됨.
# 지금은 새로 만든 청크 하나만 파일로 추가 저장하고, 아주 작은 메타 정보만
# pickle로 갱신 → 청크 개수가 많아져도 저장 속도가 항상 일정함.
# ═══════════════════════════════════════════
PROGRESS_DIR_EN  = "progress_en_chunks"
PROGRESS_META_EN = "progress_en_meta.pkl"


def save_progress_chunk(idx: int, pcm: bytes, chapter: str):
    os.makedirs(PROGRESS_DIR_EN, exist_ok=True)
    with open(os.path.join(PROGRESS_DIR_EN, f"chunk_{idx:05d}.pcm"), "wb") as f:
        f.write(pcm)
    with open(PROGRESS_META_EN, "wb") as f:
        pickle.dump({"done": idx + 1, "chapter": chapter}, f)


def load_progress_en():
    if not os.path.exists(PROGRESS_META_EN):
        return None
    try:
        with open(PROGRESS_META_EN, "rb") as f:
            meta = pickle.load(f)
    except Exception:
        return None
    pcm_list = []
    done = meta.get("done", 0)
    for i in range(done):
        path = os.path.join(PROGRESS_DIR_EN, f"chunk_{i:05d}.pcm")
        try:
            with open(path, "rb") as f:
                pcm_list.append(f.read())
        except Exception:
            # 파일이 없거나 손상됨 → 그 지점까지만 인정하고 나머지는 다시 생성
            meta["done"] = i
            break
    meta["pcm_list"] = pcm_list
    return meta


def clear_progress_en():
    if os.path.exists(PROGRESS_META_EN):
        os.remove(PROGRESS_META_EN)
    if os.path.isdir(PROGRESS_DIR_EN):
        for fn in os.listdir(PROGRESS_DIR_EN):
            try:
                os.remove(os.path.join(PROGRESS_DIR_EN, fn))
            except Exception:
                pass


# ═══════════════════════════════════════════
# UI 헬퍼 — 단계마다 수정/복사/저장을 한 세트로 제공
# ═══════════════════════════════════════════
def edit_copy_save_block(label: str, session_key: str, widget_key: str, height: int, filename: str) -> str:
    """텍스트 영역(수정) + 복사용 코드뷰(복사) + 다운로드(저장)를 한 번에 제공"""
    text_val = st.session_state.get(session_key, "")
    edited = st.text_area(label, value=text_val, height=height, key=widget_key)
    st.session_state[session_key] = edited

    c1, c2 = st.columns(2)
    with c1:
        with st.popover("📋 복사", use_container_width=True):
            st.caption("오른쪽 위 아이콘을 클릭하면 클립보드에 복사됩니다.")
            st.code(edited, language=None)
    with c2:
        st.download_button("⬇️ 저장", data=edited.encode("utf-8"), file_name=filename,
                            mime="text/plain", use_container_width=True, key=f"dl_{widget_key}")
    return edited


def direct_tag_input_block(lang_label: str, mode_key: str, session_key: str):
    """이미 태그가 붙은 원고가 있으면 AI 변환 없이 바로 붙여넣기"""
    direct_text = st.text_area(f"{lang_label} 태그 원고 붙여넣기", height=200,
        placeholder="[M] 텍스트...\n[W] 텍스트...", key=f"{mode_key}_text")
    cd1, cd2 = st.columns(2)
    with cd1:
        if st.button("✅ 확인", type="primary", use_container_width=True, key=f"{mode_key}_confirm"):
            if direct_text.strip():
                st.session_state[session_key] = direct_text
                st.session_state[mode_key] = False
                st.rerun()
    with cd2:
        if st.button("❌ 취소", use_container_width=True, key=f"{mode_key}_cancel"):
            st.session_state[mode_key] = False
            st.rerun()


def render_english_audio_generation(voices: dict, tts_model: str, max_chunk_chars: int,
                                     project_name: str, chapter_name: str, api_key: str, has_tagged: bool):
    """이어서 생성(재개) 기능을 갖춘 영어 오디오 생성 UI"""
    segs, total_chunks, saved_prog = [], 0, None
    resume_from = None

    if has_tagged:
        segs, total_chunks = plan_tts_chunks(st.session_state['tagged_en'], voices, max_chunk_chars)
        st.caption(f"청크 {total_chunks}개 (최대 {max_chunk_chars}자)")

        saved_prog = load_progress_en()
        has_progress = saved_prog is not None and saved_prog.get('chapter') == chapter_name
        if has_progress:
            done_so_far = saved_prog.get('done', 0)
            st.warning(f"⏸️ 이전 작업: {done_so_far}/{total_chunks} 청크에서 중단됨")
            cb1, cb2 = st.columns(2)
            with cb1:
                start_btn = st.button("▶️ 이어서 생성", type="primary", disabled=not api_key,
                                       use_container_width=True, key="resume_en")
            with cb2:
                if st.button("🔄 처음부터", use_container_width=True, key="restart_en"):
                    clear_progress_en()
                    st.rerun()
            resume_from = done_so_far if start_btn else None
        else:
            start_btn = st.button("🎙️ 영어 오디오 생성", type="primary", disabled=not api_key,
                                   use_container_width=True, key="gen_en")
            resume_from = 0 if start_btn else None
    else:
        st.button("🎙️ 영어 오디오 생성", type="primary", disabled=True,
                  use_container_width=True, key="gen_en_disabled")

    if resume_from is not None:
        gen_start = time.time()
        client = make_client(api_key)
        progress = st.progress(0)
        status = st.empty()
        pcm_list = list((saved_prog or {}).get('pcm_list', [])) if resume_from > 0 else []

        error_flag = False
        done = resume_from
        chunk_idx = 0

        for seg in segs:
            voice = get_voice_for_speaker(seg['speaker'], voices)
            chunks = chunk_segment_lines(seg['lines'], max_chunk_chars)
            for chunk in chunks:
                if chunk_idx < resume_from:
                    chunk_idx += 1
                    continue
                chars = sum(len(l['text']) for l in chunk)
                status.markdown(
                    f"🎙️ [{seg['speaker']}] ({voice}) — {done+1}/{total_chunks} ({chars}자)"
                )
                progress.progress(done / total_chunks)
                try:
                    script = build_speaker_script(chunk)
                    seed = SEED_BASE + chunk_idx
                    pcm = call_tts(client, script, voice, tts_model, seed=seed, status=status)
                    pcm_list.append(pcm)
                    done += 1
                    save_progress_chunk(chunk_idx, pcm, chapter_name)
                    chunk_idx += 1
                except Exception as e:
                    st.error(f"❌ [{seg['speaker']}] {e}")
                    st.info(f"💾 {done}청크까지 저장됨. [▶️ 이어서 생성]으로 재시작하세요.")
                    error_flag = True
                    break
            if error_flag:
                break

        if not error_flag and pcm_list:
            status.markdown("🔗 MP3로 합치는 중...")
            mp3 = merge_to_mp3(pcm_list)
            st.session_state['en_audio'] = mp3
            st.session_state['en_seconds'] = pcm_duration_seconds(pcm_list)
            clear_progress_en()
            progress.progress(1.0)
            status.markdown(f"🎧 완료! (길이 {format_duration(st.session_state['en_seconds'])}, "
                             f"소요시간 {format_duration(time.time() - gen_start)})")

    if 'en_audio' in st.session_state:
        mp3 = st.session_state['en_audio']
        fname = f"{project_name}_{chapter_name}_EN.mp3"
        st.success(f"✅ {len(mp3)/1024/1024:.1f} MB  |  🎵 {format_duration(st.session_state.get('en_seconds', 0))}")
        st.audio(mp3, format="audio/mp3")
        st.download_button(f"⬇️ {fname} 저장", data=mp3, file_name=fname,
                            mime="audio/mpeg", use_container_width=True, key="dl_en")


# ═══════════════════════════════════════════
# 페이지 설정 & CSS
# ═══════════════════════════════════════════
st.set_page_config(page_title="한영소리 · KOEN Audio", page_icon="📓", layout="wide")

NAVY = "#0f3460"

if st.session_state.pop('_pending_reset', False):
    for _k in ['translated_text', 'analysis_result_en', 'analysis_text_en', 'accepted_fixes_en',
               'issue_filter_en', 'translated_checked', 'tagged_en', 'en_audio', 'en_seconds',
               'direct_input_mode_en']:
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
    check_model = st.selectbox("🔍 영어 품질검사", ["gemini-2.5-pro", "gemini-2.5-flash"],
                                index=0, key="check_model",
                                help="Pro: 정확도 우선 (추천)\nFlash: 빠른 검사")
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
          한국어 소설을 세계로 (영어 오디오 전용)
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    proj_display = f"**{project_name}**" if project_name else "*(프로젝트명 없음)*"
    st.caption(f"프로젝트: {proj_display}  |  EN M={en_m_voice}/W={en_w_voice}")
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
            for _k in ['analysis_result_en', 'analysis_text_en', 'accepted_fixes_en',
                       'translated_checked', 'tagged_en', 'en_audio']:
                st.session_state.pop(_k, None)
            status.update(label="✅ 번역 완료", state="complete")
        except Exception as e:
            status.update(label="❌ 오류 발생", state="error")
            st.error(f"❌ {e}")

if 'translated_text' in st.session_state:
    edited_translation = edit_copy_save_block(
        "영어 번역 결과 (수정 가능)", 'translated_text', 'translated_text_edit', 250,
        f"{project_name}_{chapter_name}_영어번역.txt"
    )
    st.markdown(f"<p style='font-size:14px;color:{NAVY};margin:4px 0'>번역 글자 수: {len(edited_translation):,}자</p>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════
# STEP 3: 영어 원고 품질검사
# ══════════════════════════════════════════
st.markdown(step_header("3", "영어 원고 품질검사", "어색한 표현·AI패턴·맞춤법 검사 후 자동으로 다음 단계로"),
            unsafe_allow_html=True)

has_translation = bool(st.session_state.get('translated_text', '').strip())

col_q1, col_q2 = st.columns(2)
with col_q1:
    btn_label = "🔍 품질 검사 시작" if has_translation else "✏️ 먼저 번역을 완료하세요"
    if st.button(btn_label, type="primary" if has_translation else "secondary",
                 disabled=not (api_key and has_translation), use_container_width=True, key="check_en_btn"):
        with st.status("🔍 영어 번역문 품질 분석 중...", expanded=True) as status:
            st.write("Gemini가 문장을 분석하고 있습니다. (30초~1분 소요)")
            try:
                src_text = st.session_state.get('translated_text', '')
                result = analyze_manuscript_en(api_key, src_text, check_model)
                st.session_state['analysis_result_en'] = result
                st.session_state['analysis_text_en'] = src_text
                st.session_state['accepted_fixes_en'] = {}
                st.session_state.pop('translated_checked', None)
                issues_count = len(result.get('issues', []))
                status.update(label=f"✅ 분석 완료 — {issues_count}개 발견", state="complete")
            except Exception as e:
                status.update(label="❌ 오류 발생", state="error")
                st.error(f"❌ {e}")
with col_q2:
    if st.button("⏭️ 검사 건너뛰기",
                 disabled=not has_translation, use_container_width=True, key="skip_check_en_btn"):
        st.session_state['translated_checked'] = st.session_state.get('translated_text', '')
        st.session_state.pop('analysis_result_en', None)
        st.rerun()

if 'analysis_result_en' in st.session_state and 'translated_checked' not in st.session_state:
    result = st.session_state['analysis_result_en']
    issues = result.get('issues', [])
    summary = result.get('summary', '')

    if summary:
        st.info(f"📊 {summary}")

    if not issues:
        st.success("✅ 문제없음! 아래 단계로 진행하세요.")
        st.session_state['translated_checked'] = st.session_state['analysis_text_en']
    else:
        types_ = [i.get('type', '') for i in issues]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("전체", len(issues))
        c2.metric("어색함 🟡", types_.count("어색함"))
        c3.metric("AI패턴 🔴", types_.count("AI패턴"))
        c4.metric("맞춤법 🟠", types_.count("맞춤법"))

        accepted = st.session_state.get('accepted_fixes_en', {})
        color_map = {"어색함": "🟡", "AI패턴": "🔴", "맞춤법": "🟠"}

        flt = st.session_state.get('issue_filter_en', '전체')
        cnt_all = len(issues)
        cnt_awk = types_.count("어색함")
        cnt_ai = types_.count("AI패턴")
        cnt_spell = types_.count("맞춤법")

        f0, f1, f2, f3 = st.columns(4)
        if f0.button(f"전체 ({cnt_all})",
                     type="primary" if flt == '전체' else "secondary",
                     use_container_width=True, key="flt_all_en"):
            st.session_state['issue_filter_en'] = '전체'; st.rerun()
        if f1.button(f"어색함🟡 ({cnt_awk})",
                     type="primary" if flt == '어색함' else "secondary",
                     use_container_width=True, key="flt_awk_en"):
            st.session_state['issue_filter_en'] = '어색함'; st.rerun()
        if f2.button(f"AI패턴🔴 ({cnt_ai})",
                     type="primary" if flt == 'AI패턴' else "secondary",
                     use_container_width=True, key="flt_ai_en"):
            st.session_state['issue_filter_en'] = 'AI패턴'; st.rerun()
        if f3.button(f"맞춤법🟠 ({cnt_spell})",
                     type="primary" if flt == '맞춤법' else "secondary",
                     use_container_width=True, key="flt_spell_en"):
            st.session_state['issue_filter_en'] = '맞춤법'; st.rerun()

        if flt != '전체':
            if st.button(f"✅ '{flt}' 전체 → 제안으로 일괄 적용",
                         use_container_width=True, key="apply_all_type_en"):
                for j, iss in enumerate(issues):
                    if iss.get('type') == flt:
                        accepted[j] = {
                            'type': 'suggestion',
                            'text': iss.get('suggestion', ''),
                            'original': iss.get('original', '')
                        }
                st.session_state['accepted_fixes_en'] = accepted
                st.rerun()

        st.markdown("---")

        filtered = [(j, iss) for j, iss in enumerate(issues)
                    if flt == '전체' or iss.get('type') == flt]

        for i, issue in filtered:
            orig = issue.get('original', '')
            sugg = issue.get('suggestion', '')
            itype = issue.get('type', '')
            reason = issue.get('reason', '')
            icon = color_map.get(itype, "⚪")

            cur = accepted.get(i, {})
            sel_type = cur.get('type', None)

            if sel_type == 'original':
                status_txt = "📌 원본"
            elif sel_type == 'suggestion':
                status_txt = "✅ 제안"
            elif sel_type == 'custom':
                status_txt = "✏️ 직접"
            else:
                status_txt = "⬜ 미선택"

            with st.expander(f"{icon} [{itype}]  {orig[:45]}  —  {status_txt}", expanded=True):
                st.caption(f"💡 {reason}")
                co, cs, cc = st.columns(3)

                with co:
                    is_sel = sel_type == "original"
                    st.markdown(
                        f"<div style='background:{'#ebf8ff' if is_sel else '#fff5f5'};"
                        f"border:{'2px solid #2b6cb0' if is_sel else '1px solid #feb2b2'};"
                        f"border-radius:8px;padding:8px 8px 4px;font-size:13px'>"
                        f"<b style='color:#2d3748'>원본</b></div>",
                        unsafe_allow_html=True
                    )
                    st.text_area("", value=orig, height=80, disabled=True,
                                 label_visibility="collapsed", key=f"orig_disp_en_{i}")
                    if st.button("👆 원본 선택", key=f"sel_o_en_{i}", use_container_width=True):
                        accepted[i] = {'type': 'original', 'text': orig, 'original': orig}
                        st.session_state['accepted_fixes_en'] = accepted
                        st.rerun()

                with cs:
                    is_sel = sel_type == "suggestion"
                    st.markdown(
                        f"<div style='background:{'#f0fff4' if is_sel else '#f9fff9'};"
                        f"border:{'2px solid #276749' if is_sel else '1px solid #9ae6b4'};"
                        f"border-radius:8px;padding:8px 8px 4px;font-size:13px'>"
                        f"<b style='color:#2d3748'>제안</b> "
                        f"<span style='font-size:11px;color:#888'>(수정 가능)</span></div>",
                        unsafe_allow_html=True
                    )
                    sugg_edited = st.text_area("", value=cur.get('text', sugg) if is_sel else sugg,
                                                height=80, label_visibility="collapsed", key=f"sugg_inp_en_{i}")
                    if st.button("✅ 제안 선택", key=f"sel_s_en_{i}", use_container_width=True):
                        accepted[i] = {'type': 'suggestion', 'text': sugg_edited, 'original': orig}
                        st.session_state['accepted_fixes_en'] = accepted
                        st.rerun()

                with cc:
                    is_sel = sel_type == "custom"
                    cust_val = cur.get('text', '') if is_sel else ''
                    st.markdown(
                        f"<div style='background:{'#fffbeb' if is_sel else '#fff'};"
                        f"border:{'2px solid #d97706' if is_sel else '1px solid #fde68a'};"
                        f"border-radius:8px;padding:8px 8px 4px;font-size:13px'>"
                        f"<b style='color:#2d3748'>직접 수정</b></div>",
                        unsafe_allow_html=True
                    )
                    cust_input = st.text_area("", value=cust_val, height=80,
                                               placeholder="직접 입력...", label_visibility="collapsed",
                                               key=f"custom_inp_en_{i}")
                    if st.button("✏️ 직접수정 선택", key=f"sel_c_en_{i}", use_container_width=True):
                        if cust_input.strip():
                            accepted[i] = {'type': 'custom', 'text': cust_input, 'original': orig}
                            st.session_state['accepted_fixes_en'] = accepted
                            st.rerun()

        st.markdown("---")
        applied = len(accepted)
        total = len(issues)
        if st.button(f"✅ 검사 완료 → 다음 단계  ({applied}/{total}개 선택됨)",
                     type="primary", use_container_width=True, key="finish_check_en"):
            final = st.session_state['analysis_text_en']
            for idx, fix in accepted.items():
                if fix.get('type') in ('suggestion', 'custom'):
                    final = final.replace(fix['original'], fix['text'], 1)
            st.session_state['translated_checked'] = final
            st.rerun()


# ══════════════════════════════════════════
# STEP 4: 영어 화자 태그 변환
# ══════════════════════════════════════════
st.markdown(step_header("4", "영어 화자 태그 변환", "남/여 대사 태그 — 감정 태그 없음"),
            unsafe_allow_html=True)

has_checked = bool(st.session_state.get('translated_checked', '').strip())

col_tag1, col_tag2 = st.columns(2)
with col_tag1:
    if st.button("🏷️ 화자 태그 변환", type="primary" if has_checked else "secondary",
                 disabled=not (api_key and has_checked), use_container_width=True, key="tag_en_btn"):
        with st.status("🏷️ 화자 분석 중...", expanded=True) as status:
            try:
                tagged = convert_tags(api_key, st.session_state['translated_checked'], tag_model)
                st.session_state['tagged_en'] = tagged
                st.session_state.pop('en_audio', None)
                status.update(label="✅ 완료", state="complete")
            except Exception as e:
                status.update(label="❌ 오류 발생", state="error")
                st.error(f"❌ {e}")
with col_tag2:
    if st.button("📋 태그 직접 입력", use_container_width=True, key="direct_en_btn"):
        st.session_state['direct_input_mode_en'] = True
        st.session_state.pop('tagged_en', None)
        st.rerun()

if st.session_state.get('direct_input_mode_en'):
    direct_tag_input_block("영어", 'direct_input_mode_en', 'tagged_en')

if 'tagged_en' in st.session_state:
    edited_en = edit_copy_save_block(
        "영어 화자 태그 (수정 가능)", 'tagged_en', 'tagged_en_edit', 250,
        f"{project_name}_{chapter_name}_영어태그.txt"
    )
    en_lines = parse_tagged_script(edited_en)
    if en_lines:
        sc = {}
        for l in en_lines:
            sc[l['speaker']] = sc.get(l['speaker'], 0) + 1
        cols = st.columns(len(sc) if sc else 1)
        for i, (spk, cnt) in enumerate(sc.items()):
            cols[i].metric(spk, cnt)


# ══════════════════════════════════════════
# STEP 5: 영어 MP3 오디오 생성 (중단 시 이어서 생성 가능)
# ══════════════════════════════════════════
st.markdown(step_header("5", "영어 MP3 오디오 생성",
            "생성이 중단돼도 처음부터 다시 하지 않고 이어서 생성할 수 있습니다"), unsafe_allow_html=True)

has_tagged_en = bool(st.session_state.get('tagged_en', '').strip())

render_english_audio_generation(
    en_voices, tts_model, max_chunk_chars, project_name, chapter_name, api_key, has_tagged_en
)
