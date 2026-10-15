import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업 기준 기간의 마지막 해
MIN_DAYS = 300        # 관측일이 이 값보다 적은 해는 제외
RECENT_YEARS = 20     # 최근 기울기를 구할 때 쓰는 해의 범위

WARM_DOT = "#E07A3F"
WARM_LINE = "#B23A48"
WARM_BG = "#FFF8F0"
WARM_TEXT = "#5A3E36"

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {WARM_BG}; }}
    .big-box {{
        background: #FFE8D2; border-radius: 16px; padding: 24px 12px;
        text-align: center; margin: 8px 0 16px 0;
    }}
    .big-box .label {{ font-size: 1.1rem; color: {WARM_TEXT}; }}
    .big-box .value {{ font-size: 4rem; font-weight: 800; color: {WARM_LINE}; line-height: 1.2; }}
    .cmp-box {{
        background: #FFE8D2; border-radius: 16px; padding: 18px 10px;
        text-align: center; border: 2px solid transparent;
    }}
    .cmp-box.recent {{ background: #FFD9C2; border-color: {WARM_LINE}; }}
    .cmp-box .label {{ font-size: 1.15rem; font-weight: 700; color: {WARM_TEXT}; }}
    .cmp-box .period {{ font-size: 0.9rem; color: {WARM_TEXT}; opacity: 0.8; }}
    .cmp-box .value {{ font-size: 2.6rem; font-weight: 800; color: {WARM_LINE}; line-height: 1.3; }}
    .cmp-box .unit {{ font-size: 0.95rem; color: {WARM_TEXT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중...")
def load_yearly() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df = df[df["날짜"].dt.year <= LAST_YEAR]
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    yearly = yearly[yearly["관측일수"] >= MIN_DAYS].reset_index(drop=True)
    return yearly


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온으로 회귀 직선을 만들고, 고른 해의 기온을 예상해 봅니다.")

try:
    yearly = load_yearly()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 2:
    st.error("조건을 만족하는 해가 2개 미만이라 직선을 만들 수 없어요.")
    st.stop()

x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])
n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 100년당 기울기 (전체 기간 / 최근 20년)
slope100 = slope * 100
recent = yearly[yearly["연도"] > end_year - RECENT_YEARS].reset_index(drop=True)
n_recent = len(recent)
if n_recent >= 2:
    recent_slope, recent_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1,
    )
    recent_slope100 = recent_slope * 100
    recent_start = int(recent["연도"].min())
else:
    recent_slope = recent_intercept = recent_slope100 = None
    recent_start = None

# ── 데이터 요약 ─────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
st.caption(
    f"{LAST_YEAR}년까지의 데이터 중 관측일이 {MIN_DAYS}일 이상인 해만 사용했어요."
)

# ── 100년에 몇 도 오르는가 (크게) ───────────────────────────
st.markdown(
    f"""
    <div class="big-box">
        <div class="label">전체 기간({start_year}~{end_year}년) 직선의 기울기를 100년으로 환산하면</div>
        <div class="value">100년에 {slope100:+.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 전체 기간 vs 최근 20년 ─────────────────────────────────
st.subheader("기울기 비교: 전체 기간 vs 최근 20년")
cmp1, cmp2 = st.columns(2)
cmp1.markdown(
    f"""
    <div class="cmp-box">
        <div class="label">전체 기간</div>
        <div class="period">{start_year}~{end_year}년 · {n_years}개 해</div>
        <div class="value">{slope100:+.2f} °C</div>
        <div class="unit">100년당</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if recent_slope100 is not None:
    cmp2.markdown(
        f"""
        <div class="cmp-box recent">
            <div class="label">최근 {RECENT_YEARS}년</div>
            <div class="period">{recent_start}~{end_year}년 · {n_recent}개 해</div>
            <div class="value">{recent_slope100:+.2f} °C</div>
            <div class="unit">100년당</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    diff = recent_slope100 - slope100
    ratio_text = ""
    if slope100 > 0:
        ratio_text = f" (전체 기간의 약 {recent_slope100 / slope100:.1f}배)"
    st.write(
        f"최근 {RECENT_YEARS}년의 기울기가 전체 기간보다 "
        f"**100년당 {abs(diff):.2f} °C {'더 가팔라요' if diff > 0 else '더 완만해요'}**{ratio_text}."
    )
    st.caption(
        f"최근 {RECENT_YEARS}년은 점이 {n_recent}개뿐이라, 해마다의 들쭉날쭉함에 기울기가 많이 흔들릴 수 있어요. "
        "'100년당'은 지금의 변화 속도가 100년 동안 이어진다고 가정한 환산값입니다."
    )
else:
    cmp2.warning(f"최근 {RECENT_YEARS}년 안에 해가 2개 미만이라 기울기를 구할 수 없어요.")

# ── 슬라이더와 예상 기온 ────────────────────────────────────
year = st.slider("예상할 연도를 고르세요", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div class="big-box">
        <div class="label">{year}년의 예상 연평균기온</div>
        <div class="value">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 그래프 ─────────────────────────────────────────────────
line_x = np.array([min(start_year, year), max(end_year, year)], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(color=WARM_DOT, size=8, opacity=0.85),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color=WARM_LINE, width=3),
        hoverinfo="skip",
    )
)
if recent_slope is not None:
    rx = np.array([recent_start, end_year], dtype=float)
    fig.add_trace(
        go.Scatter(
            x=rx,
            y=recent_slope * rx + recent_intercept,
            mode="lines",
            name=f"최근 {RECENT_YEARS}년 직선",
            line=dict(color="#2A6F97", width=3, dash="dash"),
            hoverinfo="skip",
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(color=WARM_LINE, size=16, symbol="star", line=dict(color="white", width=1.5)),
        hovertemplate="%{x}년 예상<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온과 회귀 직선 (상관계수 r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    paper_bgcolor=WARM_BG,
    plot_bgcolor="white",
    font=dict(color=WARM_TEXT),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=80, b=10),
)
st.plotly_chart(fig, width="stretch")

# ── 상관계수와 직선 식 ─────────────────────────────────────
c4, c5 = st.columns(2)
c4.metric("상관계수 (r)", f"{r:.3f}")
c5.metric("기울기 (100년당)", f"{slope100:+.2f} °C")
st.write(f"회귀 직선: **기온 = {slope:.4f} × 연도 + ({intercept:.2f})**")

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(
        yearly.rename(columns={"연평균기온": "연평균기온(°C)"}).round({"연평균기온(°C)": 2}),
        width="stretch",
        hide_index=True,
    )
