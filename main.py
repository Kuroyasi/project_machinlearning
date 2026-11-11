
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ==================================================
# 1. 페이지 기본 설정
# ==================================================

st.set_page_config(
    page_title="디지털 문해력 점수 예측",
    layout="wide"
)

st.title("디지털 문해력 점수 예측")
st.caption(
    "Random Forest Regressor 기반 디지털 문해력 예측 시스템"
)

st.divider()


# ==================================================
# 2. 한글 폰트 설정
# ==================================================

def set_korean_font():

    font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc"
    ]

    font_path = None

    for path in font_candidates:

        if Path(path).exists():
            font_path = path
            break

    # 다른 Linux 경로 검색
    if font_path is None:

        candidates = list(
            Path("/usr/share/fonts").rglob("NanumGothic.ttf")
        )

        if candidates:
            font_path = str(candidates[0])

    if font_path is None:
        return None

    fm.fontManager.addfont(str(font_path))

    font_name = fm.FontProperties(
        fname=str(font_path)
    ).get_name()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    return font_name


FONT_NAME = set_korean_font()


# ==================================================
# 3. 그래프 디자인 설정
# ==================================================

GRAPH_BLUE = "#4C78A8"
GRAPH_ORANGE = "#F28E2B"
GRAPH_TEAL = "#59A5A5"
GRAPH_GRAY = "#A0A0A0"

graph_rc = {
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.2,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
}

if FONT_NAME:
    graph_rc["font.family"] = FONT_NAME

plt.rcParams.update(graph_rc)

sns.set_theme(
    style="whitegrid",
    palette=[
        GRAPH_BLUE,
        GRAPH_ORANGE,
        GRAPH_TEAL
    ],
    rc=graph_rc
)


# ==================================================
# 4. 데이터셋 불러오기
# ==================================================

DATASET_DIR = Path(__file__).parent / "dataset"


@st.cache_data
def load_data(file_path):

    df = pd.read_csv(
        file_path,
        encoding="utf-8-sig"
    )

    required_columns = [
        "age",
        "edu_level",
        "online_class",
        "study_times",
        "digital_score"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV 파일에 필요한 열이 없습니다: {missing_columns}"
        )

    for col in required_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 머신러닝에 필요한 값이 없는 행 제거
    df = df.dropna(
        subset=required_columns
    ).copy()

    return df


csv_files = sorted(
    DATASET_DIR.glob("*.csv")
)


if not csv_files:

    st.error(
        "dataset 폴더에서 CSV 파일을 찾을 수 없습니다."
    )

    st.info(
        "프로젝트 폴더 안에 dataset 폴더를 만들고 "
        "CSV 파일을 넣어주세요."
    )

    st.stop()


# CSV 파일이 여러 개일 경우 선택
if len(csv_files) > 1:

    selected_file = st.sidebar.selectbox(
        "데이터셋 선택",
        csv_files,
        format_func=lambda x: x.name
    )

else:

    selected_file = csv_files[0]


try:

    df = load_data(str(selected_file))

except Exception as e:

    st.error(
        f"데이터셋을 불러오는 중 오류가 발생했습니다: {e}"
    )

    st.stop()


if len(df) < 5:

    st.error(
        "학습을 진행하기 위한 데이터가 부족합니다."
    )

    st.stop()


# ==================================================
# 5. 독립변수와 종속변수 설정
# ==================================================

features = [
    "age",
    "edu_level",
    "online_class",
    "study_times"
]

target = "digital_score"

X = df[features]
Y = df[target]


# ==================================================
# 6. 팝업창 설정
# ==================================================

@st.dialog("입력 오류")
def show_input_error():

    st.warning(
        "0~100사이의 수를 입력해주세요."
    )


@st.dialog("비율 설정 오류")
def show_split_error():

    st.warning(
        "학습과 테스트를 위해 0보다 크고 "
        "100보다 작은 수를 입력해주세요."
    )


# ==================================================
# 7. 데이터 분할 및 사용자 정보 입력
# ==================================================

st.divider()

left_col, right_col = st.columns(
    2,
    gap="large"
)


# --------------------------------------------------
# 왼쪽 : 테스트 데이터 비율 설정
# --------------------------------------------------

with left_col:

    st.subheader(
        "테스트 데이터 비율 (%)"
    )

    if "test_size_percent" not in st.session_state:

        st.session_state.test_size_percent = 30.0

    with st.form("test_size_form"):

        test_size_input = st.text_area(
            "테스트 데이터 비율",
            value=str(
                st.session_state.test_size_percent
            ),
            height=80,
            label_visibility="collapsed",
            help="0부터 100까지의 숫자를 입력하세요."
        )

        apply_test_size = st.form_submit_button(
            "분할 비율 적용",
            use_container_width=True
        )


    # 비율 입력 검사
    if apply_test_size:

        try:

            percent = float(
                test_size_input.strip()
            )

            if not np.isfinite(percent):

                show_input_error()
                st.stop()

            elif not 0 <= percent <= 100:

                show_input_error()
                st.stop()

            elif percent == 0 or percent == 100:

                show_split_error()
                st.stop()

            else:

                st.session_state.test_size_percent = percent

        except (ValueError, AttributeError):

            show_input_error()
            st.stop()


    # 백분율을 소수로 변환
    test_size = (
        st.session_state.test_size_percent / 100
    )


    st.write("현재 데이터 분할 비율")

    metric1, metric2 = st.columns(2)

    with metric1:

        st.metric(
            "학습 데이터",
            f"{100 - st.session_state.test_size_percent:.1f}%"
        )

    with metric2:

        st.metric(
            "테스트 데이터",
            f"{st.session_state.test_size_percent:.1f}%"
        )


# --------------------------------------------------
# 오른쪽 : 사용자 정보 입력
# --------------------------------------------------

with right_col:

    st.subheader(
        "사용자 정보 입력"
    )

    education_options = {
        "고등학교 졸업": 1,
        "대학교 일부 수료": 2,
        "학사": 3,
        "석사": 4,
        "박사": 5,
        "전문 자격증": 6
    }

    with st.form("prediction_form"):

        n_age = st.number_input(
            "나이",
            min_value=1,
            max_value=120,
            value=20,
            step=1
        )

        selected_education = st.selectbox(
            "최종 학력",
            list(education_options.keys())
        )

        n_online_class = st.number_input(
            "지금까지 수강한 온라인 강의 수",
            min_value=0,
            max_value=50,
            value=10,
            step=1
        )

        min_study = max(
            0.0,
            float(df["study_times"].min())
        )

        max_study = max(
            min_study + 1.0,
            float(df["study_times"].max())
        )

        default_study = float(
            np.clip(
                df["study_times"].median(),
                min_study,
                max_study
            )
        )

        n_studytime = st.number_input(
            "총 공부 시간",
            min_value=min_study,
            max_value=max_study,
            value=default_study,
            step=1.0
        )

        submit = st.form_submit_button(
            "디지털 문해력 점수 예측",
            use_container_width=True
        )


    if submit:

        st.session_state.prediction_input = {
            "age": n_age,
            "edu_level": education_options[
                selected_education
            ],
            "online_class": n_online_class,
            "study_times": n_studytime
        }


# ==================================================
# 8. 학습용 데이터와 테스트용 데이터 분리
# ==================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=test_size,
    random_state=1
)


# ==================================================
# 9. Random Forest Regressor 모델 생성 및 학습
# ==================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    Y_train
)


# ==================================================
# 10. 모델 성능 평가
# ==================================================

Y_pred = model.predict(X_test)

mse = mean_squared_error(
    Y_test,
    Y_pred
)

rmse = np.sqrt(mse)

mae = mean_absolute_error(
    Y_test,
    Y_pred
)

r2 = r2_score(
    Y_test,
    Y_pred
)


# ==================================================
# 11. 데이터셋 기본 정보
# ==================================================

st.divider()

st.header("데이터셋 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "전체 데이터 수",
    f"{len(df):,}개"
)

col2.metric(
    "학습 데이터",
    f"{len(X_train):,}개"
)

col3.metric(
    "테스트 데이터",
    f"{len(X_test):,}개"
)

col4.metric(
    "사용 변수",
    f"{len(features)}개"
)


with st.expander("데이터셋 미리보기"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ==================================================
# 12. 디지털 문해력 예측 결과
# ==================================================

if "prediction_input" in st.session_state:

    new_data = pd.DataFrame([
        st.session_state.prediction_input
    ])

    prediction = model.predict(
        new_data
    )[0]

    st.divider()

    st.header("디지털 문해력 예측 결과")

    result_col1, result_col2 = st.columns(
        [1, 2]
    )

    with result_col1:

        st.metric(
            "예상 디지털 문해력 점수",
            f"{prediction:.2f}점"
        )

    with result_col2:

        st.progress(
            float(
                np.clip(prediction / 100, 0, 1)
            )
        )

        st.caption(
            "예측 결과는 학습 데이터의 패턴을 기반으로 "
            "계산된 추정값입니다."
        )


# ==================================================
# 13. 머신러닝 모델 성능 평가
# ==================================================

st.divider()

st.header("머신러닝 모델 성능 평가")

st.write(
    "테스트 데이터의 실제 점수와 예측 점수를 비교합니다."
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "MSE",
    f"{mse:.3f}"
)

col2.metric(
    "RMSE",
    f"{rmse:.3f}"
)

col3.metric(
    "MAE",
    f"{mae:.3f}"
)

col4.metric(
    "R² Score",
    f"{r2:.3f}"
)


with st.expander("평가 지표 설명"):

    st.markdown("""
    | 평가 지표 | 설명 |
    |---|---|
    | MSE | 실제값과 예측값 차이의 제곱 평균 |
    | RMSE | MSE에 제곱근을 적용한 값 |
    | MAE | 실제값과 예측값 차이의 절댓값 평균 |
    | R² Score | 모델이 데이터의 변동성을 얼마나 설명하는지 나타내는 값 |

    MSE, RMSE, MAE는 낮을수록 좋으며,
    R² Score는 일반적으로 1에 가까울수록 좋습니다.
    """)


# --------------------------------------------------
# 실제 점수와 예측 점수 비교
# --------------------------------------------------

st.subheader("실제 점수와 예측 점수 비교")

fig, ax = plt.subplots(
    figsize=(5.5, 3.5)
)

ax.scatter(
    Y_test,
    Y_pred,
    color=GRAPH_BLUE,
    alpha=0.65,
    s=25,
    edgecolors="white",
    linewidths=0.4
)

min_value = min(
    Y_test.min(),
    Y_pred.min()
)

max_value = max(
    Y_test.max(),
    Y_pred.max()
)

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    color=GRAPH_ORANGE,
    linestyle="--",
    linewidth=1.5,
    label="이상적인 예측"
)

ax.set_title(
    "실제 점수와 예측 점수 비교",
    pad=10
)

ax.set_xlabel("실제 디지털 문해력 점수")
ax.set_ylabel("예측 디지털 문해력 점수")

ax.legend(
    frameon=False
)

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ==================================================
# 14. 변수 중요도 분석
# ==================================================

st.divider()

st.header("변수 중요도 분석")

st.write(
    "Random Forest 모델이 예측 과정에서 "
    "각 독립변수를 얼마나 중요하게 활용했는지 확인합니다."
)

feature_names = {
    "age": "나이",
    "edu_level": "최종 학력",
    "online_class": "온라인 강의 수",
    "study_times": "공부 시간"
}

importance_df = pd.DataFrame({
    "변수": [
        feature_names[col]
        for col in features
    ],
    "중요도": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="중요도",
    ascending=False
)


fig, ax = plt.subplots(
    figsize=(5.5, 3.2)
)

sns.barplot(
    data=importance_df,
    x="중요도",
    y="변수",
    color=GRAPH_BLUE,
    ax=ax,
    width=0.55
)

ax.set_title(
    "Random Forest 변수 중요도",
    pad=10
)

ax.set_xlabel("변수 중요도")
ax.set_ylabel("")

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


with st.expander("변수별 중요도 수치"):

    st.dataframe(
        importance_df,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 15. 데이터 분석
# ==================================================

st.divider()

st.header("데이터 분석")

st.write(
    "학습에 사용된 데이터를 기반으로 "
    "최종 학력과 연령대에 따른 디지털 문해력을 분석합니다."
)


# --------------------------------------------------
# 15-1. 최종 학력별 디지털 문해력 평균
# --------------------------------------------------

st.subheader(
    "최종 학력별 디지털 문해력 평균"
)

edu_avg = df.groupby(
    "edu_level"
)["digital_score"].mean()

edu_labels = {
    1: "고등학교 졸업",
    2: "대학교 일부 수료",
    3: "학사",
    4: "석사",
    5: "박사",
    6: "전문 자격증"
}

edu_avg.index = [
    edu_labels.get(
        int(x),
        str(x)
    )
    for x in edu_avg.index
]


fig, ax = plt.subplots(
    figsize=(5.5, 3.2)
)

edu_avg.plot(
    kind="bar",
    ax=ax,
    color=GRAPH_BLUE,
    width=0.6
)

ax.set_title(
    "최종 학력별 디지털 문해력 평균",
    pad=10
)

ax.set_xlabel("최종 학력")
ax.set_ylabel("디지털 문해력 점수")

ax.tick_params(
    axis="x",
    rotation=25
)

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# --------------------------------------------------
# 15-2. 나이대 구간 분류
# --------------------------------------------------

df_analysis = df.copy()

age_bins = [
    10, 20, 30, 40, 50, 60, 70, 80
]

age_labels = [
    "10대",
    "20대",
    "30대",
    "40대",
    "50대",
    "60대",
    "70대"
]

df_analysis["age_bin"] = pd.cut(
    df_analysis["age"],
    bins=age_bins,
    labels=age_labels,
    right=False
)


# --------------------------------------------------
# 15-3. 연령대별 히스토그램
# --------------------------------------------------

st.subheader(
    "연령대별 디지털 문해력 분포"
)

fig, axes = plt.subplots(
    2,
    4,
    figsize=(12, 5.5)
)

axes = axes.flatten()

for i, label in enumerate(age_labels):

    data = df_analysis[
        df_analysis["age_bin"] == label
    ]["digital_score"].dropna()

    ax = axes[i]

    if len(data) == 0:

        ax.set_title(
            f"{label} 데이터 없음"
        )

        ax.set_axis_off()

        continue

    mean = data.mean()
    std = data.std()

    if pd.isna(std):
        std = 0

    ax.hist(
        data,
        bins=10,
        color=GRAPH_BLUE,
        edgecolor="white",
        alpha=0.85
    )

    ax.axvline(
        mean,
        color=GRAPH_ORANGE,
        linestyle="--",
        linewidth=1.5,
        label="평균"
    )

    ax.axvline(
        mean - std,
        color=GRAPH_GRAY,
        linestyle=":",
        linewidth=1.2,
        label="표준편차"
    )

    ax.axvline(
        mean + std,
        color=GRAPH_GRAY,
        linestyle=":",
        linewidth=1.2
    )

    ax.set_title(
        f"{label} 디지털 문해력",
        fontsize=9
    )

    ax.set_xlabel(
        "점수",
        fontsize=8
    )

    ax.set_ylabel(
        "빈도",
        fontsize=8
    )

    ax.tick_params(
        labelsize=7
    )

    ax.legend(
        fontsize=7,
        frameon=False
    )


fig.delaxes(
    axes[-1]
)

fig.tight_layout(
    pad=1.5
)

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# --------------------------------------------------
# 15-4. 연령대별 디지털 문해력 평균
# --------------------------------------------------

st.subheader(
    "연령대별 디지털 문해력 평균"
)

age_avg = df_analysis.groupby(
    "age_bin",
    observed=False
)["digital_score"].mean()


fig, ax = plt.subplots(
    figsize=(5.5, 3.2)
)

age_avg.plot(
    kind="bar",
    ax=ax,
    color=GRAPH_BLUE,
    width=0.6
)

ax.set_title(
    "연령대별 디지털 문해력 평균",
    pad=10
)

ax.set_xlabel("연령대")
ax.set_ylabel("디지털 문해력 점수")

ax.tick_params(
    axis="x",
    rotation=0
)

fig.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ==================================================
# 16. 마무리
# ==================================================

st.divider()

st.caption(
    "Digital Literacy Prediction System | "
    "Random Forest Regressor"
)
