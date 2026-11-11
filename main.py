
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

# ==========================================
# 1. 기본 페이지 설정
# ==========================================

st.set_page_config(
    page_title="디지털 문해력 점수 예측",
    layout="wide"
)

st.title("디지털 문해력 점수 예측")
st.caption("Random Forest Regressor 기반 디지털 문해력 예측 시스템")

st.divider()




# ==========================================
# 한글 폰트 설정 (Streamlit Cloud)
# ==========================================

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

def set_korean_font():

    font_path = Path(
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    )

    if not font_path.exists():
        # 다른 설치 경로에서도 검색
        candidates = list(
            Path("/usr/share/fonts").rglob("NanumGothic.ttf")
        )

        if candidates:
            font_path = candidates[0]
        else:
            raise FileNotFoundError(
                "나눔고딕 폰트가 설치되지 않았습니다. "
                "packages.txt를 확인해주세요."
            )

    fm.fontManager.addfont(str(font_path))

    font_name = fm.FontProperties(
        fname=str(font_path)
    ).get_name()

    plt.rcParams["font.family"] = font_name
    plt.rcParams["axes.unicode_minus"] = False

    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": font_name,
            "axes.unicode_minus": False
        }
    )

    return font_name


FONT_NAME = set_korean_font()


# ==========================================
# 3. 데이터셋 불러오기
# ==========================================

DATASET_DIR = Path(__file__).parent / "dataset"

@st.cache_data
def load_data(file_path):

    df = pd.read_csv(file_path, encoding="utf-8-sig")

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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=required_columns).copy()

    return df


# dataset 폴더에서 CSV 파일 검색
csv_files = sorted(DATASET_DIR.glob("*.csv"))

if not csv_files:

    st.error(
        "dataset 폴더에서 CSV 파일을 찾을 수 없습니다. "
        "CSV 파일을 dataset 폴더에 넣어주세요."
    )

    st.stop()


# CSV가 여러 개라면 선택 가능
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
    st.error(f"데이터셋을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


if len(df) < 5:
    st.error("머신러닝 학습을 진행하기 위한 데이터가 부족합니다.")
    st.stop()


# ==========================================
# 4. 독립변수와 종속변수 설정
# ==========================================

features = [
    "age",
    "edu_level",
    "online_class",
    "study_times"
]

target = "digital_score"

X = df[features]
Y = df[target]



# ==========================================
# 5. 학습용 데이터와 테스트용 데이터 분리
# ==========================================

# 기본 테스트 데이터 비율
if "test_size_percent" not in st.session_state:
    st.session_state.test_size_percent = 30.0


# 입력 오류 팝업
@st.dialog("입력 오류")
def show_input_error():
    st.warning("0~100사이의 수를 입력해주세요.")


@st.dialog("비율 설정 오류")
def show_split_error():
    st.warning(
        "0% 또는 100%는 학습용과 테스트용 데이터를 "
        "모두 만들 수 없습니다. 0보다 크고 100보다 작은 수를 입력해주세요."
    )


st.subheader("학습용 데이터와 테스트용 데이터 분리")

with st.form("test_size_form"):

    test_size_input = st.text_area(
        "테스트 데이터 비율 (%)",
        value=str(st.session_state.test_size_percent),
        height=70,
        help="0부터 100까지의 숫자를 입력하세요."
    )

    apply_test_size = st.form_submit_button(
        "분할 비율 적용",
        use_container_width=True
    )


# 적용 버튼을 눌렀을 때 입력값 검사
if apply_test_size:

    try:
        percent = float(test_size_input.strip())

        if not 0 <= percent <= 100:
            show_input_error()
            st.stop()

        elif percent == 0 or percent == 100:
            show_split_error()
            st.stop()

        else:
            st.session_state.test_size_percent = percent

    except ValueError:
        show_input_error()
        st.stop()


# 백분율을 소수로 변환
test_size = st.session_state.test_size_percent / 100


# 현재 설정된 비율 표시
col1, col2 = st.columns(2)

with col1:
    st.metric(
        "학습 데이터 비율",
        f"{100 - st.session_state.test_size_percent:.1f}%"
    )

with col2:
    st.metric(
        "테스트 데이터 비율",
        f"{st.session_state.test_size_percent:.1f}%"
    )


# 학습용 데이터와 테스트용 데이터 분리
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=test_size,
    random_state=1
)



# ==========================================
# 6. Random Forest Regressor 모델 생성
# ==========================================

@st.cache_resource
def train_model(X_train, Y_train):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, Y_train)

    return model


model = train_model(X_train, Y_train)


# ==========================================
# 7. 모델 성능 평가
# ==========================================

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)


# ==========================================
# 8. 데이터 기본 정보
# ==========================================

st.subheader("데이터셋 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("전체 데이터 수", f"{len(df):,}개")

with col2:
    st.metric("학습 데이터", f"{len(X_train):,}개")

with col3:
    st.metric("테스트 데이터", f"{len(X_test):,}개")

with col4:
    st.metric("사용 변수", f"{len(features)}개")


with st.expander("데이터셋 미리보기"):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

st.divider()


# ==========================================
# 9. 디지털 문해력 점수 예측
# ==========================================

st.header("디지털 문해력 예측")

st.write(
    "아래 항목에 정보를 입력하면 "
    "Random Forest Regressor 모델이 "
    "예상 디지털 문해력 점수를 계산합니다."
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

    st.subheader("사용자 정보 입력")

    col1, col2 = st.columns(2)

    with col1:

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

    with col2:

        n_online_class = st.number_input(
            "지금까지 수강한 온라인 강의 수",
            min_value=0,
            max_value=50,
            value=10,
            step=1
        )

        min_study = max(0.0, float(df["study_times"].min()))
        max_study = max(min_study + 1.0, float(df["study_times"].max()))

        default_study = float(np.clip(
            df["study_times"].median(),
            min_study,
            max_study
        ))

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

    new_data = pd.DataFrame([{
        "age": n_age,
        "edu_level": education_options[selected_education],
        "online_class": n_online_class,
        "study_times": n_studytime
    }])

    prediction = model.predict(new_data)[0]

    st.divider()

    st.subheader("예측 결과")

    st.metric(
        "예상 디지털 문해력 점수",
        f"{prediction:.2f}점"
    )

    st.progress(
        float(np.clip(prediction / 100, 0, 1))
    )

    st.caption(
        "예측 점수는 학습 데이터의 패턴을 기반으로 계산된 추정값이며, "
        "실제 디지털 문해력 검사 결과와 다를 수 있습니다."
    )


st.divider()


# ==========================================
# 10. 머신러닝 모델 성능 평가
# ==========================================

st.header("머신러닝 모델 성능 평가")

st.write(
    "테스트 데이터에 대한 실제 점수와 예측 점수를 비교하여 "
    "모델의 성능을 평가합니다."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("MSE", f"{mse:.3f}")

with col2:
    st.metric("RMSE", f"{rmse:.3f}")

with col3:
    st.metric("MAE", f"{mae:.3f}")

with col4:
    st.metric("R² Score", f"{r2:.3f}")


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


# 실제값과 예측값 비교
comparison_df = pd.DataFrame({
    "실제 점수": Y_test.values,
    "예측 점수": Y_pred
})

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    comparison_df["실제 점수"],
    comparison_df["예측 점수"],
    alpha=0.6
)

min_value = min(Y_test.min(), Y_pred.min())
max_value = max(Y_test.max(), Y_pred.max())

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

ax.set_title("실제 점수와 예측 점수 비교")
ax.set_xlabel("실제 디지털 문해력 점수")
ax.set_ylabel("예측 디지털 문해력 점수")

st.pyplot(fig)
plt.close(fig)


# ==========================================
# 11. 변수 중요도 분석
# ==========================================

st.divider()

st.header("변수 중요도 분석")

st.write(
    "Random Forest 모델이 예측을 수행할 때 "
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
        feature_names[col] for col in features
    ],
    "중요도": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="중요도",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.barplot(
    data=importance_df,
    x="중요도",
    y="변수",
    ax=ax
)

ax.set_title("Random Forest 변수 중요도")
ax.set_xlabel("Feature Importance")
ax.set_ylabel("독립변수")

st.pyplot(fig)
plt.close(fig)

st.dataframe(
    importance_df,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 12. 데이터 분석
# ==========================================

st.divider()

st.header("데이터 분석")

st.write(
    "학습에 사용된 데이터를 바탕으로 "
    "나이와 최종 학력에 따른 디지털 문해력 점수를 분석합니다."
)


# ------------------------------------------
# 12-1. 최종 학력별 평균
# ------------------------------------------

st.subheader("최종 학력별 디지털 문해력 평균")

edu_avg = df.groupby("edu_level")["digital_score"].mean()

edu_labels = {
    1: "고등학교 졸업",
    2: "대학교 일부 수료",
    3: "학사",
    4: "석사",
    5: "박사",
    6: "전문 자격증"
}

edu_avg.index = [
    edu_labels.get(int(x), str(x))
    for x in edu_avg.index
]

fig, ax = plt.subplots(figsize=(9, 5))

edu_avg.plot(
    kind="bar",
    ax=ax
)

ax.set_title("최종 학력별 디지털 문해력 평균")
ax.set_xlabel("최종 학력")
ax.set_ylabel("디지털 문해력 점수")
ax.tick_params(axis="x", rotation=30)

st.pyplot(fig)
plt.close(fig)


# ------------------------------------------
# 12-2. 나이대 구분
# ------------------------------------------

st.subheader("연령대별 디지털 문해력 분포")

age_bins = [10, 20, 30, 40, 50, 60, 70, 80]

age_labels = [
    "10대",
    "20대",
    "30대",
    "40대",
    "50대",
    "60대",
    "70대"
]

df_analysis = df.copy()

df_analysis["age_bin"] = pd.cut(
    df_analysis["age"],
    bins=age_bins,
    labels=age_labels,
    right=False
)


# ------------------------------------------
# 12-3. 연령대별 히스토그램
# ------------------------------------------

fig, axes = plt.subplots(
    2, 4,
    figsize=(18, 8)
)

axes = axes.flatten()

for i, label in enumerate(age_labels):

    data = df_analysis[
        df_analysis["age_bin"] == label
    ]["digital_score"].dropna()

    ax = axes[i]

    if len(data) == 0:

        ax.set_title(f"{label} 데이터 없음")
        ax.set_axis_off()
        continue

    mean = data.mean()
    std = data.std()

    if pd.isna(std):
        std = 0

    ax.hist(
        data,
        bins=10,
        edgecolor="black",
        alpha=0.7
    )

    ax.axvline(
        mean,
        linestyle="--",
        label="평균"
    )

    ax.axvline(
        mean - std,
        linestyle=":",
        label="-1 표준편차"
    )

    ax.axvline(
        mean + std,
        linestyle=":",
        label="+1 표준편차"
    )

    ax.set_title(f"{label} 디지털 문해력 분포")
    ax.set_xlabel("디지털 문해력 점수")
    ax.set_ylabel("빈도")
    ax.legend(fontsize=7)

fig.delaxes(axes[-1])

plt.tight_layout()

st.pyplot(fig)
plt.close(fig)


# ------------------------------------------
# 12-4. 연령대별 평균 점수
# ------------------------------------------

st.subheader("연령대별 디지털 문해력 평균")

age_avg = df_analysis.groupby(
    "age_bin",
    observed=False
)["digital_score"].mean()

fig, ax = plt.subplots(figsize=(9, 5))

age_avg.plot(
    kind="bar",
    ax=ax
)

ax.set_title("연령대별 디지털 문해력 평균")
ax.set_xlabel("연령대")
ax.set_ylabel("디지털 문해력 점수")
ax.tick_params(axis="x", rotation=0)

st.pyplot(fig)
plt.close(fig)


# ==========================================
# 13. 마무리
# ==========================================

st.divider()

st.caption(
    "Digital Literacy Prediction System | "
    "Random Forest Regressor"
)
