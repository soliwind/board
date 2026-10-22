```python
import io
import json

import requests
import pandas as pd
import plotly.express as px
import streamlit as st


POPULATION_URL = (
    "https://raw.githubusercontent.com/soliwind/board//main/"
    "population_ts_yearly_gender.zip"
)
GEOJSON_URL = (
    "https://raw.githubusercontent.com/soliwind/board//main/"
    "sigungu_kr.geojson"
)


st.set_page_config(
    page_title="전국 고령화 단계구분도",
    page_icon="🗺️",
    layout="wide",
)


@st.cache_data
def load_population():
    """2026년 6월 기준 전국 읍·면·동별 인구 데이터를 불러온다."""
    response = requests.get(POPULATION_URL, timeout=60)
    response.raise_for_status()

    # 압축 CSV는 pandas가 직접 읽을 수 있다.
    df = pd.read_csv(io.BytesIO(response.content), compression="zip")

    # 연도는 숫자/문자열 어느 쪽이어도 대응
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce")
    df = df[df["연도"] == 2026].copy()

    # 코드는 반드시 문자열로 취급하고 앞의 5자리를 시군구 코드로 사용
    df["코드"] = df["코드"].astype("string").str.strip()
    df["시군구코드"] = df["코드"].str[:5]

    return df


@st.cache_data
def load_geojson():
    """시군구 경계 GeoJSON을 JSON으로 읽는다."""
    response = requests.get(GEOJSON_URL, timeout=60)
    response.raise_for_status()

    geojson = response.json()

    # GeoJSON 속성의 코드도 문자열로 통일
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        if "코드" in properties:
            properties["코드"] = str(properties["코드"]).strip().zfill(5)

    return geojson


@st.cache_data
def calculate_aging_rate(df):
    """
    시군구별 65세 이상 비율을 계산한다.

    65세 이상:
      계_65세 ~ 계_100세 이상

    전체:
      계_ 로 시작하는 모든 나이별 인구 열

    남_/여_ 열은 계산에서 제외한다.
    """
    population_columns = [
        col for col in df.columns
        if col.startswith("계_")
    ]

    elderly_columns = [
        col for col in population_columns
        if col == "계_100세 이상"
        or (
            col.startswith("계_")
            and col.endswith("세")
            and col[2:-1].isdigit()
            and int(col[2:-1]) >= 65
        )
    ]

    if not population_columns:
        raise ValueError("계_로 시작하는 인구 열을 찾지 못했습니다.")

    if not elderly_columns:
        raise ValueError("65세 이상 인구 열을 찾지 못했습니다.")

    # 숫자로 변환한 뒤 결측값은 0으로 처리
    for col in population_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    grouped = (
        df.groupby("시군구코드", as_index=False)
        .agg(
            전체인구=("__전체인구__", "sum"),
            고령인구=("__고령인구__", "sum"),
        )
    )

    grouped["고령화율"] = (
        grouped["고령인구"]
        .div(grouped["전체인구"])
        .mul(100)
    )

    return grouped


def prepare_population(df):
    """계_ 인구 열을 이용해 계산용 전체/고령 인구를 만든다."""
    population_columns = [
        col for col in df.columns
        if col.startswith("계_")
    ]

    elderly_columns = [
        col for col in population_columns
        if col == "계_100세 이상"
        or (
            col.startswith("계_")
            and col.endswith("세")
            and col[2:-1].isdigit()
            and int(col[2:-1]) >= 65
        )
    ]

    for col in population_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["__전체인구__"] = df[population_columns].sum(axis=1)
    df["__고령인구__"] = df[elderly_columns].sum(axis=1)

    return df


@st.cache_data
def make_aging_data(df):
    df = prepare_population(df.copy())

    result = (
        df.groupby("시군구코드", as_index=False)
        .agg(
            전체인구=("__전체인구__", "sum"),
            고령인구=("__고령인구__", "sum"),
        )
    )

    result["고령화율"] = (
        result["고령인구"] / result["전체인구"] * 100
    )

    # 0으로 나누어 생기는 무한대 등 제거
    result.loc[result["전체인구"] <= 0, "고령화율"] = pd.NA

    result["시군구코드"] = (
        result["시군구코드"]
        .astype("string")
        .str.strip()
        .str.zfill(5)
    )

    return result


@st.cache_data
def merge_geojson_properties(geojson, aging_df):
    """
    GeoJSON의 코드와 계산 결과의 시군구코드를 정확히 매칭한다.

    지도에 표시되지 않는 지역도 경계 자체는 남기기 위해
    GeoJSON feature에 고령화율을 주입한다.
    """
    value_map = aging_df.set_index("시군구코드")["고령화율"].to_dict()

    for feature in geojson["features"]:
        props = feature["properties"]

        code = str(props.get("코드", "")).strip().zfill(5)
        props["코드"] = code
        props["고령화율"] = value_map.get(code)

    return geojson


def main():
    st.title("전국 고령화 단계구분도")
    st.caption("2026년 6월 기준 · 시군구별 65세 이상 인구 비율")

    try:
        population = load_population()
        aging = make_aging_data(population)
        geojson = load_geojson()
        geojson = merge_geojson_properties(geojson, aging)

    except Exception as e:
        st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")
        st.stop()

    # GeoJSON의 코드와 동일한 코드만 사용
    geo_codes = {
        str(feature["properties"].get("코드", "")).strip().zfill(5)
        for feature in geojson.get("features", [])
    }

    map_df = aging[aging["시군구코드"].isin(geo_codes)].copy()

    # GeoJSON properties의 시군구 이름을 hover에 사용하기 위해
    # 코드 -> 이름 매핑을 만든다.
    name_map = {
        str(feature["properties"].get("코드", "")).strip().zfill(5):
        feature["properties"].get("시군구", "")
        for feature in geojson.get("features", [])
    }

    map_df["시군구"] = map_df["시군구코드"].map(name_map)

    fig = px.choropleth(
        map_df,
        geojson=geojson,
        locations="시군구코드",
        featureidkey="properties.코드",
        color="고령화율",
        color_continuous_scale="YlOrRd",
        range_color=(0, map_df["고령화율"].max()),
        hover_name="시군구",
        hover_data={
            "시군구코드": False,
            "고령화율": ":.2f",
        },
        labels={"고령화율": "고령화율 (%)"},
    )

    # 배경 지도/타일 없이 GeoJSON 경계만 표시
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showland=False,
        showcountries=False,
        showcoastlines=False,
        showframe=False,
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "고령화율: %{z:.2f}%"
            "<extra></extra>"
        ),
    )

    fig.update_layout(
        height=800,
        margin={"r": 0, "t": 10, "l": 0, "b": 0},
        coloraxis_colorbar={
            "title": "고령화율 (%)",
            "ticksuffix": "%",
        },
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "scrollZoom": True,
            "displaylogo": False,
        },
    )

    st.subheader("시군구별 계산 결과")

    table = (
        aging.merge(
            map_df[["시군구코드", "시군구"]],
            on="시군구코드",
            how="left",
        )
        [["시군구코드", "시군구", "전체인구", "고령인구", "고령화율"]]
        .sort_values("고령화율", ascending=False)
        .reset_index(drop=True)
    )

    table["고령화율"] = table["고령화율"].round(2)

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "시군구코드": "코드",
            "시군구": "시군구",
            "전체인구": st.column_config.NumberColumn(
                "전체 인구",
                format="%d",
            ),
            "고령인구": st.column_config.NumberColumn(
                "65세 이상 인구",
                format="%d",
            ),
            "고령화율": st.column_config.NumberColumn(
                "고령화율 (%)",
                format="%.2f",
            ),
        },
    )


if __name__ == "__main__":
    main()
```