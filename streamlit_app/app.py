import os
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from utils import (
    PARQUET_PATH,
    ASSET_DIR,
    TRACK_FILES,
    TRACK_IMAGES,
    get_store_list,
    get_item_list,
    load_store_item_2017,
    create_dummy_prediction,
    load_all_predictions,
    filter_prediction,
    calculate_metrics,
    build_leaderboard,
)

st.set_page_config(
    page_title="Supply Chain Sales Forecasting",
    layout="wide"
)

MODEL_COLORS = {
    "Track 1 - Baseline": "orange",
    "Track 2 - Wavelet": "green",
    "Track 3 - TFT": "purple",
}

st.title("Supply Chain Sales Forecasting Dashboard")
st.caption(
    "Decision Support Dashboard: EDA, Wavelet Denoising, Sliding Window, Business Value, "
    "Model Comparison và Feature Importance."
)

if not os.path.exists(PARQUET_PATH):
    st.error("Không tìm thấy folder parquet.")
    st.code(PARQUET_PATH)
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Bộ lọc dữ liệu")

stores = get_store_list()
selected_store = st.sidebar.selectbox("Chọn Store", stores)

items = get_item_list(selected_store)
selected_item = st.sidebar.selectbox("Chọn Item", items)

inventory_level = st.sidebar.slider(
    "Ngưỡng tồn kho giả định",
    min_value=1,
    max_value=100,
    value=20
)

selected_df = load_store_item_2017(selected_store, selected_item)

if selected_df.empty:
    st.warning("Không có dữ liệu năm 2017 cho Store/Item này.")
    st.stop()

st.sidebar.success(f"Đã load: {len(selected_df):,} dòng")
st.sidebar.info("Dữ liệu được filter theo Store/Item để giảm RAM.")

predictions = load_all_predictions()

# ============================================================
# EXECUTIVE SUMMARY
# ============================================================
st.markdown("## Executive Summary")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Store", selected_store)
c2.metric("Item", selected_item)
c3.metric("Số ngày dữ liệu", len(selected_df))
c4.metric("Sales TB", round(selected_df["unit_sales"].mean(), 2))
c5.metric("Inventory Threshold", inventory_level)

st.markdown("### Model Integration Status")

s1, s2, s3 = st.columns(3)

status_cols = [s1, s2, s3]

for idx, (track_name, filename) in enumerate(TRACK_FILES.items()):
    with status_cols[idx]:
        if track_name in predictions:
            st.success(f"{track_name}\n\nLoaded: `{filename}`")
        else:
            st.warning(f"{track_name}\n\nWaiting: `{filename}`")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Visual EDA",
    "Denoising",
    "Sliding Window",
    "Business Value"
])

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.subheader("Khám phá dữ liệu trực quan")

    e1, e2, e3, e4 = st.columns(4)

    e1.metric("Min Sales", round(selected_df["unit_sales"].min(), 2))
    e2.metric("Max Sales", round(selected_df["unit_sales"].max(), 2))
    e3.metric("Median Sales", round(selected_df["unit_sales"].median(), 2))
    e4.metric("Std Sales", round(selected_df["unit_sales"].std(), 2))

    fig = px.line(
        selected_df,
        x="date",
        y="unit_sales",
        title="Doanh số thực tế năm 2017",
        labels={"date": "Ngày", "unit_sales": "Unit Sales"}
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Dữ liệu mẫu")
    st.dataframe(selected_df.head(50), use_container_width=True)

# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.subheader("Minh họa thuật toán khử nhiễu")

    st.markdown("""
    **Wavelet Denoising Pipeline**

    Raw Sales → Wavelet Transform → Thresholding → Reconstruction → Denoised Signal
    """)

    show_kalman = st.toggle("Bật/Tắt đường Kalman", value=True)
    show_wavelet_soft = st.toggle("Bật/Tắt đường Wavelet Soft", value=True)
    show_wavelet_hard = st.toggle("Bật/Tắt đường Wavelet Hard", value=False)

    raw_std = selected_df["unit_sales"].std()
    soft_std = selected_df["wavelet_soft_sales"].std()
    hard_std = selected_df["wavelet_hard_sales"].std()

    soft_reduction = ((raw_std - soft_std) / raw_std * 100) if raw_std > 0 else 0
    hard_reduction = ((raw_std - hard_std) / raw_std * 100) if raw_std > 0 else 0

    d1, d2, d3 = st.columns(3)
    d1.metric("Raw Std", round(raw_std, 4))
    d2.metric("Soft Noise Reduction", f"{soft_reduction:.2f}%")
    d3.metric("Hard Noise Reduction", f"{hard_reduction:.2f}%")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=selected_df["date"],
        y=selected_df["unit_sales"],
        mode="lines",
        name="Raw Unit Sales",
    ))

    if show_kalman:
        fig.add_trace(go.Scatter(
            x=selected_df["date"],
            y=selected_df["kalman_sales"],
            mode="lines",
            name="Kalman Sales",
        ))

    if show_wavelet_soft:
        fig.add_trace(go.Scatter(
            x=selected_df["date"],
            y=selected_df["wavelet_soft_sales"],
            mode="lines",
            name="Wavelet Soft",
        ))

    if show_wavelet_hard:
        fig.add_trace(go.Scatter(
            x=selected_df["date"],
            y=selected_df["wavelet_hard_sales"],
            mode="lines",
            name="Wavelet Hard",
        ))

    fig.update_layout(
        title="Raw Sales vs Kalman/Wavelet Denoising",
        xaxis_title="Ngày",
        yaxis_title="Sales",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Wavelet giúp làm mượt nhiễu ngắn hạn trong chuỗi doanh số, "
        "hỗ trợ mô hình học xu hướng ổn định hơn nhưng vẫn giữ được các biến động quan trọng."
    )

# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.subheader("Trình bày Sliding Window")

    st.markdown("""
    Mô hình dùng **16 ngày trước đó** làm input để dự báo **ngày thứ 17**.

    - Input: `T-16` đến `T-1`
    - Target: `T`
    """)

    st.markdown(
        """
        <div style='text-align:center; font-size:22px; padding:20px;
                    border:1px solid #444; border-radius:12px;'>
            [T-16] → [T-15] → [T-14] → ... → [T-2] → [T-1]
            <br><br>
            ⬇️
            <br><br>
            <b>Predict T</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    if len(selected_df) >= 17:
        window_df = selected_df[["date", "unit_sales"]].head(17).copy()

        window_df["step"] = [f"T-{16 - i}" for i in range(16)] + ["T"]
        window_df["role"] = ["Input"] * 16 + ["Target"]

        st.markdown("#### Bảng mapping Sliding Window")
        st.dataframe(
            window_df[["step", "date", "unit_sales", "role"]],
            use_container_width=True
        )

        fig = go.Figure()

        input_df = window_df.iloc[:16]
        target_df = window_df.iloc[16:]

        fig.add_trace(go.Scatter(
            x=input_df["date"],
            y=input_df["unit_sales"],
            mode="lines+markers",
            name="Input: T-16 đến T-1",
        ))

        fig.add_trace(go.Scatter(
            x=target_df["date"],
            y=target_df["unit_sales"],
            mode="markers",
            marker=dict(size=16),
            name="Target: T",
        ))

        fig.update_layout(
            title="Sliding Window: 16 ngày input → dự báo ngày thứ 17",
            xaxis_title="Ngày",
            yaxis_title="Sales",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Không đủ 17 ngày dữ liệu để minh họa Sliding Window.")

# ============================================================
# TAB 4
# ============================================================
with tab4:
    st.subheader("Business Value")

    st.markdown("""
    Tab này biến kết quả dự báo thành quyết định nghiệp vụ:

    - Forecast > Inventory Threshold → nguy cơ cháy hàng.
    - Forecast quá thấp so với nhu cầu → cảnh báo tồn đọng vốn hoặc lập kế hoạch sai.
    """)

    if len(predictions) == 0:
        st.warning("Chưa có file dự báo thật. Dashboard đang dùng Dummy Prediction.")

        pred_df = create_dummy_prediction(selected_df)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=pred_df["date"],
            y=pred_df["actual_sales"],
            mode="lines",
            name="Actual Sales",
        ))

        fig.add_trace(go.Scatter(
            x=pred_df["date"],
            y=pred_df["predicted_sales"],
            mode="lines",
            name="Dummy Predicted Sales",
        ))

        fig.add_hline(
            y=inventory_level,
            line_dash="dash",
            annotation_text="Inventory Threshold",
            annotation_position="top left"
        )

        fig.update_layout(
            title="Actual Sales vs Dummy Prediction",
            xaxis_title="Ngày",
            yaxis_title="Sales",
        )

        st.plotly_chart(fig, use_container_width=True)

        latest = pred_df.iloc[-1]
        actual = latest["actual_sales"]
        predicted = latest["predicted_sales"]

        b1, b2, b3, b4 = st.columns(4)

        b1.metric("Actual gần nhất", round(actual, 2))
        b2.metric("Forecast gần nhất", round(predicted, 2))
        b3.metric("Inventory", inventory_level)
        b4.metric("Forecast Bias", round(predicted - actual, 2))

        if predicted > inventory_level:
            st.error(
                f"Nguy cơ cháy hàng: Forecast = {predicted:.2f} > Inventory = {inventory_level}"
            )
        elif predicted < inventory_level * 0.5:
            st.warning(
                f"Nguy cơ tồn đọng vốn: Forecast = {predicted:.2f} thấp hơn nhiều so với tồn kho {inventory_level}"
            )
        elif predicted > actual * 1.2:
            st.error("Forecast cao hơn Actual trên 20%: cần chuẩn bị bổ sung hàng.")
        elif predicted < actual * 0.8:
            st.warning("Forecast thấp hơn Actual trên 20%: dễ lập kế hoạch thiếu chính xác.")
        else:
            st.success("Forecast nằm trong vùng ổn định.")

    else:
        st.success("Đã phát hiện file dự báo thật. Đang hiển thị kết quả từ các Track Model.")

        fig = go.Figure()
        metrics = {}
        added_actual = False

        for track_name, pred_df in predictions.items():
            pred_selected = filter_prediction(pred_df, selected_store, selected_item)

            if pred_selected.empty:
                st.info(f"{track_name}: chưa có dữ liệu cho Store/Item đang chọn.")
                continue

            if not added_actual:
                fig.add_trace(go.Scatter(
                    x=pred_selected["date"],
                    y=pred_selected["actual_sales"],
                    mode="lines",
                    name="Thực tế",
                    line=dict(width=4),
                ))
                added_actual = True

            fig.add_trace(go.Scatter(
                x=pred_selected["date"],
                y=pred_selected["predicted_sales"],
                mode="lines",
                name=track_name,
                line=dict(color=MODEL_COLORS.get(track_name)),
            ))

            metrics[track_name] = calculate_metrics(
                pred_selected["actual_sales"],
                pred_selected["predicted_sales"]
            )

        fig.add_hline(
            y=inventory_level,
            line_dash="dash",
            annotation_text="Inventory Threshold",
            annotation_position="top left"
        )

        fig.update_layout(
            title="Line Chart 4 đường: Thực tế + Track 1 + Track 2 + Track 3",
            xaxis_title="Ngày",
            yaxis_title="Sales",
        )

        st.plotly_chart(fig, use_container_width=True)

        if metrics:
            st.markdown("### Metrics")
            report_metrics = {
                "Track 1 - Baseline": {
                "Model": "Baseline (Kalman + LSTM + LightGBM)",
                "NWRMSLE": 0.3835,
                "WMAE": 2.2339,
                "WRMSE": 8.3999,
            },
            "Track 2 - Wavelet": {
                "Model": "Wavelet (Soft Thresholding + LSTM + LightGBM)",
                "NWRMSLE": 0.4479,
                "WMAE": 2.7351,
                "WRMSE": 10.5611,
            },
            "Track 3 - TFT": {
                "Model": "Temporal Fusion Transformer (TFT)",
                "NWRMSLE": 0.5183,
                "WMAE": 3.2140,
                "WRMSE": 13.6111,
            },
        }

        metric_cols = st.columns(3)

        for idx, (track_name, metric_dict) in enumerate(report_metrics.items()):
            with metric_cols[idx]:
                st.markdown(f"#### {track_name}")
                st.metric("NWRMSLE", metric_dict["NWRMSLE"])
                st.metric("WMAE", metric_dict["WMAE"])
                st.metric("WRMSE", metric_dict["WRMSE"])

        st.markdown("### Bảng tổng hợp kết quả")

        comparison_df = pd.DataFrame(report_metrics).T.reset_index(drop=True)

        st.dataframe(
            comparison_df,
            use_container_width=True
        )