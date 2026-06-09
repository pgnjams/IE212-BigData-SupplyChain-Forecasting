# IE212-BigData-SupplyChain-Forecasting

Big Data Project: Supply Chain Sales Forecasting on Favorita Dataset (125M rows) using PySpark, LightGBM, LSTM and Temporal Fusion Transformer.

**Môn học:** IE212 — Big Data, Học kỳ 2 - 2026  
**Trường:** Trường Đại học Công nghệ Thông tin, ĐHQG-HCM  
**Giảng viên hướng dẫn (GVHD):** T.S Hà Minh Tân

## Thành viên thực hiện

| STT | Họ và tên              |   MSSV   |
| :-: | :--------------------- | :------: |
|  1  | Nguyễn Thị Ngọc Phước  | 23521235 |
|  2  | Nguyễn Huỳnh Xuân Nghi | 23521004 |
|  3  | Nguyễn Tấn Mạnh        | 23520916 |

---

## Giới thiệu

Dự án tái tạo bài báo _"Supply chain sales forecasting based on lightGBM and LSTM combination model"_ và thực hiện các cải tiến trên tập dữ liệu Corporación Favorita (125M+ records). Nhóm tập trung vào việc xử lý dữ liệu lớn trên hạ tầng Spark và áp dụng các mô hình học sâu hiện đại để tối ưu hóa dự báo doanh số.

## Liên kết hệ thống & Pipeline

Vì tập dữ liệu có dung lượng lớn, toàn bộ quá trình xử lý và huấn luyện mô hình được tự động hóa và lưu trữ trên nền tảng Kaggle:

- **Tập dữ liệu gốc (Raw Data):** [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/overview)
- **EDA:** [Kaggle Notebook - EDA](https://www.kaggle.com/code/nghuxung/ie212-eda-p1)
- **Tiền xử lý dữ liệu (PySpark):** [Kaggle Notebook - Data Preprocessing](https://www.kaggle.com/code/pgnjams/ie212-datapreprocessing-2)
- **Tập dữ liệu sau xử lý (Processed Dataset):** [Kaggle Dataset - Cleaned Data](https://www.kaggle.com/datasets/pgnjams/ie212-favorita-engineered-features-27-stores)
- **Huấn luyện mô hình (LightGBM + LSTM/Kalman):** [Kaggle Notebook - Model Training](https://www.kaggle.com/code/jamieinuit/ie212-bl-lightgbm-lstm-final)
- **Huấn luyện mô hình (LightGBM + LSTM/Wavelet):** [Kaggle Notebook - Model Training](https://www.kaggle.com/code/jamieinuit/ie212-wavelet-lightgbm-lstm-final)
- **Huấn luyện mô hình (TFT):** [Kaggle Notebook - Model Training](https://www.kaggle.com/code/nghuxung/ie212-tft-final-69e4ed)

## Cấu trúc thư mục

Kho lưu trữ được tổ chức theo cấu trúc chuẩn công nghiệp dành cho các dự án Khoa học dữ liệu (Data Science Workflow), tách biệt rõ ràng giữa mã nguồn thực nghiệm, mã nguồn ứng dụng và tài liệu học thuật:

```text
IE212-BigData-SupplyChain-Forecasting/
│
├── notebooks/
│   ├── EDA/
│   ├── Models/
│   ├── Preprocessing/
│
├── outputs/
│   ├── figures/
│   └── metrics/
│
│
├── streamlit_app/              #Dashboard Streamlit
    ├── predictions/
│   ├── app.py
│   └── requirements.txt
│
├── docs/                       # Tài liệu báo cáo
│   ├── reports/
│   └── slides/
│
├── .gitignore
└── README.md
```

## Phương pháp triển khai

- **Baseline:** Tái tạo mô hình kết hợp **LSTM** (trích xuất Net Features) và **LightGBM** (Gradient Boosting).
- **Cải tiến 1:** Ứng dụng **Wavelet Transform** thay thế cho bộ lọc Kalman truyền thống để khử nhiễu dữ liệu chuỗi thời gian mà không làm mất các tín hiệu biến động mạnh (peaks).
- **Cải tiến 2:** Triển khai mô hình **Temporal Fusion Transformer (TFT)** nhằm tận dụng cơ chế Multi-head Attention để tăng khả năng diễn giải mô hình (Interpretability).

## Tài liệu tham khảo

1. Weng, T., Liu, W., & Xiao, J. (2020). _Supply chain sales forecasting based on lightGBM and LSTM combination model_. Industrial Management & Data Systems.
2. Kaggle Competition. Corporación Favorita Grocery Sales Forecasting: Can you predict how many grocery items will be sold? Nền tảng Kaggle, 2017. Đường dẫn: https://www.kaggle.com/c/favorita-grocery-sales-forecasting.
