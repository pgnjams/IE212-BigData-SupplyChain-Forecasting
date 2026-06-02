# IE212-BigData-SupplyChain-Forecasting
Big Data Project: Supply Chain Sales Forecasting on Favorita Dataset (12.5M rows) using PySpark, LightGBM, and Temporal Fusion Transformer.

**Môn học:** IE212 — Big Data, Học kỳ 2 - 2026  
**Trường:** Trường Đại học Công nghệ Thông tin, ĐHQG-HCM  
**Giảng viên hướng dẫn (GVHD):** T.S Hà Minh Tân

## Thành viên thực hiện

| STT | Họ và tên | MSSV |
| :---: | :--- | :---: |
| 1 |  Phước | 23521235 |
| 2 |  Nguyễn Huỳnh Xuân Nghi | 23521004 |
| 3 |  Mạnh | 23520916 |



---

## Giới thiệu
Dự án tái tạo bài báo *"Supply chain sales forecasting based on lightGBM and LSTM combination model"* và thực hiện các cải tiến trên tập dữ liệu Corporación Favorita (12.5M+ records). Nhóm tập trung vào việc xử lý dữ liệu lớn trên hạ tầng Spark và áp dụng các mô hình học sâu hiện đại để tối ưu hóa dự báo doanh số.

## Dữ liệu
**Lưu ý:** Dữ liệu gốc và các file trung gian (`.csv`, `.parquet`) được loại bỏ khỏi Git do kích thước lớn. 

* **Nguồn dữ liệu:** [Corporación Favorita Grocery Sales Forecasting trên Kaggle](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting/data)
* **Quy mô:** 12,549,704 bản ghi (Tập huấn luyện).

## Cấu trúc thư mục
*(cập nhật sau)*

## Phương pháp triển khai
* **Baseline:** Tái tạo mô hình kết hợp **LSTM** (trích xuất Net Features) và **LightGBM** (Gradient Boosting).
* **Cải tiến 1:** Ứng dụng **Wavelet Transform** thay thế cho bộ lọc Kalman truyền thống để khử nhiễu dữ liệu chuỗi thời gian mà không làm mất các tín hiệu biến động mạnh (peaks).
* **Cải tiến 2:** Triển khai mô hình **Temporal Fusion Transformer (TFT)** nhằm tận dụng cơ chế Multi-head Attention để tăng khả năng diễn giải mô hình (Interpretability).

## Setup & Chạy dự án
*( cập nhật sau)*

## Tài liệu tham khảo
1. Weng, T., Liu, W., & Xiao, J. (2020). *Supply chain sales forecasting based on lightGBM and LSTM combination model*. Industrial Management & Data Systems.
