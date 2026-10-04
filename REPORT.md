# Lab 16 — Báo cáo: Cloud AI Environment trên AWS (CPU + LightGBM)

- **Ngày chạy:** 04/10/2026 · **Cloud:** AWS, region `ap-southeast-2` (Sydney) · **Hạ tầng:** Terraform (`terraform/`, commit gốc `55539f6`)
- **Compute node:** `c7i-flex.large` (2 vCPU, 3,7 GB RAM), Ubuntu 22.04, Python 3.10.12, LightGBM 4.7.0, scikit-learn 1.7.2, pandas 2.3.3
- **Dataset:** Kaggle `mlg-ulb/creditcardfraud` — 284.807 giao dịch × 31 cột, 492 gian lận (0,17%), không thiếu giá trị
- **Cách chia:** stratified 80/20 train+val/test (seed 42), 10% train+val làm validation cho early stopping

## Kết quả benchmark

| Metric | Kết quả |
|---|---|
| Thời gian load data | 1,028 s |
| Thời gian training | 6,863 s |
| Best iteration | 214 |
| AUC-ROC | 0,9838 |
| Accuracy | 0,9994 |
| F1-Score | 0,8280 |
| Precision | 0,8750 |
| Recall | 0,7857 |
| Inference latency (1 row) | 0,689 ms (median; p95 0,724 ms) |
| Inference throughput (1000 rows) | 7,60 ms/batch ≈ 131.574 rows/s |

Thêm: AUPRC (average precision) 0,8350 · Confusion matrix trên test (ngưỡng 0,5): TN 56.853, FP 11, FN 21, TP 77. Chi tiết trong `benchmark_result.json`.

## Nhận xét

1. Training chỉ mất ~6,9 s cho 205 nghìn dòng × 30 đặc trưng trên 2 vCPU (`top` cho thấy python3 dùng ~150% CPU và ~0,5 GB RAM trên tổng 3,7 GB) — với dữ liệu bảng cỡ này, LightGBM trên CPU nhỏ là đủ, không cần GPU.
2. Load CSV 144 MB mất ~1 s; toàn bộ benchmark chạy dưới 15 s, phần lớn thời gian của lab là dựng hạ tầng chứ không phải tính toán.
3. AUC-ROC 0,984 cho thấy mô hình xếp hạng giao dịch gian lận rất tốt; AUPRC 0,835 là chỉ số sát thực tế hơn với dữ liệu mất cân bằng (Kaggle khuyến nghị dùng AUPRC cho bộ này).
4. Accuracy 0,9994 gần như vô nghĩa: đoán "không gian lận" cho mọi giao dịch đã đạt 0,9983. F1/Precision/Recall mới phản ánh chất lượng: bắt được 77/98 vụ gian lận (Recall 0,79) với 11 cảnh báo nhầm (Precision 0,875).
5. Early stopping ban đầu theo logloss/AUC trên tập validation (chỉ ~40 ca gian lận) dừng ở vòng ~51 và cho test AUC chỉ 0,919; chuyển sang dừng theo AUPRC (patience 100) thì mô hình học đến vòng 214 và đạt AUC 0,984.
6. Inference 1 dòng ~0,7 ms (≈1.450 dự đoán/s nếu gọi từng dòng), còn batch 1000 dòng đạt ~131 nghìn dòng/s — nhanh hơn ~90 lần nhờ chia sẻ overhead gọi hàm; khi phục vụ thực tế nên gom batch.
7. Ngưỡng 0,5 là mặc định; nếu ưu tiên bắt gian lận (Recall) hơn cảnh báo nhầm thì có thể hạ ngưỡng.

## Khác biệt so với README và lý do

| README | Thực tế | Lý do |
|---|---|---|
| Region `us-east-1` | `ap-southeast-2` (đặt bằng `TF_VAR_aws_region`, không sửa code) | Tài khoản AWS Free plan bị Service Control Policy chặn EC2/ELB ở mọi region trừ Sydney (`UnauthorizedOperation ... explicit deny in a service control policy`) |
| Compute node `t3.medium` | `c7i-flex.large` (đặt bằng `TF_VAR_cpu_instance_type`) | `terraform apply` bị từ chối: *"The specified instance type is not eligible for Free Tier"*; `c7i-flex.large` có cùng 2 vCPU / 4 GB và nằm trong danh sách Free plan |
| IAM Group `AI-Lab-Group` | Gắn 4 policy (EC2, VPC, ELB, IAM FullAccess) trực tiếp cho user `ai-lab-user` | Console IAM của tài khoản không có mục *User groups* |
| SSH Bastion từ 1 IP `/32` | 2 IP `/32` | Mạng nhà ra Internet bằng 2 IP public khác nhau |
| `ssh` 2 bước qua Bastion | `ssh -F terraform/ssh_config lab-cpu` (ProxyJump) | Theo runbook: Bastion không có private key |

## Thời gian và chi phí

- `terraform apply` bắt đầu 10:33:17 → hạ tầng sẵn sàng 10:42:23 (gồm lần apply đầu bị từ chối `t3.medium` và lần apply lại); SSH và cài môi trường xong ~10:44; `terraform destroy` 11:08:33 → 11:09:51 (*Destroy complete! Resources: 27 destroyed*), đã kiểm tra không còn instance/NAT/EIP/ALB/VPC.
- Tổng thời gian hạ tầng chạy ~36 phút; chi phí ước tính < $0,20, trừ vào credit Free plan (xem `screenshots/04_billing.png`).

## File nộp

- `benchmark.py`, `benchmark_result.json`, `REPORT.md`
- `screenshots/01_benchmark_output.png` — output `python3 benchmark.py` trên compute node
- `screenshots/02_top.png` — `top` trong lúc benchmark chạy
- `screenshots/03_free_iplink.png` — `free -h` và `ip -s link`
- `screenshots/04_billing.png` — AWS Billing
- `terraform.zip` — thư mục `terraform/` đã chạy (bỏ SSH key, state, `.terraform/`)
