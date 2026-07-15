"""
generate_shipment_data.py

Sinh synthetic dataset cho bài toán dự đoán ETA & lan truyền delay
(Safiri AI Take-Home Assignment).

Mỗi shipment có 4 stage:
    1. Departure (khởi hành từ origin)
    2. Port Arrival (đến cảng/điểm trung chuyển)
    3. Customs Clearance (thông quan)
    4. Final Delivery (giao hàng cuối)

Delay được sinh CÓ TÍNH LAN TRUYỀN: delay ở stage sau phụ thuộc một phần
vào delay ở stage trước, cộng thêm ảnh hưởng từ congestion/weather.

Chạy: python generate_shipment_data.py
Kết quả: shipments.csv (mặc định 250 dòng)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# Cấu hình chung
# ----------------------------------------------------------------------
np.random.seed(42)
N_SHIPMENTS = 250

ROUTES = [
    # (origin, destination, route_type, base_transit_days)
    ("Shanghai", "Los Angeles", "sea", 14),
    ("Shanghai", "Rotterdam", "sea", 30),
    ("Singapore", "Dubai", "sea", 10),
    ("Ho Chi Minh City", "Tokyo", "sea", 6),
    ("Hamburg", "New York", "sea", 9),
    ("Hong Kong", "Sydney", "sea", 8),
    ("Mumbai", "London", "air", 1),
    ("Bangkok", "Frankfurt", "air", 1),
    ("Ho Chi Minh City", "Los Angeles", "sea", 18),
    ("Busan", "Long Beach", "sea", 12),
]

# Hệ số lan truyền delay giữa các stage (GIẢ ĐỊNH - nêu rõ trong report)
PROP_DEPARTURE_TO_PORT = 0.30     # delay departure ảnh hưởng đến port arrival
PROP_PORT_TO_CUSTOMS = 0.40       # delay port ảnh hưởng đến customs
PROP_CUSTOMS_TO_FINAL_NOISE = 1.0 # customs delay cộng dồn thẳng vào final


# ----------------------------------------------------------------------
# Hàm sinh dữ liệu
# ----------------------------------------------------------------------
def generate_shipment(shipment_id: int) -> dict:
    origin, destination, route_type, base_transit_days = ROUTES[
        np.random.randint(len(ROUTES))
    ]

    # Yếu tố ngoại sinh: congestion (0-1), weather severity (0-1)
    congestion_index = np.clip(np.random.beta(2, 5), 0, 1)
    weather_index = np.clip(np.random.beta(2, 6), 0, 1)

    # Sea route chịu ảnh hưởng congestion/weather nhiều hơn air route
    route_factor = 1.5 if route_type == "sea" else 0.5

    # --- Scheduled timestamps ---
    scheduled_departure = datetime(2025, 1, 1) + timedelta(
        days=int(np.random.randint(0, 300))
    )
    # Phân bổ tổng transit time cho 3 chặng: dep->port, port->customs, customs->final
    leg1 = base_transit_days * 0.5
    leg2 = base_transit_days * 0.35
    leg3 = base_transit_days * 0.15

    scheduled_port_arrival = scheduled_departure + timedelta(days=leg1)
    scheduled_customs = scheduled_port_arrival + timedelta(days=leg2)
    scheduled_final = scheduled_customs + timedelta(days=leg3)

    # --- Delay generation (có lan truyền) ---
    # 1) Delay tại departure: nhiễu cơ bản, không phụ thuộc stage nào trước đó
    delay_departure = np.random.exponential(scale=0.5)  # ngày

    # 2) Delay tại port: phụ thuộc congestion + một phần delay departure
    delay_port = (
        np.random.exponential(scale=0.5)
        + PROP_DEPARTURE_TO_PORT * delay_departure
        + congestion_index * route_factor * 1.5
    )

    # 3) Delay tại customs: phụ thuộc delay port + yếu tố riêng (thủ tục giấy tờ...)
    delay_customs = (
        np.random.exponential(scale=0.4)
        + PROP_PORT_TO_CUSTOMS * delay_port
        + weather_index * 0.5  # thời tiết xấu có thể làm chậm xử lý giấy tờ gián tiếp
    )

    # 4) Delay tại final delivery: cộng dồn toàn bộ chuỗi + nhiễu cuối
    delay_final = (
        PROP_CUSTOMS_TO_FINAL_NOISE * delay_customs
        + 0.2 * delay_port
        + 0.1 * delay_departure
        + np.random.normal(0, 0.3)
    )
    delay_final = max(delay_final, 0)  # không cho delay âm

    # --- Actual timestamps ---
    actual_departure = scheduled_departure + timedelta(days=delay_departure)
    actual_port_arrival = scheduled_port_arrival + timedelta(days=delay_port)
    actual_customs = scheduled_customs + timedelta(days=delay_customs)
    actual_final = scheduled_final + timedelta(days=delay_final)

    # --- Mô phỏng missing data (thử thách "incomplete intermediate data") ---
    # ~8% khả năng thiếu actual_customs (ví dụ: dữ liệu không được ghi nhận)
    missing_customs = np.random.rand() < 0.08
    if missing_customs:
        actual_customs_out = None
        delay_customs_out = None
    else:
        actual_customs_out = actual_customs
        delay_customs_out = round(delay_customs, 2)

    return {
        "shipment_id": shipment_id,
        "origin": origin,
        "destination": destination,
        "route_type": route_type,
        "congestion_index": round(congestion_index, 3),
        "weather_index": round(weather_index, 3),

        "scheduled_departure": scheduled_departure,
        "actual_departure": actual_departure,
        "delay_departure_days": round(delay_departure, 2),

        "scheduled_port_arrival": scheduled_port_arrival,
        "actual_port_arrival": actual_port_arrival,
        "delay_port_days": round(delay_port, 2),

        "scheduled_customs_clearance": scheduled_customs,
        "actual_customs_clearance": actual_customs_out,
        "delay_customs_days": delay_customs_out,

        "scheduled_final_delivery": scheduled_final,
        "actual_final_delivery": actual_final,
        "delay_final_days": round(delay_final, 2),  # <-- target chính (ground truth)

        # Nhãn phụ cho bài toán classification: delay đáng kể hay không (>1 ngày)
        "is_delayed": int(delay_final > 1.0),
    }


def main():
    records = [generate_shipment(i) for i in range(1, N_SHIPMENTS + 1)]
    df = pd.DataFrame(records)

    output_path = "shipments.csv"
    df.to_csv(output_path, index=False)

    print(f"Đã sinh {len(df)} shipments -> {output_path}")
    print("\nThống kê nhanh:")
    print(df[["delay_departure_days", "delay_port_days",
              "delay_customs_days", "delay_final_days"]].describe())
    print(f"\nTỷ lệ shipment bị delay (>1 ngày): {df['is_delayed'].mean():.2%}")
    print(f"Tỷ lệ thiếu dữ liệu customs: {df['actual_customs_clearance'].isna().mean():.2%}")


if __name__ == "__main__":
    main()
