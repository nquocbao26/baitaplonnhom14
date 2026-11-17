
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==================================
# 1. CAU HINH TRANG WEB
# ==================================

st.set_page_config(
    page_title="Du doan huy dat phong",
    page_icon="🏨",
    layout="wide"
)

# ==================================
# 2. TAI MO HINH MACHINE LEARNING
# ==================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "hotel_cancellation_model.joblib"
)

if not MODEL_PATH.exists():
    st.error("Khong tim thay file mo hinh .joblib!")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

package = load_model()

model = package["model"]
features = package["features"]
categorical_cols = package["categorical_cols"]
category_options = package["category_options"]
numeric_defaults = package["numeric_defaults"]

# ==================================
# 3. TEN CAC TRUONG DU LIEU
# ==================================

labels = {
    "hotel": "Loại khách sạn",
    "lead_time": "Số ngày đặt trước",
    "arrival_date_month": "Tháng nhận phòng",
    "stays_in_weekend_nights": "Số đêm cuối tuần",
    "stays_in_week_nights": "Số đêm ngày thường",
    "adults": "Số người lớn",
    "children": "Số trẻ em",
    "babies": "Số em bé",
    "meal": "Loại bữa ăn",
    "market_segment": "Phân khúc thị trường",
    "distribution_channel": "Kênh đặt phòng",
    "is_repeated_guest": "Khách đã từng đặt phòng?",
    "previous_cancellations": "Số lần hủy trước đây",
    "previous_bookings_not_canceled": "Số lần không hủy trước đây",
    "reserved_room_type": "Loại phòng đã đặt",
    "deposit_type": "Loại tiền đặt cọc",
    "customer_type": "Loại khách hàng",
    "required_car_parking_spaces": "Số chỗ đỗ xe",
    "total_of_special_requests": "Số yêu cầu đặc biệt"
}

# ==================================
# 4. HAM TAO O NHAP
# ==================================

def create_input(field):

    label = labels.get(field, field)

    if field in categorical_cols:
        return st.selectbox(
            label,
            options=category_options[field],
            key=field
        )

    if field == "is_repeated_guest":
        return st.selectbox(
            label,
            options=[0, 1],
            format_func=lambda x: "Có" if x == 1 else "Không",
            key=field
        )

    default = max(
        0,
        int(round(numeric_defaults[field]))
    )

    return st.number_input(
        label,
        min_value=0,
        value=default,
        step=1,
        key=field
    )

# ==================================
# 5. GIAO DIEN CHINH
# ==================================

st.title("🏨 HỆ THỐNG DỰ ĐOÁN HỦY ĐẶT PHÒNG")

st.markdown(
    "### Ứng dụng Machine Learning trong quản lý khách sạn"
)

st.info(
    "Nhập thông tin đặt phòng để dự đoán "
    "khả năng khách hàng hủy phòng."
)

st.divider()

main_features = [
    "hotel",
    "lead_time",
    "arrival_date_month",
    "adults",
    "children",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "deposit_type"
]

other_features = [
    field for field in features
    if field not in main_features
]

inputs = {}

# ==================================
# 6. FORM NHAP THONG TIN
# ==================================

with st.form("hotel_form"):

    st.subheader("📋 Thông tin đặt phòng")

    col1, col2 = st.columns(2)

    for i, field in enumerate(main_features):
        with (col1 if i % 2 == 0 else col2):
            inputs[field] = create_input(field)

    with st.expander("Thông tin bổ sung"):

        col3, col4 = st.columns(2)

        for i, field in enumerate(other_features):
            with (col3 if i % 2 == 0 else col4):
                inputs[field] = create_input(field)

    submitted = st.form_submit_button(
        "🔍 DỰ ĐOÁN KHẢ NĂNG HỦY PHÒNG",
        type="primary",
        use_container_width=True
    )

# ==================================
# 7. DU DOAN MACHINE LEARNING
# ==================================

if submitted:

    input_data = pd.DataFrame(
        [{field: inputs[field] for field in features}],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    st.divider()
    st.subheader("📊 KẾT QUẢ DỰ ĐOÁN")

    st.metric(
        "Xác suất hủy phòng ước tính",
        f"{probability * 100:.2f}%"
    )

    st.progress(probability)

    if prediction == 1:

        st.error(
            "⚠️ KHÁCH HÀNG CÓ KHẢ NĂNG HỦY PHÒNG"
        )

        st.warning(
            "Khách sạn nên xác nhận lại "
            "thông tin đặt phòng với khách hàng."
        )

    else:

        st.success(
            "✅ KHÁCH HÀNG CÓ KHẢ NĂNG KHÔNG HỦY PHÒNG"
        )

    st.caption(
        "Kết quả mang tính dự đoán thống kê, "
        "không phải kết luận chắc chắn."
    )

st.divider()

st.caption(
    "Đề tài 14 - Dự đoán khách hàng hủy đặt phòng khách sạn"
)
