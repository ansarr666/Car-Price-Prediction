"""
Car Price Predictor — Streamlit Application
===========================================
Frontend-only Streamlit application that communicates with the FastAPI
backend for model inference via HTTP requests.
"""

# ──────────────────────────────────────────────
# 1. Imports
# ──────────────────────────────────────────────

import pathlib
import pandas as pd
import streamlit as st
import requests

# ──────────────────────────────────────────────
# 2. Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="Car Price Predictor"
)

# Resolve paths relative to this script
BASE_DIR  = pathlib.Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "car_price_prediction.csv"

# FastAPI backend URL (change this when deploying to production)
API_BASE_URL = "http://127.0.0.1:8000"

# Request timeout in seconds (connect, read)
API_TIMEOUT = (5, 30)

# ──────────────────────────────────────────────
# 3. Data Loader & API Helpers
# ──────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_reference_data() -> pd.DataFrame:
    """Load and clean the CSV using the cleaning logic from train.py."""
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}.")
        st.stop()
    df = pd.read_csv(DATA_PATH)

    # Import clean_data from train.py to keep the code DRY
    from train import clean_data
    return clean_data(df)


def get_options(df: pd.DataFrame) -> dict:
    """Extract sorted unique values for each categorical column."""
    return {
        "make":              sorted(df["make"].unique()),
        "model":             sorted(df["model"].unique()),
        "engine_fuel_type":  sorted(df["engine_fuel_type"].unique()),
        "transmission_type": sorted(df["transmission_type"].unique()),
        "driven_wheels":     sorted(df["driven_wheels"].unique()),
        "vehicle_size":      sorted(df["vehicle_size"].unique()),
        "vehicle_style":     sorted(df["vehicle_style"].unique()),
    }


def check_api_health() -> bool:
    """Ping the FastAPI /health endpoint to verify the backend is reachable.

    Returns True if the API responds with a healthy status, False otherwise.
    """
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=API_TIMEOUT)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_predict_api(payload: dict) -> dict:
    """Send a prediction request to the FastAPI /predict endpoint.

    Args:
        payload: Dictionary matching the CarFeatures schema expected by the API.

    Returns:
        Parsed JSON response containing 'predicted_price'.

    Raises:
        requests.exceptions.ConnectionError: Backend is unreachable.
        requests.exceptions.Timeout: Request exceeded the timeout limit.
        requests.exceptions.HTTPError: API returned a non-2xx status code.
    """
    resp = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=API_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────
# 4. Format Helpers
# ──────────────────────────────────────────────

def pretty_label(raw: str) -> str:
    """Convert snake_case value to a user-friendly label."""
    return raw.replace("_", " ").title()


# ──────────────────────────────────────────────
# 5. Sidebar (Model Metadata & Project Context Only)
# ──────────────────────────────────────────────

def render_sidebar() -> None:
    """Render a minimal, uncluttered sidebar with model statistics and information."""
    with st.sidebar:
        st.subheader("About the Model")
        st.markdown(
            "**Algorithm:** Random Forest Regressor  \n"
            "**Estimators:** 100 trees  \n"
            "**Training Size:** 11,914 vehicles  \n"
            "**R² Accuracy:** 0.9534  \n"
            "**MAPE:** 7.71%"
        )
        
        st.write("")
        st.subheader("Project Details")
        st.markdown(
            "**Target:** MSRP (Manufacturer Suggested Retail Price)  \n"
            "**Features Used:** 13 total  \n"
            "**Preprocessing:** OneHotEncoder, Passthrough  \n"
            "**Engineered Features:** age, hp_per_cylinder"
        )
        
        st.write("")
        st.subheader("Tech Stack")
        st.markdown("Python, scikit-learn, FastAPI, Streamlit, pandas, NumPy")


# ──────────────────────────────────────────────
# 6. Main Application UI
# ──────────────────────────────────────────────

def main() -> None:
    # Render layout elements
    render_sidebar()

    # Load reference data for dropdown options
    ref_df = load_reference_data()
    options = get_options(ref_df)

    # Build dynamic make -> model mapping
    make_model_map = {}
    for m in options["make"]:
        make_model_map[m] = sorted(ref_df[ref_df["make"] == m]["model"].unique())

    # --- Header ---
    st.title("Car Price Predictor")
    st.caption("Estimate a vehicle's Manufacturer Suggested Retail Price using a Random Forest model.")
    st.write("")

    # --- API Health Status Banner ---
    if not check_api_health():
        st.warning(
            "⚠️ The prediction backend is currently unavailable. "
            "Please ensure the FastAPI server is running "
            f"(`uvicorn main:app --reload` at `{API_BASE_URL}`)."
        )

    # --- Input Form Block ---
    with st.container(border=True):
        st.subheader("Vehicle Specifications")
        st.write("")

        # Row 1: Make & Model (Equal Width)
        col1, col2 = st.columns(2)
        with col1:
            selected_make = st.selectbox(
                "Make (Manufacturer)",
                options=options["make"],
                format_func=pretty_label,
                index=options["make"].index("toyota") if "toyota" in options["make"] else 0,
            )
        with col2:
            available_models = make_model_map.get(selected_make, options["model"])
            selected_model = st.selectbox(
                "Model",
                options=available_models,
                format_func=pretty_label,
            )

        # Row 2: Year Slider (Full Width with inline label/value display)
        # Note: Slider displays current value inline on the right side of the label automatically
        selected_year = st.slider(
            "Manufacturing Year",
            min_value=1990,
            max_value=2017,
            value=2015,
            help="Dataset contains model years from 1990 to 2017.",
        )

        # Row 3: Fuel Type, Transmission, Doors (3 columns)
        col3, col4, col5 = st.columns(3)
        with col3:
            selected_fuel = st.selectbox(
                "Engine Fuel Type",
                options=options["engine_fuel_type"],
                format_func=pretty_label,
                index=options["engine_fuel_type"].index("regular_unleaded")
                if "regular_unleaded" in options["engine_fuel_type"]
                else 0,
            )
        with col4:
            selected_transmission = st.selectbox(
                "Transmission Type",
                options=options["transmission_type"],
                format_func=pretty_label,
                index=options["transmission_type"].index("automatic")
                if "automatic" in options["transmission_type"]
                else 0,
            )
        with col5:
            selected_doors = st.selectbox(
                "Number of Doors",
                options=[2.0, 4.0],
                index=1,
                format_func=lambda x: str(int(x)),
            )

        # Row 4: Horsepower, Cylinders, Driven Wheels (3 columns)
        col6, col7, col8 = st.columns(3)
        with col6:
            selected_hp = st.number_input(
                "Engine Horsepower",
                min_value=50.0,
                max_value=1100.0,
                value=240.0,
                step=5.0,
                help="Engine horsepower rating",
            )
        with col7:
            selected_cylinders = st.selectbox(
                "Engine Cylinders",
                options=[0, 3, 4, 5, 6, 8, 10, 12, 16],
                index=4,
                help="0 indicates an Electric Vehicle",
            )
        with col8:
            selected_wheels = st.selectbox(
                "Driven Wheels",
                options=options["driven_wheels"],
                format_func=pretty_label,
                index=options["driven_wheels"].index("front_wheel_drive")
                if "front_wheel_drive" in options["driven_wheels"]
                else 0,
            )

        # Row 5: Size, Style, City MPG (3 columns)
        col9, col10, col11 = st.columns(3)
        with col9:
            selected_size = st.selectbox(
                "Vehicle Size",
                options=options["vehicle_size"],
                format_func=pretty_label,
                index=options["vehicle_size"].index("midsize")
                if "midsize" in options["vehicle_size"]
                else 0,
            )
        with col10:
            selected_style = st.selectbox(
                "Vehicle Style",
                options=options["vehicle_style"],
                format_func=pretty_label,
            )
        with col11:
            selected_city_mpg = st.number_input(
                "City MPG",
                min_value=5,
                max_value=140,
                value=18,
                step=1,
            )

    st.write("")
    st.divider()
    predict_clicked = st.button("Predict Price", use_container_width=True)
    st.divider()
    st.write("")

    # --- Results Block ---
    if predict_clicked:
        # Simple input validation
        if selected_hp <= 0:
            st.error("Engine horsepower must be positive.")
            return
        if selected_city_mpg <= 0:
            st.error("City MPG must be positive.")
            return

        # Build the JSON payload matching the FastAPI CarFeatures schema
        payload = {
            "make":              selected_make,
            "model":             selected_model,
            "engine_fuel_type":  selected_fuel,
            "engine_hp":         float(selected_hp),
            "engine_cylinders":  float(selected_cylinders),
            "transmission_type": selected_transmission,
            "driven_wheels":     selected_wheels,
            "number_of_doors":   float(selected_doors),
            "vehicle_size":      selected_size,
            "vehicle_style":     selected_style,
            "city_mpg":          int(selected_city_mpg),
            "year":              selected_year,
        }

        try:
            with st.spinner("Running model inference..."):
                result = call_predict_api(payload)
                predicted_price = result["predicted_price"]

            st.subheader("Results")
            st.write("")

            # Display Predicted Price as a large metric
            st.metric(
                label="Estimated MSRP",
                value=f"${predicted_price:,.2f}"
            )
            st.write("")

            # Metric details columns
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Make", pretty_label(selected_make))
            col_b.metric("Horsepower", f"{selected_hp:.0f} HP")
            col_c.metric("Age", f"{2017 - selected_year} yrs")
            col_d.metric("City MPG", f"{selected_city_mpg} mpg")
            st.write("")

            # Model error disclaimer note
            st.info(
                "This prediction is generated by a Random Forest model with a "
                "Mean Absolute Percentage Error (MAPE) of 7.71%."
            )

        except requests.exceptions.ConnectionError:
            st.error(
                "🔌 **Connection failed.** Could not reach the prediction backend. "
                f"Please ensure the FastAPI server is running at `{API_BASE_URL}`."
            )
        except requests.exceptions.Timeout:
            st.error(
                "⏱️ **Request timed out.** The prediction backend took too long to respond. "
                "Please try again shortly."
            )
        except requests.exceptions.HTTPError as http_err:
            # Extract the detail message from the FastAPI error response if available
            try:
                detail = http_err.response.json().get("detail", str(http_err))
            except Exception:
                detail = str(http_err)
            st.error(f"🚫 **Prediction failed:** {detail}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
