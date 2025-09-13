import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import pytz
import json
import io
import re
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gspread
from google.oauth2.service_account import Credentials

# -------------------------
# IMPORTANT: page config must be first Streamlit call
# -------------------------
st.set_page_config(
    page_title="Owolawi Compound - Electricity Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Google Sheets setup
# -------------------------
# Define scope
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"  # This was missing!
]

# trying loading credentials from secrets and authorize gspread client
sheet = None
client = None
try:
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=scope
    )
    client = gspread.authorize(creds)
    SHEET_NAME = "ElectricityConsumption"
    sheet = client.open(SHEET_NAME).sheet1
    
    st.success("✅ Successfully connected to Google Sheets!")
    
except Exception as e:
    # helpful error message to the user
    st.error(
        "Error connecting to Google Sheets. Check your service account credentials in Streamlit Secrets and ensure the sheet "
        f"named '{SHEET_NAME}' exists and is shared with the service account email.\n\nDetails: " + str(e)
    )
    
# -------------------------
# Config / Constants
# -------------------------
# local time
wat_tz = pytz.timezone('Africa/Lagos')

# -------------------------
# Session state initialization
# -------------------------
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'rate_per_kwh': 250,
        'primary_color': '#2E86AB',
        'secondary_color': '#1f77b4',
        'success_color': '#28a745',
        'occupants': [
            {'name': 'Mr Chidi', 'icon': '👤'},
            {'name': 'Mr Olisa', 'icon': '👤'},
            {'name': 'Mr Martin', 'icon': '👤'}
        ],
        'compound_name': 'Owolawi Compound',
        'currency': '₦'
    }

# -------------------------
# Utilities: persistence
# -------------------------

# --- robust create_calculation_data ---
def create_calculation_data(timestamp, initial_readings, final_readings,
                            consumptions, costs, total_costs, water_consumed,
                            water_cost, rate, total_amount):
    """Create calculation data dictionary (robust to missing keys)."""
    data = {
        'timestamp': timestamp,
        'timestamp_iso': datetime.now(wat_tz).isoformat(),
        'water_consumed': water_consumed,
        'water_cost': water_cost,
        'rate_per_kwh': rate,
        'total_amount': total_amount
    }

    occupants = st.session_state.get('settings', {}).get('occupants', [])
    for i, occupant in enumerate(occupants):
        key = f"occupant_{i}"
        # use .get to avoid KeyError for placeholder dicts
        data[f'{key}_initial'] = initial_readings.get(key, 0)
        data[f'{key}_final'] = final_readings.get(key, 0)
        data[f'{key}_consumed'] = consumptions.get(key, 0)
        data[f'{key}_total'] = total_costs.get(key, 0)

    # ensure water keys exist (use .get to be safe)
    data['water_initial'] = initial_readings.get('water', 0)
    data['water_final'] = final_readings.get('water', 0)

    return data

# --- helper to build consistent sheet headers ---
def get_sheet_headers():
    """Return a consistent list of headers for the sheet based on current occupants."""
    occupants = st.session_state.get('settings', {}).get('occupants', [])
    n = len(occupants)
    # placeholders must include 'water'
    initial = {f"occupant_{i}": 0 for i in range(n)}
    initial['water'] = 0
    final = initial.copy()
    consumptions = initial.copy()
    costs = initial.copy()
    totals = initial.copy()
    headers = list(create_calculation_data(
        "timestamp", initial, final, consumptions, costs, totals,
        0, 0, st.session_state.get('settings', {}).get('rate_per_kwh', 0), 0
    ).keys())
    return headers

# -------------------------
# Export helpers
# -------------------------

def export_to_json(df: pd.DataFrame) -> str:
    """Export DataFrame to formatted JSON string with pretty indentation."""
    if df.empty:
        return "[]"
    df = _coerce_history_numeric(df)
    records = df.to_dict(orient="records")
    return json.dumps(records, indent=2, ensure_ascii=False)    

def load_history():
    """Load all history records from Google Sheets into a list of dicts."""
    if sheet is None:
        return []
    try:
        records = sheet.get_all_records()
        return records if records else []
    except Exception as e:
        st.error(f"Error loading history from Google Sheets: {e}")
        return []

def save_history(history_data):
    """Overwrite Google Sheet with full history (not used often)."""
    if sheet is None:
        st.error("Google Sheet not initialized. Cannot save history.")
        return False
    try:
        sheet.clear()
        if not history_data:
            return True
        # Use the canonical headers for this app
        headers = get_sheet_headers()
        sheet.append_row(headers)
        for record in history_data:
            row = [record.get(h, "") for h in headers]
            sheet.append_row(row)
        return True
    except Exception as e:
        st.error(f"Error saving history to Google Sheets: {e}")
        return False

def reset_history():
    """Clear all history from Google Sheets and session state."""
    if sheet is None:
        st.error("Google Sheet not initialized. Cannot reset history.")
        return False
    try:
        sheet.clear()
        headers = get_sheet_headers()
        sheet.append_row(headers)
        st.session_state.consumption_history = []
        return True
    except Exception as e:
        st.error(f"Error clearing history from Google Sheets: {e}")
        return False

# -------------------------
# Initialize session state
# -------------------------
if "consumption_history" not in st.session_state:
    st.session_state["consumption_history"] = load_history()

# -------------------------
# Initialize sheet headers if empty
# -------------------------
def init_sheet():
    """Ensure the Google Sheet has headers (called at startup)."""
    if sheet is None:
        return
    try:
        vals = sheet.get_all_values()
        if not vals or len(vals) == 0 or (len(vals) == 1 and all([str(x).strip() == "" for x in vals[0]])):
            headers = get_sheet_headers()
            sheet.append_row(headers)
    except Exception as e:
        st.warning(f"Could not initialize sheet headers: {e}")

# -------------------------
# Dynamic CSS based on settings
# -------------------------
def get_dynamic_css():
    settings = st.session_state.settings
    return f"""
<style>
    .main-header {{
        text-align: center;
        color: {settings['primary_color']};
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }}
    .occupant-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid {settings['primary_color']};
    }}
    .water-card {{
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid {settings['secondary_color']};
    }}
    .summary-card {{
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid {settings['success_color']};
    }}
    .designer-credit {{
        text-align: center;
        font-style: italic;
        color: #666;
        margin-top: 3rem;
        border-top: 1px solid #ddd;
        padding-top: 1rem;
    }}
    .settings-card {{
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }}
    .stButton > button {{
        background-color: {settings['primary_color']};
        color: white;
        border: none;
        border-radius: 5px;
    }}
</style>
"""

st.markdown(get_dynamic_css(), unsafe_allow_html=True)

# -------------------------
# Helpers for calculations / latest readings
# -------------------------
def get_latest_readings():
    """Get the latest readings from history (final readings) with numeric coercion."""
    history = load_history()
    if history:
        df_history = pd.DataFrame(history)
        df_history = _coerce_history_numeric(df_history)
        latest = df_history.iloc[-1].to_dict()
        result = {}
        for i, occupant in enumerate(st.session_state.settings['occupants']):
            key = f"occupant_{i}_final"
            result[key] = float(latest.get(key, 0)) if latest.get(key) is not None else 0.0
        result['water_final'] = float(latest.get('water_final', 0)) if latest.get('water_final') is not None else 0.0
        return result

    # Defaults if no history
    result = {}
    for i, occupant in enumerate(st.session_state.settings['occupants']):
        result[f"occupant_{i}_final"] = 0.0
    result['water_final'] = 0.0
    return result

# Helper to convert Google Sheet string values to numeric before calculations/plots
def _coerce_history_numeric(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Convert columns to numeric if most values look numeric (default 70%).
    """
    for col in df.columns:
        non_null = df[col].dropna().astype(str)
        numeric_like = non_null.str.match(r"^-?\d+(\.\d+)?$")
        if len(non_null) > 0 and (numeric_like.mean() >= threshold):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# -------------------------
# Export helpers: Excel (existing) and combined image (new)
# -------------------------
def export_to_excel(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to Excel format (bytes).
    Includes raw history and summary sheet.
    """
    if df.empty:
        return None

    df = _coerce_history_numeric(df)
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Consumption History', index=False)

            # Add summary sheet
            summary_data = []
            occupants = st.session_state.settings['occupants']

            for i, occupant in enumerate(occupants):
                consumed_col = f"occupant_{i}_consumed"
                total_col = f"occupant_{i}_total"

                if consumed_col in df.columns and total_col in df.columns:
                    avg_consumption = df[consumed_col].mean()
                    avg_cost = df[total_col].mean()
                    total_consumption = df[consumed_col].sum()
                    total_cost = df[total_col].sum()

                    summary_data.append({
                        'Occupant': occupant['name'],
                        'Average Consumption (kWh)': round(avg_consumption, 2),
                        'Average Cost': round(avg_cost, 2),
                        'Total Consumption (kWh)': round(total_consumption, 2),
                        'Total Cost': round(total_cost, 2)
                    })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

        output.seek(0)
        return output.getvalue()
    except Exception as e:
        st.error(f"Error exporting to Excel: {e}")
        return None

def create_combined_image(df_summary, fig_pie, fig_bar, settings):
    """
    Create a single image (JPEG) combining:
    - pie chart image (plotly png)
    - bar chart image (plotly png)
    - summary table image (matplotlib table)
    - footer credit
    Returns bytes (JPEG).
    """
    try:
        # Create PNG images from plotly figures
        pie_png = pio.to_image(fig_pie, format="png", scale=2)  # bytes
        bar_png = pio.to_image(fig_bar, format="png", scale=2)

        pie_img = Image.open(io.BytesIO(pie_png)).convert("RGB")
        bar_img = Image.open(io.BytesIO(bar_png)).convert("RGB")

        # Create table image using matplotlib
        fig, ax = plt.subplots(figsize=(8, max(1.5, len(df_summary) * 0.5 + 1)))
        ax.axis('off')
        # build table
        table = ax.table(cellText=df_summary.values,
                         colLabels=df_summary.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        plt.close(fig)
        buf.seek(0)
        table_img = Image.open(buf).convert("RGB")

        # Determine final width = max widths
        width = max(pie_img.width + bar_img.width, table_img.width)
        # We'll stack pie+bar horizontally on top, table below, and footer at bottom
        # First create top combined (pie + bar side-by-side)
        top_height = max(pie_img.height, bar_img.height)
        top_combined = Image.new("RGB", (pie_img.width + bar_img.width, top_height), color="white")
        top_combined.paste(pie_img, (0, 0))
        top_combined.paste(bar_img, (pie_img.width, 0))

        # Final image height
        spacing = 20
        footer_height = 40
        final_width = max(top_combined.width, table_img.width)
        final_height = top_combined.height + spacing + table_img.height + spacing + footer_height

        final_img = Image.new("RGB", (final_width, final_height), color="white")
        # center top_combined
        final_img.paste(top_combined, ((final_width - top_combined.width) // 2, 0))
        # paste table centered
        final_img.paste(table_img, ((final_width - table_img.width) // 2, top_combined.height + spacing))

        # draw footer text
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(final_img)
        # try to load a default font; if not available, Pillow will fallback
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        footer_text = f"Designed by {settings.get('compound_name', '')} - Generated: {datetime.now(wat_tz).strftime('%d-%m-%Y %I:%M')}"
        text_w, text_h = draw.textsize(footer_text, font=font)
        draw.text(((final_width - text_w) // 2, top_combined.height + spacing + table_img.height + (spacing // 2)),
                  footer_text, fill=(80, 80, 80), font=font)

        # Convert to JPEG bytes
        out_buf = io.BytesIO()
        final_img.save(out_buf, format="JPEG", quality=90)
        out_buf.seek(0)
        return out_buf.getvalue()
    except Exception as e:
        st.error(f"Error creating combined image: {e}")
        return None

# -------------------------
# Calculation helpers (save_calculation uses create_calculation_data)
# -------------------------
def save_calculation(timestamp, initial_readings, final_readings,
                     consumptions, costs, total_costs, water_consumed,
                     water_cost, rate, total_amount):
    """Save a single calculation as a new row in Google Sheets (consistent with headers)."""
    if sheet is None:
        st.error("Google Sheet not initialized. Cannot save calculation.")
        return

    try:
        calculation_data = create_calculation_data(
            timestamp, initial_readings, final_readings,
            consumptions, costs, total_costs, water_consumed,
            water_cost, rate, total_amount
        )

        # Always enforce the canonical header order
        headers = get_sheet_headers()
        existing_values = sheet.get_all_values()
        if not existing_values:
            # Add headers if sheet is completely empty
            sheet.append_row(headers)
        else:
            # If headers differ, reconcile them
            sheet_headers = sheet.row_values(1)
            if sheet_headers != headers:
                st.warning("⚠️ Sheet headers differ from expected schema — reconciling.")
                # Reset to canonical headers
                sheet.clear()
                sheet.append_row(headers)

        # Build row in strict header order
        row = [calculation_data.get(h, "") for h in headers]

        # Convert floats to round numbers where appropriate (avoid "123.0" as text)
        row = [round(v, 2) if isinstance(v, (int, float)) else v for v in row]

        sheet.append_row(row)
        st.success("✅ Calculation saved to Google Sheets!")
    except Exception as e:
        st.error(f"❌ Failed to save calculation: {e}")

# -------------------------
# Pages
# -------------------------
def calculation_page():
    settings = st.session_state.settings
    st.subheader("📊 New Electricity Consumption Calculation & Insight")

    # Quick load from last reading
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📋 Use Last Readings as Initial", key="use_last_readings"):
            latest = get_latest_readings()
            for key, value in latest.items():
                st.session_state[f"{key}_session"] = float(value) if value is not None else 0.0
            st.success("✅ Last readings loaded as initial values!")

    # Input section
    st.subheader("📝 Enter Initial Readings (kWh)")

    occupants = settings['occupants']
    num_cols = len(occupants) + 1
    cols = st.columns(num_cols)

    initial_readings = {}
    final_readings = {}

    for i, occupant in enumerate(occupants):
        with cols[i]:
            st.markdown('<div class="occupant-card">', unsafe_allow_html=True)
            st.markdown(f"{occupant['icon']} **{occupant['name']}**")
            key = f"occupant_{i}"
            initial_readings[key] = st.number_input(
                "Initial Reading",
                min_value=0.0,
                value=float(st.session_state.get(f"{key}_final_session", 0.0)),
                step=0.1,
                key=f"{key}_init"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with cols[-1]:
        st.markdown('<div class="water-card">', unsafe_allow_html=True)
        st.markdown("💧 **Water Pump**")
        initial_readings['water'] = st.number_input(
            "Initial Reading",
            min_value=0.0,
            value=st.session_state.get('water_final_session', 0.0),
            step=0.1,
            key="water_init"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("📝 Enter Final Readings (kWh)")
    cols = st.columns(num_cols)

    for i, occupant in enumerate(occupants):
        with cols[i]:
            st.markdown('<div class="occupant-card">', unsafe_allow_html=True)
            key = f"occupant_{i}"
            final_readings[key] = st.number_input(
                "Final Reading",
                min_value=float(initial_readings[key]),
                value=initial_readings[key],
                step=0.1,
                key=f"{key}_final"
            )
            st.markdown('</div>', unsafe_allow_html=True)

    with cols[-1]:
        st.markdown('<div class="water-card">', unsafe_allow_html=True)
        final_readings['water'] = st.number_input(
            "Final Reading",
            min_value=float(initial_readings['water']),
            value=float(initial_readings['water']),
            step=0.1,
            key="water_final"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("💰 Rate Configuration")
    rate_per_kwh = st.number_input(
        f"Rate per kWh ({settings['currency']})",
        min_value=1,
        value=settings['rate_per_kwh'],
        step=1,
        key="rate_input"
    )

    if st.button("🧮 Calculate Bills", type="primary", key="calculate_bills"):
        calculate_and_display_results(initial_readings, final_readings, rate_per_kwh)

def calculate_and_display_results(initial_readings, final_readings, rate):
    settings = st.session_state.settings
    occupants = settings['occupants']
    currency = settings['currency']

    consumptions = {}
    costs = {}

    for i, occupant in enumerate(occupants):
        key = f"occupant_{i}"
        consumption = final_readings[key] - initial_readings[key]
        consumptions[key] = consumption
        costs[key] = consumption * rate

    water_consumed = final_readings['water'] - initial_readings['water']
    water_cost = water_consumed * rate
    water_cost_per_person = water_cost / len(occupants) if len(occupants) > 0 else 0

    total_costs = {}
    for i, occupant in enumerate(occupants):
        key = f"occupant_{i}"
        total_costs[key] = costs[key] + water_cost_per_person

    total_amount = sum(total_costs.values())

    st.subheader("🧮 Calculation Breakdown")
    st.markdown("### Step-by-Step Calculation:")

    with st.expander("📋 View Detailed Calculations", expanded=True):
        consumption_text = "**Individual Consumption:**\n"
        for i, occupant in enumerate(occupants):
            key = f"occupant_{i}"
            consumption_text += f"- {occupant['name']}: {final_readings[key]:.1f} - {initial_readings[key]:.1f} = **{consumptions[key]:.1f} kWh**\n"
        consumption_text += f"- Water Pump: {final_readings['water']:.1f} - {initial_readings['water']:.1f} = **{water_consumed:.1f} kWh**\n\n"

        cost_text = f"**Individual Costs (at {currency}{rate}/kWh):**\n"
        for i, occupant in enumerate(occupants):
            key = f"occupant_{i}"
            cost_text += f"- {occupant['name']}: {consumptions[key]:.1f} × {currency}{rate} = **{currency}{costs[key]:,.0f}**\n"
        cost_text += f"- Water Pump: {water_consumed:.1f} × {currency}{rate} = **{currency}{water_cost:,.0f}**\n\n"

        water_text = f"**Water Cost Split:**\n"
        water_text += f"- Water cost per person: {currency}{water_cost:,.0f} ÷ {len(occupants)} = **{currency}{water_cost_per_person:,.0f}**\n\n"

        final_text = "**Final Amount Per Person:**\n"
        for i, occupant in enumerate(occupants):
            key = f"occupant_{i}"
            final_text += f"- {occupant['name']}: {currency}{costs[key]:,.0f} + {currency}{water_cost_per_person:,.0f} = **{currency}{total_costs[key]:,.0f}**\n"

        breakdown_text = consumption_text + cost_text + water_text + final_text
        st.markdown(breakdown_text)

        plain_text = re.sub(r"\*\*(.*?)\*\*", r"\1", breakdown_text)
        st.session_state["breakdown_text"] = plain_text

    st.subheader("📊 Final Summary")
    current_timestamp = datetime.now(wat_tz).strftime("%a, %d/%m/%Y %I:%M %p")

    summary_data = {
        'Occupant': [occupant['name'] for occupant in occupants],
        'Initial (kWh)': [initial_readings[f"occupant_{i}"] for i in range(len(occupants))],
        'Final (kWh)': [final_readings[f"occupant_{i}"] for i in range(len(occupants))],
        'Consumed (kWh)': [consumptions[f"occupant_{i}"] for i in range(len(occupants))],
        f'Personal Cost ({currency})': [f"{currency}{costs[f'occupant_{i}']:,.0f}" for i in range(len(occupants))],
        f'Water Share ({currency})': [f"{currency}{water_cost_per_person:,.0f}"] * len(occupants),
        f'Total Amount ({currency})': [f"{currency}{total_costs[f'occupant_{i}']:,.0f}" for i in range(len(occupants))]
    }

    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True)

    # Summary cards
    num_cards = len(occupants) + 3
    cols = st.columns(min(4, num_cards))
    col_idx = 0

    with cols[col_idx % len(cols)]:
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        st.metric("💧 Water Pump Cost", f"{currency}{water_cost:,.0f}", f"{water_consumed:.1f} kWh")
        st.markdown('</div>', unsafe_allow_html=True)
        col_idx += 1

    with cols[col_idx % len(cols)]:
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        total_consumption = sum(consumptions.values()) + water_consumed
        st.metric("⚡ Total Consumption", f"{total_consumption:.1f} kWh")
        st.markdown('</div>', unsafe_allow_html=True)
        col_idx += 1

    with cols[col_idx % len(cols)]:
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        st.metric("💰 Total Amount", f"{currency}{total_amount:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)
        col_idx += 1

    if col_idx < 4:
        with cols[col_idx % len(cols)]:
            st.markdown('<div class="summary-card">', unsafe_allow_html=True)
            st.markdown(f"**📅 {current_timestamp}**", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Charts
    st.subheader("📊 Consumption Breakdown Chart")
    chart_persons = [occupant['name'] for occupant in occupants] + ['Water Pump']
    chart_consumptions = [consumptions[f"occupant_{i}"] for i in range(len(occupants))] + [water_consumed]
    chart_costs = [costs[f"occupant_{i}"] for i in range(len(occupants))] + [water_cost]

    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366', '#B366FF', '#66FFB3', '#FF66B3']

    fig_pie = px.pie(
        values=chart_consumptions,
        names=chart_persons,
        title="Energy Consumption Distribution",
        color_discrete_sequence=colors[:len(chart_persons)]
    )

    fig_bar = px.bar(
        x=chart_persons,
        y=chart_costs,
        title="Individual Costs (Before Water Split)",
        labels={'x': 'Person', 'y': f'Cost ({currency})'},
        color=chart_persons,
        color_discrete_sequence=colors[:len(chart_persons)]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.plotly_chart(fig_bar, use_container_width=True)

    # Save / Export / Image operations
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("💾 Save This Calculation", key="save_calc"):
            # timestamp should be consistent and human-friendly
            timestamp = current_timestamp
            save_calculation(timestamp, initial_readings, final_readings,
                             consumptions, costs, total_costs, water_consumed,
                             water_cost, rate, total_amount)

    with col2:
        df_history = pd.DataFrame(load_history())
        if not df_history.empty:
            excel_data = export_to_excel(df_history)
            if excel_data:
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name=f"{settings['compound_name']}_electricity_history_{datetime.now().strftime('%d%m%Y')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
        else:
            st.warning("No history to export!")

    with col3:
        # Allow download of a combined JPEG (table + breakdown + chart)
        if df_summary is not None and not df_summary.empty:
            def export_summary_as_image(df, breakdown_text):
                # Round numeric columns to 1 decimal place where appropriate
                for col in ["Initial (kWh)", "Final (kWh)", "Consumed (kWh)"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float).round(1)
            
                fig = plt.figure(figsize=(9, 14))  # Taller canvas for better spacing
                gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 2, 2.5])  # allocate more space for breakdown + table
            
                # 1. Horizontal bar chart
                ax1 = fig.add_subplot(gs[0])
                bars = sns.barplot(
                    ax=ax1,
                    data=df,
                    y="Occupant",
                    x="Consumed (kWh)",
                    palette="Blues_d"
                )
                ax1.set_title(f"Owolawi Compd Electricity Consumption Breakdown: {datetime.now(wat_tz).strftime('%a, %d-%m-%Y %I:%M %p')}", fontsize=13, weight="bold")
                ax1.set_xlabel("Consumption (kWh)")
                ax1.set_ylabel("")
            
                # Data labels on bars
                for bar in bars.patches:
                    width = bar.get_width()
                    ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                             f"{width:.1f}", va="center", fontsize=9)
            
                # 2. Detailed breakdown text
                ax2 = fig.add_subplot(gs[1])
                ax2.axis("off")
                ax2.text(0, 1, breakdown_text, ha="left", va="top",
                         fontsize=10, family="monospace", wrap=True)
            
                # 3. Summary table
                ax3 = fig.add_subplot(gs[2])
                ax3.axis("off")
                table = ax3.table(
                    cellText=df.values,
                    colLabels=df.columns,
                    cellLoc="center",
                    loc="center"
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.3)  # widen rows for readability
            
                plt.tight_layout()
            
                buf = io.BytesIO()
                plt.savefig(buf, format="jpeg", dpi=200, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                return buf.getvalue()
    
            jpeg_bytes = export_summary_as_image(df_summary, st.session_state["breakdown_text"])
    
            st.download_button(
                label="📷 Download Report JPEG",
                data=jpeg_bytes,
                file_name=f"{settings['compound_name']}_Electricity_Consumption_report_{datetime.now(wat_tz).strftime('%d-%m-%Y_%I-%M%p')}.jpg",
                mime="image/jpeg",
                key="download_jpg"
            )
    

def history_page():
    st.header("📈 Consumption History & Analytics")

    history = load_history()

    # -------------------------
    # Convert to DataFrame + coerce numerics early
    # -------------------------
    df_history = pd.DataFrame(history)
    df_history = _coerce_history_numeric(df_history)

    # Export & reset options
    col1, col2, col3 = st.columns(3)
    with col1:
        if not df_history.empty:
            json_data = export_to_json(df_history)
            st.download_button(
                label="📥 Download History (JSON)",
                data=json_data,
                file_name=f"electricity_history_{datetime.now().strftime('%d%m%Y')}.json",
                mime="application/json",
                key="download_history_json"
            )
        else:
            st.warning("No history to download!")
    
    with col2:
        if not df_history.empty:
            excel_data = export_to_excel(df_history)
            if excel_data:
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name=f"{st.session_state.settings['compound_name']}_history_{datetime.now().strftime('%d%m%Y')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_history_excel"
                )
            else:
                st.error("Excel export failed. Please try again.")
            
    with col3:
        if st.button("🗑️ Reset History (Google Sheets + session)", key="reset_history_confirm"):
            if reset_history():
                st.success("All history cleared from Google Sheets and session.")
                st.rerun()

    # Display recent history table
    display_columns = ['timestamp']
    occupants = st.session_state.settings['occupants']

    for i, occupant in enumerate(occupants):
        display_columns.append(f'occupant_{i}_total')
    display_columns.extend(['water_cost', 'total_amount'])

    if not df_history.empty and all(col in df_history.columns for col in display_columns):
        display_df = df_history[display_columns].tail(10)
        rename_dict = {'timestamp': 'Date/Time', 'water_cost': 'Water Cost', 'total_amount': 'Total'}
        for i, occupant in enumerate(occupants):
            rename_dict[f'occupant_{i}_total'] = f"{occupant['name']} ({st.session_state.settings['currency']})"
        display_df = display_df.rename(columns=rename_dict)
        st.dataframe(display_df.iloc[::-1], use_container_width=True)
    else:
        st.warning("History exists but expected columns are missing in some records. This may happen if older records used a different schema.")

    # -------------------------
    # Analytics charts if >1 record
    # -------------------------
    if len(df_history) > 1:
        st.subheader("📊 Consumption Trends")
        try:
            if 'timestamp_iso' in df_history.columns:
                df_history['date'] = pd.to_datetime(
                    df_history['timestamp_iso'], errors="coerce"
                ).dt.date
            elif 'timestamp' in df_history.columns:
                # fallback if some old rows only have human-readable timestamp
                df_history['date'] = pd.to_datetime(
                    df_history['timestamp'], errors="coerce"
                ).dt.date
            else:
                st.warning("No valid timestamp column found for trend analysis.")
                df_history['date'] = pd.NaT
    
            # Consumption trends
            fig_line = go.Figure()
            colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FFB366', '#B366FF']
    
            for i, occupant in enumerate(occupants):
                col_name = f'occupant_{i}_consumed'
                if col_name in df_history.columns:
                    fig_line.add_trace(go.Scatter(
                        x=df_history['date'],
                        y=df_history[col_name],
                        mode='lines+markers',
                        name=occupant['name'],
                        line=dict(color=colors[i % len(colors)])
                    ))
    
            if 'water_consumed' in df_history.columns:
                fig_line.add_trace(go.Scatter(
                    x=df_history['date'],
                    y=df_history['water_consumed'],
                    mode='lines+markers',
                    name='Water Pump',
                    line=dict(color='#FFCC99')
                ))
    
            fig_line.update_layout(
                title="Consumption Trends Over Time",
                xaxis_title="Date",
                yaxis_title="Consumption (kWh)",
                hovermode='x'
            )
            st.plotly_chart(fig_line, use_container_width=True)
    
            # Cost trends
            fig_cost = go.Figure()
            for i, occupant in enumerate(occupants):
                col_name = f'occupant_{i}_total'
                if col_name in df_history.columns:
                    fig_cost.add_trace(go.Scatter(
                        x=df_history['date'],
                        y=df_history[col_name],
                        mode='lines+markers',
                        name=f"{occupant['name']} Total",
                        line=dict(color=colors[i % len(colors)])
                    ))
    
            fig_cost.update_layout(
                title="Total Cost Trends Over Time",
                xaxis_title="Date",
                yaxis_title=f"Cost ({st.session_state.settings['currency']})",
                hovermode='x'
            )
            st.plotly_chart(fig_cost, use_container_width=True)
    
        except Exception as e:
            st.error(f"Error rendering trend charts: {e}")

    # -------------------------
    # Summary statistics
    # -------------------------
    st.subheader("📈 Summary Statistics")
    if not df_history.empty:
        cols = st.columns(min(4, len(occupants) + 1))
        for i, occupant in enumerate(occupants):
            if i < 4:
                with cols[i]:
                    consumed_col = f'occupant_{i}_consumed'
                    if consumed_col in df_history.columns:
                        avg_consumption = df_history[consumed_col].mean()
                        st.metric(f"Average - {occupant['name']}", f"{avg_consumption:.1f} kWh")

        # Water pump average
        if len(occupants) < 4:
            with cols[len(occupants)]:
                if 'water_consumed' in df_history.columns:
                    avg_water = df_history['water_consumed'].mean()
                    st.metric("Average - Water Pump", f"{avg_water:.1f} kWh")

# settings page
def settings_page():
    st.header("⚙️ Settings & Data Management")

    st.subheader("📊 Current Status")
    history = load_history()
    df_history = pd.DataFrame(history)
    df_history = _coerce_history_numeric(df_history)

    st.info(f"Total calculations saved: {len(df_history)}")

    if not df_history.empty:
        latest = df_history.iloc[-1].to_dict()
        st.success(f"Last calculation: {latest.get('timestamp', 'No timestamp')}")
        occupants = st.session_state.settings['occupants']
        cols = st.columns(min(4, len(occupants) + 1))
        for i, occupant in enumerate(occupants):
            if i < 4:
                with cols[i]:
                    final_key = f'occupant_{i}_final'
                    if final_key in latest:
                        st.metric(f"{occupant['name']} - Last Final", f"{latest[final_key]:.1f} kWh")
        if len(occupants) < 4:
            with cols[len(occupants)]:
                if 'water_final' in latest:
                    st.metric("Water Pump - Last Final", f"{latest['water_final']:.1f} kWh")

    st.subheader("🗂️ Data Management")
    col1, col2, col3 = st.columns(3)
    with col1:
        if not df_history.empty:
            json_data = export_to_json(df_history)
            st.download_button(
                label="📥 Download History (JSON)",
                data=json_data,
                file_name=f"electricity_history_{datetime.now().strftime('%d%m%Y')}.json",
                mime="application/json",
                key="download_history_json"
            )
        else:
            st.warning("No history to download!")
    
    with col2:
        if not df_history.empty:
            excel_data = export_to_excel(df_history)
            if excel_data:
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name=f"{st.session_state.settings['compound_name']}_history_{datetime.now().strftime('%d%m%Y')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_history_excel"
                )
            else:
                st.error("Excel export failed. Please try again.")

    with col3:
        if st.button("🗑️ Clear All History (session only)", type="secondary", key="clear_session_only"):
            st.session_state.consumption_history = []
            st.success("Session history cleared (Google Sheets remains). Reload to fetch latest.")
        if st.button("⚠️ Reset History (Google Sheets + session)", type="secondary", key="clear_file_and_session"):
            if reset_history():
                st.success("All history cleared from Google Sheets and session.")
                st.rerun()

    st.subheader("ℹ️ About This App")
    compound_name = st.session_state.settings['compound_name']
    st.markdown(f"""
    **{compound_name} Electricity Tracker Insight** helps:
    - Track individual electricity consumption
    - Split water pump costs fairly among occupants
    - Maintain historical records in **Google Sheets** with Excel/JSON export
    - Visualize consumption patterns
    - Calculate transparent billing
    - Customize occupants, colors, and rates

    **Features:**
    - 💾 Data stored in Google Sheets (persists across sessions)
    - 📊 Interactive charts and analytics
    - 📱 Mobile-friendly design
    - 🔄 Quick loading from previous readings
    - 📈 Trend analysis
    - 📊 Excel & JSON export functionality
    - 🎨 Customizable interface

    **Note:** All history is persisted in Google Sheets. You can export your data to Excel or JSON for backup.
    """)

def customization_page():
    settings = st.session_state.settings
    st.header("🎨 Customization & Configuration")

    # Basic Settings
    st.subheader("🏠 Basic Settings")
    col1, col2 = st.columns(2)
    with col1:
        settings['compound_name'] = st.text_input(
            "Compound Name",
            value=settings['compound_name'],
            key="compound_name_input"
        )
        settings['currency'] = st.text_input(
            "Currency Symbol",
            value=settings['currency'],
            key="currency_input"
        )
    with col2:
        settings['rate_per_kwh'] = st.number_input(
            "Default Rate per kWh",
            value=settings['rate_per_kwh'],
            min_value=1,
            step=1,
            key="default_rate_input"
        )

    # Color Customization
    st.subheader("🎨 Color Theme")
    col1, col2, col3 = st.columns(3)
    with col1:
        settings['primary_color'] = st.color_picker(
            "Primary Color",
            value=settings['primary_color'],
            key="primary_color_picker"
        )
    with col2:
        settings['secondary_color'] = st.color_picker(
            "Secondary Color",
            value=settings['secondary_color'],
            key="secondary_color_picker"
        )
    with col3:
        settings['success_color'] = st.color_picker(
            "Success Color",
            value=settings['success_color'],
            key="success_color_picker"
        )

    # Occupants Management
    st.subheader("👥 Occupants Management")
    st.write("**Current Occupants:**")
    for i, occupant in enumerate(settings['occupants']):
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            settings['occupants'][i]['name'] = st.text_input(
                f"Name {i+1}",
                value=occupant['name'],
                key=f"occupant_name_{i}"
            )
        with col2:
            settings['occupants'][i]['icon'] = st.selectbox(
                f"Icon {i+1}",
                options=['👤', '👨', '👩', '🧑', '👦', '👧', '🧔', '👱', '🏠', '⚡'],
                index=['👤', '👨', '👩', '🧑', '👦', '👧', '🧔', '👱', '🏠', '⚡'].index(occupant.get('icon', '👤')),
                key=f"occupant_icon_{i}"
            )
        with col3:
            if len(settings['occupants']) > 1:
                if st.button("🗑️", key=f"delete_occupant_{i}"):
                    settings['occupants'].pop(i)
                    st.rerun()
        with col4:
            if i == len(settings['occupants']) - 1:
                if st.button("➕", key="add_occupant"):
                    settings['occupants'].append({'name': f'Occupant {len(settings["occupants"]) + 1}', 'icon': '👤'})
                    st.rerun()

    # Preset Themes
    st.subheader("🌈 Preset Themes")
    themes = {
        "Default Blue": {
            'primary_color': '#2E86AB',
            'secondary_color': '#1f77b4',
            'success_color': '#28a745'
        },
        "Green Energy": {
            'primary_color': '#28a745',
            'secondary_color': '#20c997',
            'success_color': '#198754'
        },
        "Electric Purple": {
            'primary_color': '#6f42c1',
            'secondary_color': '#6610f2',
            'success_color': '#198754'
        },
        "Solar Orange": {
            'primary_color': '#fd7e14',
            'secondary_color': '#ff8c00',
            'success_color': '#28a745'
        },
        "Night Mode": {
            'primary_color': '#495057',
            'secondary_color': '#6c757d',
            'success_color': '#28a745'
        }
    }

    col1, col2, col3, col4, col5 = st.columns(5)
    for i, (theme_name, theme_colors) in enumerate(themes.items()):
        with [col1, col2, col3, col4, col5][i]:
            if st.button(theme_name, key=f"theme_{i}"):
                settings.update(theme_colors)
                st.success(f"Applied {theme_name} theme!")
                st.rerun()

    # Save Settings
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💾 Save All Customizations", type="primary", key="save_customizations"):
            # settings are stored in session state already, give user feedback
            st.success("✅ All customizations saved successfully!")
            st.info("🔄 Refresh the page to see all changes")

    # Live preview
    st.subheader("👀 Live Preview")
    preview_css = f"""
    <style>
    .preview-card {{
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid {settings['primary_color']};
    }}
    .preview-header {{
        color: {settings['primary_color']};
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }}
    </style>
    """
    st.markdown(preview_css, unsafe_allow_html=True)
    st.markdown(f'<div class="preview-header">{settings["compound_name"]} - Preview</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.write(f"{settings['occupants'][0]['icon']} **{settings['occupants'][0]['name']}**")
        st.write(f"Rate: {settings['currency']}{settings['rate_per_kwh']}/kWh")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.write("💧 **Water Pump**")
        st.write(f"Shared among {len(settings['occupants'])} occupants")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="preview-card">', unsafe_allow_html=True)
        st.write("📊 **Summary**")
        st.write(f"Colors: Primary, Secondary, Success")
        st.markdown('</div>', unsafe_allow_html=True)

    # Import/Export settings
    st.subheader("🔄 Import/Export Settings")
    col1, col2 = st.columns(2)
    with col1:
        settings_json = json.dumps(settings, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Export Settings",
            data=settings_json,
            file_name=f"{settings['compound_name']}_settings_{datetime.now().strftime('%d%m%Y')}.json",
            mime="application/json",
            key="download_settings"
        )
    with col2:
        uploaded_file = st.file_uploader("📤 Import Settings", type=['json'], key="upload_settings")
        if uploaded_file is not None:
            try:
                imported_settings = json.load(uploaded_file)
                if st.button("Apply Imported Settings", key="apply_imported_settings"):
                    st.session_state.settings.update(imported_settings)
                    st.success("✅ Settings imported and applied!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error importing settings: {e}")

# -------------------------
# App entry
# -------------------------
def main():
    settings = st.session_state.settings
    st.markdown(f'<h1 class="main-header">⚡ {settings["compound_name"]} Electricity Tracker</h1>', unsafe_allow_html=True)

    # Sidebar for navigation
    st.sidebar.title("🏠 Navigation")
    page = st.sidebar.selectbox("Choose Page", ["📊 New Calculation", "📈 History & Charts", "⚙️ Settings", "🎨 Customization"])

    if page == "📊 New Calculation":
        calculation_page()
    elif page == "📈 History & Charts":
        history_page()
    elif page == "⚙️ Settings":
        settings_page()
    else:
        customization_page()

if __name__ == "__main__":
    # Initialize Google Sheet (adds headers if sheet is empty)
    init_sheet()
    main()

# Footer
st.markdown('<div class="designer-credit">Designed by **Arthur_Techy**</div>', unsafe_allow_html=True)


