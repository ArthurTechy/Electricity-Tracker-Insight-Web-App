import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import pytz
import json
import io
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# -------------------------
# Config / Constants
# -------------------------
HISTORY_FILENAME = "consumption_history.json"

# Page configuration
st.set_page_config(
    page_title="Owolawi Compound - Electricity Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Initialize consumption history in session state
if 'consumption_history' not in st.session_state:
    st.session_state.consumption_history = []

# -------------------------
# Utilities: persistence
# -------------------------
def load_history_from_file():
    """Load history from JSON file into session state (called on startup & when needed)."""
    try:
        if os.path.exists(HISTORY_FILENAME):
            with open(HISTORY_FILENAME, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    st.session_state.consumption_history = data.copy()
                else:
                    # if file contents are invalid, reset to empty
                    st.session_state.consumption_history = []
        else:
            st.session_state.consumption_history = []
    except Exception as e:
        # If something goes wrong reading, keep session state list and show debug
        st.error(f"Error loading history file: {e}")
        st.session_state.consumption_history = []

def save_history_to_file():
    """Write session state's consumption_history to disk (JSON)."""
    try:
        with open(HISTORY_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(st.session_state.consumption_history, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving history file: {e}")
        return False

# Wrap load/save functions used by the app (keeps names similar to original)
def load_history():
    """Return consumption history (session-state-backed, ensures file load)."""
    # Make sure session history is loaded once from file
    if 'history_loaded_from_file' not in st.session_state:
        load_history_from_file()
        st.session_state.history_loaded_from_file = True
    return st.session_state.consumption_history

def save_history(history_data):
    """Save history both to session state and to JSON file."""
    try:
        st.session_state.consumption_history = history_data.copy()
        saved = save_history_to_file()
        return saved
    except Exception as e:
        st.error(f"Error saving data: {e}")
        return False

def reset_history():
    """Clear session and file history."""
    st.session_state.consumption_history = []
    # remove file if exists
    try:
        if os.path.exists(HISTORY_FILENAME):
            os.remove(HISTORY_FILENAME)
        return True
    except Exception as e:
        st.error(f"Error clearing history file: {e}")
        return False

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
    """Get the latest readings from history (final readings)."""
    history = load_history()
    if history:
        latest = history[-1]
        result = {}
        for i, occupant in enumerate(st.session_state.settings['occupants']):
            key = f"occupant_{i}_final"
            # In stored data we use occupant_{i}_final as key (create_calculation_data)
            result[key] = latest.get(f"{key}", 0)
        result['water_final'] = latest.get('water_final', 0)
        return result
    # Defaults
    result = {}
    for i, occupant in enumerate(st.session_state.settings['occupants']):
        result[f"occupant_{i}_final"] = 0
    result['water_final'] = 0
    return result

# -------------------------
# Export helpers: Excel (existing) and combined image (new)
# -------------------------
def export_to_excel(history_data):
    """Export history data to Excel format (bytes)."""
    if not history_data:
        return None

    df = pd.DataFrame(history_data)
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Consumption History', index=False)

            # Add summary sheet
            if len(df) > 0:
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
        footer_text = f"Designed by {settings.get('compound_name', '')} - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
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
# Calculation helpers
# -------------------------
def create_calculation_data(timestamp, initial_readings, final_readings,
                            consumptions, costs, total_costs, water_consumed,
                            water_cost, rate, total_amount):
    """Create calculation data dictionary (same structure as original code)."""
    data = {
        'timestamp': timestamp,
        'water_consumed': water_consumed,
        'water_cost': water_cost,
        'rate_per_kwh': rate,
        'total_amount': total_amount
    }

    occupants = st.session_state.settings['occupants']
    for i, occupant in enumerate(occupants):
        key = f"occupant_{i}"
        data.update({
            f'{key}_initial': initial_readings[key],
            f'{key}_final': final_readings[key],
            f'{key}_consumed': consumptions[key],
            f'{key}_total': total_costs[key]
        })

    data['water_initial'] = initial_readings['water']
    data['water_final'] = final_readings['water']

    return data

def save_calculation(timestamp, initial_readings, final_readings,
                     consumptions, costs, total_costs, water_consumed,
                     water_cost, rate, total_amount):
    """Save calculation to history (session + file)."""
    calculation_data = create_calculation_data(
        timestamp, initial_readings, final_readings,
        consumptions, costs, total_costs, water_consumed,
        water_cost, rate, total_amount
    )

    history = load_history()
    history.append(calculation_data)

    if save_history(history):
        st.success("✅ Calculation saved successfully!")
    else:
        st.error("❌ Failed to save calculation")

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
                # store into session so defaults pick them up (keeps original naming)
                st.session_state[f"{key}_session"] = float(value) if value is not None else 0.0
            st.success("✅ Last readings loaded as initial values!")

    # Input section
    st.subheader("📝 Enter Initial Readings (kWh)")

    # Dynamic columns based on occupants
    occupants = settings['occupants']
    num_cols = len(occupants) + 1  # +1 for water pump
    cols = st.columns(num_cols)

    initial_readings = {}
    final_readings = {}

    # Occupant inputs
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

    # Water pump input
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

    # Final readings for occupants
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

    # Water pump final reading
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

    # Rate per kWh
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

    # Calculate consumption and costs
    consumptions = {}
    costs = {}

    for i, occupant in enumerate(occupants):
        key = f"occupant_{i}"
        consumption = final_readings[key] - initial_readings[key]
        consumptions[key] = consumption
        costs[key] = consumption * rate

    # Water calculations
    water_consumed = final_readings['water'] - initial_readings['water']
    water_cost = water_consumed * rate
    water_cost_per_person = water_cost / len(occupants) if len(occupants) > 0 else 0

    # Total costs per person
    total_costs = {}
    for i, occupant in enumerate(occupants):
        key = f"occupant_{i}"
        total_costs[key] = costs[key] + water_cost_per_person

    total_amount = sum(total_costs.values())

    # Display calculation breakdown
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

        st.markdown(consumption_text + cost_text + water_text + final_text)

    # Summary table & timestamp
    st.subheader("📊 Final Summary")
    wat_tz = pytz.timezone('Africa/Lagos')
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
        history = load_history()
        if history:
            excel_data = export_to_excel(history)
            if excel_data:
                st.download_button(
                    label="📊 Export to Excel",
                    data=excel_data,
                    file_name=f"{settings['compound_name']}_electricity_history_{datetime.now().strftime('%d%m%Y')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )

    with col3:
        # Allow download of a combined JPEG (charts + table + footer)
        if st.button("📷 Download Report Image (JPEG)", key="download_image_btn"):
            # Build a display-friendly df for the table (avoid very long currency formatting)
            df_for_image = df_summary.copy()
            # Optionally convert costs to plain strings already done in summary creation
            image_bytes = create_combined_image(df_for_image, fig_pie, fig_bar, st.session_state.settings)
            if image_bytes:
                st.download_button(
                    label="⬇️ Download Report JPEG",
                    data=image_bytes,
                    file_name=f"{st.session_state.settings['compound_name']}_report_{datetime.now().strftime('%d%m%Y_%H%M')}.jpg",
                    mime="image/jpeg",
                    key="download_jpg"
                )

def history_page():
    st.header("📈 Consumption History & Analytics")

    history = load_history()

    # Debug information (helpful for troubleshooting)
    st.write(f"Debug: Session state keys: {list(st.session_state.keys())}")
    st.write(f"Debug: History length: {len(history)}")
    if history:
        st.write(f"Debug: Last calculation timestamp: {history[-1].get('timestamp', 'No timestamp')}")

    if not history:
        st.info("No calculation history found. Please make some calculations first!")
        # still show reset option: sometimes file exists but list empty
        if st.button("🔄 Reset History (clear file & session)", key="reset_history_from_empty"):
            if reset_history():
                st.success("History reset.")
                st.experimental_rerun()
        return

    # Export & reset options
    col1, col2, col3 = st.columns(3)
    with col1:
        json_data = json.dumps(history, indent=2, ensure_ascii=False)
        st.download_button(
            label="📥 Download History (JSON)",
            data=json_data,
            file_name=f"electricity_history_{datetime.now().strftime('%d%m%Y')}.json",
            mime="application/json",
            key="download_history_json"
        )
    with col2:
        excel_data = export_to_excel(history)
        if excel_data:
            st.download_button(
                label="📊 Export to Excel",
                data=excel_data,
                file_name=f"{st.session_state.settings['compound_name']}_electricity_history_{datetime.now().strftime('%d%m%Y')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_history_excel"
            )
    with col3:
        if st.button("🗑️ Reset History (clear file & session)", key="reset_history_confirm"):
            if reset_history():
                st.success("All history cleared!")
                st.experimental_rerun()

    # Display history table (recent)
    st.subheader("📋 Recent Calculations")
    df_history = pd.DataFrame(history)

    display_columns = ['timestamp']
    occupants = st.session_state.settings['occupants']

    for i, occupant in enumerate(occupants):
        display_columns.append(f'occupant_{i}_total')

    display_columns.extend(['water_cost', 'total_amount'])

    if all(col in df_history.columns for col in display_columns):
        display_df = df_history[display_columns].tail(10)
        rename_dict = {'timestamp': 'Date/Time', 'water_cost': 'Water Cost', 'total_amount': 'Total'}
        for i, occupant in enumerate(occupants):
            rename_dict[f'occupant_{i}_total'] = f"{occupant['name']} ({st.session_state.settings['currency']})"
        display_df = display_df.rename(columns=rename_dict)
        st.dataframe(display_df.iloc[::-1], use_container_width=True)
    else:
        st.warning("History exists but expected columns are missing in some records. This can happen if older records have a different schema.")

    # Analytics charts if >1 record
    if len(history) > 1:
        st.subheader("📊 Consumption Trends")
        try:
            df_history['date'] = pd.to_datetime(df_history['timestamp']).dt.date

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
            fig_line.update_layout(title="Consumption Trends Over Time", xaxis_title="Date", yaxis_title="Consumption (kWh)", hovermode='x')
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
            fig_cost.update_layout(title="Total Cost Trends Over Time", xaxis_title="Date", yaxis_title=f"Cost ({st.session_state.settings['currency']})", hovermode='x')
            st.plotly_chart(fig_cost, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering trend charts: {e}")

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    if len(history) >= 1:
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

def settings_page():
    st.header("⚙️ Settings & Data Management")

    st.subheader("📊 Current Status")
    history = load_history()
    st.info(f"Total calculations saved: {len(history)}")

    if history:
        latest = history[-1]
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
        if st.button("📥 Download History (JSON)", type="secondary", key="settings_download_json"):
            if history:
                json_data = json.dumps(history, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download JSON File",
                    data=json_data,
                    file_name=f"electricity_history_{datetime.now().strftime('%d%m%Y')}.json",
                    mime="application/json",
                    key="download_json_from_settings"
                )
            else:
                st.warning("No history to download!")

    with col2:
        if st.button("📊 Export to Excel", type="secondary", key="settings_export_excel"):
            if history:
                excel_data = export_to_excel(history)
                if excel_data:
                    st.download_button(
                        label="Download Excel File",
                        data=excel_data,
                        file_name=f"{st.session_state.settings['compound_name']}_history_{datetime.now().strftime('%d%m%Y')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_from_settings"
                    )
            else:
                st.warning("No history to export!")

    with col3:
        if st.button("🗑️ Clear All History (session only)", type="secondary", key="clear_session_only"):
            # Clear only session (file remains) - useful for testing
            st.session_state.consumption_history = []
            st.success("Session history cleared (file still present). Reload to re-load file.")
        if st.button("⚠️ Reset History (clear file + session)", type="secondary", key="clear_file_and_session"):
            if reset_history():
                st.success("All history cleared (file removed).")
                st.experimental_rerun()

    st.subheader("ℹ️ About This App")
    compound_name = st.session_state.settings['compound_name']
    st.markdown(f"""
    **{compound_name} Electricity Tracker Insight** helps:
    - Track individual electricity consumption
    - Split water pump costs fairly among occupants
    - Maintain historical records with Excel export
    - Visualize consumption patterns
    - Calculate transparent billing
    - Customize occupants, colors, and rates

    **Features:**
    - 💾 Session-based data storage (persists during your browser session) + optional JSON file persistence
    - 📊 Interactive charts and analytics
    - 📱 Mobile-friendly design
    - 🔄 Quick loading from previous readings
    - 📈 Trend analysis
    - 📊 Excel export functionality
    - 🎨 Customizable interface

    **Note:** History is saved to a local JSON file (`{HISTORY_FILENAME}`) so it can persist across Streamlit session restarts where the environment allows filesystem persistence. Export your data if you need to move it elsewhere.
    """)

def customization_page():
    st.header("🎨 Customization & Configuration")
    settings = st.session_state.settings

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
                    st.experimental_rerun()
        with col4:
            if i == len(settings['occupants']) - 1:
                if st.button("➕", key="add_occupant"):
                    settings['occupants'].append({'name': f'Occupant {len(settings["occupants"]) + 1}', 'icon': '👤'})
                    st.experimental_rerun()

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
                st.experimental_rerun()

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
                    st.experimental_rerun()
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
    # Ensure history is loaded once on startup
    load_history_from_file()
    main()

# Footer
st.markdown('<div class="designer-credit">Designed by **Arthur_Techy**</div>', unsafe_allow_html=True)

