import io
from datetime import date
import pandas as pd
import streamlit as st

# Custom CSS for visual enhancements
st.markdown(
    """
    <style>
    /* General styling */
    .stApp {
        background-color: #041A00;
        font-family: 'Arial', sans-serif;
    }
    h1, h2 {
        color: #1e3a8a;
        font-weight: 600;
    }
    .stMarkdown p, .stCaption {
        color: #4b5e7b;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }
    .stDataFrame, .stDataEditor {
        border: 1px solid #d1d5db;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stDataFrame table, .stDataEditor table {
        width: 100%;
        border-collapse: collapse;
    }
    .stDataFrame th, .stDataEditor th {
        background-color: #eff6ff;
        color: #1e3a8a;
        padding: 10px;
    }
    .stDataFrame td, .stDataEditor td {
        padding: 10px;
        border-bottom: 1px solid #e5e7eb;
    }
    .stDataFrame tr:nth-child(even), .stDataEditor tr:nth-child(even) {
        background-color: #f9fafb;
    }
    .stDataFrame tr:hover, .stDataEditor tr:hover {
        background-color: #e5e7eb;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 1px solid #d1d5db;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    .stSidebar h1, .stSidebar h2 {
        color: #00802B;
    }
    .stSidebar .stMarkdown p {
        color: #4b5e7b;
    }
    /* Metrics styling */
    .stMetric {
        background-color: #111827;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stMetric label {
        color: #991900;
        font-weight: 600;
    }
    .stMetric .metric-value {
        color: #111827;
        font-size: 1.2em;
    }
    /* Alerts styling */
    .stAlert {
        border-radius: 8px;
        padding: 15px;
    }
    .stAlert[role="alert"] {
        background-color: #fef2f2;
        color: #991b1b;
    }
    .stAlert[role="info"] {
        background-color: #eff6ff;
        color: #1e3a8a;
    }
    .stAlert[role="success"] {
        background-color: #f0fdf4;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Weekly Payroll Calculator", page_icon="💵", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def coerce_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def update_reports_totals(df):
    df = df.copy()
    df["Total"] = df[WEEKDAYS].sum(axis=1)
    df["Payment"] = df["Total"] / 2
    return df

def compute_earnings(row, reports_payment, employees_df):
    total = sum(coerce_float(row.get(day, 0)) for day in WEEKDAYS)
    name = row["Name"]
    matching_employee = employees_df[employees_df["Name"] == name]
    multiplier = coerce_float(matching_employee["Multiplier"].iloc[0]) if not matching_employee.empty else 0.5
    payment = total * multiplier
    cash = payment - coerce_float(reports_payment)
    return pd.Series({"Total": round(total, 2), "Payment": round(payment, 2), "Cash": round(cash, 2)})

def build_base_earnings(employees):
    base = employees[["First Name", "Last Name"]].copy()
    for d in WEEKDAYS:
        base[d] = 0.0
    base["First Name"] = base["First Name"].fillna("").astype(str)
    base["Last Name"] = base["Last Name"].fillna("").astype(str)
    base["Name"] = (base["First Name"].str.strip() + " " + base["Last Name"].str.strip()).str.strip()
    return base

def sync_earnings_with_employees(current_earnings, employees):
    new_base = build_base_earnings(employees)
    if current_earnings is None or current_earnings.empty:
        return new_base
    
    existing_data = {}
    for _, row in current_earnings.iterrows():
        if 'Name' in row:
            key = str(row['Name']).strip()
        else:
            first = str(row.get("First Name", "")).strip()
            last = str(row.get("Last Name", "")).strip()
            key = f"{first} {last}".strip()
        
        existing_data[key] = {d: coerce_float(row.get(d, 0.0)) for d in WEEKDAYS}
    
    for idx, row in new_base.iterrows():
        key = row["Name"]
        if key in existing_data:
            for d in WEEKDAYS:
                new_base.at[idx, d] = existing_data[key][d]
    
    return new_base

# -----------------------------
# Sidebar: Settings
# -----------------------------
with st.sidebar:
    st.title("⚙️ Payroll Settings")
    st.markdown('<p style="color: #4b5e7b;">Adjust global settings for the payroll run.</p>', unsafe_allow_html=True)
    if "default_week_ending" not in st.session_state:
        st.session_state.default_week_ending = date.today()
        st.session_state.week_ending_edited = False
    
    payroll_week_ending = st.date_input("📅 Week ending", value=date.today(), help="Select the payroll week ending date.")
    st.session_state.week_ending_edited = payroll_week_ending != st.session_state.default_week_ending

    st.markdown("---")
    st.subheader("🧮 Taxes (Optional)")
    fed_tax_pct = st.number_input("Federal tax % (flat)", value=0.0, min_value=0.0, step=0.1, help="Enter federal tax percentage.")
    state_tax_pct = st.number_input("State tax % (flat)", value=0.0, min_value=0.0, step=0.1, help="Enter state tax percentage.")
    st.markdown('<small style="color: #4b5e7b;">These are placeholders for quick estimates.</small>', unsafe_allow_html=True)

# -----------------------------
# Employees
# -----------------------------
st.title("💵 Weekly Payroll Calculator")
st.markdown("---")
st.header("1) 💼 Employees")
col_left, col_right = st.columns([3, 1], gap="medium")

with col_left:
    emp_upload = st.file_uploader("📤 Upload Employees CSV", type=["csv"], key="emp_csv", help="Upload CSV with First Name, Last Name, Multiplier.")
    if "employees" not in st.session_state:
        st.session_state.employees = pd.DataFrame([
            {"First Name": "Austria", "Last Name": "Brito", "Multiplier": 0.5},
            {"First Name": "Anatalia", "Last Name": "Dolores", "Multiplier": 0.6},
            {"First Name": "Leydys", "Last Name": "", "Multiplier": 0.5},
            {"First Name": "Yoeny", "Last Name": "Peraza", "Multiplier": 0.5},
            {"First Name": "Roseline", "Last Name": "", "Multiplier": 0.5},
            {"First Name": "Luan", "Last Name": "Nguyen", "Multiplier": 0.5},
            {"First Name": "Thuong", "Last Name": "Ho", "Multiplier": 0.6},
            {"First Name": "Huynh", "Last Name": "Hoang", "Multiplier": 0.6},
            {"First Name": "Tan", "Last Name": "Ta", "Multiplier": 0.6},
            {"First Name": "Phuong", "Last Name": "Le", "Multiplier": 0.6},
            {"First Name": "Long", "Last Name": "Nguyen", "Multiplier": 0.6},
            {"First Name": "Tuyet Nhung", "Last Name": "Nguyen", "Multiplier": 0.6},
            {"First Name": "Lee", "Last Name": "", "Multiplier": 0.6},
            {"First Name": "Zhixin", "Last Name": "Meng", "Multiplier": 0.6}
        ])
        st.session_state.employees["Name"] = (
            st.session_state.employees["First Name"].str.strip() + " " +
            st.session_state.employees["Last Name"].str.strip()
        ).str.strip()

    if emp_upload:
        try:
            df_emp = pd.read_csv(emp_upload)
            for col in ["First Name", "Last Name"]:
                if col not in df_emp.columns:
                    df_emp[col] = ""
            if "Multiplier" not in df_emp.columns:
                df_emp["Multiplier"] = 0.5
            df_emp["Name"] = (df_emp["First Name"].str.strip() + " " + df_emp["Last Name"].str.strip()).str.strip()
            st.session_state.employees = df_emp[["First Name", "Last Name", "Multiplier", "Name"]].copy()
            st.success("✅ Employees loaded successfully.")
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")

    st.session_state.employees["Multiplier"] = pd.to_numeric(st.session_state.employees["Multiplier"], errors='coerce').fillna(0.5)

    num_rows = len(st.session_state.employees)
    table_height = (num_rows * 36) + 40 + 20

    edited_employees = st.data_editor(
        st.session_state.employees[["First Name", "Last Name", "Multiplier"]],
        num_rows="dynamic",
        use_container_width=True,
        key="emp_editor",
        column_config={
            "First Name": st.column_config.TextColumn("First Name", width="medium"),
            "Last Name": st.column_config.TextColumn("Last Name", width="medium"),
            "Multiplier": st.column_config.NumberColumn("Multiplier", min_value=0.0, max_value=10.0, step=0.05, width="small"),
        },
        height=table_height
    )
    
    if not edited_employees.equals(st.session_state.employees[["First Name", "Last Name", "Multiplier"]]):
        edited_employees["Name"] = (edited_employees["First Name"].str.strip() + " " + edited_employees["Last Name"].str.strip()).str.strip()
        st.session_state.employees = edited_employees[["First Name", "Last Name", "Multiplier", "Name"]].copy()
        st.session_state.employees_need_sync = True
        st.rerun()

with col_right:
    st.info("**CSV Requirements**: First Name, Last Name, Multiplier. Download template below.", icon="ℹ️")
    sample_emp = pd.DataFrame([{"First Name": "", "Last Name": "", "Multiplier": 0.5}])
    st.download_button("📥 Download Employees Template", sample_emp.to_csv(index=False).encode("utf-8"),
                       file_name="employees_template.csv", mime="text/csv")

# -----------------------------
# Daily Earnings
# -----------------------------
st.markdown("---")
st.header("2) 🕒 Daily Earnings (Mon–Sun)")
sheet_upload = st.file_uploader("📤 Upload Earnings CSV", type=["csv"], key="sheet_csv", help="Upload CSV with First Name, Last Name, Mon–Sun earnings.")

need_sync = st.session_state.get("employees_need_sync", False)

if "edited_earnings" not in st.session_state:
    st.session_state.edited_earnings = build_base_earnings(st.session_state.employees)
    st.session_state.employees_need_sync = False
elif need_sync:
    st.session_state.edited_earnings = sync_earnings_with_employees(
        st.session_state.edited_earnings, st.session_state.employees
    )
    st.session_state.employees_need_sync = False

if sheet_upload:
    try:
        df_sheet = pd.read_csv(sheet_upload)
        for d in WEEKDAYS:
            if d not in df_sheet.columns:
                df_sheet[d] = 0.0
        for d in WEEKDAYS:
            df_sheet[d] = pd.to_numeric(df_sheet[d], errors='coerce').fillna(0.0)
        
        new_earnings = build_base_earnings(st.session_state.employees)
        
        for idx, emp_row in new_earnings.iterrows():
            matching_rows = df_sheet[
                (df_sheet["First Name"].astype(str).str.strip() == emp_row["First Name"]) & 
                (df_sheet["Last Name"].astype(str).str.strip() == emp_row["Last Name"])
            ]
            if not matching_rows.empty:
                match_row = matching_rows.iloc[0]
                for d in WEEKDAYS:
                    new_earnings.at[idx, d] = coerce_float(match_row[d])
        
        st.session_state.edited_earnings = new_earnings
        st.success("✅ Earnings CSV loaded and synced.")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Could not read earnings CSV: {e}")

display_earnings = st.session_state.edited_earnings[["Name"] + WEEKDAYS].copy()

for d in WEEKDAYS:
    display_earnings[d] = pd.to_numeric(display_earnings[d], errors='coerce').fillna(0.0)

num_rows = len(display_earnings)
table_height = (num_rows * 36) + 40 + 20

edited_earnings_display = st.data_editor(
    display_earnings,
    num_rows="dynamic",
    use_container_width=True,
    key="timesheet_editor",
    column_config={
        "Name": st.column_config.TextColumn("Name", width="medium"),
        **{d: st.column_config.NumberColumn(d, min_value=0.0, step=0.01, format="$%.2f", width="small") for d in WEEKDAYS},
    },
    height=table_height
)

if not edited_earnings_display.equals(display_earnings):
    for col in ["Name"] + WEEKDAYS:
        st.session_state.edited_earnings[col] = edited_earnings_display[col]
    st.rerun()

# -----------------------------
# Reports Table
# -----------------------------
st.markdown("---")
st.header("3) 📊 Reports (Separate - Modified Daily Earnings)")

if "reports_raw" not in st.session_state:
    reports_base = st.session_state.edited_earnings[["Name"] + WEEKDAYS].copy()
    st.session_state.reports_raw = reports_base
else:
    current_reports = st.session_state.reports_raw.copy()
    reports_base = st.session_state.edited_earnings[["Name"] + WEEKDAYS].copy()
    
    for idx, row in reports_base.iterrows():
        matching_old = current_reports[current_reports["Name"] == row["Name"]]
        if not matching_old.empty:
            old_row = matching_old.iloc[0]
            for d in WEEKDAYS:
                if pd.notna(old_row[d]):
                    reports_base.at[idx, d] = coerce_float(old_row[d])
    
    st.session_state.reports_raw = reports_base

reports_display = st.session_state.reports_raw.copy()
reports_display = update_reports_totals(reports_display)

num_rows = len(reports_display)
table_height = (num_rows * 36) + 40 + 20

edited_reports = st.data_editor(
    reports_display,
    num_rows="dynamic",
    use_container_width=True,
    key="reports_editor",
    column_config={
        "Name": st.column_config.TextColumn("Name", width="medium"),
        **{d: st.column_config.NumberColumn(d, min_value=0.0, step=0.01, format="$%.2f", width="small") for d in WEEKDAYS},
        "Total": st.column_config.NumberColumn("Total", disabled=True, format="$%.2f", width="small"),
        "Payment": st.column_config.NumberColumn("Payment", disabled=True, format="$%.2f", width="small"),
    },
    height=table_height
)

if not edited_reports[["Name"] + WEEKDAYS].equals(st.session_state.reports_raw[["Name"] + WEEKDAYS]):
    st.session_state.reports_raw = edited_reports[["Name"] + WEEKDAYS].copy()
    st.rerun()

st.session_state.reports = update_reports_totals(st.session_state.reports_raw)

# -----------------------------
# Sales Summary
# -----------------------------
st.markdown("---")
st.header("4) 💰 Sales Summary")

if "sales_raw" not in st.session_state:
    sales_types = ["Total Sales", "Cash", "Credit Card", "Gift Certificate Redeem"]
    sales_base = pd.DataFrame([{"Type": t, **{d: 0.0 for d in WEEKDAYS}} for t in sales_types])
    st.session_state.sales_raw = sales_base
else:
    sales_base = st.session_state.sales_raw.copy()
    for d in WEEKDAYS:
        if d not in sales_base.columns:
            sales_base[d] = 0.0
    if "Type" not in sales_base.columns:
        sales_base.insert(0, "Type", ["Total Sales", "Cash", "Credit Card", "Gift Certificate Redeem"][:len(sales_base)])
    st.session_state.sales_raw = sales_base[["Type"] + WEEKDAYS]

editable_rows = ["Cash", "Credit Card", "Gift Certificate Redeem"]
editable_mask = st.session_state.sales_raw["Type"].isin(editable_rows)
sales_editable = st.session_state.sales_raw.loc[editable_mask].reset_index(drop=True)

num_rows_editable = len(sales_editable)
editable_height = (num_rows_editable * 36) + 40 + 20

with st.expander("✏️ Edit Cash / Credit Card / Gift Certificate Redeem", expanded=False):
    edited_sales_subset = st.data_editor(
        sales_editable,
        num_rows="fixed",
        use_container_width=True,
        key="sales_editor_subset",
        column_config={
            "Type": st.column_config.TextColumn("Type", disabled=True, width="medium"),
            **{d: st.column_config.NumberColumn(d, min_value=0.0, step=0.01, format="$%.2f", width="small") for d in WEEKDAYS},
        },
        height=editable_height
    )
    if not edited_sales_subset.equals(sales_editable):
        for i, row in edited_sales_subset.iterrows():
            row_type = row["Type"]
            mask = st.session_state.sales_raw["Type"] == row_type
            for d in WEEKDAYS:
                st.session_state.sales_raw.loc[mask, d] = coerce_float(row[d])
        st.rerun()

sales_display = st.session_state.sales_raw.copy()
sales_display["Total"] = sales_display[WEEKDAYS].sum(axis=1)

num_rows_sales = len(sales_display)
sales_height = (num_rows_sales * 36) + 40 + 20

st.dataframe(
    sales_display,
    use_container_width=True,
    height=sales_height,
    column_config={
        "Type": st.column_config.TextColumn("Type", width="medium"),
        **{d: st.column_config.NumberColumn(d, format="$%.2f", width="small") for d in WEEKDAYS},
        "Total": st.column_config.NumberColumn("Total", format="$%.2f", width="small"),
    }
)

st.session_state.sales = sales_display.copy()

# -----------------------------
# Sales validation panel
# -----------------------------
_sales = st.session_state.get("sales", None)
_validation_tol = 0.01

if _sales is None or _sales.empty:
    st.info("Sales summary is empty — validation not run.", icon="ℹ️")
else:
    try:
        sales_indexed = _sales.set_index("Type")
    except Exception:
        sales_indexed = None

    required_rows = ["Total Sales", "Cash", "Credit Card", "Gift Certificate Redeem"]
    missing_rows = []
    if sales_indexed is None:
        missing_rows = required_rows
    else:
        for r in required_rows:
            if r not in sales_indexed.index:
                missing_rows.append(r)

    if missing_rows:
        st.warning(f"❌ Sales validation could not run — missing rows: {missing_rows}. Ensure Sales Summary has rows: {required_rows}", icon="⚠️")
    else:
        mismatches = []
        check_columns = WEEKDAYS + ["Total"]
        for col in check_columns:
            total_sales_val = coerce_float(sales_indexed.at["Total Sales", col]) if col in sales_indexed.columns else 0.0
            parts_sum = (
                coerce_float(sales_indexed.at["Cash", col]) +
                coerce_float(sales_indexed.at["Credit Card", col]) +
                coerce_float(sales_indexed.at["Gift Certificate Redeem", col])
            )
            diff = total_sales_val - parts_sum
            if abs(diff) > _validation_tol:
                mismatches.append((col, total_sales_val, parts_sum, diff))

        if mismatches:
            lines = ["❌ Sales validation failed — differences detected (Total Sales − (Cash+Credit+Gift Certificate Redeem)):"]
            for col, total_val, parts_val, diff in mismatches:
                lines.append(f"- {col}: Total Sales = ${total_val:,.2f}; Parts sum = ${parts_val:,.2f}; Difference = ${diff:,.2f}")
            st.warning("\n".join(lines), icon="⚠️")
        else:
            st.success("Sales validation OK — Total Sales equals Cash + Credit Card + Gift Certificate Redeem.", icon="✅")

# -----------------------------
# Results Table
# -----------------------------
st.markdown("---")
st.header("5) 📈 Results")

earnings_data = st.session_state.edited_earnings[["Name"] + WEEKDAYS].copy()

reports_payments = {}
if "reports" in st.session_state and not st.session_state.reports.empty:
    for _, row in st.session_state.reports.iterrows():
        reports_payments[row["Name"]] = coerce_float(row.get("Payment", 0.0))

results_list = []
for _, row in earnings_data.iterrows():
    total = sum(coerce_float(row.get(day, 0)) for day in WEEKDAYS)
    name = row["Name"]
    matching_employee = st.session_state.employees[st.session_state.employees["Name"] == name]
    multiplier = coerce_float(matching_employee["Multiplier"].iloc[0]) if not matching_employee.empty else 0.5
    payment = total * multiplier
    reports_payment = reports_payments.get(row["Name"], 0.0)
    cash = payment - reports_payment
    check = payment - cash
    
    results_list.append({
        "Total": round(total, 2), 
        "Payment": round(payment, 2), 
        "Cash": round(cash, 2),
        "Check": round(check, 2)
    })

results = pd.DataFrame(results_list)
final = pd.concat([earnings_data[["Name"] + WEEKDAYS], results], axis=1)

totals = {"Name": "TOTAL"}
for col in WEEKDAYS + ["Total", "Payment", "Cash", "Check"]:
    totals[col] = round(pd.to_numeric(final[col], errors="coerce").sum(), 2)

try:
    if "sales_raw" in st.session_state and "Type" in st.session_state.sales_raw.columns:
        mask = st.session_state.sales_raw["Type"].astype(str).str.strip() == "Total Sales"
        if mask.any():
            current_row = st.session_state.sales_raw.loc[mask, WEEKDAYS].iloc[0].astype(float).tolist()
            new_values = [float(totals[d]) for d in WEEKDAYS]
            changed = any(abs(coerce_float(curr) - new) > 0.005 for curr, new in zip(current_row, new_values))
            if changed:
                for i, d in enumerate(WEEKDAYS):
                    st.session_state.sales_raw.loc[mask, d] = new_values[i]
                st.rerun()
except Exception:
    pass

display_with_total = pd.concat([final, pd.DataFrame([totals])], ignore_index=True)
display_formatted = display_with_total.copy()

for col in WEEKDAYS + ["Total", "Payment", "Cash", "Check"]:
    display_formatted[col] = display_formatted[col].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) and x != 0 else "$0.00"
    )

num_rows = len(display_with_total)
table_height = (num_rows * 36) + 40 + 20

st.dataframe(
    display_formatted,
    use_container_width=True,
    height=table_height,
    column_config={
        "Name": st.column_config.TextColumn("Name", width="medium"),
        **{d: st.column_config.TextColumn(d, width="small") for d in WEEKDAYS},
        "Total": st.column_config.TextColumn("Total", width="small"),
        "Payment": st.column_config.TextColumn("Payment", width="small"),
        "Cash": st.column_config.TextColumn("Cash", width="small"),
        "Check": st.column_config.TextColumn("Check", width="small"),
    }
)

# -----------------------------
# Summary metrics
# -----------------------------
st.markdown("---")
st.header("6) 📊 Summary Metrics")
total_employees = len(final)
total_week_earnings = totals["Total"]
total_payment = totals["Payment"]
total_cash = totals["Cash"]
total_check = totals["Check"]

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Employees", f"{total_employees}")
    col2.metric("💸 Total Sales", f"${total_week_earnings:,.2f}")
    col3.metric("💵 Total Cash", f"${total_cash:,.2f}")
    col4.metric("📝 Total Check", f"${total_check:,.2f}")

    st.markdown("---")
    col_check = st.columns([1, 2, 1, 1])[1]
    col_check.metric("📈 Week Earnings", f"${total_week_earnings - total_payment:,.2f}")

# -----------------------------
# Export
# -----------------------------
st.markdown("---")
st.header("7) 📥 Export")
pay_week_label = payroll_week_ending.strftime("%Y-%m-%d")

if not st.session_state.get("week_ending_edited", False):
    st.warning("Please select a 'Week ending' date in the Settings sidebar before exporting.", icon="⚠️")
else:
    col_csv, col_excel = st.columns(2)
    with col_csv:
        csv_buffer = io.StringIO()
        display_formatted.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download CSV", data=csv_buffer.getvalue().encode("utf-8"),
                           file_name=f"payroll_{pay_week_label}.csv", mime="text/csv")
    
    with col_excel:
        excel_buffer = io.BytesIO()
        _excel_ok = False

        try:
            with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                combined_df = pd.concat([
                    display_formatted,
                    pd.DataFrame([[""] * len(display_formatted.columns)], columns=display_formatted.columns),
                    pd.DataFrame([["Sales Summary"]], columns=["Name"]),
                    sales_display.rename(columns={"Type": "Name"})
                ], ignore_index=True)
                combined_df.to_excel(writer, index=False, sheet_name="Payroll_and_Sales")

                credit_card_row = sales_display[sales_display["Type"] == "Credit Card"].copy()
                cash_row = sales_display[sales_display["Type"] == "Cash"].copy()
                for col in WEEKDAYS + ["Total"]:
                    cash_row[col] = 0.0
                reports_combined = pd.concat([
                    reports_display,
                    pd.DataFrame([[""] * len(reports_display.columns)], columns=reports_display.columns),
                    pd.DataFrame([["Sales Summary"]], columns=["Name"]),
                    credit_card_row.rename(columns={"Type": "Name"}),
                    cash_row.rename(columns={"Type": "Name"})
                ], ignore_index=True)
                reports_combined.to_excel(writer, index=False, sheet_name="Reports")

            _excel_ok = True
        except ModuleNotFoundError:
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    combined_df = pd.concat([
                        display_formatted,
                        pd.DataFrame([[""] * len(display_formatted.columns)], columns=display_formatted.columns),
                        pd.DataFrame([["Sales Summary"]], columns=["Name"]),
                        sales_display.rename(columns={"Type": "Name"})
                    ], ignore_index=True)
                    combined_df.to_excel(writer, index=False, sheet_name="Payroll_and_Sales")

                    credit_card_row = sales_display[sales_display["Type"] == "Credit Card"].copy()
                    cash_row = sales_display[sales_display["Type"] == "Cash"].copy()
                    for col in WEEKDAYS + ["Total"]:
                        cash_row[col] = 0.0
                    reports_combined = pd.concat([
                        reports_display,
                        pd.DataFrame([[""] * len(reports_display.columns)], columns=reports_display.columns),
                        pd.DataFrame([["Sales Summary"]], columns=["Name"]),
                        credit_card_row.rename(columns={"Type": "Name"}),
                        cash_row.rename(columns={"Type": "Name"})
                    ], ignore_index=True)
                    reports_combined.to_excel(writer, index=False, sheet_name="Reports")

                _excel_ok = True
            except ModuleNotFoundError:
                _excel_ok = False

        if _excel_ok:
            st.download_button("📥 Download Excel (.xlsx)", data=excel_buffer.getvalue(),
                               file_name=f"payroll_{pay_week_label}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning(
                "❌ Excel export requires **openpyxl** or **XlsxWriter**.\n\n"
                "Install one of them:\n"
                "`pip install openpyxl` or `pip install XlsxWriter`",
                icon="⚠️"
            )