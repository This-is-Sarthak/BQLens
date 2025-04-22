import streamlit as st
import os
import tempfile
import pyaudio
import wave
import time
import pandas as pd
import altair as alt
import google.generativeai as genai
import json
import requests
import datetime

from google.cloud import speech_v1p1beta1 as speech
from google.cloud import bigquery
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# ---- CONFIG ----
GEMINI_API_KEY = "API KEY "
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
PROJECT_ID = "YOUR-PROJECT-ID"
MAX_HISTORY = 3

# ---- GLOBAL STATE ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_schema" not in st.session_state:
    st.session_state.show_schema = False
if "clicked_table" not in st.session_state:
    st.session_state.clicked_table = None

# ---- AUTH ----
@st.cache_resource
def get_credentials():
    flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
    creds = flow.run_local_server(port=0)
    return creds

# ---- AUDIO RECORDING ----
def record_audio(file_path, duration=5, sample_rate=16000):
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1

    p = pyaudio.PyAudio()
    stream = p.open(format=fmt, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)

    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(fmt))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()

# ---- TRANSCRIBE AUDIO ----
def transcribe_audio(file_path, creds):
    client = speech.SpeechClient(credentials=creds)

    with open(file_path, "rb") as f:
        audio_content = f.read()

    audio = {"content": audio_content}
    config = {
        "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
        "sample_rate_hertz": 16000,
        "language_code": "en-US"
    }

    response = client.recognize(config=config, audio=audio)
    if not response.results:
        return ""

    return " ".join([res.alternatives[0].transcript for res in response.results])

# ---- GET FULL SCHEMA ----
def get_schema(dataset_name, client):
    query = f"""
        SELECT table_name, column_name, data_type
        FROM `{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
        ORDER BY table_name
    """
    df = client.query(query).result().to_dataframe()
    schema_text = f"You are a data analyst assistant. Dataset: `{dataset_name}` contains the following tables:\n\n"
    for table in df['table_name'].unique():
        schema_text += f"{table}:\n"
        for _, row in df[df['table_name'] == table].iterrows():
            schema_text += f"  - {row['column_name']} ({row['data_type']})\n"
        schema_text += "\n"
    return schema_text

# ---- LIST DATASETS ----
def list_datasets(client):
    datasets = client.list_datasets()
    return [dataset.dataset_id for dataset in datasets]

# ---- LIST TABLES ----
def list_tables(dataset_name, client):
    tables = client.list_tables(dataset_name)
    return [table.table_id for table in tables]

# ---- GET SCHEMA OF SPECIFIC TABLE ----
def get_table_schema(dataset_name, table_name, client):
    query = f"""
        SELECT column_name, data_type
        FROM `{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_name}'
        ORDER BY ordinal_position
    """
    df = client.query(query).result().to_dataframe()
    return df

# ---- GEMINI SQL GEN WITH HISTORY ----
def generate_sql_with_gemini(question, schema_context, dataset):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("models/gemini-1.5-pro")

    history_text = ""
    for i, (past_q, past_sql) in enumerate(st.session_state.chat_history[-MAX_HISTORY:]):
        history_text += f"Previous Question {i+1}: {past_q}\n"
        history_text += f"SQL {i+1}: {past_sql}\n\n"

    prompt = f"""
{schema_context}
You are an expert SQL assistant for BigQuery. You’ll help generate SQL based on natural questions.

Here are some previous questions and your responses to maintain context:

{history_text}

Now respond to this question:
{question}

Always use fully qualified table names like `{dataset}.table_name`. Only return the SQL code, no explanations.
"""

    response = model.generate_content(prompt)
    return response.text.replace("```sql", "").replace("```", "").strip()

# ---- RUN BQ QUERY ----
def run_query(sql, creds, project_id):
    access_token = creds.token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": sql,
        "useLegacySql": False
    }

    response = requests.post(
        f"https://bigquery.googleapis.com/bigquery/v2/projects/{project_id}/queries",
        headers=headers,
        data=json.dumps(payload)
    )

    if response.status_code != 200:
        raise Exception(response.text)

    result = response.json()
    fields = [f["name"] for f in result["schema"]["fields"]]
    rows = result.get("rows", [])
    return pd.DataFrame([{k: v["v"] for k, v in zip(fields, row["f"])} for row in rows])

# ---- CHART GENERATOR ----
def interactive_streamlit_viz(df):
    st.subheader("Interactive Visualization")

    chart_types = ['Bar', 'Line', 'Scatter', 'Area', 'Histogram', 'Boxplot', 'Pie', 'Heatmap']
    df = df.copy()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    cols = df.columns.tolist()
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    chart_type = st.selectbox("Chart Type", chart_types)
    x_axis = st.selectbox("X-Axis", cols)
    y_axis = st.selectbox("Y-Axis", numeric_cols if numeric_cols else cols)
    color = st.color_picker("Color", "#1f77b4")

    base = alt.Chart(df).encode(
        x=alt.X(x_axis),
        y=alt.Y(y_axis)
    ).properties(width=700, height=400)

    if chart_type == "Bar":
        chart = base.mark_bar(color=color)
    elif chart_type == "Line":
        chart = base.mark_line(color=color) + base.mark_point(color=color)
    elif chart_type == "Scatter":
        chart = base.mark_circle(size=80, color=color)
    elif chart_type == "Area":
        chart = base.mark_area(color=color)
    elif chart_type == "Histogram":
        chart = alt.Chart(df).mark_bar(color=color).encode(x=alt.X(y_axis, bin=True), y='count()')
    elif chart_type == "Boxplot":
        chart = alt.Chart(df).mark_boxplot(color=color).encode(x=x_axis, y=y_axis)
    elif chart_type == "Pie":
        chart = alt.Chart(df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field=y_axis, type="quantitative"),
            color=alt.Color(field=x_axis, type="nominal")
        )
    elif chart_type == "Heatmap":
        chart = alt.Chart(df).mark_rect().encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color(y_axis, aggregate='mean')
        )
    else:
        st.warning("Unsupported chart type selected.")
        return

    st.altair_chart(chart, use_container_width=True)

# ---- MAIN APP ----
def main():
    st.set_page_config("Voice-to-SQL for BigQuery", layout="centered")
    st.title("BQLens")

    creds = get_credentials()
    bq_client = bigquery.Client(credentials=creds, project=PROJECT_ID)

    # ---- SIDEBAR ----
    with st.sidebar:
        with st.expander("🧠 Show Chat History", expanded=False):
            for i, (q, a) in enumerate(st.session_state.chat_history[-MAX_HISTORY:][::-1], 1):
                st.markdown(f"**Q{i}:** {q}")
                st.code(a, language="sql")

        if st.button("🗑️ Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.df = None
            st.session_state.spoken_question = ""
            st.session_state.show_schema = False
            st.session_state.clicked_table = None
            st.rerun()

        st.markdown("---")
        st.header("📊 Dataset & Table")
        datasets = list_datasets(bq_client)
        selected_dataset = st.selectbox("Choose Dataset", datasets)
        dataset_fqdn = f"{PROJECT_ID}.{selected_dataset}"

        tables = list_tables(dataset_fqdn, bq_client)
        selected_table = st.selectbox("Choose Table", tables if tables else ["No tables found"])

        # Determine if user clicked for schema
        button_key = f"show_schema_{selected_table}"
        clicked = st.button("📘 Know Your Schema", key=button_key)

        if clicked:
            st.session_state.show_schema = True
            st.session_state.clicked_table = selected_table

        if selected_table != st.session_state.get("clicked_table"):
            st.session_state.show_schema = False

        if st.session_state.show_schema and selected_table == st.session_state.get("clicked_table"):
            schema_df = get_table_schema(dataset_fqdn, selected_table, bq_client)
            st.markdown("**🔍 Table Schema:**")
            st.dataframe(schema_df, use_container_width=True)

    if st.button("🎙️ Speak Your Question"):
        filename = f"recorded_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        file_path = os.path.join("PASTE-YOUR-PATH-HERE", filename)

        st.info(f"Recording for 5 seconds...\nSaving to: {file_path}")
        record_audio(file_path, duration=5)
        st.success("Recording complete. Transcribing...")

        question = transcribe_audio(file_path, creds)
        if question:
            st.session_state["spoken_question"] = question
            st.success(f"You said: {question}")
        else:
            st.error("Couldn't capture any voice. Try again.")

    question = st.text_input("Type or Edit your Question", value=st.session_state.get("spoken_question", ""))

    if st.button("🚀 Generate SQL and Run"):
        st.info("Fetching schema...")
        schema_context = get_schema(dataset_fqdn, bq_client)

        st.info("Generating SQL using Gemini...")
        sql = generate_sql_with_gemini(question, schema_context, dataset_fqdn)
        st.code(sql, language="sql")

        st.info("Running query in BigQuery...")
        df = run_query(sql, creds, PROJECT_ID)
        st.session_state.df = df
        st.success("Query completed!")

        st.session_state.chat_history.append((question, sql))

    if st.session_state.get("df") is not None:
        st.dataframe(st.session_state.df)
        interactive_streamlit_viz(st.session_state.df)

if __name__ == "__main__":
    main()
