import streamlit as st
import pandas as pd
import os, io, time, contextlib, threading, pickle, psutil, matplotlib.pyplot as plt
from langchain_ollama import ChatOllama

STORE_DIR = "store"
CHAT_FILE = os.path.join(STORE_DIR, "chats.pkl")

def save_chats():
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(CHAT_FILE, "wb") as f:
        pickle.dump(st.session_state.chats, f)

def load_chats():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "rb") as f:
            st.session_state.chats = pickle.load(f)
    else:
        st.session_state.chats = {}
    st.session_state.active_chat = None

# Load Dataset + Model
@st.cache_resource
def load_dataset(store_dir=STORE_DIR):
    csv_path = os.path.join(store_dir, "merged_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Merged CSV not found at {csv_path}. Please upload dataset first.")
    df = pd.read_csv(csv_path)
    llm = ChatOllama(model="gpt-oss:20b", temperature=0.1)
    return df, llm

df, llm = load_dataset()

st.set_page_config(page_title="Chatbot", layout="wide")
col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
with col2:
    st.title("Chatbot (RAG & GPT-OSS:20B LLM)")
    st.caption(f"Ask anything — analytical, descriptive, or visual. | Dataset: {df.shape[0]} rows and {df.shape[1]} columns")

if "chats" not in st.session_state:
    load_chats()
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False
if "edit_index" not in st.session_state:
    st.session_state.edit_index = None

# Sidebar 
with st.sidebar:
    st.header("Chats")

    if st.button("New Chat"):
        st.session_state.active_chat = None
        st.rerun()

    for chat_key, messages in st.session_state.chats.items():
        first_q = messages[0]["question"] if messages else "Untitled Chat"
        is_active = st.session_state.active_chat == chat_key

        col1, col2 = st.columns([0.8, 0.15])
        with col1:
            if st.button(first_q[:15] + ("..." if len(first_q) > 15 else ""), key=f"select_{chat_key}", use_container_width=True):
                st.session_state.active_chat = chat_key
                st.rerun()
        # Remove/Delete the chats
        # with col2:
        #     if st.button("⛔", key=f"delete_{chat_key}", help="Delete chat"):
        #         del st.session_state.chats[chat_key]
        #         save_chats()
        #         st.session_state.active_chat = None
        #         st.rerun()

def stop_generation():
    st.session_state.stop_flag = True

def reset_edit():
    st.session_state.edit_index = None
    st.session_state.question_edit = ""

#  Display Chat 
if st.session_state.active_chat:
    for i, chat in enumerate(st.session_state.chats[st.session_state.active_chat]):
        col1, col2, col3 = st.columns([0.2, 0.5, 0.2])
        with col2:
            with st.chat_message("user"):
                st.markdown(f"{chat['question']}")
                # st.caption(
                #     f"SEC🕐 LLM: {chat.get('llmCodeGenTime', 0):.2f}s | Exec: {chat.get('codeExceTime', 0):.2f}s | Total: {chat.get('time', 0):.2f}s | "
                #     f" CPU: {chat.get('cpu_usage', 0):.1f}% |  Memory: {chat.get('memory_usage', 0):.1f} MB"
                # )

                st.caption(
                    f"⏱️ LLM code Gen: {int(chat.get('llmCodeGenTime',0))//60}:{int(chat.get('llmCodeGenTime',0)%60)} sec | "
                    f"Exec: {chat.get('codeExceTime',0):.2f} sec | "
                    f"Total {int(chat.get('time',0)//60)}:{int(chat.get('time',0)%60)} sec | "
                    f"CPU: {chat.get('cpu_usage', 0):.2f}% | Memory: {chat.get('memory_usage', 0):.2f} MB"
                )

                # Edit question button
                if st.button("🖍", key=f"edit_{i}", help="Edit this question"):
                    st.session_state.edit_index = i
                    st.session_state.question_edit = chat["question"]
                    st.session_state.run_mode = "edit"
                    st.rerun()

            with st.chat_message("assistant"):
                answer = chat["answer"]
                if isinstance(answer, pd.DataFrame):
                    st.dataframe(answer.set_index(pd.Index(range(1, len(answer) + 1))))
                elif isinstance(answer, plt.Figure):
                    st.pyplot(answer)
                elif isinstance(answer, (dict, list)):
                    st.dataframe(answer)
                elif isinstance(answer, str) and answer.startswith("⚠️"):
                    st.warning(f"Error occured in execution!Please Try again or start a new question!")
                else:
                    st.markdown(answer)
                # st.markdown(answer)


if st.session_state.edit_index is not None:
    st.divider()
    edited_question = st.text_input("Edit Question", value=st.session_state.question_edit)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("➤"):
            st.session_state.question_to_run = edited_question
            st.session_state.temp_edit_index = st.session_state.edit_index
            reset_edit()
            st.rerun()
    with col_b:
        if st.button("✖"):
            reset_edit()
            st.rerun()

# User Input 
if st.session_state.edit_index is None:
    user_question = st.chat_input("Ask a question about your dataset...")
    if user_question:
        st.session_state.question_to_run = user_question
        st.rerun()
# Processing Section 
if "question_to_run" in st.session_state:
    question = st.session_state.question_to_run
    st.session_state.pop("question_to_run")

    # Create chat only when first Q&A happens
    if not st.session_state.active_chat:
        chat_name = f"Chat_{int(time.time())}"
        st.session_state.chats[chat_name] = []
        st.session_state.active_chat = chat_name

    start_time = time.time()
    st.chat_message("user").markdown(question)
    stop_col, _ = st.columns([0.3, 0.7])
    with stop_col:
        st.button("Stop", on_click=stop_generation)
    st.session_state.stop_flag = False
    container = {"result": None, "code": ""}

    def process_question():
        cols = ", ".join(df.columns)
        sample = df.head(3).to_markdown(index=False)

        prompt = f"""
        You are a professional data analyst working with pandas DataFrame named df.
        Columns: {cols}
        Example rows:
        {sample}
       
        You MUST follow these rules STRICTLY:

            1. Output ONLY executable Python code.
            2. Do NOT include markdown, backticks, comments, explanations, or text of any kind.
            3. Use ONLY ASCII characters. Do NOT use Unicode characters such as “–”, “≤”, “≥”, “≠”, “×”, “•”.
            4. Use only ASCII operators: <=  >=  !=  ==  -
            5. The code must be valid to run inside Python exec() without modification.
            6. Use only pandas and matplotlib (if a plot is needed).
            7. The DataFrame is available as the variable df.
            8. Assign the final answer to a variable named result.
            9. Never print anything. Never ask for input. Never show a plot; instead assign the figure to result if needed.
            Write Python (pandas + matplotlib if needed) code to answer:
        "{question}"
        """

        t_llm_start = time.time()
        resp = llm.invoke(prompt)
        t_llm_end = time.time()
        container["llm_time"] = t_llm_end - t_llm_start

        code = str(resp.content if hasattr(resp, "content") else resp)
        # If you want llm generated code ,print below line
        # print(f"Question : {question} \n llm generated code : \n{code}")
        code = code.replace("```python", "").replace("```", "").strip()
        local_env = {"df": df.copy(), "pd": pd, "plt": plt}
        try:
            t_exec_start = time.time()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code, {}, local_env)
            container["result"] = local_env.get("result", buf.getvalue())
            container["exec_time"] = time.time() - t_exec_start
        except Exception as e:
            container["result"] = f"⚠️ Error executing code: {e}\n\nCode:\n{code}"
            container["exec_time"] = 0

    thread = threading.Thread(target=process_question)
    thread.start()
    with st.spinner("Generating answer..."):
        while thread.is_alive():
            time.sleep(0.2)
            if st.session_state.stop_flag:
                st.warning("Stopped by user.")
                break
        thread.join()

    st.session_state.stop_flag = False
    result = container["result"]
    code = container["code"]
    llm_time = container.get("llm_time", 0)
    exec_time = container.get("exec_time", 0)
    total_time = time.time() - start_time

    process = psutil.Process(os.getpid())
    cpu_percent = process.cpu_percent(interval=0.3)
    mem_usage_mb = process.memory_info().rss / (1024 * 1024)

# NOTE : Dont store llm generated code in record,it adds pickle file ,so it correpts the file data if it contains graphs,dataframes ,then we can't access the chat history
    record = {
        "question": question,
        "answer": result,
        "time": total_time,
        "llmCodeGenTime": llm_time,
        "codeExceTime": exec_time,
        "cpu_usage": cpu_percent,
        "memory_usage": mem_usage_mb
    }

    # Handle edit vs new
    if st.session_state.get("run_mode") == "edit" and "temp_edit_index" in st.session_state:
        idx = st.session_state.temp_edit_index
        st.session_state.chats[st.session_state.active_chat][idx] = record
        st.session_state.pop("temp_edit_index", None)
        reset_edit()
    else:
        st.session_state.chats[st.session_state.active_chat].append(record)

    save_chats()
    st.rerun()


