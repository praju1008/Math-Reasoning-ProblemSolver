import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper

st.set_page_config(page_title="Math & Reasoning Problem Solver", page_icon=":robot:")
st.title("Math & Reasoning Problem Solver 🤖")

# ----- API KEY -----
groq_api_key = st.sidebar.text_input("Enter your Groq API Key", type="password")
if not groq_api_key:
    st.warning("Please enter your Groq API Key to proceed.")
    st.stop()

# ----- LLM -----
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0,
)

# ----- TOOLS (manual) -----
wikipedia = WikipediaAPIWrapper()

math_prompt_tmpl = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a math expert.\n"
        "Solve this problem step by step and give the final numeric answer on the last line as:\n"
        "Final Answer: <answer>\n\n"
        "Question: {question}"
    ),
)

reason_prompt_tmpl = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a helpful AI assistant that can solve math and reasoning problems.\n\n"
        "Question: {question}\n"
        "Answer with clear reasoning and a final conclusion."
    ),
)

def handle_question(question: str) -> str:
    q = question.lower()

    # Wikipedia mode
    if any(k in q for k in ["wiki", "who is", "where is", "what is", "tell me about"]):
        try:
            return wikipedia.run(question)
        except Exception as e:
            return f"Error while calling Wikipedia: {e}"

    # Math mode
    if any(k in q for k in ["calculate", "+", "-", "*", "/", "solve", "area", "perimeter", "percentage"]):
        prompt = math_prompt_tmpl.format(question=question)
        return llm.invoke(prompt).content

    # Reasoning mode
    prompt = reason_prompt_tmpl.format(question=question)
    return llm.invoke(prompt).content

# ----- CHAT STATE -----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am your Math Problem Solver AI. "
                "I can calculate, look up Wikipedia, and reason through problems."
            ),
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# ----- UI -----
question = st.text_area(
    "Enter your Question here:",
    "Look at this series: 36, 34, 30, 28, 24, ... What number should come next?",
)

if st.button("Submit Question"):
    if question.strip():
        with st.spinner("Thinking..."):
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            response = handle_question(question)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
    else:
        st.warning("Please enter a question before submitting.")
                        