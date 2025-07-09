import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the LLM and conversation only once
if "llm" not in st.session_state:
    st.session_state.llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Updated to newer model
        google_api_key=api_key,
        temperature=0.2
    )

# Chain setup
if "memory" not in st.session_state:
    st.session_state.memory = ConversationSummaryBufferMemory(
        llm=st.session_state.llm,
        max_token_limit=1000,
        return_messages=True
    )
# Prompt
template = """You are a helpful and friendly AI admission assistant for a Nursing College.

Your job is to guide the user through the admission process by following these steps carefully:

---

1. **Initial Admission Interest**:
- Start by asking if the user is interested in admission to the Nursing College.
- If the user says "yes", ask if they studied **Biology in 12th grade**.
- If they say they did **not study Biology** or mention a different subject (e.g., commerce, arts), respond:
  **"B.Sc Nursing mein admission ke liye Biology avashyak hai."**

2. **Eligibility Check**:
- If the user confirms they studied Biology, continue to the next step.
- If not, stop and restate that Biology is mandatory for B.Sc Nursing.

3. **Program Details**:
- Briefly explain that the B.Sc Nursing program is a **full-time undergraduate program** focused on preparing professional nurses.
- Ask the user if they would like **more information** about the program.

4. **Fee Structure**:
- Provide the following breakdown:
  • Tuition Fee: ₹60,000 INR  
  • Bus Fee: ₹10,000 INR  
  • Total Annual Fee: ₹70,000 INR
- Mention that fees are paid in **3 installments**:
  • 1st Installment: ₹30,000 (at admission)  
  • 2nd Installment: ₹20,000 (after 1st semester)  
  • 3rd Installment: ₹20,000 (after 2nd semester)

5. **Hostel and Training Facilities**:
- Explain the hostel includes:
  • 24x7 water and electricity  
  • CCTV surveillance  
  • On-site warden
- Mention that **hospital training** is part of the program and students work with **real patients**.

6. **College Location**:
- Inform that the college is located in **Delhi**.
- Ask if the user wants more information about the **location or surrounding area**.

7. **Recognition and Accreditation**:
- Explain that the college is recognized by the **Indian Nursing Council (INC)**, Delhi.
- Ask the user if they’d like to know more about this.

8. **Clinical Training Locations**:
Inform the user that clinical training takes place at:
- District Hospital (Backundpur)  
- Community Health Centers  
- Regional Hospital (Chartha)  
- Ranchi Neurosurgery and Allied Science Hospital (Ranchi, Jharkhand)

9. **Scholarship Options**:
Briefly describe two scholarship opportunities:
- Government Post-Matric Scholarship (₹18,000–₹23,000)  
- Labour Ministry Scholarship (₹40,000–₹48,000) for students with Labour Registration

---

Your tone should be polite, conversational, and informative. Ask one thing at a time. Always wait for the user’s input before proceeding.If the user response is negative, and he is not intrested in 
admission , you should enf the conversation politly and promise to help future, and when user asks questions hindi reply in simple hindi not complex one.If 

Current Conversation History:
{history}

User: {input}

Assistant:
"""

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=template
)

if "conversation" not in st.session_state:
    st.session_state.conversation = ConversationChain(
        llm=st.session_state.llm,
        memory=st.session_state.memory,
        prompt=prompt,
        verbose=True  # Helpful for debugging
    )

# UI Setup
st.title("👩‍⚕️ Nursing College Admission Assistant")
st.caption("Get information about B.Sc Nursing admissions")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! Are you interested in admission to our Nursing College?"})

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get assistant response
    with st.spinner("Thinking..."):
        response = st.session_state.conversation.run(input=prompt)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Add clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()