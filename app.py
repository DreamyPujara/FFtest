import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import re
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from rag_chain import AIModel
import asyncio

UPLOAD_FOLDER = "docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ai_model = None

FORMULA_CONFIG = {
    "Absence": {
        "types": [
            "Global Absence Plan Duration", "Global Absence Entry Validation", "Global Absence Type Duration", "Global Absence Band Entitlement", "Global Absence Accrual Matrix", "Global Absence Discretionary Disbursement Rule", "Global Absence Carryover", "Global Absence Plan Enrollment End", "Global Absence Plan Entitlement", "Global Absence Plan Roll Forward Start", "Global Absence Partial Period Accrual Rate", "Global Absence Plan Enrollment Start", "Global Absence Proration", "Global Absence Plan Period Anniversary Event Date"
        ]
    },
    "Compensation": {
        "types": ["Compensation Default and Override","Compensation Person Selection","Total Compensation Item","Compensation Currency Selection" ,"Compensation Hierarchy Determination" ,]
    },
    "Benefits": {
        "types": [
            "Person Change Causes Life Event", "Participation And Rate Eligibility", "Rounding", "Age Calculation", "Person Selection",  "Life Event Reason Timeliness", "Coverage Amount Calculation", "Rate Value Calculation",  "Age Determination Date", "Enrollment Opportunity", "Beneficiary Certification Required", "Compensation Calculation", "Post Election Edit","Enrollment Period Start Date","Default Enrollment","Evaluate Life Event","Waiting Period Value And UOM","Coverage Amount Limit","Rate Periodization","Length Of Service Date To Use","Enrollment Certification Required","Coverage Upper Limit","Rate Start Date","Enrollment Coverage Start Date","Default To Assign Pending Action","Extra Input"
        ]
    },
    "ORC": {
        "types": [
            "Recruiting Job Requisition","Recruitiong Candidate Selection Process"
        ]
    }
}

async def main():
    st.set_page_config(
        page_title="Mastek_Fast_Formula_Generator",
        page_icon="📘",
        layout="wide"
    )
    
    # logo_path = "image.png" #need to add----->
    # st.image(logo_path, width=150)
    st.title("Fast Formula Generator")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "module_selected" not in st.session_state:
        st.session_state.module_selected = None
    if "formula_type" not in st.session_state:
        st.session_state.formula_type = None
    if "conversation_started" not in st.session_state:
        st.session_state.conversation_started = False

    with st.sidebar:
        st.subheader("Select Formula")
        module = st.selectbox(
            "Select Module",
            list(FORMULA_CONFIG.keys()),
            index=None,
            placeholder="Choose a module..."
        )
        
        if module:
            st.session_state.module_selected = module
            formula_type = None
            if len(FORMULA_CONFIG[module]["types"]) > 0:
                formula_type = st.selectbox(
                    "Select Formula Type",
                    FORMULA_CONFIG[module]["types"],
                    index=None,
                    placeholder="Choose a formula type..."
                )
            else:
                # ai_model = AIModel(module,module)
                pass
            
            if formula_type:
                st.session_state.formula_type = formula_type
                ai_model = AIModel(module,formula_type)


    for role, content in st.session_state.messages:
        with st.chat_message(role):
            st.markdown(content)

    with stylable_container(
        key="bottom_content",
        css_styles="""
            {
                position: fixed;
                bottom: 120px;
            }
            """,
    ):
        if st.checkbox("Include chat history"):
            st.session_state.is_chat_history = True
        else:
            st.session_state.is_chat_history = False

    if prompt := st.chat_input("What is up?"):
        if not st.session_state.formula_type:
            st.warning("Please select a formula type before submitting a prompt.")
        else:
            st.session_state.messages.append(("human", prompt))
            with st.chat_message("human"):
                st.markdown(prompt)

            if len(st.session_state.messages) > 0:
                with st.chat_message("ai"):
                    if st.session_state.is_chat_history:
                        lst = st.session_state.messages
                        chat_history = ([lst[-3], lst[-2]] if len(lst) > 2 else [])
                    else:
                        chat_history = []

                    stream = ai_model.chat(st.session_state.messages[-1][1], chat_history)
                    response = st.write_stream(stream)
                    st.session_state.messages.append(("ai", response))

if __name__ == '__main__':
    asyncio.run(main())
