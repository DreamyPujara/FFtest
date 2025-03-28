from langchain.schema import BaseRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_chroma import Chroma
from model import llm
from prompts import contextualize_q_prompt, get_qa_prompt
from get_embedding_function import get_embedding_function
from structure import get_formula_details
from multi_retriever import MultiCollectionRetriever
absence_map = {
    "Global Absence Plan Duration": "Global_Absence_Plan_Duration",
    "Global Absence Entry Validation": "Global_Absence_Entry_Validation",
    "Global Absence Type Duration": "Global_Absence_Type_Duration",
    "Global Absence Band Entitlement": "Global_Absence_Band_Entitlement",
    "Global Absence Accrual Matrix": "Global_Absence_Accrual_Matrix",
    "Global Absence Discretionary Disbursement Rule": "Global_Absence_Discretionary_Disbursement_Rule",
    "Global Absence Carryover": "Global_Absence_Carryover",
    "Global Absence Plan Enrollment End": "Global_Absence_Plan_Enrollment_End",
    "Global Absence Plan Entitlement": "Global_Absence_Plan_Entitlement",
    "Global Absence Plan Roll Forward Start": "Global_Absence_Plan_Roll_Forward_Start",
    "Global Absence Partial Period Accrual Rate": "Global_Absence_Partial_Period_Accrual_Rate",
    "Global Absence Plan Enrollment Start": "Global_Absence_Plan_Enrollment_Start",
    "Global Absence Proration": "Global_Absence_Proration",
    "Global Absence Plan Period Anniversary Event Date": "Global_Absence_Plan_Period_Anniversary_Event_Date",
    "Compensation Default and Override": "Compensation_Default_and_Override",
    "Compensation Person Selection": "Compensation_Person_Selection",
    "Total Compensation Item": "Total_Compensation_Item",
    "Compensation Currency Selection" : "Compensation_Currency_Selection",
    "Compensation Hierarchy Determination" : "Compensation_Hierarchy_Determination",
    "Age Calculation" : "Age_Calculation",
    "Age Determination Date" : "Age_Determination_Date",
    "Beneficiary Certification Required" : "Beneficiary_Certification_Required",
    "Compensation Calculation" : "Compensation_Calculation",
    "Coverage Amount Calculation" : "Coverage_Amount_Calculation",
    "Coverage Amount Limit" : "Coverage_Amount_Limit",
    "Coverage Upper Limit" : "Coverage_Upper_Limit",
    "Default Enrollment" : "Default_Enrollment",
    "Default To Assign Pending Action" : "Default_To_Assign_Pending_Action",
    "Enrollment Certification Required" : "Enrollment_Certification_Required",
    "Enrollment Coverage Start Date" : "Enrollment_Coverage_Start_Date",
    "Enrollment Opportunity" : "Enrollment_Opportunity",
    "Enrollment Period Start Date" : "Enrollment_Period_Start_Date",
    "Evaluate Life Event" : "Evaluate_Life_Event",
    "Extra Input" : "Extra_Input",
    "Length Of Service Date To Use" : "Length_Of_Service_Date_To_Use",
    "Life Event Reason Timeliness" : "Life_Event_Reason_Timeliness",
    "Participation And Rate Eligibility" : "Participation_And_Rate_Eligibility",
    "Person Change Causes Life Event" : "Person_Change_Causes_Life_Event",
    "Person Selection" : "Person_Selection",
    "Post Election Edit" : "Post_Election_Edit",
    "Rate Periodization" : "Rate_Periodization",
    "Rate Start Date" : "Rate_Start_Date",
    "Rate Value Calculation" : "Rate_Value_Calculation",
    "Rounding" : "Rounding",
    "Waiting Period Value And UOM" : "Waiting_Period_Value_And_UOM",
    "Recruiting Job Requisition" : "Recruiting_Job_Requisition",
    "Recruiting Candidate Selection Process" : "Recruiting_Candidate_Selection_Process"
}

class AIModel:
    def __init__(self, module, formula_type_name):
        # Map the formula type name to the corresponding key in absence_map
        formula_type = absence_map.get(formula_type_name)
        
        # Debug: Print the formula_type to verify it's correct
        print(f"Formula type from absence_map: {formula_type}")
        
        if not formula_type:
            raise ValueError(f"Formula type '{formula_type_name}' not found in absence_map.")
        
        self.chain = self._setup_rag_chain(module, formula_type)

    def _setup_rag_chain(self, module, formula_type):
        # Debug: Print the formula_type being used
        print(f"Setting up RAG chain for formula type: {formula_type}")
        
        # Get formula details
        formula_data = get_formula_details(formula_type)
        
        # Debug: Print formula_data to verify it's not None
        print(f"Formula data: {formula_data}")
        
        # Check if formula_data is None
        if not formula_data:
            raise ValueError(f"No formula data found for formula type: {formula_type}")
        
        # Ensure formula_data has the required keys
        required_keys = ["FORMULA_TYPE_NAME", "Context", "Input Values"]
        for key in required_keys:
            if key not in formula_data:
                raise KeyError(f"Missing required key '{key}' in formula_data for formula type: {formula_type}")
        
        # Get the QA prompt
        qa_prompt = get_qa_prompt(
            module,
            formula_data["FORMULA_TYPE_NAME"],
            formula_data["Context"],
            formula_data["Input Values"],
            formula_data["Return Variables"]
        )
        
        try:
            embedding_function = get_embedding_function()
            CHROMA_PATH = "chroma"
            
            # Initialize Chroma retriever
            FF_collection = Chroma(
                collection_name=formula_type,
                embedding_function=embedding_function,
                persist_directory=CHROMA_PATH
            ).as_retriever(search_kwargs={"k": 1})
            
            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm=llm,
                retriever=FF_collection,
                prompt=contextualize_q_prompt
            )
            
            # Create question-answering chain
            question_answer_chain = create_stuff_documents_chain(
                llm=llm,
                prompt=qa_prompt
            )
            
            # Create the final RAG chain
            return create_retrieval_chain(
                history_aware_retriever,
                question_answer_chain
            )
        except Exception as e:
            print(f"Error setting up RAG chain: {str(e)}")
            raise

    def chat(self, query: str, chat_history):
        try:
            for chunk in self.chain.stream({"input": query, "chat_history": chat_history}):
                if 'answer' in chunk:
                    yield chunk["answer"]
                else:
                    print(chunk)
        except Exception as e:
            print(f"Error during chat: {str(e)}")
            raise
