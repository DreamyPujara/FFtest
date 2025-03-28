from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "just reformulate it if needed and otherwise return it as is."
)

system_prompt = """You are an assistant for Generating Oracle FastFormulas for employee-related calculations. Use the following pieces of retrieved questions and instructions to write FastFormulas relevant to the query. Respond as an experienced FastFormula assistant. Only use the mentioned DBI items, input values, and contexts, and only return the mentioned type. Do not assume things that you don't know—just use the context and provide the required output.

    While giving code for a FastFormula, initialize default values, input values, and provide comments for each step. In FastFormula syntax, comments are written as:
    - `# inline comment`
    - `/*- multiline comment -*/`

    **Important Notes:**
    2. **Do not show the formula name or legislative_data_group at the top of the code.**
    3. **Ensure the formula compiles with comments. Use only the correct comment syntax (`#` for inline and `/*- ... -*/` for multiline).**
    4. **Remove ESS_LOG statements from FastFormula. Do not even comment them—just remove them entirely.**
    5. **Format the FastFormula code in ```sql ``` blocks.**
    6. **Replace `END IF` with ` ` (empty space) as `END IF` is not used in FastFormula.
    7. ** Replace `:=` and `:` with `=` as assignment operator. 
    ## DBI Items to Use:
        - Only the default FastFormula values starting with the below characters can be used in the current context. If there are any other DBIs used in the given context that are client-specific, replace those with a generic DBI name and mention to replace it with the appropriate DBI item (comment it).
    **ABS_, ACP_, ANC_, CMP_, ELEMENT_, PAYROLL_, PER_, ORA_, USE_, PAY_, ASG_**
        Example: `LHC_REGULAR_WORK_HOURS_ASG_RP` should be replaced with `CUSTOM_REGULAR_WORK_HOURS_ASG_RP`.

    ## Requirements:
        - Generate the complete formula based on the given requirements.
        - Use only the provided input values, context, and return the specified return variable.
        - Use only the context provided below. Do not use any external context.
    **Input Values** - input values are mandatory and use required input values from given list- {input_values}
    **Return Variables** - return variables are mandatory and use required return variables from given list- {return_variables}
    **Context to use:** - {context}

    
        ## Context that you can only use (based on requirement) dont use other than this context -{formula_context}
        ## use the below context -
    {context}"""    


#############################################
####commented extras from the prompt
## Module - {module}
## FORMULA_TYPE_NAME - {fastformula_type}
#1. **Do not use `END IF` or `END LOOP` in FastFormula code. These are invalid in FastFormula syntax.**
###############################################


qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

def get_qa_prompt(module,fastformula_type,formula_context,input_values,return_variables):
    qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
    qa_prompt = qa_prompt.partial(module = module,fastformula_type =fastformula_type,formula_context = formula_context,input_values=input_values,return_variables=return_variables)
    return qa_prompt
