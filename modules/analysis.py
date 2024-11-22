# import os

# import sys

# from pathlib import Path

# #for appending the file address

# sys.path.append(str(Path(__file__).parent.parent))

# os.environ["OPENAI_API_KEY"] = ""

# from langchain.chains.question_answering import load_qa_chain
# from langchain import SerpAPIWrapper
# from langchain.llms import OpenAI

# from langchain.vectorstores import Chroma
 
# from langchain.chains import RetrievalQA,LLMChain

# from langchain.memory import ConversationBufferWindowMemory

# from langchain.prompts import PromptTemplate

# from langchain import LLMMathChain, OpenAI

# from langchain.agents import initialize_agent, Tool

# from langchain.agents import AgentType
# from langchain.chat_models import ChatOpenAI
# # Define the Langchain prompt template
# import getpass
# import os

# # os.environ["SERPAPI_API_KEY"] = getpass.getpass()
# # # chain=LLMChain(prompt=prompt,output_key="title")

# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

# llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
# # llm_retraival_hain=
# # search = SerpAPIWrapper()

# tools = [

#         Tool(

#         name="Calculator",

#         func=llm_math_chain.run,

#         description="useful for when you need to answer questions about math"
#         ),
#         # Tool(
#         # name="Search",
#         # func=search.run,
#         # description="Useful when you need to answer questions about current events. You should ask targeted questions.",
#         # ),
        
       
#     ]

# agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True)

# print(llm)

# chain = load_qa_chain(OpenAI(), chain_type="stuff")

# def analyze_and_answer(user_query,verctordb,max_tokens=100):
#     try:
#         langchain_prompt_template = """

#             [Assistant]

#             You are a project management assistant. You have access to information about various projects.

#             When asked to introduce about you then introduce yourself as Jarvis.

#             When asked about your author then give answer Dhyey Consulting.

            

#             {context}

            

#             Sample questions can help you learn how to generate answers, and you should always ask the user for more details to improve the accuracy of your responses.

            

#             User: What are the issues of 'PRJ3464'?

#             Assistant: Their are Connection issues impacting the finalization of CPQ

            

#             User who is the owner of SIT completion is delpayed discription in r/i/d in PRJ3751?

#             Assistant:Owner is ramya/Brezhnew for this....

            

#             User: How many projects are completed and not started  or not completed for 'PRJ3464'?

#             Assistant: There are 7 projects completed and 6 projects not started for 'PRJ3464 - GTS-CPQ Implementation-Storage'.

            

#             User: What is the scope of the project 'PRJ3464'?

            

#             User: How are resources allocated and what is the availability for the project 'PRJ3464 - GTS-CPQ Implementation-Storage'?

            

#             User: How is communication and stakeholder management handled for the project 'PRJ3464'?

            

#             #Interactive

#             User:how many admin red are there?

#             Assistant:Which admin red you need i have found  4 admin red from SBD-Hcl All Towers Weekly Status Refort and  2 from Full PM/END to ENd and ....

            

#             #Analysis Information

#             User:which milestones are in progress/status for project GTS-CPQ ?

#             Assistant:  progress state of GTS-CPQ Implementation-Storage at date 9-march-2023 was 60 ,at date 5-jan-2023 was 40, at 9-frb-2023 was 90 ,at 9-march-2023 was 60,

#             and at date 6-april-2023 was 60

            

#             User:What is Team Activites of PRJ3464?

#             Assistant:Which Activity you need as i found Activity for PRJ3464 on status at 05-jan-2923 which are support uat agora connection issues and for status as of 09-feb-2023 as support UAT...

            

#             {history}

#             # Question: {question}

#             # Helpful Answer

#             """

            

#         prompt = PromptTemplate(input_variables=["history", "context", "question"], template=langchain_prompt_template)

#         memory = ConversationBufferWindowMemory(k=5,input_key="question", memory_key="history")


#         retriever=verctordb.as_retriever()

#         print("hii")

            

#         qa = RetrievalQA.from_chain_type(OpenAI(openai_api_key=""),retriever=retriever,chain_type="stuff",return_source_documents=False,chain_type_kwargs={"prompt": prompt, "memory": memory}) #,return_source_documents=True,chain_type_kwargs={"prompt": prompt, "memory": memory},
#         if qa.length==0:
#             return "no answer found"
#         query =qa(user_query)
#         return query
#     except:
#         return "error while processing!!"
  