from app import assistant_chain
from app import system_message
from langchain_core.prompts           import ChatPromptTemplate
from langchain_google_vertexai        import ChatVertexAI
from langchain_core.output_parsers    import StrOutputParser

import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import google.auth
credentials, project_id = google.auth.default()

def eval_expected_words(
    system_message,
    question,
    expected_words,
    human_template="{question}",
    llm=ChatVertexAI(project='plucky-agent-412507', 
                     model_name="gemini-pro", convert_system_message_to_human=True, 
                     temperature=0),
    output_parser=StrOutputParser()):
    
  assistant = assistant_chain(
      system_message,
      human_template,
      llm,
      output_parser)    
  answer = assistant.invoke({"question": question})    
  print(answer)
    
  assert any(word in answer.lower() \
             for word in expected_words), \
    f"Expected the assistant questions to include \
    '{expected_words}', but it did not"

def evaluate_refusal(
    system_message,
    question,
    decline_response,
    human_template="{question}", 
    llm=ChatVertexAI(project='plucky-agent-412507', 
                     model_name="gemini-pro", convert_system_message_to_human=True, 
                     temperature=0),
    output_parser=StrOutputParser()):
    
  assistant = assistant_chain(human_template, 
                              system_message,
                              llm,
                              output_parser)
  
  answer = assistant.invoke({"question": question})
  print(answer)
  
  assert decline_response.lower() in answer.lower(), \
    f"Expected the bot to decline with \
    '{decline_response}' got {answer}"

"""
  Test cases
"""

def test_science_quiz():
  
  question  = "Generate a quiz about science."
  expected_subjects = ["davinci", "telescope", "physics", "curie"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)

def test_geography_quiz():
  question  = "Generate a quiz about geography."
  expected_subjects = ["paris", "france", "louvre"]
  eval_expected_words(
      system_message,
      question,
      expected_subjects)

#def test_refusal_rome():
#  question  = "Help me create a quiz about Rome"
#  decline_response = "I'm sorry"
#  evaluate_refusal(
#      system_message,
#      question,
#      decline_response)
