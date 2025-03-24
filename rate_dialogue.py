from openai import OpenAI
import json
import os
import sys
import logging
from dotenv import load_dotenv
import time
import pandas as pd


def get_json(obj):
    return json.loads(obj.model_dump_json())

def show_json(obj):
    print(json.loads(obj.model_dump_json()))

def submit_message(assistant_id, thread1, user_message):
    # add a message to the thread
    message1 = client.beta.threads.messages.create(
        thread_id=thread1.id, role="user", content=user_message
    )
    run1 = client.beta.threads.runs.create(
        thread_id=thread1.id,
        assistant_id=assistant_id,
    )
    return [message1, run1]

def wait_on_run(run1, thread1):
    while run1.status == "queued" or run1.status == "in_progress":
        run1 = client.beta.threads.runs.retrieve(
            thread_id=thread1.id,
            run_id=run1.id,
        )
        time.sleep(0.5)
    return run

def get_response(thread1, message1):
    return client.beta.threads.messages.list(thread_id=thread1.id, order="asc", after=message1.id)


load_dotenv()
openai_api_key = os.environ['OPENAI_API_KEY']
if openai_api_key:
    client = OpenAI(api_key=openai_api_key)
else:
    logging.info("openai_api_key not found")
    sys.exit(1)


# path to conversation log
conv_log_path = "conversation_logs/"

#output path
output_path = "output/quality_rating"

# create a dataframes to store the rating
primary_measures_df = pd.DataFrame(columns=["Doctor", "Patient", "Intent", "Session_ID", "Timestamp",
                                            "Accuracy1_rating", "Accuracy1_reasoning",
                                            "Relevance_rating", "Relevance_reasoning",
                                            "Accuracy2_rating", "Accuracy2_reasoning",
                                            "Accuracy3_rating", "Accuracy3_reasoning",
                                            "Accuracy_rating", "Response_quality"])


# assistant instructions
instructions = ("You are annotating a conversation between a speech-language therapist and a virtual patient. " +
                "Provide your answers using the following JSON format: " +
                "{\n \"question1\": {\n \"rating\": [Number of goals],\n \"reasoning\":[String: reasoning for response to question1]\n},\n " +
                "\"question2\": {\n \"rating\": [\"Yes\" or \"No\"],\n \"reasoning\":[String: reasoning for response to question2]\n},\n " +
                "\"question3\": {\n \"rating\": [\"Yes\" or \"No\"],\n \"reasoning\":[String: reasoning for response to question3]\n},\n " +
                "\"question4\": {\n \"rating\": [\"Yes\" or \"No\"],\n \"reasoning\":[String: reasoning for response to question4]\n}\n }" +
                "\n You will be given an exchange between a speech-language therapist and a virtual patient. " +
                "\n Answer question 1 about what the speech-language therapist said: " +
                "\n Question 1: How many major goals does the speech-language therapist's turn have out of these goals: " +
                "\n1. expressing empathy, 2. greeting or introduction, 3. asking for information, 4. asking for preferences or 5. providing explanations. " +
                "\n If the speech-language therapist asked for multiple pieces of information, count it as one major goal: asking for information. " +
                "\n Typically the speech-language therapist's turn contains one major goal. " +
                "\n Occasionally the speech-language therapist's turn can contain two major goals such as expressing empathy and asking for information, " +
                "\n as in this example: 'I’m sorry to hear that. How long have you had this problem?'" +
                "\n Only assess what the speech-language therapist said to answer this question. " +
                "\n Do not take into account what the virtual patient said. " +
                "\n State how many major goals the speech-language therapist's turn contains. List the goals you identified in the reasoning for your answer."
                "\n Answer questions 2, 3 and 4 about the virtual patient's response: " +
                "\n Question 2: Was the virtual patient's response at least partially relevant to one of the goals or topics in the speech-language therapist's turn? An incomplete response which does not fully address the goal or topic is ok."
                "\n State 'Yes' or 'No' and give reasoning. " +
                "\n Responding with a 'Thank you' to an expression of empathy is fine. Rate this as 'Yes' for question 2. " +
                "\n Question 3: Is the phrasing of the virtual patient's response correct? Does it make sense in light of what the speech-language therapist said? " +
                "\n State 'Yes' or 'No' and give reasoning. If you answered 'No' to question 2, state 'No' for question 3." +
                "\n For example, the Speech-language therapist said: 'Do you produce normal amounts of saliva?' " +
                "\n The Virtual Patient replied:  'I think so... I don’t ever have much saliva so I’m always drinking a lot of water.' " +
                "\n Rate it as 'No' for question 3, as the response does not make sense in light of the therapist’s question. " +
                "\n The Virtual Patient talks about saliva but responds inaccurately with an 'I think so' rather than an 'I don’t think so' answer. "+
                "\n Rate it as 'Yes' for question 2, as the response is on the right topic of saliva production. " +
                "\n Question 4: If the speech-language therapist asked a question, for example 'Which foods give you more trouble?', " +
                "\n or made a request, for example 'Tell me about the food you can eat', did the virtual patient respond fully to the question or request? " +
                "\n State 'Yes' or 'No' and give reasoning. " +
                "\n Responding with a 'Thank you' to an expression of empathy is fine. Rate this as 'Yes' for question 4.")

print(instructions)
# Define the assistant
assistant = client.beta.assistants.create(
    name="Automated conversation annotator",
    instructions=instructions,
    temperature=0,
    model="gpt-4o-mini"
)


#list of session IDs
session_IDs = ["Amani71069",
"Jay53526",
"Lilly771507"]


for session_ID in session_IDs:
    print(session_ID)

    timestring = time.strftime("%Y%m%d-%H%M%S")
    file_name = output_path + session_ID +'_auto_rating_' + timestring  +'.csv'

    #read the conversation log
    # grab the whole conversation from a csv conversation log
    conversation_file = conv_log_path + session_ID + ".csv"
    logging.info(f"Trying to access file {conversation_file}")
    conversation_df = []
    try:
        conversation_df = pd.read_csv(conversation_file)
    except OSError as err:
        logging.error(f"OS Error: {err}")
        sys.exit(1)
    logging.info(f"Accessed file {conversation_file}")

    #remove the task description line from the conversation transcript
    if conversation_df["Intent"].values[0] != "not applicable":
        conversation_df = conversation_df.drop(conversation_df.index[conversation_df['Intent'] ==
                                               "Task Description"]).reset_index()
        #remove the final goodbye that was added by Andrew's script
        last_line_index = len(conversation_df)-1
        if conversation_df['Doctor'].values[last_line_index] == "Goodbye":
            conversation_df = conversation_df.drop(last_line_index).reset_index()

    # create a new thread passing the example annotation file
    # Create a thread and attach the file to the message
    thread = client.beta.threads.create()

    for i in range(0, len(conversation_df)):

        relevance_rating = ""
        relevance_reasoning = ""
        accuracy_rating = ""
        accuracy1_rating = ""
        accuracy1_reasoning = ""
        accuracy2_rating = ""
        accuracy2_reasoning = ""
        accuracy3_rating = ""
        accuracy3_reasoning = ""
        use_LLM = True

        #check for fallback intent
        # for 2024 use intent Default Fallback Intent
        if conversation_df["Intent"].values[i] != "not applicable":
            if conversation_df["Intent"].values[i] == "Default Fallback Intent":
                relevance_rating = "N/A"
                accuracy_rating = "N/A"
                use_LLM = False
        else:
            if conversation_df["Patient"].values[i] == "Sorry, I didn't get that. Can you rephrase?" or conversation_df["Patient"].values[i] == "I'm sorry what do you mean?":
                relevance_rating = "N/A"
                accuracy_rating = "N/A"
                accuracy1_rating = 0
                use_LLM = False

        if use_LLM:
            prompt = ("The speech-language therapist said: '" + conversation_df["Doctor"].values[i] + "'"
                      + " The virtual patient responded: '" + conversation_df["Patient"].values[i] + "'"
                      + " Answer question 1 about what the speech-language therapist said. " +
                      "Answer questions 2, 3 and 4 about the virtual patient's response.")
            [message, run] = submit_message(assistant.id, thread, prompt)
            run = wait_on_run(run, thread)
            response = get_response(thread, message)
            json_response = json.loads(response.data[0].content[0].text.value)

            #did the SLT's turn contain one goal: Yes/No
            #if question1 is No, accuracy will be set to No, as the VPs are not
            #able to respond to multiple goals
            accuracy1_rating = json_response['question1']['rating']
            accuracy1_reasoning = json_response['question1']['reasoning']

            #question2: did the VP address one of the topics or goals in SLT's turn? Yes/No
            relevance_rating = json_response['question2']['rating']
            relevance_reasoning = json_response['question2']['reasoning']

            #question3: did the VP's response make sense in light of what the SLT said? Yes/No
            #we are looking at the phrasing here and assessing if it's accurate
            accuracy2_rating = json_response['question3']['rating']
            accuracy2_reasoning = json_response['question3']['reasoning']

            #question4: did the virtual patient respond to the SLT's question: Yes/No
            accuracy3_rating = json_response['question4']['rating']
            accuracy3_reasoning = json_response['question4']['reasoning']

            show_json(response)

        #populate output columns
        slt_said = conversation_df["Doctor"].values[i]
        vp_said = conversation_df["Patient"].values[i]
        intent = conversation_df["Intent"].values[i]
        conv_timestamp = conversation_df["TimeStamp"].values[i]

        #check if the VP responded to the question fully if the SLT asked a question
        check_accuracy3 = False
        if "ask" in accuracy1_reasoning:
            check_accuracy3 = True

        if check_accuracy3:
            if accuracy2_rating == "Yes" and accuracy3_rating == "Yes":
                accuracy_rating = "Yes"
        else:
            if accuracy2_rating == "Yes":
                accuracy_rating = "Yes"

        if accuracy_rating == "":
            accuracy_rating = "No"

        #check if it's a short answer, change relevance rating to 'No topic' for answers
        #Yes, No or Okay
        #that were considered accurate
        if accuracy_rating == "Yes":
            if vp_said.startswith("Yes") or vp_said.startswith("No") or vp_said.startswith("Okay"):
                if len(vp_said) < 12:
                    relevance_rating = "No topic"
            if vp_said.startswith("Yes it is tough") or vp_said.startswith("Yes sometimes") or vp_said.startswith("Oh that's interesting") or vp_said.startswith("I don't think so") or vp_said.startswith("Ok doctor"):
                relevance_rating = "No topic"

        # populate response_quality
        response_quality = ""
        if relevance_rating == "Yes" and accuracy_rating == "Yes":
            response_quality = "1. Relevant and accurate"
        elif relevance_rating == "Yes" and accuracy_rating == "No":
            response_quality = "2. Relevant but not accurate"
        elif relevance_rating == "No topic" and accuracy_rating == "Yes":
            response_quality = "3. Accurate short answers"
        elif relevance_rating in ["No","No topic"] and accuracy_rating == "No":
            response_quality = "4. Not relevant or accurate"
        elif relevance_rating == "N/A" and accuracy_rating == "N/A":
            response_quality = "5. Did not understand"


        primary_measures_df.loc[0] = [slt_said, vp_said, intent,
                                            session_ID, conv_timestamp,
                                            accuracy1_rating, accuracy1_reasoning,
                                            relevance_rating, relevance_reasoning,
                                            accuracy2_rating, accuracy2_reasoning,
                                            accuracy3_rating, accuracy3_reasoning,
                                            accuracy_rating, response_quality]

        if i == 0:
            primary_measures_df.to_csv(file_name, mode='w', index=False, header=True)
        else:
            primary_measures_df.to_csv(file_name, mode='a', index=False, header=False)
