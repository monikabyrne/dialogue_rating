# dialogue_rating
Python script to rate Virtual Patient responses

It looks for the conversation file (or files) in the "conversation_logs" folder named “<session_ID>.csv” for session IDs in this list: "session_IDs" 


It will output the Virtual Patient response ratings in folder “output/quality_rating” as a csv file

add a .env file in the project root directory dialogue_rating/ with the following environment variables:
OPENAI_API_KEY=your_key
TOKENIZERS_PARALLELISM=False

replace 'your_key' with the OpenAI Key for your account from https://platform.openai.com/account/api-keys
