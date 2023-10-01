import os
import whisper
from llama_index import VectorStoreIndex, SimpleDirectoryReader, set_global_service_context, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index.llms import openai
from chat import Chatbot
from transcriber import transcriber

# get your own api key I only got $15 in there!!!
api_key = "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"

os.environ['OPENAI_API_KEY'] = api_key

audio_text = "results.txt"
lons_audio_files = os.listdir("audio")  # add file names here


# turn audio files into txt
def audios2txt(loa):
    """
    Turn audio files into .txt files
    :param loa: [str], list of audio files to convert into txt
    :return: None
    """
    scribe = transcriber(model="base.en", save_file=audio_text)
    if len(loa) == 0:
        pass
    for a in loa:
        scribe.transcribe(os.path.join("audio",a), verbose=True)





def main():
    llm = openai.OpenAI(model="gpt-3.5-turbo", temperature=0)
    service_context = ServiceContext.from_defaults(llm=llm)
    set_global_service_context(service_context)

    # try:
    #     storage_context = StorageContext.from_defaults(persist_dir="./storage")
    #     index = load_index_from_storage(storage_context)
    # except:
    documents = SimpleDirectoryReader(os.path.join(os.getcwd(), "training data")).load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist()

    bot = Chatbot(api_key, index=index)
    bot.load_chat_history("chat_history.json")

    while True:
        user_input = input("Enter Text Here: ")
        if user_input.lower() in ["bye", "goodbye", "exit"]:
            print("Bot: Goodbye!")
            bot.save_chat_history("chat_history.json")
            break
        response = bot.generate_response(user_input)
        print(f"Bot: {response['content']}")
    exit()


# DRIVER CODE
# audios2txt(lons_audio_files) done! only took 4.5 hours!
main()
