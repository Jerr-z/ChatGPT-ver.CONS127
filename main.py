import os
import whisper
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader
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

    try:
        index = GPTSimpleVectorIndex.load_from_disk("index.json")
    except:
        documents = SimpleDirectoryReader(os.path.join(os.getcwd(), "training data")).load_data()
        index = GPTSimpleVectorIndex.from_documents(documents)
        index.save_to_disk("index.json")

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
