import whisper
import os

# MODELS
# tiny.en
# base.en
# small.en
# medium.en

class transcriber:

    def __init__(self, save_file="results.txt", model="medium.en"):
        """
        :param save_file: str, directory for save file
        :param model: whisper model used for transcribing, options include: tiny.en, base.en, small.en etc.
        """

        self.save_file = save_file
        self.model = model
        self.output_folder = "training data"

    def transcribe(self, audio_file, verbose=True, language="english"):
        """
        Turns speech into text, saves it into a text file
        :param verbose: True, False or None. Adjusts log info output level
        :param language: language for the model to recognize
        :param word_timestamps: Don't know what it does :)
        :param audio_file: directory of the audio file
        :return: None
        """

        save_file = self.save_file
        model = whisper.load_model(self.model)

        result = model.transcribe(audio=audio_file, language=language,  verbose=verbose,
                                  condition_on_previous_text=True)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            print("Created Output Folder.\n")

        with open(os.path.join(self.output_folder, save_file), "a", encoding="utf-8") as file:
            file.write(result["text"])
