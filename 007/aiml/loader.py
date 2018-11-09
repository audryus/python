import aiml
import os
import pyttsx3

current_file = os.path.abspath(os.path.dirname(__file__))

kernel = aiml.Kernel()
kernel.setBotPredicate("name", "Chief")

kernel.learn(current_file + "/startup.xml")
kernel.respond("load aiml")
engine = pyttsx3.init()


def answer(question):
    ans = kernel.respond(question)
    engine.say(ans)
    engine.runAndWait()
    return ans


while True:
    print(answer(input("> ")))
