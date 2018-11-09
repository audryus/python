import aiml
import os
import pyttsx3

current_file = os.path.abspath(os.path.dirname(__file__))

kernel = aiml.Kernel()
kernel.setBotPredicate("name", "Chief")

kernel.learn(current_file + "/startup.xml")
kernel.respond("load aiml")

while True:
    print(kernel.respond(input("> ")))
