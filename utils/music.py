import os
import random
from playsound import playsound
def playmusic(emotion):
    path="D:/Project_music/"+emotion+'/'
    os.chdir(path)
    list=os.listdir(path)
    l=len(list)
    n=random.randint(1,l-1)
    print("Playing : "+ list[n])
    playsound(list[n])

