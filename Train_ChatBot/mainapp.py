import json 
import requests
import pandas as pd
import speech_recognition as sr
from ModelTraining import extract_departure_destination

with open('traincodelist.json') as station_list:
    station_data = json.load(station_list)

name_to_code = {stationcode['name'].lower():stationcode['code'] for stationcode in station_data['data']}

def get_code(name):
    return name_to_code.get(name.lower())

def get_train_data(departure, destination):
    departurecode = get_code(departure)
    destinationcode = get_code(destination)
    url = f'https://indian-railway-api.cyclic.app/trains/betweenStations/?from={departurecode}&to={destinationcode}'
    response = requests.get(url)
    data = json.loads(response.text)
    stationdetails = data['data']
    traindata = []
    for traindetail in stationdetails:
        data = dict(
            train_number=traindetail['train_base']['train_no'],
            train_name=traindetail['train_base']['train_name'],
            departure_time=traindetail['train_base']['from_time'],
            destination_time=traindetail['train_base']['to_time'],
            travel_duration=traindetail['train_base']['travel_time']
        )
        traindata.append(data)
    df = pd.DataFrame(traindata)
    print(df)

def text_input():
    query = input("Enter your query (departure and destination): ")
    return query

def speech_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say your query:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio)
        print("You said:", query)
        return query
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand what you said.")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

def main():
    print("Choose input method:")
    print("1. Text")
    print("2. Speech")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        query = text_input()
    elif choice == "2":
        query = speech_input()
    else:
        print("Invalid choice.")
        return

    if query:
        departure, destination = extract_departure_destination(query)
        if departure is not None and destination is not None:    
            get_train_data(departure, destination)
        else:
            print('No Trains Available')

if __name__ == "__main__":
    main()
