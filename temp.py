import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import webbrowser
import datetime

# Load your model, labels, and other necessary components
model = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Ferver")

if "run" not in st.session_state:
    st.session_state["run"] = "true"

try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not emotion:
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Define a list of emotions to evaluate
emotions = ["happy", "sad", "angry", "surprise"]

class EmotionProcessor:
    def __init__(self):
        self.video_displayed = True
        self.performance_metrics = []

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1,-1)

            pred = label[np.argmax(model.predict(lst))]

            # Inside the if block where you make a prediction
            for emotion in emotions:
                true_emotion = emotion.capitalize()  # Assuming labels are in title case
                accuracy = accuracy_score([true_emotion], [pred])
                precision = precision_score([true_emotion], [pred], average='micro')
                recall = recall_score([true_emotion], [pred], average='micro')
                f1 = f1_score([true_emotion], [pred], average='micro')

                performance_str = f"Emotion: {emotion}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}"
                self.performance_metrics.append({
                    'emotion': emotion,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                })
                print(performance_str)

            cv2.putText(frm, pred, (50,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)

            np.save("emotion.npy", np.array([pred]))

            # Hide video stream
            self.video_displayed = False

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
                                landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
                                connection_drawing_spec=drawing.DrawingSpec(thickness=1))
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

processor = EmotionProcessor()

# The rest of your code remains unchanged...
# Define a dictionary of artists by language
language_artists = {
    "Hindi": ["Lata Mangeshkar", "Arijit Singh", "Kishore Kumar", "Shreya Ghoshal", "Rahat Fateh Ali Khan", "Neha Kakkar", "Sonu Nigam", "Mohammed Rafi", 
				"Sunidhi Chauhan", "Atif Aslam", "Kumar Sanu", "Alka Yagnik", "Asha Bhosle", "Udit Narayan", "S.P. Balasubrahmanyam", "Anuradha Paudwal", 
				"Amit Kumar", "Hariharan", "Kailash Kher", "Kavita Krishnamurthy", "Shaan", "Kunal Ganjawala", "Sukhwinder Singh", "Javed Ali", "Shankar Mahadevan",
				"Amaal Mallik", "Shalmali Kholgade", "Neeti Mohan", "Palak Muchhal", "Monali Thakur", "Badshah", "Yo Yo Honey Singh", "Daler Mehndi", "Mika Singh",
				"Guru Randhawa", "Jubin Nautiyal", "Himesh Reshammiya", "Vishal Dadlani", "Shekhar Ravjiani", "Rahul Vaidya", "Asees Kaur", "Neha Bhasin", 
				"Shirley Setia", "Jyotica Tangri", "Darshan Raval", "Armaan Malik", "Prakriti Kakar", "Tulsi Kumar", "Jonita Gandhi", "Adnan Sami", "Kavita Seth", 
				"Benny Dayal", "Mohit Chauhan", "Abhijeet", "Aditya Narayan", "Jagjit Singh", "Suresh Wadkar", "Papon", "Kunal Ganjawala", "Sadhana Sargam", 
				"Usha Uthup", "Kanika Kapoor", "Kailash Kher", "Harshdeep Kaur", "K. S. Chithra", "Richa Sharma", "Kumar Sanu", "Amit Trivedi", "Sukriti Kakar", 
				"Ankit Tiwari", "Nakash Aziz", "Jassie Gill", "Mohammed Irfan","K. J. Yesudas", "S. P. Balasubrahmanyam", "P. Susheela", "Vani Jairam", "K. S. Chithra",
				"Hariharan", "Unnikrishnan", "S. Janaki", "S. Janaki", "Sujatha Mohan", "Karthik", "Shreya Ghoshal",
				"Kailash Kher", "Vijay Yesudas", "G. V. Prakash Kumar", "Andrea Jeremiah", "Anuradha Sriram", "Hemachandra",
				"Chinmayi", "Hari Charan", "Chinna Ponnu", "Madhu Balakrishnan", "Madhumitha", "M. G. Sreekumar", "Gopika Poornima",
				"Mano", "Kavita Paudwal", "Sangeetha", "Krish", "P. Jayachandran", "Suchitra Karthik Kumar", "Shweta Mohan",
				"Ranjith", "G. K. Venkatesh", "Vijay Prakash", "Kousalya", "Srinivas", "Rita Thyagarajan", "Rita", "Sailaja",
				"S. P. Sailaja", "Vidhu Prathap", "Swetha Mohan", "Rajalakshmi", "Swarnalatha", "Swarnalatha", "Suresh Peters",
				"T. L. Maharajan", "Karthik", "Anuradha Sriram"],
    "English": ["Ed Sheeran", "Adele", "Taylor Swift", "Justin Bieber", "Lady Gaga", "Bruno Mars","Katy Perry", 
				"Rihanna", "Sam Smith", "Coldplay", "Shawn Mendes", "Alicia Keys", "Drake", "Beyoncé", "John Legend", "Sia", "Billie Eilish", 
				"Elton John", "Whitney Houston", "Dua Lipa", "Maroon 5", "Ariana Grande", "Chris Brown", "Madonna", "The Weeknd", "Jason Derulo", "Pink", 
				"Michael Jackson", "Mariah Carey", "Selena Gomez", "Usher", "Miley Cyrus", "Lana Del Rey", "Khalid", "Christina Aguilera", "George Michael", 
				"Nicki Minaj", "David Bowie", "Eminem", "Rita Ora", "Lorde", "Zayn Malik", "Calvin Harris", "Jennifer Lopez", "Harry Styles", "Kelly Clarkson", 
				"SZA", "Imagine Dragons", "One Direction", "Tina Turner", "Stevie Wonder", "Gwen Stefani", "Ella Fitzgerald", "Prince", "Aretha Franklin", 
				"Frank Sinatra", "Justin Timberlake", "Britney Spears", "Elvis Presley", "Enrique Iglesias", "Meghan Trainor", "Charlie Puth", 
				"Justin Timberlake", "Dolly Parton", "George Ezra", "Olivia Rodrigo", "Neil Diamond", "Kesha", "Avril Lavigne", "Norah Jones", 
				"The Beatles", "Hozier", "Lizzo", "Rascal Flatts", "Camila Cabello", "Julia Michaels", "Jason Mraz", "Celine Dion", "Troye Sivan", 
				"Bon Jovi", "Alanis Morissette", "Annie Lennox", "John Mayer", "Randy Newman", "Sheryl Crow", "Mumford & Sons", "Tom Petty", 
				"Demi Lovato", "Anne-Marie", "Christine and the Queens", "The Killers", "Halsey", "James Blunt", "John Lennon", "Sting", 
				"Carly Rae Jepsen", "The Rolling Stones", "Sade", "Randy Travis", "Cyndi Lauper", "Robbie Williams", "Tom Jones", 
				"Liam Payne", "Duran Duran", "Tina Arena", "Stevie Nicks", "Paul McCartney", "Tracy Chapman", "Billy Joel", "Alan Jackson", "Marvin Gaye"]
}

lang = st.selectbox("Language", list(language_artists.keys()))
if lang:
    singer_options = language_artists[lang]
    singer = st.selectbox("Singer", singer_options)
    
    # Display video stream
    video_display = st.empty()
    webrtc_ctx = webrtc_streamer(key="key", desired_playing_state=True, video_processor_factory=lambda: processor)

    if not processor.video_displayed:
        webrtc_ctx.video_transformer(
            fps=0.1,
            key_points=lambda frame: frame,
        )

    btn = st.button("Recommend me songs")
    if btn:
        if not emotion:
            st.warning("Let me capture your emotion first")
            st.session_state["run"] = "true"
        else:
            search_query = f"{lang}+{emotion}+song+{singer}"
            webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")
            np.save("emotion.npy", np.array([""]))
            st.session_state["run"] = "false"

    # Hide video stream after emotion is captured
    if not processor.video_displayed:
        video_display.empty()

# ...
# [Your previous code]

st.write("Developed by Al-Ain")

# Debugging: Print the length of processor.performance_metrics
print("Length of performance_metrics:", len(processor.performance_metrics))

# Check if processor.performance_metrics is empty
if len(processor.performance_metrics) > 0:
    accuracies = [metric['accuracy'] for metric in processor.performance_metrics if 'accuracy' in metric and not np.isnan(metric['accuracy'])]
    precisions = [metric['precision'] for metric in processor.performance_metrics if 'precision' in metric and not np.isnan(metric['precision'])]
    recalls = [metric['recall'] for metric in processor.performance_metrics if 'recall' in metric and not np.isnan(metric['recall'])]
    f1_scores = [metric['f1'] for metric in processor.performance_metrics if 'f1' in metric and not np.isnan(metric['f1'])]

    avg_accuracy = np.mean(accuracies) if accuracies else 0
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_f1 = np.mean(f1_scores) if f1_scores else 0

    avg_metrics_str = f"Average Accuracy: {avg_accuracy}\nAverage Precision: {avg_precision}\nAverage Recall: {avg_recall}\nAverage F1-Score: {avg_f1}"

    # Save metrics to a text file with timestamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%d-%m-%Y_%H:%M:%S")
    file_path = f"performanceMatrix.txt"
    with open(file_path, "a") as file:
        file.write(timestamp)
        file.write(avg_metrics_str)

    print(avg_metrics_str)
    print(f"Metrics saved to {file_path}")