import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import numpy as np
import os
import joblib
import pandas as pd
import threading
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageTk
import customtkinter as ctk
from ttkthemes import ThemedStyle

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


class SignLanguageApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Sign Language Interpreter")
        self.geometry("1200x700")
        ctk.set_appearance_mode("dark")
        self.configure(bg="#1a1a1a")

        # Initialize variables
        self.cap = None
        self.running = False
        self.model = None
        self.scaler = None

        # Setup UI
        self.create_widgets()
        self.load_model()

    def create_widgets(self):
        # Style configuration
        style = ThemedStyle(self)
        style.set_theme("arc")

        # Header
        header_frame = ctk.CTkFrame(self, height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        title_label = ctk.CTkLabel(
            header_frame,
            text="AI-Powered Sign Language Interpreter",
            font=("Helvetica", 24, "bold"),
            text_color="#00ffff"
        )
        title_label.pack(pady=20)

        # Main content frame
        content_frame = ctk.CTkFrame(self)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left control panel
        control_frame = ctk.CTkFrame(content_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Video display
        self.video_frame = ctk.CTkFrame(content_frame)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Control buttons
        btn_style = {"width": 250, "height": 50, "font": ("Arial", 14)}

        ctk.CTkButton(
            control_frame,
            text="📊 Collect Data",
            command=self.open_data_collection,
            fg_color="#3366cc",
            hover_color="#25478c",
            **btn_style
        ).pack(pady=15)

        ctk.CTkButton(
            control_frame,
            text="🧠 Train Model",
            command=self.train_model,
            fg_color="#28a745",
            hover_color="#1e7e34",
            **btn_style
        ).pack(pady=15)

        ctk.CTkButton(
            control_frame,
            text="▶️ Start Interpreter",
            command=self.toggle_interpreter,
            fg_color="#ffc107",
            hover_color="#e0a800",
            text_color="#000000",
            **btn_style
        ).pack(pady=15)

        # Status bar
        self.status_bar = ctk.CTkLabel(
            self,
            text="Ready",
            anchor="w",
            font=("Arial", 12),
            text_color="#999999"
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

    def load_model(self):
        try:
            if os.path.exists("asl_svm_model.pkl") and os.path.exists("scaler.pkl"):
                self.model = joblib.load("asl_svm_model.pkl")
                self.scaler = joblib.load("scaler.pkl")
                self.status("Model loaded successfully")
            else:
                self.status("Model files not found - Please train model first")
        except Exception as e:
            self.status(f"Error loading model: {str(e)}")

    def status(self, message):
        self.status_bar.configure(text=message)

    def open_data_collection(self):
        DataCollectionWindow(self)

    def train_model(self):
        def train():
            try:
                data_files = [f for f in os.listdir() if f.startswith("data_") and f.endswith(".csv")]
                if not data_files:
                    self.status("No data files found. Please collect data first.")
                    return

                self.status("Loading data...")
                df = pd.concat([pd.read_csv(f, header=None) for f in data_files], ignore_index=True)

                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                self.status("Training SVM classifier...")
                model = SVC(kernel='linear', probability=True)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)

                acc = accuracy_score(y_test, model.predict(X_test))
                self.status(f"Model trained with {acc * 100:.2f}% accuracy")

                joblib.dump(model, "asl_svm_model.pkl")
                joblib.dump(scaler, "scaler.pkl")
                self.load_model()

            except Exception as e:
                self.status(f"Training error: {str(e)}")

        threading.Thread(target=train).start()

    def toggle_interpreter(self):
        if not self.running:
            if self.model is None or self.scaler is None:
                messagebox.showerror("Error", "Model not loaded. Please train the model first.")
                return

            self.running = True
            self.cap = cv2.VideoCapture(0)
            self.update_video_feed()
            self.status("Interpreter running")
        else:
            self.running = False
            if self.cap:
                self.cap.release()
            self.status("Interpreter stopped")

    def update_video_feed(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                predicted_sign = ""

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks = np.array(landmarks).reshape(1, -1)
                        landmarks_scaled = self.scaler.transform(landmarks)
                        predicted_sign = self.model.predict(landmarks_scaled)[0]
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Draw prediction
                cv2.putText(frame, f"Sign: {predicted_sign}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Convert to PIL image
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.thumbnail((self.video_frame.winfo_width(), self.video_frame.winfo_height()))
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=img_tk)
                self.video_label.image = img_tk

            self.after(10, self.update_video_feed)


class DataCollectionWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Data Collection")
        self.geometry("500x300")
        self.resizable(False, False)

        self.create_widgets()

    def create_widgets(self):
        ctk.CTkLabel(self, text="Collect Training Data", font=("Arial", 18, "bold")).pack(pady=15)

        form_frame = ctk.CTkFrame(self)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)

        # Label input
        ctk.CTkLabel(form_frame, text="Sign Label:").pack(anchor="w")
        self.label_entry = ctk.CTkEntry(form_frame, placeholder_text="Enter sign (A-Z, 0-9)")
        self.label_entry.pack(fill=tk.X, pady=(0, 15))

        # Sample count
        ctk.CTkLabel(form_frame, text="Number of Samples:").pack(anchor="w")
        self.samples_spinbox = tk.Spinbox(
            form_frame,
            from_=50,
            to=500,
            increment=10,
            font=("Arial", 14),
            background="#2a2a2a",
            foreground="white"
        )
        self.samples_spinbox.pack(fill=tk.X, pady=(0, 15))

        # Start button
        ctk.CTkButton(
            form_frame,
            text="Start Collection",
            command=self.start_collection,
            fg_color="#3366cc",
            hover_color="#25478c"
        ).pack(pady=20)

    def start_collection(self):
        label = self.label_entry.get().strip()
        samples = int(self.samples_spinbox.get())

        if not label:
            messagebox.showerror("Error", "Please enter a sign label")
            return

        self.destroy()
        self.parent.status(f"Collecting {samples} samples for '{label}'")

        # Run collection in separate thread
        threading.Thread(target=self.collect_data, args=(label, samples)).start()

    def collect_data(self, label, num_samples):
        cap = cv2.VideoCapture(0)
        data_file = f"data_{label}.csv"

        with open(data_file, 'w') as f:
            collected = 0
            while collected < num_samples and self.parent.running:
                ret, frame = cap.read()
                if not ret:
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        landmarks_str = ",".join(map(str, landmarks))
                        f.write(f"{landmarks_str},{label}\n")
                        collected += 1
                        self.parent.status(f"Collected sample {collected}/{num_samples} for {label}")

                # Display collection progress
                cv2.putText(frame, f"Collecting: {label} | {collected}/{num_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Data Collection", frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.parent.status(f"Data collection completed for {label}")


if __name__ == "__main__":
    app = SignLanguageApp()
    app.mainloop()