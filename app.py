import tensorflow as tf
import numpy as np
from tkinter import *
from PIL import Image, ImageDraw, ImageOps

# Load pre-trained model
try:
    model = tf.keras.models.load_model("digit_recognizer_model.h5")
    print("Model loaded successfully!")
except:
    print("Error: Model file 'digit_recognizer_model.h5' not found.")
    print("Run 'model_trainer.py' first to train and save the model.")
    exit()

# GUI Setup
class App(Tk):
    def __init__(self):
        super().__init__()
        self.title("Handwritten Digit Recognizer")
        self.canvas_width = 200
        self.canvas_height = 200

        # Canvas for drawing
        self.canvas = Canvas(self, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=10)

        # Label for prediction
        self.prediction_label = Label(self, text="Draw a digit and click Predict", font=("Helvetica", 16))
        self.prediction_label.pack(pady=10)

        # Buttons
        btn_frame = Frame(self)
        btn_frame.pack()

        btn_predict = Button(btn_frame, text="Predict", command=self.predict_digit, width=10, bg="green", fg="white")
        btn_predict.pack(side=LEFT, padx=5, pady=5)

        btn_clear = Button(btn_frame, text="Clear", command=self.clear_canvas, width=10, bg="red", fg="white")
        btn_clear.pack(side=RIGHT, padx=5, pady=5)

        # PIL image for drawing
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image1)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Draw on canvas
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill=255)
        self.prediction_label.config(text="Draw a digit and click Predict")

    def predict_digit(self):
        img = self.image1.convert('L')
        img = ImageOps.invert(img)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        self.prediction_label.config(
            text=f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%", fg="blue"
        )


# Run app
if __name__ == "__main__":
    app = App()
    app.mainloop()
