# Machine learning: Mnist Project
# image recognition
# Mnist --> recognizing handwritten digits

# 1. get all data and resize them to all same size
# 2. convolution --> activation function + pooling
# 3. flatten + classify into a singular smaller matrix

# Classification vs regression
# classification: identify which group the object belongs in
# regression: predicition a value based on given data

# brainstorming:
# Canvas to draw
# draw bounding boxes around digit
# predict digit using model



# GUI for MNIST prediction
from tkinter import * # * means all
import tkinter as tk
from keras.models import load_model
from PIL import Image, ImageDraw
import numpy as np


model = load_model('Test_model.h5') # load the model

def paint(event): # coordiunates of cursor and draw from point
    x1, y1 = (event.x-1), (event.y-1)
    x2, y2 = (event.x+1), (event.y+1)
    canvas.create_oval(x1, y1, x2, y2, fill='black', width=20) # makes the cursor touch an oval


def handwriting_digits(): # saves the image of the shape
    canvas.postscript(file="digit.eps", colormode='color')
    print("digit saved!!")
    process_and_predict()
def process_and_predict(): # processes the image and puts it through the model
    image = Image.open("digit.eps")
    image = image.resize((28,28))
    image = image.convert('L')
    digit_array = np.array(image)
    digit_array = digit_array.reshape(1,28,28)
    prediction = model.predict(image)
    prediction_digit = np.argmax(prediction)

    print(prediction_digit)

# gui stuff
root = tk.Tk()
root.title("Handwritten Digits <3")

canvas = tk.Canvas(root, width=300, height=300, bg="white")
canvas.pack(expand=tk.YES, fill=tk.BOTH)
canvas.bind("<B1-Motion>", paint)

recognize_button = tk.Button(root, text = "Recognize", command=handwriting_digits)
recognize_button.pack(side=tk.RIGHT)

mainloop()

