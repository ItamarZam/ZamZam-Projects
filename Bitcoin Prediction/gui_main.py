# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 08:51:36 2025

@author: itama

"""

from bitcoin_data_creation import generate_single_graph, initialize_client
import tkinter as tk
from model import run_model_gui
from tkinter import filedialog, messagebox, ttk
import numpy as np
from PIL import Image
from PIL import ImageTk
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import os
import threading
import random  
import tensorflow as tf
from tensorflow.keras.saving import register_keras_serializable
import tkinter.simpledialog as simpledialog
import graphviz


api_key = None
api_secret = None

client = initialize_client(api_key, api_secret)
image_size_x = 224
image_size_y = 224
channels = 3
mobilenet_feature_dim = 1280
fun_facts = [
    "Bitcoin was created in 2009!",
    "The smallest unit of Bitcoin is called a Satoshi.",
    "There will only ever be 21 million Bitcoins.",
    "Bitcoin transactions are recorded on a public ledger called the blockchain.",
    "The creator of Bitcoin is known as Satoshi Nakamoto.",
    "The first ever Bitcoin transaction was made for two Pizzas:)",
    "Bitcoin peak as of right now is 114,000$"
]

mobilenet_layer = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", trainable=False)


@register_keras_serializable()
def apply_mobilenet_per_frame(x):
    batch_size = tf.shape(x)[0]
    steps = tf.shape(x)[1]
    x_new = tf.reshape(x, (-1, image_size_x, image_size_y, channels))
    features = mobilenet_layer(x_new)
    features = tf.reshape(features, (batch_size, steps, mobilenet_feature_dim))
    return features


try:
    model = load_model(
        'C:/Users/itama/Bitcoin Prediction/gui/model.keras',
        custom_objects={'apply_mobilenet_per_frame': apply_mobilenet_per_frame}
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Model Loading Error: Failed to load model: {e}")
    model = None
    
    
def upload_m():
    global model
    m_path=filedialog.askopenfilename(title="Select model file", filetypes=[("keras models", "*.h5 *.keras *.tf"), ("All Files","*.*")])
    if not m_path:
        return
    try:
        new_model= load_model(m_path,custom_objects={'apply_mobilenet_per_frame': apply_mobilenet_per_frame,'KerasLayer': hub.KerasLayer})
        model=new_model
        model_status.config(text=f"Model Loaded: {os.path.basename(m_path)}")
        messagebox.showinfo("Success","Model Loaded")
    except Exception as e:
        messagebox.showerror("Error ",f"Failed to load the model:{str(e)}")
        model_status.config(text="model failed to load")
    
root = tk.Tk()
root.title("BTC Prediction App")
root.geometry("500x300")
root.configure(bg="#161616")

s = ttk.Style()
s.theme_use("clam")  
s.configure("TFrame", background="#161616")
s.configure("TLabel", background="#161616", foreground="#FFD700", font=("Segoe UI", 12))
s.configure("TButton", background="#232323", foreground="#FFD700", font=("Segoe UI", 12, "bold"))

def train_model():
    try:
        accuracy,loss=run_model_gui()
        messagebox.showinfo("Test Accuracy:", accuracy)
        messagebox.showinfo("Test Loss:", loss)
    except Exception as e: 
        messagebox.showinfo("error has accured")

def end_app():
    if messagebox.askokcancel("Quit", "Do you really wish to quit?"):
        root.destroy()

def show_fun_fact(event=None):
    fact= random.choice(fun_facts)
    messagebox.showinfo("Fun Fact", fact)

def pulse_lbl(lbl, count=0):  
    colors = ["#FFD700", "#FFFACD", "#FFD700"]
    lbl.config(foreground=colors[count % len(colors)])
    if count < 6:
        lbl.after(100, pulse_lbl, lbl, count + 1)
        
def upload_predict_image():
    global img_tk # keeping reference to avoid garbage collection
    if model is None:
        prediction_lbl.config(text="Model Not Loaded: Cannot predict.") 
        return
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])

    if not file_path:
        return

    try:
        show_loading()  
        img= Image.open(file_path).convert( 'RGB') 
        img= img.resize((image_size_x, image_size_y)) 
        img_tk= ImageTk.PhotoImage(img)
        image_lbl.config(image= img_tk)
        img_array= np.array(img) / 255.0
        img_array= np.expand_dims(img_array , axis= 0)
        seq_len= 12
        img_seq= np.tile(img_array, (1, seq_len, 1, 1, 1))
        pred = model.predict(img_seq)
        confidence = float(pred[0][0])
        if confidence > 0.5:
            predicted='Up'
        else:
            predicted= 'Down'

        output_text= f"prediction: {predicted} (Confidence: {confidence:.2f})" 
        prediction_lbl.config(text=output_text)
        pulse_lbl(prediction_lbl)
        history_listbox.insert(0, output_text)
    except Exception as e:
        prediction_lbl.config(text=f"Prediction Error: {e}")
    finally:
        hide_loading() 
        
        
def graph_parameters():
    symbol= simpledialog.askstring("Symbol", "Enter your market symbol(BTCUSD)", parent= root )
    if not symbol:
        return
    time_frame = simpledialog.askstring("Time Frame", "Enter you wished time frame(1h)", parent= root)
    if not time_frame:
        return
    start_date=simpledialog.askstring("Start Full Date","Enter strat date (YYYY-MM-DD HH:MM:SS" )
    if not start_date:
        return
    
    try: 
        show_loading()
        filepath= generate_single_graph(client,symbol,time_frame,start_date)
        hide_loading()
        
        if filepath:
            messagebox.showinfo("Success", f"Graph saved to:\n{filepath}")
        else:
            messagebox.showwarning("No Data", "No data available for the selected timeframe.")
    except Exception as e:
        hide_loading()
        messagebox.showerror("Error", f"Failed to generate graph:\n{str(e)}")
        


main_frame= ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

try:
    logo_img= Image.open("bitcoin_logo.png").resize((60, 60))
    logo_tk=ImageTk.PhotoImage(logo_img)
    logo_lbl= ttk.Label(main_frame, image=logo_tk, background="#161616")
    logo_lbl.image= logo_tk  
    logo_lbl.pack(pady= 5)
except Exception as e:
    print(f"Could not load logo: {e}")
    logo_lbl= ttk.Label(main_frame, text="BTC", background="#161616", foreground="#FFD700", font=("Segoe UI", 28, "bold"))
    logo_lbl.pack(pady=5)

info_lbl= ttk.Label(main_frame, text= "Click to predict from uploaded image")
info_lbl.pack(pady=20)

upload_button= ttk.Button(main_frame, text="Upload Image" , command= upload_predict_image)
upload_button.pack(pady= 10) 
def on_enter(e):
    upload_button.config(style= "Hover.TButton")
def on_leave(e):
    upload_button.config(style= "TButton")
def on_press(e):
    upload_button.config(style= "Pressed.TButton")
def on_release(e):
    upload_button.config(style= "Hover.TButton")

s.configure("Hover.TButton", background="#FFD700", foreground= "#232323")
s.configure("Pressed.TButton", background= "#FFA500" , foreground="#232323")

history_lbl= ttk.Label(main_frame, text="Prediction History:")
history_lbl.pack(pady=(10,0))

history_listbox= tk.Listbox(main_frame, width= 40, height= 6 )
history_listbox.pack(pady=(0,10))

model_button=ttk.Button(main_frame,text="Upload custom model", command=upload_m)
model_button.pack(pady=10)


history_scroll= ttk.Scrollbar( main_frame, orient= "vertical", command= history_listbox.yview)
history_listbox.config( yscrollcommand= history_scroll.set)
history_scroll.pack(side= "right", fill= "y")


upload_button.bind("<Enter>" , on_enter)
upload_button.bind("<Leave>" , on_leave)
upload_button.bind("<ButtonPress-1>" , on_press)
upload_button.bind("<ButtonRelease-1>", on_release)

end_button= ttk.Button(main_frame, text="End" , command= end_app)
end_button.pack(pady= 10)
s.configure("End.TButton", background="#c0392b" , foreground= "#fff" , font= ("Segoe UI", 12, "bold"))
end_button.config(style="End.TButton")

model_status= ttk.Label(main_frame, text= "Model Status: Loading.....")
model_status.pack( pady= 5)

prediction_lbl= ttk.Label(main_frame , text ="" , font =("Arial", 12))  
prediction_lbl.pack(pady= 10) 

image_lbl= ttk.Label(main_frame)
image_lbl.pack(pady= 10)

graph_button= ttk.Button(main_frame, text= "Generate Candlestick Graph", command= graph_parameters)
graph_button.pack(pady=10)

train_button= ttk.Button(main_frame,text="Train the model",command=run_model_gui)
train_button.pack(pady=10)

root.bind('<Control-e>', show_fun_fact)
root.focus_set()

prog = ttk.Progressbar(main_frame , mode= 'indeterminate')

def show_loading():
    prog.pack(pady= 10)
    prog.start()
def hide_loading():
    prog.stop()
    prog.pack_forget()
    
if model is not None:
    model_status.config(text= "Model Loaded with Success!")
else:
    model_status.config(text= "Model Failed to Load:)")

    

root.mainloop()

"""


allows uploading a adataset which I don't see why it is needed, also I might want to add to the train model some epoch and stuff shit
def upload_predict_image():
    global img_tk  # keeping reference to avoid garbage collection
    if model is None:
        prediction_lbl.config(text="Model Not Loaded: Cannot predict.")
        return
    file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_paths:
        return
    try:
        show_loading()
        for file_path in file_paths:
            img = Image.open(file_path).convert('RGB')
            img = img.resize((image_size_x, image_size_y))
            img_tk = ImageTk.PhotoImage(img)
            image_lbl.config(image=img_tk)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            seq_len = 12
            img_seq = np.tile(img_array, (1, seq_len, 1, 1, 1))
            pred = model.predict(img_seq)
            confidence = float(pred[0][0])
            if confidence > 0.5:
                predicted = 'Up'
            else:
                predicted = 'Down'
            output_text = f"prediction: {predicted} (Confidence: {confidence:.2f})"
            prediction_lbl.config(text=output_text)
            pulse_lbl(prediction_lbl)
            history_listbox.insert(0, f"{os.path.basename(file_path)}: {output_text}")
    except Exception as e:
        prediction_lbl.config(text=f"Prediction Error: {e}")
    finally:
        hide_loading()
"""