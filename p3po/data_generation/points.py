import os
from PIL import Image

import numpy as np
from IPython.display import display, Javascript
import ipywidgets as widgets
from ipycanvas import Canvas, hold_canvas
import pickle
import cv2
from pathlib import Path

import io
import asyncio
import logging

# Define an async function to wait for button click
async def wait_for_click(button):
    # Create a future object
    future = asyncio.Future()
    # Define the click event handler
    def on_button_clicked(b):
        future.set_result(None)
    # Attach the event handler to the button
    button.on_click(on_button_clicked)
    # Wait until the future is set
    await future

class Points():
    def __init__(self, env_name, img, coordinates_path, size_multiplier=1):
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Starting the Points class")
        self.img = img
        self.size_multiplier = size_multiplier
        self.coordinates_path = coordinates_path
        self.env_name = env_name

        # Save the image to a bytes buffer
        image = Image.fromarray(self.img)
        size = img.shape
        image = image.resize((size[1] * self.size_multiplier, size[0] * self.size_multiplier))
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)

        # Create an IPyWidgets Image widget
        self.canvas = Canvas(width=size[1] * self.size_multiplier, height=size[0] * self.size_multiplier)
        # Define the size of each cell

        self.canvas.put_image_data(np.array(image), 0, 0)

        # Display coordinates
        coords_label = widgets.Label(value="Click on the image to select the coordinates")

        # Define the click event handler
        self.coords = []
        def on_click(x, y):
            coords_label.value = f"Coordinates: ({x}, {y})"
            self.coords.append((0, x, y))

            with hold_canvas(self.canvas):
                self.canvas.put_image_data(np.array(image), 0, 0)  # Redraw the original image

                self.canvas.fill_style = 'red'
                for coord in self.coords:
                    x, y = coord[1] // self.size_multiplier, coord[2] // self.size_multiplier
                    self.canvas.fill_circle(x, y, 2)

        # Connect the click event to the handler
        self.canvas.on_mouse_down(on_click)

        self.button = widgets.Button(description="Save Points")

        # Display the widgets
        self.vbox = widgets.VBox([self.canvas, coords_label, self.button])

        # # Display the widget
        display(self.vbox)

    def on_done(self):
        logging.info("saving")
        Path(self.coordinates_path + "/coords/").mkdir(parents=True, exist_ok=True)
        with open(self.coordinates_path + "/coords/" + self.env_name + ".pkl", 'wb') as f:
            try:
                pickle.dump(self.coords, f)
            except Exception as e:
                logging.info(e)
        Path(self.coordinates_path + "/images/").mkdir(parents=True, exist_ok=True)
        with open(self.coordinates_path + "/images/" + self.env_name + ".png", 'wb') as f:
            try:
                image = Image.fromarray(self.img)
                image.save(f)
            except Exception as e:
                logging.info(e)
        logging.info("saved")