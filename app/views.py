# myapp/views.py
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        image_path = fs.save(uploaded_file.name, uploaded_file)

        # Load the pre-trained model
        model = load_model('dl_model.h5')

        # Process the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Make a prediction
        prediction = model.predict(image)

        # Interpret the prediction
        result = "normal" if prediction[0][0] > prediction[0][1] else "affect"

        return render(request, 'result.html', {'result': result})
    return render(request, 'upload.html')

