# TinyML Model Training & TensorFlow.js Conversion

This project demonstrates how to train a lightweight TensorFlow model and convert it into a TensorFlow.js graph model for browser-based inference. The workflow includes installing dependencies, training the model, and exporting it for web deployment.

## Installation

Install the required Python packages:

pip install tensorflow==2.13 tensorflowjs
pip install pdf2image pillow

## Train the Model

Run the training script to generate the Keras `.h5` model:

python train.py

This will produce a model file such as:

tiny_model.h5

## Convert Model to TensorFlow.js Format

Use the TensorFlow.js converter to generate a browser-ready model with float16 quantization:

tensorflowjs_converter --input_format=keras --quantize_float16 --output_format=tfjs_graph_model tiny_model.h5 <tiny_ml_model/js>

The converted model can then be loaded directly in web applications using TensorFlow.js.

## Output Structure Example

tiny_ml_model/
└── js/
    ├── model.json
    ├── group1-shard1of5.bin
    ├── group1-shard2of5.bin
    └── ...

## Summary

- Train a TinyML model using TensorFlow  
- Export it as a `.h5` Keras model  
- Convert it to TensorFlow.js for browser inference  
- Use float16 quantization to reduce model size and improve efficiency
