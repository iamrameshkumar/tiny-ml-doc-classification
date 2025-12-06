pip install tensorflow==2.13 tensorflowjs
pip install pdf2image pillow

python train.py

tensorflowjs_converter --input_format=keras --quantize_float16 --output_format=tfjs_graph_model tiny_model.h5 <tiny_ml_model\js>
