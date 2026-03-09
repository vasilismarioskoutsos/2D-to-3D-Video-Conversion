import onnxruntime as ort
import numpy as np
import os

def check_transnet_model(model_path):
    
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"File Size: {size_mb:.2f} MB")

    # load model
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"Could not load model \n{e}")
        return

    # shape = [Batch, 100 Frames, 27 Height, 48 Width, 3 Channels]
    input_info = session.get_inputs()[0]
    input_shape = input_info.shape
    input_name = input_info.name
    print(f"Input Shape Detected: {input_shape}")
    print(f"Data type: {input_info.type}")

    # simulation
    # we create a fake block of video (all zeros) to see if it crashes
    print("Running dummy inference...")
    
    dummy_input = np.zeros((1, 32, 3, 518, 518), dtype=np.float32)

    print(input_info.shape[3])
    
    try:
        outputs = session.run(None, {input_name: dummy_input})
        
        predictions = outputs[0]
        print(f"Inference successful! Output shape: {predictions.shape}")
        
        # check if values are probabilitiess
        if np.isnan(predictions).any():
             print("Model output contains nans")
        else:
             print("The model is valid and working correctly")
             
    except Exception as e:
        print(f"Error during inference: {e}")

check_transnet_model(r"C:\proj\2d_to_3d\depth_estimation\video_depth_anything_vits_input518.onnx")