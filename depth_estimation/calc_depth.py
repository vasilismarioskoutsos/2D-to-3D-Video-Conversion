import numpy as np
import onnxruntime as ort
import numpy as np
import subprocess as sp
import time

FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"

INPUT_VIDEO = r"C:\proj\2d_to_3d\videos\bike.mp4"
ONNX_PATH = r"depth_estimation\video_depth_anything_vits_input518.onnx"

OUTPUT_FILE = r"C:\proj\2d_to_3d\videos\bike_depth.raw"

try:
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Could not load model \n{e}")

input_info = session.get_inputs()[0]
input_shape = input_info.shape
# both inputs 518
W = input_info.shape[4]
H = input_info.shape[3]

command = [FFMPEG_PATH,
            '-ss', '30', # skip the first 30 seconds
            '-t', '10', # stop reading after 10 seconds
            '-f', 'mp4',
            '-i', INPUT_VIDEO,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo',
            '-vf', f'scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2', # preserve aspect ratio without distorting the image
            'pipe:1']

pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)

# reshape raw bytes into video frames
frame_size = W * H * 3

# standart ImageNet mean and std
mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
std = np.array([0.229, 0.224, 0.225]).astype(np.float32)

FRAMES = 32
chunk = []
start = time.time()
# the keyframes should be 12 and 24 frames from the end of the previous chunk
# index values
tk0 = 8
tk1 = 20
t0 = 8
first_batch = True
with open(OUTPUT_FILE, 'wb') as f_out:
    while True:
        data = pipe.stdout.read(frame_size)
        if not data: # if there is no more data
            break
        video = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 3)
        chunk.append(video)
        if len(chunk) == FRAMES:
            np_chunk = np.array(chunk)
            np_chunk = np_chunk.transpose(0, 3, 1, 2) # put the dimensions in correct order
            input_batch = np_chunk[np.newaxis, ...].astype(np.float32) / 255.0 # convert each batch to save memory

            input_batch = (input_batch - mean[None, None, :, None, None]) / std[None, None, :, None, None]

            # run prediction
            outputs = session.run(
                [o.name for o in session.get_outputs()], 
                {session.get_inputs()[0].name: input_batch}
            )
            batch_preds = outputs[0][0]

            if first_batch:
                f_out.write(batch_preds.astype(np.float32).tobytes())
                first_batch = False
            else:
                f_out.write(batch_preds[10:].astype(np.float32).tobytes())

            keyframe0 = chunk[tk0]
            keyframe1 = chunk[tk1]
            new_chunk = [keyframe0, keyframe1] + chunk[FRAMES - t0:]
            chunk = new_chunk

    if len(chunk) > 0:
        original_len = len(chunk)

        # add padding to complete 32 frames
        padding = [chunk[-1]] * (FRAMES - original_len) # duplicate last frame
        chunk = chunk + padding

        np_chunk = np.array(chunk)
        np_chunk = np_chunk.transpose(0, 3, 1, 2) # put the dimensions in correct order
        input_batch = np_chunk[np.newaxis, ...].astype(np.float32) / 255.0 # convert each batch to save memory
        input_batch = (input_batch - mean[None, None, :, None, None]) / std[None, None, :, None, None]

        # run prediction
        outputs = session.run(
            [o.name for o in session.get_outputs()], 
            {session.get_inputs()[0].name: input_batch}
        )
        batch_preds = outputs[0][0]

        # save only valid frames
        if first_batch:
            f_out.write(batch_preds[:original_len].astype(np.float32).tobytes())
        else:
            valid_len = original_len - 10
            if valid_len > 0:
                f_out.write(batch_preds[10 : 10 + valid_len].astype(np.float32).tobytes())

pipe.stdout.close()
pipe.terminate()

end = time.time()
print(f"Total time for inference = {end - start}")