import numpy as np
import onnxruntime as ort
import numpy as np
import subprocess as sp
import time

FFMPEG_PATH = r"C:\ffmpeg-2026-02-04-git-627da1111c-full_build\bin\ffmpeg.exe"

INPUT_VIDEO = r"C:\proj\2d_to_3d\videos\bbc_02_clip.mp4"
ONNX_PATH = r"scene_detection\transnetv2.onnx"
PADDED_FRAMES = 25 # how many frames we repeat the first and last frame

# load model
try:
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Could not load model \n{e}")

input_info = session.get_inputs()[0]
input_shape = input_info.shape
W = input_info.shape[3]
H = input_info.shape[2]
start_ffmpeg = time.time()

# resize the video
command_resize = [FFMPEG_PATH,
            '-f', 'mp4', # expect mp4 as input format
            '-i', INPUT_VIDEO,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', 
            '-vf', f'scale={W}:{H}', # scale to 48x27 to match requirement
            'pipe:1']

pipe = sp.Popen(command_resize, stdout=sp.PIPE, bufsize=10**8)
data = pipe.stdout.read()

end_ffmpeg = time.time()

video_raw= np.frombuffer(data, dtype=np.uint8) # the pipe send uint8
video = video_raw.reshape((-1, H, W, 3))

pipe.stdout.close()
pipe.terminate()

# get first and last frames
first_frame = video[0]
last_frame = video[-1]

start_padding = np.tile(first_frame, (PADDED_FRAMES, 1, 1, 1))
end_padding = np.tile(last_frame, (PADDED_FRAMES, 1, 1, 1))

final_video = np.concatenate((start_padding, video, end_padding), axis=0)
final_video = final_video.astype(np.float32)

WINDOW_SIZE = 100
STRIDE = 50

all_predictions = []
chunk = []
left = 0
start = time.time()
for right in range(len(final_video)):
    chunk.append(final_video[right])
    if right - left == WINDOW_SIZE - 1:
        np_chunk = np.array(chunk)
        input_batch = np_chunk[np.newaxis, ...].astype(np.float32) # make (1, 100, 27, 48, 3) format

        # run prediction
        outputs = session.run(
            [o.name for o in session.get_outputs()], 
            {session.get_inputs()[0].name: input_batch}
        )
        out1 = outputs[0][0].flatten() # hard cuts
        out2 = outputs[1][0].flatten() # soft cuts
        #batch_preds = out1 if np.max(out1) > np.max(out2) else out2
        batch_preds = out1
        # we want the center 25 frames
        all_predictions.append(batch_preds[(WINDOW_SIZE - STRIDE) // 2 : ((WINDOW_SIZE - STRIDE) // 2) + STRIDE])
        # move left
        for _ in range(STRIDE):
            chunk.pop(0)
            left += 1

end = time.time()
final_predictions = np.concatenate(all_predictions, axis=0)
print(f"Total time for FFmpeg = {end_ffmpeg - start_ffmpeg}")
print(f"Total time for inference = {end - start}")

# generate result txt
output_file = "predictions.txt"

CUT_THRESHOLD = 0.5

scenes = []
start_scene_frame = 0
predictions_flat = final_predictions.flatten()
previous_frame_prob = 0
for i in range(len(predictions_flat)):    
    # check for cut
    if predictions_flat[i] > CUT_THRESHOLD and previous_frame_prob == 0: # save only if previous frame was a 0 (below threshold) to avoid long streches of frame cuts
        if i > 0: # prevents a bug if the video starts exactly on a cut
            scenes.append((start_scene_frame, i - 1))
        previous_frame_prob = 1
    if predictions_flat[i] <= CUT_THRESHOLD and previous_frame_prob == 1: 
        previous_frame_prob = 0
        start_scene_frame = i

# handle the final scene
last_frame_index = len(predictions_flat) - 1
if start_scene_frame <= last_frame_index:
    scenes.append((start_scene_frame, last_frame_index))

with open(output_file, "w") as f:
    for start, end in scenes:
        f.write(f"{start}\t{end}\n")

print(f"Max Probability: {np.max(final_predictions)}")
print(f"Average Probability: {np.mean(final_predictions)}")
print(f"Number of frames > 0.1: {np.sum(final_predictions > 0.1)}")
