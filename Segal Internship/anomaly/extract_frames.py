import cv2, os, glob

video_folder = "UCSDped2/Test"
output_root = "UCSDped2/Test_frames"
os.makedirs(output_root, exist_ok=True)

video_paths = glob.glob(os.path.join(video_folder, "*.avi"))
print(f"Found {len(video_paths)} test videos.")

for video_path in video_paths:
    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        fname = os.path.join(out_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(fname, frame)
        idx += 1
    cap.release()
    print(f"{name}: {idx-1} frames extracted")
print("Done extracting frames")

