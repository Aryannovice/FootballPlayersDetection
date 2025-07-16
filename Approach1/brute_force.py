import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import cv2, numpy as np, itertools, argparse
from utils import load_model, detect_players

def color_histogram(img, size=(64,64)):
    img = cv2.resize(img, size)
    hist = cv2.calcHist([img],[0,1,2],None,[8,8,8],[0,256]*3).flatten()
    return hist / (np.linalg.norm(hist)+1e-6)

def similarity(pA, pB):
    
    return 0.5*np.sum(((pA-pB)**2)/(pA+pB+1e-6))

def brute_force_pairing(listA, listB):
    matches = []
    for i, j in itertools.product(range(len(listA)), range(len(listB))):
        d = similarity(listA[i]["hist"], listB[j]["hist"])
        matches.append((d, i, j))
    matches.sort()                      
    paired_B = set()
    result = {}
    for d,i,j in matches:
        if j in paired_B: continue
        result[i] = j
        paired_B.add(j)
    return result

def run(broadcast_path, tacticam_path, device="cpu"):
    print("Loading model...")
    model = load_model(device)
    print("Model loaded.")
    capA, capB = cv2.VideoCapture(broadcast_path), cv2.VideoCapture(tacticam_path)
    print(f"Opened videos: {broadcast_path}, {tacticam_path}")
    while True:
        retA, frameA = capA.read(); retB, frameB = capB.read()
        if not (retA and retB):
            print("End of video or failed to read frames.")
            break
        detA = detect_players(frameA, model); detB = detect_players(frameB, model)
    for p in detA: p["hist"] = color_histogram(p["crop"])
    for p in detB: p["hist"] = color_histogram(p["crop"])
    mapping = brute_force_pairing(detA, detB)
    # Draw IDs
    for idxA, player in enumerate(detA):
        x1,y1,x2,y2 = player["bbox"]
        cv2.rectangle(frameA,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frameA,f"ID{idxA}",(x1,y1-5),0,0.5,(0,255,0),2)
        if idxA in mapping:
            idxB = mapping[idxA]
            x1b,y1b,x2b,y2b = detB[idxB]["bbox"]
            cv2.rectangle(frameB,(x1b,y1b),(x2b,y2b),(0,255,0),2)
            cv2.putText(frameB,f"ID{idxA}",(x1b,y1b-5),0,0.5,(0,255,0),2)
    # RESIZE FRAMES HERE
    display_width, display_height = 640, 360
    frameA = cv2.resize(frameA, (display_width, display_height))
    frameB = cv2.resize(frameB, (display_width, display_height))
    cv2.imshow("Broadcast", frameA)
    cv2.imshow("Tacticam", frameB)
    key = cv2.waitKey(1)
    if key == 27:
        capA.release()
        capB.release() 
        cv2.destroyAllWindows()
        break

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--broadcast_path",default="videos/broadcast.mp4")
    parser.add_argument("--tacticam_path",default="videos/tacticam.mp4")
    parser.add_argument("--device",default="cpu")
    run(**vars(parser.parse_args()))
