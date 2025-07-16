import cv2, numpy as np, argparse
from utils import load_model, detect_players

class SimpleDB:
    def __init__(self): self.bank = []; self.next_id=0
    def match(self, hist, thr=0.3):
        if not self.bank: return None
        dists=[float(np.linalg.norm(hist-h)) for _,h in self.bank]
        m=min(dists); idx=dists.index(m)
        return self.bank[idx][0] if m<thr else None
    def add(self, hist):
        pid=self.next_id; self.bank.append((pid,hist)); self.next_id+=1; return pid

def color_hist(img): 
    h=cv2.calcHist([cv2.resize(img,(64,64))],[0,1,2],None,[8,8,8],[0,256]*3).flatten()
    return h/(np.linalg.norm(h)+1e-6)

def run(video_path, device="cpu"):
    db, model = SimpleDB(), load_model(device)
    cap=cv2.VideoCapture(video_path)
    print("Video opened:", cap.isOpened())
    while True:
        ok,frame=cap.read()
        if not ok:
            print("End of video or failed to read frames.")
            break
        players=detect_players(frame,model)
        print(f"Detected {len(players)} players in frame")
        for p in players:
            p["hist"]=color_hist(p["crop"])
            pid=db.match(p["hist"])
            if pid is None: pid=db.add(p["hist"])
            x1,y1,x2,y2=p["bbox"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(frame,f"ID{pid}",(x1,y1-5),0,0.6,(0,255,255),2)
        frame = cv2.resize(frame, (640, 360))  # Resize for visibility
        cv2.imshow("Brute-Force ReID",frame)
        if cv2.waitKey(1)==27: break
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--video_path", default="videos/15sec_input_720p.mp4")
    parser.add_argument("--device", default="cpu")
    run(**vars(parser.parse_args()))
