import os
import torch
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# ================== INIT MODEL ==================
model = InceptionResnetV1(pretrained='vggface2').eval()

dataset_path = "dataset"
embeddings = {}

print("Generating embeddings... ??")

# ================== LOOP DATASET ==================
for user in os.listdir(dataset_path):
    user_path = os.path.join(dataset_path, user)

    if not os.path.isdir(user_path):
        continue

    user_embeddings = []

    for pose in os.listdir(user_path):
        pose_path = os.path.join(user_path, pose)

        for img_name in os.listdir(pose_path):
            img_path = os.path.join(pose_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((160,160))

                # convert ke tensor
                img = np.array(img) / 255.0
                img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0)

                # ================== EMBEDDING ==================
                with torch.no_grad():
                    emb = model(img)

                emb = emb.squeeze().numpy()
                user_embeddings.append(emb)

            except Exception as e:
                print(f"Error {img_path}: {e}")

    embeddings[user] = user_embeddings
    print(f"[DONE] {user} ? {len(user_embeddings)} embeddings")

# ================== SAVE ==================
with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)

print("?? Embedding saved to embeddings.pkl")
