import os.path

import numpy as np
def generate_clip_embedding(cfg,checkpoint_dir,class_id_list,class_name_list):
    clip_embeddings = np.zeros((len(class_id_list), 512))

    for idx,class_name in enumerate(class_name_list):
        class_embedding=np.load(os.path.join(cfg["clip_embedding_root"],class_name+".npy"))
        clip_embeddings[idx,:]=class_embedding

    np.save(os.path.join(checkpoint_dir,"clip_embedding.npy"),clip_embeddings)

    return clip_embeddings