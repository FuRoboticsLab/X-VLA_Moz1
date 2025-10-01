python -m eval.agibot.agibot_server \
    --pretrained /home/admin123/xvla/model/rel_joint/ckpt-100000/model.safetensors \
    --device cuda:0 \
    --action_mode agibot_joint \
    --port 7000 \
    --use_proprio 1 \
    --use_local_vlm /home/admin123/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594