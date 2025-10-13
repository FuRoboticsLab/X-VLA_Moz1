python -m eval.agibot.agibot_server \
    --pretrained /home/agiuser/AIR-DREAM/checkpoints/challenge-1002/rel_ee/ckpt-final/model.safetensors \
    --device cuda:0 \
    --action_mode ee6d \
    --port 7000 \
    --use_proprio 0 \
    --use_local_vlm /home/agiuser/.cache/huggingface/hub/models--microsoft--Florence-2-large/snapshots/21a599d414c4d928c9032694c424fb94458e3594