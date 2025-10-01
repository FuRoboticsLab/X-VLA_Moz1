# ------------------------------------------------------------------------------
# Copyright (c) 2022–∞, 2toINF
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0∂
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations
from json import encoder

from typing import Any, Dict, Tuple

import logging
import traceback

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from safetensors.torch import load_file
import json_numpy
from transformers import AutoModelForCausalLM
from components.transformer import SoftPromptedTransformer
from components.losses import EE6DLoss, JointLoss, AGIBOTJointLoss
from components.preprocessor import LanguagePreprocessor, ImagePreprocessor

LOSS_HUB = {
    "ee6d": EE6DLoss,
    "joint": JointLoss,
    "agibot_joint": AGIBOTJointLoss
}


class XVLA(nn.Module):
    """
    XVLA

    Overview
    --------
    - Visual-Language Encoder (VLMEncoder): encodes multi-view images and language tokens.
    - SoftPromptedTransformer: fuses (noisy) action sequences, proprioception, time, and VLM features.
    - Loss: fixed-parameter loss (EE6DLoss or JointLoss), no runtime config objects.

    Notes
    -----
    - Gripper channel indices are taken from the selected loss class (`.GRIPPER_IDX`).
    - During training/inference, gripper channels in proprio and noisy actions are zeroed.
    - At inference, a sigmoid is applied only on gripper channels.
    """

    def __init__(
        self,
        encoder_name: str = "microsoft/Florence-2-large",
        *,
        depth: int = 24,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_actions: int = 30,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        use_hetero_proj: bool = False,
        action_mode: str = "ee6d",
        use_proprio: bool = True,
        version: str = "v1"
    ):
        """
        Parameters
        ----------
        encoder_name : str
            Name or path passed to `VLMEncoder`.
        depth : int
            Transformer depth.
        hidden_size : int
            Transformer hidden size.
        num_heads : int
            Number of attention heads.
        num_actions : int
            Temporal length of the action sequence (T).
        num_domains : int
            Number of domains/task IDs for domain conditioning.
        len_soft_prompts : int
            Length of soft prompt tokens.
        use_hetero_proj : bool
            Whether to use heterogeneous projection heads in the transformer.
        action_mode : {"ee6d","joint"}
            Layout for actions/proprio; controls channel dimensions and loss type.
        use_proprio : bool
            Whether to use proprio info.
        """
        super().__init__()

        action_mode = action_mode.lower()
        assert action_mode in {"ee6d", "joint", "agibot_joint"}, "action_mode must be 'ee6d' or 'joint'"

        # Channel layout derived from mode
        self.criterion = LOSS_HUB[action_mode]()
        self.num_actions = num_actions
        self.use_proprio = use_proprio
        if not use_proprio: print(">>> disable proprioception <<<")
        
        # Modules
        assert 'Florence' in encoder_name, "Only microsoft/Florence-2-base and microsoft/Florence-2-large are supported."
        self.vlm = AutoModelForCausalLM.from_pretrained(
            encoder_name,
            torch_dtype="auto",
            trust_remote_code=True
        )
        # Remove decoder-specific components to reduce memory and ensure we only
        # use encoder pathways. Guard these in case internals change.
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head
                
        
        self.transformer = SoftPromptedTransformer(
            hidden_size=hidden_size,
            multi_modal_input_size = self.vlm.config.projection_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            num_domains=num_domains,
            dim_action=self.criterion.DIM_ACTION,
            dim_propio=self.criterion.DIM_ACTION,
            len_soft_prompts=len_soft_prompts,
            dim_time=32,
            max_len_seq=512,
            use_hetero_proj=use_hetero_proj,
        )
        

        # I/O preprocessors (implementations are project-specific)
        self.text_preprocessor = LanguagePreprocessor(encoder_name=encoder_name)
        self.image_preprocessor = ImagePreprocessor(version=version)
        self.app: FastAPI | None = None

    # ------------------------------ utilities -------------------------------
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V], bool or 0/1
    ) -> Dict[str, torch.Tensor]:
        """
        Produce VLM token features and auxiliary visual tokens from multi-view inputs.

        Parameters
        ----------
        input_ids : LongTensor, shape [B, L]
            Token ids for the text prompt/instruction.
        pixel_values : FloatTensor, shape [B, V, C, H, W]
            Image batch with V views per sample.
        image_mask : Tensor, shape [B, V]
            Mask indicating which views are valid (True/1) vs padded (False/0).

        Returns
        -------
        Dict[str, Tensor]
            {
              "vlm_features": FloatTensor [B, T_enc, D],  # encoder token sequence
              "aux_visual_inputs": FloatTensor [B, (V-1)*N, D]  # flattened features for views 1..V-1
            }
        """
        B, V = pixel_values.shape[:2]

        # Flatten views, select valid images, encode
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)                    # [B*V, C, H, W]

        num_valid = int(flat_mask.sum().item())
        assert num_valid > 0, "At least one image must be valid."

        valid_images = flat_images[flat_mask]                   # [#valid, C, H, W]
        valid_feats = self.vlm._encode_image(valid_images)    # [#valid, N, D]
        N, D = valid_feats.shape[1:]

        # Reconstruct dense [B, V, N, D] tensor
        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)        # [B, V, N, D]

        # Text embeddings
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids) # [B, L, D]

        # Merge first-view visual tokens with text embeddings
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],  # [B, N, D]
            inputs_embeds,         # [B, L, D]
        )

        # Run encoder to get token-level features
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask,
            inputs_embeds=merged_embeds,
        )[0]  # [B, T_enc, D]

        # Remaining views (1..V-1) flattened as auxiliary inputs
        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)  # [B, (V-1)*N, D]

        return {
            "vlm_features": enc_out,
            "aux_visual_inputs": aux_visual_inputs,
        }


    # ------------------------------ utilities -------------------------------
    def mask_gripper(self, proprio: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zero gripper channels in `proprio` and `action`.

        Parameters
        ----------
        proprio : Tensor, shape [B, dim_proprio]
        action : Tensor, shape [B, T, dim_action]

        Returns
        -------
        proprio_m : Tensor
            Proprio with gripper channels zeroed.
        action_m : Tensor
            Action with gripper channels zeroed.
        """
        idx = self.criterion.GRIPPER_IDX
        proprio_m = proprio.clone()
        action_m = action.clone()
        proprio_m[..., idx] = 0.0
        action_m[..., idx] = 0.0
        if not self.use_proprio: proprio_m = torch.zeros_like(proprio_m)
        return proprio_m, action_m

    # ------------------------------ training --------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.Tensor,
        proprio: torch.Tensor,
        action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Parameters
        ----------
        input_ids : LongTensor, shape [B, L]
            Token IDs for language input.
        image_input : FloatTensor, shape [B, V, C, H, W]
            Multi-view images.
        image_mask : Tensor, shape [B, V]
            Mask for views (1 = valid, 0 = pad).
        domain_id : LongTensor, shape [B]
            Domain/task IDs.
        proprio : Tensor, shape [B, dim_proprio]
            Proprioceptive state.
        action : Tensor, shape [B, T=num_actions, dim_action]
            Ground-truth action sequence.

        Returns
        -------
        Dict[str, Tensor]
            Loss dictionary (keys depend on loss type), including "total_loss".
        """
        enc = self.forward_vlm(input_ids=input_ids, pixel_values=image_input, image_mask=image_mask)

        B = input_ids.shape[0]
        # Per-sample time in (0, 1); avoid exactly 1.0
        time = (torch.rand(1, device=input_ids.device) + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        # Mix GT action with noise
        action_with_noise = torch.randn_like(action) * time.view(-1, 1, 1) + action * (1 - time).view(-1, 1, 1)

        # Mask gripper channels
        proprio_m, action_with_noise_m = self.mask_gripper(proprio, action_with_noise)

        # Predict actions
        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_with_noise_m,
            t=time,
            proprio=proprio_m,
            **enc,
        )
        return self.criterion(pred_action, action)

    # ------------------------------ inference -------------------------------
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Iterative denoising sampler (linear schedule).

        Parameters
        ----------
        input_ids : LongTensor, shape [B, L]
            Token IDs for language input.
        image_input : FloatTensor, shape [B, V, C, H, W]
            Multi-view images.
        image_mask : Tensor, shape [B, V]
            Mask for views (1 = valid, 0 = pad).
        domain_id : LongTensor, shape [B]
            Domain/task IDs.
        proprio : Tensor, shape [B, dim_proprio]
            Proprioceptive state.
        steps : int
            Number of denoising iterations.

        Returns
        -------
        Tensor, shape [B, T=num_actions, dim_action]
            Predicted action sequence; sigmoid applied only on gripper channels.
        """
        self.eval()
        enc = self.forward_vlm(input_ids=input_ids, pixel_values=image_input, image_mask=image_mask)

        B = input_ids.shape[0]
        device = input_ids.device
        x1 = torch.randn(B, self.num_actions, self.criterion.DIM_ACTION, device=device)
        action = torch.zeros(B, self.num_actions, self.criterion.DIM_ACTION, device=device)

        steps = max(int(steps), 1)
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=device)
            action_with_noise = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, action_with_noise_m = self.mask_gripper(proprio, action_with_noise)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=action_with_noise_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        idx = self.criterion.GRIPPER_IDX
        action[..., idx] = torch.sigmoid(action[..., idx])
        return action

    # ------------------------------ minimal service -------------------------
    def _build_app(self):
        """
        Create a minimal FastAPI app with a single `/act` endpoint.

        Request JSON
        ------------
        language_instruction : str
            Natural language instruction.
        image0/1/2 : ndarray/list or file path
            Up to 3 images.
        proprio : array-like
            Proprioceptive vector of length `dim_proprio`.
        domain_id : int
            Domain/task ID.
        steps : int, optional
            Number of denoising steps (default 10).

        Response JSON
        -------------
        { "action": List[List[float]] }
            Predicted action sequence.
        """
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()

                image_list = []
                for key in ("image0", "image1", "image2"):
                    if key in payload:
                        v = json_numpy.loads(payload[key])
                        if isinstance(v, (list, np.ndarray)):
                            image_list.append(Image.fromarray(np.array(v)))
                        else:
                            image_list.append(Image.open(v))

                language_inputs = self.text_preprocessor.encode_language(
                    [payload["language_instruction"]]
                )
                image_inputs = self.image_preprocessor(image_list)

                proprio_np = np.asarray(json_numpy.loads(payload["proprio"]), dtype=np.float32)
                domain_id_val = int(payload["domain_id"])

                to_cuda = (lambda t: t.cuda(non_blocking=True)) if torch.cuda.is_available() else (lambda t: t)
                inputs = {
                    **{k: to_cuda(v) for k, v in language_inputs.items()},
                    **{k: to_cuda(v) for k, v in image_inputs.items()},
                    "proprio": to_cuda(torch.tensor(proprio_np, dtype=torch.float32).unsqueeze(0)),
                    "domain_id": to_cuda(torch.tensor([domain_id_val], dtype=torch.long)),
                }

                steps = int(payload.get("steps", 10))
                action = self.generate_actions(
                    input_ids=inputs["input_ids"],
                    image_input=inputs["pixel_values"],
                    image_mask=inputs["image_mask"],
                    domain_id=inputs["domain_id"],
                    proprio=inputs["proprio"],
                    steps=steps,
                ).squeeze(0).cpu().numpy()

                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                msg = "Request failed; ensure payload fields are valid."
                return JSONResponse({"error": msg}, status_code=400)

        self.app = app

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch the FastAPI service.

        Parameters
        ----------
        host : str
            Bind address.
        port : int
            Port number.
        """
        self._build_app()
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)


def build_xvla(device: str = "cuda", 
         num_actions = 30,
         pretrained: str | None = None, 
         action_mode = 'ee6d', 
         use_local_vlm = None, 
         use_proprio = True,
         version = "v1",
         **kwargs):
    """
    Factory for an XVLA preset using Florence-2-large and a deeper transformer.

    Parameters
    ----------
    device : str
        Device to move the model to (e.g., "cuda", "cpu").
    pretrained : str | None
        Path to a safetensors checkpoint to load with `strict=False`.
    **kwargs
        Extra keyword args forwarded to `XVLA`.

    Returns
    -------
    nn.Module
        Constructed XVLA model.
    """
    encoder_name = use_local_vlm if use_local_vlm else "microsoft/Florence-2-large"
    model = XVLA(
        encoder_name=encoder_name,
        
        depth=24,
        hidden_size=1024,
        num_heads=16,
        num_domains=30,
        len_soft_prompts=32,
        use_hetero_proj=False,
        
        num_actions=num_actions,
        action_mode=action_mode,
        use_proprio=use_proprio,
        version = version
    )

    if isinstance(pretrained, str):
        print(f">>>>>> load pretrain from {pretrained}")
        pretrained_ckpt = load_file(pretrained)
        print(model.load_state_dict(pretrained_ckpt, strict=False))

    if device:
        model = model.to(device)
    return model
