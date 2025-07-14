import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # ==================================================
        # ADDED FOR PEZ: Store token embeddings from CLIP model
        # ==================================================
        # PEZ의 Projection을 위해 전체 어휘집(vocabulary)의 임베딩을 저장합니다.
        self.token_embedding = clip_model.token_embedding.weight.detach()
        # ==================================================

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # This is the continuous prompt embedding P in PEZ algorithm
        self.ctx = nn.Parameter(ctx_vectors)

        # CSC=True is assumed as per the request
        # class-specific context: a prompt for each class
        # This will be (n_cls, n_ctx, ctx_dim)
        self.ctx = nn.Parameter(torch.randn(n_cls, n_ctx, ctx_dim, dtype=dtype))
        nn.init.normal_(self.ctx, std=0.02)
        
        # We don't need classname embeddings if we are learning the full description.
        # However, for compatibility with the original CoOp structure, we keep it.
        # The prompt will be [CLASS_SPECIFIC_PROMPT].
        # If you want to prepend "a photo of", you'd concatenate it here.
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.csc = True # As per request
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION
        self.tokenized_prompts = tokenized_prompts

    # ==================================================
    # ADDED FOR PEZ: Projection function
    # ==================================================
    def project_to_vocabulary(self, continuous_prompt_embeddings: torch.Tensor):
        """
        Projects continuous embeddings to the nearest neighbors in the vocabulary.
        This is the Proj_E function in the PEZ algorithm.
        Uses Straight-Through Estimator to allow gradient flow.
        """
        flat_continuous = continuous_prompt_embeddings.reshape(-1, continuous_prompt_embeddings.size(-1))

        # Get the device AND dtype from the continuous prompt embeddings
        device = continuous_prompt_embeddings.device
        dtype = continuous_prompt_embeddings.dtype  # CHANGED: 현재 데이터 타입(dtype) 가져오기

        # Ensure token_embedding is on the same device and has the same dtype
        # CHANGED: to() 메서드 하나로 디바이스와 dtype을 동시에 동기화
        token_embedding_on_device = self.token_embedding.to(device=device, dtype=dtype)
        
        # Calculate distances
        distances = torch.cdist(flat_continuous, token_embedding_on_device, p=2)
        
        # Find the index of the nearest neighbor
        _, nearest_indices = torch.min(distances, dim=1)
        
        # Get the embeddings of the nearest neighbors
        projected_embeddings_flat = token_embedding_on_device[nearest_indices]

        # Reshape back
        projected_embeddings = projected_embeddings_flat.reshape(continuous_prompt_embeddings.shape)
        
        # Straight-Through Estimator (STE)
        return continuous_prompt_embeddings + (projected_embeddings - continuous_prompt_embeddings).detach()
    # ==================================================

    # ==================================================
    # MODIFIED FOR PEZ: Apply projection during forward pass
    # ==================================================
    def forward(self):
        # self.ctx is the continuous prompt P
        # We project it to get the hard prompt P'
        projected_ctx = self.project_to_vocabulary(self.ctx)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # The prompt is constructed using the projected (hard) embeddings
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                projected_ctx, # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts
    
    # ==================================================
    # ADDED FOR PEZ: Utility to check the final hard prompts
    # ==================================================
    def get_hard_prompt_texts(self):
        """
        Returns the text of the optimized hard prompts after training.
        """
        hard_prompt_texts = []
        with torch.no_grad():
            # Final projection to get the discrete token embeddings
            
            # CHANGED: Move to CPU and convert dtype to float32 to match vocab_embeddings
            continuous_prompts = self.ctx.cpu().to(torch.float32)
            
            flat_continuous = continuous_prompts.reshape(-1, continuous_prompts.size(-1))
            
            # vocab_embeddings is already on CPU and is float32
            vocab_embeddings = self.token_embedding.cpu()
            
            distances = torch.cdist(flat_continuous, vocab_embeddings, p=2)
            _, nearest_indices = torch.min(distances, dim=1)
            
            # Reshape indices to (n_cls, n_ctx)
            prompt_indices = nearest_indices.reshape(self.n_cls, self.n_ctx)

            # Decode indices to text
            for i in range(self.n_cls):
                tokens = prompt_indices[i].tolist()
                text = _tokenizer.decode(tokens)
                hard_prompt_texts.append(text)
        
        return hard_prompt_texts
    # ==================================================


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        optimized_prompts = self.model.prompt_learner.get_hard_prompt_texts()
        classnames = self.dm.dataset.classnames
        for classname, prompt in zip(classnames, optimized_prompts):
            print(f"Class: {classname}, Optimized Prompt: '{prompt}'")
            
        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
