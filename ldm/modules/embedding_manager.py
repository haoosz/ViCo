import torch
from torch import nn

from ldm.data.personalized import per_img_token_list
from transformers import CLIPTokenizer
from functools import partial
from ldm.modules.encoders.modules import FrozenClipImageFeatureExtractor


DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

def get_bert_token_for_string(tokenizer, string):
    token = tokenizer(string)
    assert torch.count_nonzero(token) == 3, f"String '{string}' maps to more than a single token. Please use another string"

    token = token[0, 1]

    return token

def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))[0]


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class EmbeddingManager(nn.Module):
    def __init__(
            self,
            embedder,
            placeholder_strings=None,
            initializer_words=None,
            per_image_tokens=False,
            num_vectors_per_token=1,
            progressive_words=False,
            **kwargs
    ):
        super().__init__()

        self.string_to_token_dict = {}
        
        self.string_to_param_dict = nn.ParameterDict()

        self.initial_embeddings = nn.ParameterDict() # These should not be optimized

        self.progressive_words = progressive_words
        self.progressive_counter = 0

        self.max_vectors_per_token = num_vectors_per_token

        if hasattr(embedder, 'tokenizer'): # using Stable Diffusion's CLIP encoder
            self.is_clip = True
            get_token_for_string = partial(get_clip_token_for_string, embedder.tokenizer)
            get_embedding_for_tkn = partial(get_embedding_for_clip_token, embedder.transformer.text_model.embeddings.token_embedding)
            token_dim = 768
        else: # using LDM's BERT encoder
            self.is_clip = False
            get_token_for_string = partial(get_bert_token_for_string, embedder.tknz_fn)
            get_embedding_for_tkn = embedder.transformer.token_emb
            token_dim = 1280

        if per_image_tokens:
            placeholder_strings.extend(per_img_token_list)

        for idx, placeholder_string in enumerate(placeholder_strings):
            
            token = get_token_for_string(placeholder_string)

            if initializer_words and idx < len(initializer_words):
                init_word_token = get_token_for_string(initializer_words[idx])
                
                with torch.no_grad():
                    init_word_embedding = get_embedding_for_tkn(init_word_token.cpu())

                token_params = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=True)
                self.initial_embeddings[placeholder_string] = torch.nn.Parameter(init_word_embedding.unsqueeze(0).repeat(num_vectors_per_token, 1), requires_grad=False)
            else:
                token_params = torch.nn.Parameter(torch.rand(size=(num_vectors_per_token, token_dim), requires_grad=True))
            
            self.string_to_token_dict[placeholder_string] = token
            self.string_to_param_dict[placeholder_string] = token_params
        
        ################ 
        # add visual emb
        # self.img_token = get_token_for_string("#")
        # self.mlp = nn.Sequential(
        #     nn.Linear(768, 96),
        #     nn.GELU(),
        #     nn.Linear(96, 768))
        ################
        
    def forward(
            self,
            tokenized_text,
            embedded_text,
    ):  
        b, n, device = *tokenized_text.shape, tokenized_text.device
        
        ################ 
        # add visual emb
        # img_emb = torch.load('image_feature/clock.pt')
        # img_token_idx = torch.where(tokenized_text == self.img_token.to(device))
        # img_emb_proj = self.mlp(img_emb.to(device).float())
        # embedded_text[img_token_idx] = img_emb_proj
        ################
        
        for placeholder_string, placeholder_token in self.string_to_token_dict.items():

            placeholder_embedding = self.string_to_param_dict[placeholder_string].to(device)

            if self.max_vectors_per_token == 1: # If there's only one vector per token, we can do a simple replacement
                self.placeholder_idx = torch.where(tokenized_text == placeholder_token.to(device))
                embedded_text[self.placeholder_idx] = placeholder_embedding
            else: # otherwise, need to insert and keep track of changing indices
                if self.progressive_words:
                    self.progressive_counter += 1
                    max_step_tokens = 1 + self.progressive_counter // PROGRESSIVE_SCALE
                else:
                    max_step_tokens = self.max_vectors_per_token

                num_vectors_for_token = min(placeholder_embedding.shape[0], max_step_tokens)

                placeholder_rows, placeholder_cols = torch.where(tokenized_text == placeholder_token.to(device))

                if placeholder_rows.nelement() == 0:
                    continue

                sorted_cols, sort_idx = torch.sort(placeholder_cols, descending=True)
                sorted_rows = placeholder_rows[sort_idx]

                for idx in range(len(sorted_rows)):
                    row = sorted_rows[idx]
                    col = sorted_cols[idx]

                    new_token_row = torch.cat([tokenized_text[row][:col], placeholder_token.repeat(num_vectors_for_token).to(device), tokenized_text[row][col + 1:]], axis=0)[:n]
                    new_embed_row = torch.cat([embedded_text[row][:col], placeholder_embedding[:num_vectors_for_token], embedded_text[row][col + 1:]], axis=0)[:n]

                    embedded_text[row]  = new_embed_row
                    tokenized_text[row] = new_token_row

        return embedded_text

    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict}, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    def get_embedding_norms_squared(self):
        all_params = torch.cat(list(self.string_to_param_dict.values()), axis=0) # num_placeholders x embedding_dim
        param_norm_squared = (all_params * all_params).sum(axis=-1)              # num_placeholders

        return param_norm_squared

    def embedding_parameters(self):
        return self.string_to_param_dict.parameters()

    # def mlp_parameters(self):
    #     return self.mlp.parameters()
    
    def embedding_to_coarse_loss(self):
        
        loss = 0.
        num_embeddings = len(self.initial_embeddings)

        for key in self.initial_embeddings:
            optimized = self.string_to_param_dict[key]
            coarse = self.initial_embeddings[key].clone().to(optimized.device)

            loss = loss + (optimized - coarse) @ (optimized - coarse).T / num_embeddings

        return loss
    
    ######
    # reg
    def get_reg_loss(self, cond):
        
        loss = 0.
        optimized = cond[self.placeholder_idx]

        clip_vis = torch.load("clip_feature/clock/1.pt").to(optimized.device)
        text_proj = torch.load("presave/text_proj.pt").to(optimized.device)
        
        optimized_proj = text_proj(optimized)
        
        image_feature = clip_vis / clip_vis.norm(dim=-1, keepdim=True)
        text_feature = optimized_proj / optimized_proj.norm(dim=-1, keepdim=True)

        loss = loss + (text_feature - image_feature) @ (text_feature - image_feature).T
            
        return loss.mean() * 0.01
    ######