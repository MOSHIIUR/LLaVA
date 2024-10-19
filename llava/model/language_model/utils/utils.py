import torch

def split_seqeunce(text_split, img_seq, input_embeds):
    text_hidden_states, img_hidden_states = [], []
    for idx, input_embed in enumerate(input_embeds):
        txt_seq_len = sum(text_split[idx])
        input_seq_len = input_embeds[idx].shape[0]
        split_size = [txt_seq_len, img_seq[idx], input_seq_len]
        print(f'split size: {split_size}')
        text_tokens, img_tokens, padded_tokens = torch.split(input_embed, split_size, dim=0)
        text_tokens = torch.cat((text_tokens, padded_tokens), dim=0)
        text_hidden_states.append(text_tokens)
        if img_seq == 0:
            print(f'image hidden feature: {img_tokens.shape}')
        img_hidden_states.append(img_tokens)

    new_text_hidden_states = torch.stack(text_hidden_states, dim=0)
    new_img_hidden_states = torch.stack(img_hidden_states, dim=0)

    return new_text_hidden_states, new_img_hidden_states

    
    

