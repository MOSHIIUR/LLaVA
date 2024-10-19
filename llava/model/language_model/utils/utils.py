import torch

def split_seqeunce(text_splits, img_sequences, input_embeds):
    text_hidden_states, img_hidden_states = [], []
    for text_split, img_sequence, input_embed in zip(text_splits, img_sequences, input_embeds):
        txt_seq_len = sum(text_split)
        total_seq_len = input_embed.shape[0]
        padded_sequence = total_seq_len - (txt_seq_len + txt_seq_len)

        if txt_seq_len + img_sequence + padded_sequence != total_seq_len:
            raise ValueError(f'Split sizes do not match the total sequence length'
                             f'expected {total_seq_len}, but got {txt_seq_len + img_sequence + padded_sequence}')

        split_size = [txt_seq_len, img_sequence, padded_sequence]
        
        print(f'split size: {split_size}')

        text_tokens, img_tokens, padded_tokens = torch.split(input_embed, split_size, dim=0)
        text_tokens = torch.cat((text_tokens, padded_tokens), dim=0)
        text_hidden_states.append(text_tokens)
        
        if img_sequence == 0:
            print(f'image hidden feature: {img_tokens.shape}')
        
        img_hidden_states.append(img_tokens)

    new_text_hidden_states = torch.stack(text_hidden_states, dim=0)
    new_img_hidden_states = torch.stack(img_hidden_states, dim=0)

    return new_text_hidden_states, new_img_hidden_states

    
    

