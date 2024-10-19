import torch

def split_seqeunce(text_splits, img_sequences, input_embeds):
    text_hidden_states, img_hidden_states = [], []
    for text_split, img_sequence, input_embed in zip(text_splits, img_sequences, input_embeds):
        txt_seq_len = sum(text_split)
        total_seq_len = input_embed.shape[0]
        padded_sequence = total_seq_len - (txt_seq_len + img_sequence)

        if txt_seq_len + img_sequence + padded_sequence != total_seq_len:
            raise ValueError(f'Split sizes do not match the total sequence length'
                             f'expected {total_seq_len}, but got {txt_seq_len + img_sequence + padded_sequence}')

        split_size = [txt_seq_len, img_sequence, padded_sequence]
        # print(f'split size: {split_size}')

        

        text_tokens, img_tokens, padded_tokens = torch.split(input_embed, split_size, dim=0)
        text_hidden_states.append(text_tokens)
        
        img_hidden_states.append(img_tokens)

    return text_hidden_states, img_hidden_states



def pad_sequence(new_input_embeds, padding_side):

    max_len = max(x.shape[0] for x in new_input_embeds)

    new_input_embeds_padded = []

    for i, cur_new_embed in enumerate(new_input_embeds):
        cur_len = cur_new_embed.shape[0]
        if padding_side == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))

        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))


    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    return new_input_embeds

    
    

