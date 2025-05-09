import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import TensorDataset, DataLoader

class BertRNNAttn(nn.Module):
    def __init__(self, nlayer, nclass, dropout=0.5, nfinetune=0, speaker_info='none', topic_info='none', emb_batch=0):
        super(BertRNNAttn, self).__init__()

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False
        n_layers = 12
        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers - 1, n_layers - 1 - nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        self.encoder = nn.GRU(nhid, nhid // 2, num_layers=nlayer, dropout=dropout, bidirectional=True)
        self.attn_fc = nn.Linear(nhid, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(nhid, nclass)

        self.speaker_emb = nn.Embedding(3, nhid)
        self.topic_emb = nn.Embedding(100, nhid)

        self.nclass = nclass
        self.speaker_info = speaker_info
        self.topic_info = topic_info
        self.emb_batch = emb_batch

    def forward(self, input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels):
        chunk_lens = chunk_lens.to('cpu')
        batch_size, chunk_size, seq_len = input_ids.shape
        speaker_ids = speaker_ids.reshape(-1)
        chunk_lens = chunk_lens.reshape(-1)
        topic_labels = topic_labels.reshape(-1)

        input_ids = input_ids.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)

        if self.training or self.emb_batch == 0:
            embeddings = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)[0][:, 0]
        else:
            embeddings_ = []
            dataset2 = TensorDataset(input_ids, attention_mask)
            loader = DataLoader(dataset2, batch_size=self.emb_batch)
            for _, batch in enumerate(loader):
                emb = self.bert(batch[0], attention_mask=batch[1], output_hidden_states=True)[0][:, 0]
                embeddings_.append(emb)
            embeddings = torch.cat(embeddings_, dim=0)

        nhid = embeddings.shape[-1]

        if self.speaker_info == 'emb_cls':
            speaker_embeddings = self.speaker_emb(speaker_ids)
            embeddings += speaker_embeddings
        if self.topic_info == 'emb_cls':
            topic_embeddings = self.topic_emb(topic_labels)
            embeddings += topic_embeddings

        embeddings = embeddings.reshape(-1, chunk_size, nhid).permute(1, 0, 2)

        packed_input = pack_padded_sequence(embeddings, chunk_lens, enforce_sorted=False)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(packed_input)
        outputs, _ = pad_packed_sequence(outputs)

        if outputs.shape[0] < chunk_size:
            pad = torch.zeros(chunk_size - outputs.shape[0], batch_size, nhid, device=outputs.device)
            outputs = torch.cat([outputs, pad], dim=0)

        outputs = self.dropout(outputs)

        # Attention per time step (utterance-level attention)
        attn_scores = self.attn_fc(outputs)              # (chunk_size, bs, 1)
        attn_weights = torch.softmax(attn_scores, dim=0) # (chunk_size, bs, 1)
        attn_outputs = outputs * attn_weights            # apply attention
        attn_outputs = attn_outputs.permute(1, 0, 2)     # (bs, chunk_size, nhid)
        attn_outputs = attn_outputs.reshape(-1, nhid)    # (bs * chunk_size, nhid)

        logits = self.fc(attn_outputs)                   # (bs * chunk_size, nclass)
        return logits
