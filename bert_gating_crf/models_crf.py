import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF
from torch.utils.data import TensorDataset, DataLoader

class BertRNN(nn.Module):
    def __init__(self, nlayer, nclass, dropout=0.5, nfinetune=0, speaker_info='none', topic_info='none', emb_batch=0):
        super(BertRNN, self).__init__()

        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained('roberta-base')
        nhid = self.bert.config.hidden_size

        for param in self.bert.parameters():
            param.requires_grad = False
        n_layers = 12
        if nfinetune > 0:
            for param in self.bert.pooler.parameters():
                param.requires_grad = True
            for i in range(n_layers-1, n_layers-1-nfinetune, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True

        self.encoder = nn.GRU(nhid, nhid//2, num_layers=nlayer, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(nhid, nclass)

        self.speaker_emb = nn.Embedding(3, nhid)
        self.topic_emb = nn.Embedding(100, nhid)
        self.gate = nn.Linear(nhid * 2, nhid)

        self.dropout = nn.Dropout(p=dropout)
        self.nclass = nclass
        self.speaker_info = speaker_info
        self.topic_info = topic_info
        self.emb_batch = emb_batch

        # CRF layer
        self.crf = CRF(nclass, batch_first=True)

    def forward(self, input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels, labels=None):
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
                embeddings = self.bert(batch[0], attention_mask=batch[1], output_hidden_states=True)[0][:, 0]
                embeddings_.append(embeddings)
            embeddings = torch.cat(embeddings_, dim=0)

        nhid = embeddings.shape[-1]

        if self.speaker_info == 'emb_cls':
            speaker_embeddings = self.speaker_emb(speaker_ids)
            gate_input = torch.cat((embeddings, speaker_embeddings), dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            embeddings = embeddings + gate * speaker_embeddings

        if self.topic_info == 'emb_cls':
            topic_embeddings = self.topic_emb(topic_labels)
            gate_input = torch.cat((embeddings, topic_embeddings), dim=-1)
            gate = torch.sigmoid(self.gate(gate_input))
            embeddings = embeddings + gate * topic_embeddings

        embeddings = embeddings.reshape(-1, chunk_size, nhid)
        embeddings = embeddings.permute(1, 0, 2)

        embeddings = pack_padded_sequence(embeddings, chunk_lens, enforce_sorted=False)
        self.encoder.flatten_parameters()
        outputs, _ = self.encoder(embeddings)
        outputs, _ = pad_packed_sequence(outputs)

        if outputs.shape[0] < chunk_size:
            outputs_padding = torch.zeros(chunk_size - outputs.shape[0], batch_size, nhid, device=outputs.device)
            outputs = torch.cat([outputs, outputs_padding], dim=0)

        outputs = self.dropout(outputs)
        emissions = self.fc(outputs)  # (chunk_size, batch_size, nclass)
        emissions = emissions.permute(1, 0, 2)  # (batch_size, chunk_size, nclass)

        if labels is not None:
            labels = labels.reshape(batch_size, chunk_size)
            mask = labels != -1
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss
        else:
            predictions = self.crf.decode(emissions)
            max_len = emissions.shape[1]
            decoded = torch.full((batch_size, max_len), fill_value=-1, dtype=torch.long, device=emissions.device)
            for i, seq in enumerate(predictions):
                decoded[i, :len(seq)] = torch.tensor(seq, device=emissions.device)
            return decoded.reshape(-1)
