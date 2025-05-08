import torch
import torch.nn as nn
import os
import numpy as np
import wandb
from datasets import data_loader
from model_attn import BertRNNAttn as BertRNN  
from sklearn.metrics import accuracy_score
import copy

class Engine:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        train_loader = data_loader(args.corpus, 'train', args.batch_size, args.chunk_size, shuffle=True) if args.mode != 'inference' else None
        val_loader = data_loader(args.corpus, 'val', args.batch_size_val, args.chunk_size) if args.mode != 'inference' else None
        test_loader = data_loader(args.corpus, 'test', args.batch_size_val, args.chunk_size)

        print('Done\n')

        if torch.cuda.device_count() > 0:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")

        print('Initializing model....')
        model = BertRNN(
            nlayer=args.nlayer,
            nclass=args.nclass,
            dropout=args.dropout,
            nfinetune=args.nfinetune,
            speaker_info=args.speaker_info,
            topic_info=args.topic_info,
            emb_batch=args.emb_batch,
        )

        model = nn.DataParallel(model)
        model.to(device)
        params = model.parameters()

        from torch.optim import AdamW
        optimizer = AdamW(params, lr=args.lr, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

        if args.mode == 'train':
            wandb.init(project="dialogue-act-classification", config=vars(args))
            wandb.run.name = f"{args.corpus}_attn_run"
            wandb.run.save()

    def train(self):
        best_epoch = 0
        best_epoch_acc = 0
        best_epoch_test_acc = 0
        best_acc = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.args.epochs):
            print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
            loss = self.train_epoch()
            acc = self.eval()
            test_acc = self.eval(val=False)
            if acc > best_epoch_acc:
                best_epoch = epoch
                best_epoch_acc = acc
                best_epoch_test_acc = test_acc
                best_state_dict = copy.deepcopy(self.model.state_dict())
            if test_acc > best_acc:
                best_acc = test_acc
            print(f'Epoch {epoch + 1}\tTrain Loss: {loss:.3f}\tVal Acc: {acc:.3f}\tTest Acc: {test_acc:.3f}\n'
                  f'Best Epoch: {best_epoch + 1}\tBest Val Acc: {best_epoch_acc:.3f}\t'
                  f'Best Test Acc: {best_epoch_test_acc:.3f}, Max Test Acc: {best_acc:.3f}\n')

            wandb.log({
                'epoch': epoch + 1,
                'train_loss': loss,
                'val_acc': acc,
                'test_acc': test_acc,
                'best_epoch': best_epoch + 1,
                'best_val_acc': best_epoch_acc,
                'best_test_acc': best_epoch_test_acc
            })

            if epoch - best_epoch >= 10:
                break

        print('Saving the best checkpoint....')
        torch.save(best_state_dict, f"ckp/model_{self.args.corpus}_attn.pt")
        self.model.load_state_dict(best_state_dict)
        acc = self.eval(val=False)
        print(f'Test Acc: {acc:.3f}')
        wandb.save(f"ckp/model_{self.args.corpus}_attn.pt")

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            chunk_lens = batch['chunk_lens']
            speaker_ids = batch['speaker_ids'].to(self.device)
            topic_labels = batch['topic_labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels)
            labels = labels.reshape(-1)
            loss_act = self.criterion(outputs, labels)
            loss = loss_act
            loss.backward()
            self.optimizer.step()
            interval = max(len(self.train_loader) // 20, 1)
            if i % interval == 0 or i == len(self.train_loader) - 1:
                print(f'Batch: {i + 1}/{len(self.train_loader)}\tloss: {loss.item():.3f}\tloss_act:{loss_act.item():.3f}')
            epoch_loss += loss.item()
        return epoch_loss / len(self.train_loader)

    def eval(self, val=True, inference=False):
        self.model.eval()
        y_pred, y_true = [], []
        loader = self.val_loader if val else self.test_loader
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                chunk_lens = batch['chunk_lens']
                speaker_ids = batch['speaker_ids'].to(self.device)
                topic_labels = batch['topic_labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask, chunk_lens, speaker_ids, topic_labels)
                y_pred.append(outputs.detach().cpu().argmax(dim=1).numpy())
                labels = labels.reshape(-1)
                y_true.append(labels.detach().cpu().numpy())

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        mask = y_true != -1
        acc = accuracy_score(y_true[mask], y_pred[mask])

        if inference:
            import pickle
            pickle.dump(y_pred[mask].tolist(), open('preds_on_new.pkl', 'wb'))

        return acc

    def inference(self):
        self.model.load_state_dict(torch.load(f"ckp/model_{self.args.corpus}_attn.pt"))
        self.eval(val=False, inference=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default='swda')
    parser.add_argument('--mode', type=str, choices=('train', 'inference'), default='train')
    parser.add_argument('--nclass', type=int, default=43)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--batch_size_val', type=int, default=32)
    parser.add_argument('--emb_batch', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nlayer', type=int, default=1)
    parser.add_argument('--chunk_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--speaker_info', type=str, choices=('none', 'emb_cls'), default='none')
    parser.add_argument('--topic_info', type=str, choices=('none', 'emb_cls'), default='none')
    parser.add_argument('--nfinetune', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(args)
    engine = Engine(args)
    if args.mode == 'train':
        engine.train()
    else:
        engine.inference()
