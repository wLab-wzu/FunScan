
import os
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from model import Transformer
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


parser = argparse.ArgumentParser(description="""FunScan_Train_py""")
parser.add_argument('--threads', help='number of threads to use', type=int, default=64)
parser.add_argument('--trainfolder', help='results file of train', type=str, default='FunScan_train_sentence/')
parser.add_argument('--evalfolder', help='results file of eval', type=str, default='FunScan_eval_sentence/')
parser.add_argument('--trainoutdir', help='eval_model result file', type=str, default='FunScan_train_result/')

inputs = parser.parse_args()
train_fn = inputs.trainfolder
eval_fn = inputs.evalfolder
out_fn = inputs.trainoutdir


if not os.path.isdir(out_fn):
    os.makedirs(out_fn)



pcs2idx = pkl.load(open(f'{train_fn}/pc2wordsid.dict', 'rb'))
num_pcs = len(set(pcs2idx.keys()))

device = torch.device("cuda")
if device.type == 'cpu':
    print("running with cpu")
    torch.set_num_threads(inputs.threads)


src_pad_idx = 0
src_vocab_size = num_pcs + 1

def set_model():
    model = Transformer(
                src_vocab_size,
                src_pad_idx,
                device=device,
                max_length=100,
                dropout=0.5
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    loss_func = loss_func.to(device)
    return model, optimizer, loss_func

def return_batch(train_sentence, label, flag):
    X_train = torch.from_numpy(train_sentence).to(device)
    y_train = label.to(device)
    train_dataset = Data.TensorDataset(X_train, y_train)
    training_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=200,
        shuffle=flag,
        num_workers=0
    )
    return training_loader

def return_tensor(var, device):
    return torch.from_numpy(var).to(device)
model, optimizer, loss_func = set_model()
model = model.to(device)

train_sentence = pkl.load(open(f'{train_fn}/sentence.feat', 'rb'))
contig2pcs = pkl.load(open(f'{train_fn}/contig2pcs.dict', 'rb'))
def train_label():
    labels = []
    for contig in contig2pcs:
        if 'fungi' in contig:
            labels.append(1)
        else:
            labels.append(0)
    label_array = np.array(labels)
    label_array = np.asarray(label_array, np.int64)
    tensor_label = torch.from_numpy(label_array)
    return tensor_label
train_labels = train_label()
training_loader = return_batch(train_sentence, train_labels, flag=True)



eval_sentence = pkl.load(open(f'{eval_fn}/sentence.feat', 'rb'))
eval_id2contig = pkl.load(open(f'{eval_fn}/id2contig.dict', 'rb'))
contig2pcs = pkl.load(open(f'{eval_fn}/contig2pcs.dict', 'rb'))
def eval_label():
    labels = []
    for contig in contig2pcs:
        if 'fungi' in contig:
            labels.append(1)
        else:
            labels.append(0)
    label_array = np.array(labels)
    label_array = np.asarray(label_array, np.int64)
    tensor_label = torch.from_numpy(label_array)
    return tensor_label
eval_labels = eval_label()
eval_label = eval_labels.numpy()
eval_loader = return_batch(eval_sentence, eval_labels, flag=True)



num_epochs = 100
loss_min = float('inf')
best_epoch = 0
epoch_max = 0
train_loss = []
eval_loss = []
epoch_results = []
for epoch in range(num_epochs):
    epoch_train_loss = []
    epoch_eval_loss = []
    train_correct_predictions = 0
    train_total_samples = 0
    eval_total_samples = 0
    eval_correct_predictions = 0
    all_pred = []
    all_score = []
    all_true = []
    print(f"Epoch {epoch + 1}/{num_epochs}")
    _ = model.train()
    for step, (batch_x, batch_y) in enumerate(training_loader):
        batch_x = batch_x.long().to(device)
        optimizer.zero_grad()
        prediction = model(batch_x)
        batch_y = batch_y.float()
        loss = loss_func(prediction.squeeze(1), batch_y.to(device))
        loss.backward()
        optimizer.step()
        epoch_train_loss.append(loss.item())
        pred_labels = (prediction.squeeze(1) > 0.5).float()
        train_correct_predictions += (pred_labels == batch_y.to(device)).sum().item()
        train_total_samples += batch_y.size(0)
    avg_train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
    train_acc = train_correct_predictions / train_total_samples
    with torch.no_grad():
        _ = model.eval()
        for step, (batch_x_eval, batch_y_eval) in enumerate(eval_loader):
            batch_x_eval = batch_x_eval.long()
            batch_y_eval = batch_y_eval.float().to(device)
            eval_prediction = model(batch_x_eval)
            eval_loss = loss_func(eval_prediction.squeeze(1), batch_y_eval.to(device))
            epoch_eval_loss.append(eval_loss.item())
            eval_binary_prediction = (eval_prediction.squeeze(1) > 0.5).float()
            # batch_y_test = batch_y_test.reshape(-1, 1)
            eval_correct_predictions += (eval_binary_prediction == batch_y_eval.to(device)).sum().item()
            eval_total_samples += batch_y_eval.size(0)

    eval_acc = eval_correct_predictions / eval_total_samples
    avg_eval_loss = sum(epoch_eval_loss) / len(epoch_eval_loss)
    epoch_result = {
        "Epoch": epoch,
        "Train Loss": avg_train_loss,
        "Eval Loss": avg_eval_loss,
        "Train Accuracy": train_acc,
        "Eval Accuracy": eval_acc

    }
    epoch_results.append(epoch_result)

    print(
        f"Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Eval Accuracy: {eval_acc}, Train Accuracy: {train_acc}")
    if avg_eval_loss < loss_min:
        loss_min = avg_eval_loss
        best_epoch = epoch
        epoch_max = 0
        torch.save(model.state_dict(), f"{out_fn}/transformer.pth")  # "transformerL2.pth" 想测试添加L2的
    else:
        epoch_max += 1

    if epoch_max >= 10:
        model.load_state_dict(torch.load(f"{out_fn}/transformer.pth"))
        print(f"Best epoch {best_epoch}: Valid Loss {loss_min}")
        break

results_csv_filename = f"{out_fn}/training_results.csv"
results_df = pd.DataFrame(epoch_results)
results_df.to_csv(results_csv_filename, index=False)
torch.save(model.state_dict(), f"{out_fn}/transformer.pth")
