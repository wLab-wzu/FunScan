
import os
import torch
from torch import nn
from torch import optim
import torch.utils.data as Data
import torch.distributed as dist
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
from model import Transformer
from torch.distributed import all_reduce, ReduceOp
from torch.distributed import all_gather_object

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def parse_args():
    parser = argparse.ArgumentParser(description="""FunScan_Train_py""")
    parser.add_argument('--threads', help='number of threads to use', type=int, default=64)
    parser.add_argument('--trainfolder', help='results file of train', type=str, default='FunScan_train_sentence/')
    parser.add_argument('--evalfolder', help='results file of eval', type=str, default='FunScan_eval_sentence/')
    parser.add_argument('--trainoutdir', help='eval_model result file', type=str, default='FunScan_train_result/')
    return parser.parse_args()

inputs = parse_args()
train_fn = inputs.trainfolder
eval_fn = inputs.evalfolder
out_fn = inputs.trainoutdir


if not os.path.isdir(out_fn):
    os.makedirs(out_fn)

def train(rank, world_size):
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    rank = torch.distributed.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    pcs2idx = pkl.load(open(f'{train_fn}/pc2wordsid.dict', 'rb'))
    num_pcs = len(set(pcs2idx.keys()))

    src_pad_idx = 0
    src_vocab_size = num_pcs + 1


    model = Transformer(

        src_vocab_size,
        src_pad_idx,
        device=device,
        max_length=100,
        dropout=0.5
    ).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank,
                                                      find_unused_parameters=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.BCEWithLogitsLoss()
    loss_func = loss_func.to(device)

    def return_batch(train_sentence, label):
        X_train = torch.from_numpy(train_sentence).to(device)
        y_train = label.to(device)
        train_dataset = Data.TensorDataset(X_train, y_train)
        return train_dataset

    train_sentence = pkl.load(open(f'{train_fn}/sentence.feat', 'rb'))
    contig2pcs = pkl.load(open(f'{train_fn}/contig2pcs.dict', 'rb'))

    def label():
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

    train_labels = label()
    training_loader = return_batch(train_sentence, train_labels)

    sampler = torch.utils.data.distributed.DistributedSampler(training_loader, num_replicas=world_size, rank=rank)
    train_dataloader = torch.utils.data.DataLoader(training_loader, sampler=sampler, batch_size=200)

    eval_sentence = pkl.load(open(f'{eval_fn}/sentence.feat', 'rb'))
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
    eval_loader = return_batch(eval_sentence, eval_labels)

    sampler = torch.utils.data.distributed.DistributedSampler(eval_loader, num_replicas=world_size, rank=rank)
    eval_dataloader = torch.utils.data.DataLoader(eval_loader, sampler=sampler, batch_size=200)

    num_epochs = 50
    loss_min = float('inf')
    best_epoch = 0
    epoch_max = 0
    train_losses = []
    epoch_results = []
    train_accuracies = []
    terminate_flag = torch.tensor(0, dtype=torch.int).to(device)
    for epoch in range(num_epochs):
        epoch_train_loss = []
        epoch_eval_loss = []
        train_correct_predictions = 0
        train_total_samples = 0
        eval_total_samples = 0
        eval_correct_predictions = 0
        all_pred = []
        all_score = []
        print(f"Epoch {epoch + 1}/{num_epochs}")
        _ = model.train()
        for step, (batch_x, batch_y) in enumerate(train_dataloader):
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
        train_loss_tensor = torch.tensor(epoch_train_loss, device=device)
        all_reduce(train_loss_tensor, op=ReduceOp.SUM)  # 汇总训练损失
        avg_train_loss = train_loss_tensor.mean().item() if rank == 0 else 0
        # 汇总训练准确度
        correct_predictions_tensor = torch.tensor(train_correct_predictions, device=device)
        total_predictions_tensor = torch.tensor(train_total_samples, device=device)
        all_reduce(correct_predictions_tensor, op=ReduceOp.SUM)  # 汇总正确预测数量
        all_reduce(total_predictions_tensor, op=ReduceOp.SUM)  # 汇总预测总数量

        if rank == 0:
            train_accuracy = correct_predictions_tensor.item() / total_predictions_tensor.item()
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)

        with torch.no_grad():
            _ = model.eval()
            for step, (batch_x_eval, batch_y_eval) in enumerate(eval_dataloader):
                batch_x_eval = batch_x_eval.long()
                batch_y_eval = batch_y_eval.float().to(device)
                eval_prediction = model(batch_x_eval)
                eval_loss = loss_func(eval_prediction.squeeze(1), batch_y_eval.to(device))
                epoch_eval_loss.append(eval_loss.item())
                eval_binary_prediction = (eval_prediction.squeeze(1) > 0.5).float()
                # batch_y_test = batch_y_test.reshape(-1, 1)
                eval_correct_predictions += (eval_binary_prediction == batch_y_eval.to(device)).sum().item()
                eval_total_samples += batch_y_eval.size(0)

                logit1 = torch.sigmoid(eval_prediction.squeeze(1))  # Apply the threshold
                all_pred += ['fungi' if item > 0.5 else 'other' for item in logit1]
                all_score += [float('{:.3f}'.format(i)) for i in logit1]

        local_pred = all_pred  # 当前 rank 的预测标签
        local_score = all_score  # 当前 rank 的预测分数

        # 收集所有 rank 的数据
        # world_size = get_world_size()
        gathered_preds = [None] * world_size
        gathered_scores = [None] * world_size
        all_gather_object(gathered_preds, local_pred)
        all_gather_object(gathered_scores, local_score)


        eval_loss_tensor = torch.tensor(epoch_eval_loss, device=device)
        all_reduce(eval_loss_tensor, op=ReduceOp.SUM)  # 汇总测试损失
        avg_eval_loss = eval_loss_tensor.mean().item() if rank == 0 else 0

        # 汇总评估准确度
        eval_correct_predictions_tensor = torch.tensor(eval_correct_predictions, device=device)

        eval_label_tensor = torch.tensor(eval_total_samples, device=device)

        # 汇总预测结果
        all_reduce(eval_correct_predictions_tensor, op=ReduceOp.SUM)
        all_reduce(eval_label_tensor, op=ReduceOp.SUM)

        if rank == 0:
            eval_accuracy = eval_correct_predictions_tensor.item() / eval_label_tensor.item()

            epoch_result = {
                "Epoch": epoch,
                "Train Loss": avg_train_loss,
                "Eval Loss": avg_eval_loss,
                "Train Accuracy": train_accuracy,
                "Eval Accuracy": eval_accuracy}
            epoch_results.append(epoch_result)

        if rank == 0:
            print(
                f"Train Loss: {avg_train_loss:.10f}, Eval Loss: {avg_eval_loss:.10f}, Train_Acc: {train_accuracy * 100:.2f}%, Eval_Acc: {eval_accuracy * 100:.2f}%")  # Accuracy: {accuracy:.4f}

            if avg_eval_loss < loss_min:
                loss_min = avg_eval_loss
                best_epoch = epoch
                epoch_max = 0
                results_csv_filename = f"{out_fn}/results.csv"
                results_df = pd.DataFrame(epoch_results)
                results_df.to_csv(results_csv_filename, index=False)
                torch.save(model.module.state_dict(), f"{out_fn}/transformer.pth")
            else:
                epoch_max += 1

            if epoch_max >= 10:
                # Recover the best model so far
                model.module.load_state_dict(torch.load(f"{out_fn}/transformer.pth"))
                print(f"Best epoch {best_epoch}: Valid Loss {loss_min}")
                results_csv_filename1 = f"{out_fn}/results.csv"
                results_df1 = pd.DataFrame(epoch_results)
                results_df1.to_csv(results_csv_filename1, index=False)
                torch.save(model.module.state_dict(), f"{out_fn}/transformer.pth")
                terminate_flag.fill_(1)
                # break
        # 广播终止标志到所有进程
        dist.all_reduce(terminate_flag, op=dist.ReduceOp.MAX)
        # dist.broadcast(terminate_flag, src=0)
        if terminate_flag.item() == 1:
            print("早停")
            break

    dist.destroy_process_group()
    print("整体结束：", epoch)



if __name__ == "__main__":
    args = parse_args()
    rank = args.local_rank  # 获取传递的 rank

    world_size = torch.cuda.device_count()  # 获取 GPU 数量
    train(rank, world_size)

