import util
import argparse
import torch
from model import HSTGNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="input_dim")
parser.add_argument("--channels", type=int, default=128, help="number of nodes")
parser.add_argument("--num_nodes", type=int, default=170, help="number of nodes")
parser.add_argument("--input_len", type=int, default=12, help="input_len")
parser.add_argument("--output_len", type=int, default=12, help="out_len")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument('--checkpoint', type=str,
                    default='log/2025-02-08-12:02:21-PEMS08/best_model.pth', help='')
parser.add_argument('--plotheatmap', type=str, default='True', help='')
args = parser.parse_args()

def main():
    
    if args.data == "PEMSBAY":
        args.data = "data//" + args.data
        args.num_nodes = 325

    elif args.data == "PEMS08":
        args.data = "data//" + args.data
        args.num_nodes = 170
        
    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        args.num_nodes = 307
    
    elif args.data == "PEMS08_60":
        args.data = "data//" + args.data
        args.num_nodes = 170
        args.input_len = 60
        args.output_len = 60

    
    elif args.data == "Urban_60":
        args.data = "data//" + args.data
        args.num_nodes = 304
        args.input_len = 60
        args.output_len = 60

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        args.num_nodes = 358
        args.epochs = 300
        args.es_patience = 100

    elif args.data == "Urban":
        args.data = "data//" + args.data
        args.num_nodes = 304

    elif args.data == "METRLA":
        args.data = "data/METRLA"
        args.num_nodes = 207

    elif args.data == "PEMS03_60":
        args.data = "data//"+args.data
        args.num_nodes = 358
        args.epochs = 300

    device = torch.device(args.device)


    model = HSTGNN(
            device, args.input_dim, args.channels, args.num_nodes, args.input_len, args.output_len, args.dropout
        )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    print('model load successfully')

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = model(testx).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    amae = []
    amape = []
    awmape = []
    armse = []
    
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse),np.mean(awmape)))


    realy = realy.to("cpu")
    yhat1 = scaler.inverse_transform(yhat)
    yhat1 = yhat1.to("cpu")

    print(realy.shape)
    print(yhat1.shape)

    torch.save(realy,"stamt_04real.pt")
    torch.save(yhat1,"stamt_04pred.pt")

if __name__ == "__main__":
    main()
