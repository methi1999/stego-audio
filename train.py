from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import random
import time
import torch
import torch.nn as nn
import torch.optim
import tqdm

import speech
import speech.loader as loader
import speech.models as models
import tensorboard_logger as tb

device = ("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, optimizer, train_ldr, it, avg_loss):
    model.train()
    model_t = 0.0
    data_t = 0.0
    end_t = time.time()
    tq = tqdm.tqdm(train_ldr)
    for batch in tq:
        start_t = time.time()
        optimizer.zero_grad()
        loss = model.loss(batch)
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200)
        loss = loss.item()

        optimizer.step()
        prev_end_t = end_t
        end_t = time.time()
        model_t += end_t - start_t
        data_t += start_t - prev_end_t

        exp_w = 0.99
        avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
        tb.log_value('train_loss', loss, it)
        tq.set_postfix(iter=it, loss=loss,
                       avg_loss=avg_loss, grad_norm=grad_norm,
                       model_time=model_t, data_time=data_t)
        it += 1

    return it, avg_loss


def eval_dev(model, ldr, preproc):

    with torch.no_grad():
        losses = []
        all_preds = []
        all_labels = []

        model.eval()
        for batch in tqdm.tqdm(ldr):
            preds, labels, loss = model.infer_batch(batch, calculate_loss=True)
            losses.append(loss.item())
            all_preds.extend(preds)
            all_labels.extend(labels)
    # loss = sum(losses) / len(losses)
    results = [(preproc.decode(l), preproc.decode(p)) for l, p in zip(all_labels, all_preds)]
    cer = speech.compute_cer(results)
    print("Dev: Loss {:.3f}, CER {:.3f}".format(100, cer))
    return loss, cer


def run(config, use_cuda):
    opt_cfg = config["optimizer"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    aud_cfg = config['audio']

    print("Epochs to train:", opt_cfg["epochs"])
    # Loaders
    batch_size = opt_cfg["batch_size"]
    # preprocessor stores data and functions to encode/decode text
    preproc = loader.Preprocessor(data_cfg["train_set"], aud_cfg, start_and_end=data_cfg["start_and_end"])

    # preproc_d = loader.Preprocessor(data_cfg["dev_set"],
    #                               start_and_end=data_cfg["start_and_end"])

    # Dataloader is a subclass of pytorch.utils.dataloader. Can iterate
    train_ldr = loader.make_loader(data_cfg["train_set"], preproc, batch_size)
    dev_ldr = loader.make_loader(data_cfg["dev_set"], preproc, batch_size)

    # eval('print("Hello")') will actually call print("Hello")
    model_class = eval("models." + model_cfg["class"])
    # define model
    model = model_class(preproc.input_dim, preproc.vocab_size, model_cfg)
    # model, preproc = speech.load("ctc_models_MOMENTUM", tag="best")
    model = model.cuda() if use_cuda else model.cpu()

    # can try out Adam
    optimizer = torch.optim.SGD(model.parameters(), lr=opt_cfg["learning_rate"],
                                momentum=opt_cfg["momentum"])

    speech.save(model, optimizer, preproc, config["save_path"])
    run_state = (0, 0)
    best_so_far = float("inf")
    for e in range(opt_cfg["epochs"]):
        start = time.time()

        run_state = run_epoch(model, optimizer, train_ldr, *run_state)

        msg = "Epoch {} completed in {:.2f} (s)."
        print(msg.format(e, time.time() - start))
        if (e % 10 == 0) or (e == (opt_cfg["epochs"] - 1)):
            dev_loss, dev_cer = eval_dev(model, dev_ldr, preproc)

            # Log for tensorboard
            tb.log_value("dev_loss", dev_loss, e)
            tb.log_value("dev_cer", dev_cer, e)

        speech.save(model, optimizer, preproc, config["save_path"])

        # Save the best model on the dev set
        if dev_cer < best_so_far:
            best_so_far = dev_cer
            speech.save(model, optimizer, preproc, config["save_path"], tag="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speech model")
    # path to config file
    parser.add_argument("--config", default='config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as fid:
        config = json.load(fid)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    tb.configure(config["save_path"])

    use_cuda = torch.cuda.is_available()
    # make it deterministic
    if use_cuda:
        torch.backends.cudnn.enabled = False

    run(config, use_cuda)
