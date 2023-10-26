import argparse
import json
import os
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.metric.utils import calc_wer, calc_cer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for key in ['test-other', 'test-clean']:
            logger.info(f'{key}:')
            results = []
            for batch_num, batch in enumerate(tqdm(dataloaders[key])):
                batch = Trainer.move_batch_to_device(batch, device)
                output = model(**batch)
                if type(output) is dict:
                    batch.update(output)
                else:
                    batch["logits"] = output
                batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
                batch["log_probs_length"] = model.transform_input_lengths(
                    batch["spectrogram_length"]
                )
                batch["probs"] = batch["log_probs"].exp().cpu()
                batch["argmax"] = batch["probs"].argmax(-1)
                for i in range(len(batch["text"])):
                    argmax = batch["argmax"][i]
                    argmax = argmax[: int(batch["log_probs_length"][i])]
                    ground_truth = batch["text"][i]
                    
                    pred_text_argmax = text_encoder.ctc_decode(argmax.cpu().numpy())
                    # pred_text_beam_search = text_encoder.ctc_beam_search_decode(
                    #     batch["probs"][i], batch["log_probs_length"][i], beam_size=2, use_lm=False
                    # )
                    pred_text_lm_beam_search = text_encoder.ctc_beam_search_decode(
                        batch["probs"][i], batch["log_probs_length"][i], beam_size=100
                    )
                    
                    argmax_wer = calc_wer(ground_truth, pred_text_argmax)
                    argmax_cer = calc_cer(ground_truth, pred_text_argmax)
                    # beam_search_wer = calc_wer(ground_truth, pred_text_beam_search)
                    # beam_search_cer = calc_cer(ground_truth, pred_text_beam_search)
                    lm_beam_search_wer = calc_wer(ground_truth, pred_text_lm_beam_search)
                    lm_beam_search_cer = calc_cer(ground_truth, pred_text_lm_beam_search)
                    
                    results.append(
                        {
                            "ground_truth": ground_truth,
                            "pred_text_argmax": pred_text_argmax,
                            "pred_text_beam_search": pred_text_beam_search,
                            "pred_text_lm_beam_search": pred_text_lm_beam_search,
                            "argmax_wer": argmax_wer,
                            "argmax_cer": argmax_cer,
                            # "beam_search_wer": beam_search_wer,
                            # "beam_search_cer": beam_search_cer,
                            "lm_beam_search_wer": lm_beam_search_wer,
                            "lm_beam_search_cer": lm_beam_search_cer
                        }
                    )
            table_dict = [
                {
                    'strategy': 'argmax', 
                    'wer': sum([r['argmax_wer'] for r in results]) / len(results),
                    'cer': sum([r['argmax_cer'] for r in results]) / len(results)
                },
                # {
                #     'strategy': 'beam_search',
                #     'wer': sum([r['beam_search_wer'] for r in results]) / len(results),
                #     'cer': sum([r['beam_search_cer'] for r in results]) / len(results)
                # },
                {
                    'strategy': 'lm_beam_search',
                    'wer': sum([r['lm_beam_search_wer'] for r in results]) / len(results),
                    'cer': sum([r['lm_beam_search_cer'] for r in results]) / len(results)
                }
            ]
            table = pd.DataFrame(table_dict)
            table = table.set_index('strategy')
            table.to_csv(f"{key}_metrics.csv", index=True)
            logger.info(table)
            
            
            with Path(f'{key}_{out_file}').open("w") as f:
                json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    # assert config.config.get("data", {}).get("test-clean", None) is not None

    main(config, args.output)
