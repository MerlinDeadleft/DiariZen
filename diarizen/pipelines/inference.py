# Licensed under the MIT license.
# Copyright 2020 CNRS (author: Herve Bredin, herve.bredin@irit.fr)
# Copyright 2025 Brno University of Technology (author: Jiangyu Han, ihan@fit.vut.cz)

import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any

import toml
import numpy as np
import torch
import torchaudio

from scipy.ndimage import median_filter

from huggingface_hub import snapshot_download, hf_hub_download
from pyannote.audio.pipelines import SpeakerDiarization as SpeakerDiarizationPipeline
from pyannote.audio.utils.signal import Binarize
from pyannote.database.protocol.protocol import ProtocolFile
from pyannote.audio.core.task import Problem, Resolution
from pyannote.audio.core.model import Specifications

from diarizen.pipelines.utils import scp2path

from pyannote.audio import Model
from unittest.mock import patch
from types import SimpleNamespace

DIARIZEN_LOADING = False

# Store the original method so it can be called after modifying the data
original_from_pretrained = Model.from_pretrained


@classmethod
def patched_from_pretrained(cls, *args, **kwargs):
    checkpoint_path = args[0] if args else kwargs.get("checkpoint")

    if DIARIZEN_LOADING and isinstance(checkpoint_path, (str, os.PathLike)):
        print(f"DEBUG: DiariZen patch bypassing load for: {checkpoint_path}")

        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # 1. Inject missing version metadata
        if "pyannote.audio" not in checkpoint_dict:
            checkpoint_dict["pyannote.audio"] = {"versions": {"pyannote.audio": "3.1.1"}}

        # 2. Inject architecture metadata to fix the KeyError
        # This tells pyannote which class to use to load the weights
        if "architecture" not in checkpoint_dict["pyannote.audio"]:
            checkpoint_dict["pyannote.audio"]["architecture"] = {
                "module": "diarizen.models.eend.model_wavlm_conformer",
                "class": "Model"
            }

        # 3. Locate and Parse config.toml
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.toml")

        # Fallback defaults
        chunk_size = 16.0
        max_speakers_per_chunk = 4
        max_speakers_per_frame = 4

        if os.path.exists(config_path):
            try:
                # Use tomllib (Python 3.11+) or toml package
                try:
                    import tomllib  # Standard library in 3.11+
                    with open(config_path, "rb") as f:
                        config = tomllib.load(f)
                except ImportError:
                    import toml  # Fallback for older python versions
                    config = toml.load(config_path)

                model_args = config.get("model", {}).get("args", {})
                chunk_size = model_args.get("chunk_size", chunk_size)
                max_speakers_per_chunk = model_args.get("max_speakers_per_chunk", max_speakers_per_chunk)
                max_speakers_per_frame = model_args.get("max_speakers_per_frame", max_speakers_per_frame)

                print(f"DEBUG: Dynamic patch loaded from TOML: chunk={chunk_size}, "
                      f"max_spk_chunk={max_speakers_per_chunk}, max_spk_frame={max_speakers_per_frame}")
            except Exception as e:
                print(f"WARNING: Could not parse config.toml at {config_path}: {e}")

        # 4. Inject specifications so pyannote doesn't throw a KeyError
        # Recreate the specifications object to match what the model expects
        # These values should match the model's defaults
        checkpoint_dict["pyannote.audio"]["specifications"] = Specifications(
            problem=Problem.MONO_LABEL_CLASSIFICATION,
            resolution=Resolution.FRAME,
            duration=chunk_size,
            classes=[f"speaker_{i}" for i in range(max_speakers_per_chunk)],
            powerset_max_classes=max_speakers_per_frame,
            permutation_invariant=True
        )

        # 5. Inject Lightning version metadata
        if "pytorch-lightning_version" not in checkpoint_dict:
            # Set this to a standard version to skip migration logic
            checkpoint_dict["pytorch-lightning_version"] = "2.1.0"

        # 6. Ensure weights are nested under 'state_dict' for Lightning
        if "state_dict" not in checkpoint_dict:
            # Move all weights into a nested dictionary
            # But avoid moving the metadata that was just added
            metadata_keys = {"pyannote.audio", "pytorch-lightning_version"}
            actual_weights = {k: v for k, v in checkpoint_dict.items() if k not in metadata_keys}

            # Clean the top level
            for k in list(checkpoint_dict.keys()):
                if k not in metadata_keys:
                    del checkpoint_dict[k]

            # Re-insert weights under the expected key
            checkpoint_dict["state_dict"] = actual_weights

        # 7. Use mock to return our "enriched" dictionary
        from unittest.mock import patch
        with patch("torch.load", return_value=checkpoint_dict):
            # Also clear duration/sample_rate from kwargs if pyannote injected them
            return original_from_pretrained(*args, **kwargs)

    return original_from_pretrained(*args, **kwargs)


# Replace the original method
Model.from_pretrained = patched_from_pretrained


class DiariZenPipeline(SpeakerDiarizationPipeline):
    def __init__(
            self,
            diarizen_hub,
            embedding_model,
            config_parse: Optional[Dict[str, Any]] = None,
            rttm_out_dir: Optional[str] = None,
    ):
        config_path = Path(diarizen_hub / "config.toml")
        config = toml.load(config_path.as_posix())

        if config_parse is not None:
            print('Overriding with parsed config.')
            config["inference"]["args"] = config_parse["inference"]["args"]
            config["clustering"]["args"] = config_parse["clustering"]["args"]

        inference_config = config["inference"]["args"]
        clustering_config = config["clustering"]["args"]

        print(f'Loaded configuration: {config}')

        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        global DIARIZEN_LOADING
        DIARIZEN_LOADING = True
        try:
            super().__init__(
                segmentation=str(Path(diarizen_hub / "pytorch_model.bin")),
                segmentation_step=inference_config["segmentation_step"],
                embedding=embedding_model,
                embedding_exclude_overlap=True,
                clustering=clustering_config["method"],
                embedding_batch_size=inference_config["batch_size"],
                segmentation_batch_size=inference_config["batch_size"],
            )

            # Move the entire pipeline (models) to the detected device
            self.to(self.device)
        finally:
            DIARIZEN_LOADING = False

        self.apply_median_filtering = inference_config["apply_median_filtering"]
        self.min_speakers = clustering_config["min_speakers"]
        self.max_speakers = clustering_config["max_speakers"]

        if clustering_config["method"] == "AgglomerativeClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": clustering_config["min_cluster_size"],
                    "threshold": clustering_config["ahc_threshold"],
                }
            }
        elif clustering_config["method"] == "VBxClustering":
            self.PIPELINE_PARAMS = {
                "clustering": {
                    "threshold": clustering_config["ahc_threshold"],
                    "Fa": clustering_config["Fa"],
                    "Fb": clustering_config["Fb"],
                }
            }
            self.clustering.plda_dir = str(Path(diarizen_hub / "plda"))
            self.clustering.lda_dim = clustering_config["lda_dim"]
            self.clustering.maxIters = clustering_config["max_iters"]
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_config['method']}")

        self.instantiate(self.PIPELINE_PARAMS)

        if rttm_out_dir is not None:
            os.makedirs(rttm_out_dir, exist_ok=True)
        self.rttm_out_dir = rttm_out_dir

        assert self._segmentation.model.specifications.powerset is True

    @classmethod
    def from_pretrained(
            cls,
            repo_id: str,
            cache_dir: str = None,
            rttm_out_dir: str = None,
    ) -> "DiariZenPipeline":
        diarizen_hub = snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        embedding_model = hf_hub_download(
            repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
            filename="pytorch_model.bin",
            cache_dir=cache_dir,
            local_files_only=cache_dir is not None
        )

        return cls(
            diarizen_hub=Path(diarizen_hub).expanduser().absolute(),
            embedding_model=embedding_model,
            rttm_out_dir=rttm_out_dir
        )

    def __call__(self, in_wav, sess_name=None, num_speakers=None, min_speakers=None, max_speakers=None):
        # Handle WhisperX's Dictionary input: {'waveform': tensor, 'sample_rate': 16000}
        if isinstance(in_wav, dict) and "waveform" in in_wav:
            waveform = in_wav["waveform"]
            sample_rate = in_wav.get("sample_rate", 16000)

            # Ensure it's a float tensor on the correct device if needed
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float()
            else:
                waveform = waveform.float()

        # Handle raw Array/Tensor input
        elif hasattr(in_wav, "shape") and not isinstance(in_wav, str):
            if isinstance(in_wav, torch.Tensor):
                waveform = in_wav.float()
            else:
                waveform = torch.from_numpy(in_wav).float()
            sample_rate = 16000

        # Fallback for file paths (str or ProtocolFile)
        else:
            audio_path = in_wav if not isinstance(in_wav, ProtocolFile) else in_wav['audio']
            waveform, sample_rate = torchaudio.load(audio_path)

        # Standardize shape to (1, samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 5:
            waveform = waveform.T

        # Ensure only the first channel (mono) is used for DiariZen
        waveform = waveform[0:1, :]

        # Ensure the waveform is on the correct device
        waveform = waveform.to(self.device)

        print(f'Extracting segmentations on {self.device} device.')
        segmentations = self.get_segmentations({"waveform": waveform, "sample_rate": sample_rate})

        if self.apply_median_filtering:
            segmentations.data = median_filter(segmentations.data, size=(1, 11, 1), mode='reflect')

        # binarize segmentation
        binarized_segmentations = segmentations  # powerset

        # estimate frame-level number of instantaneous speakers
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation.model._receptive_field,
            warm_up=(0.0, 0.0),
        )

        print("Extracting Embeddings.")
        embeddings = self.get_embeddings(
            {"waveform": waveform, "sample_rate": sample_rate},
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap,
        )

        effective_min = num_speakers or min_speakers or self.min_speakers
        effective_max = num_speakers or max_speakers or self.max_speakers

        # shape: (num_chunks, local_num_speakers, dimension)
        print("Clustering.")
        hard_clusters, _, _ = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            min_clusters=effective_min,
            max_clusters=effective_max
        )

        # during counting, we could possibly overcount the number of instantaneous
        # speakers due to segmentation errors, so we cap the maximum instantaneous number
        # of speakers by the `max_speakers` value
        count.data = np.minimum(count.data, self.max_speakers).astype(np.int8)

        # keep track of inactive speakers
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        #   shape: (num_chunks, num_speakers)

        # reconstruct discrete diarization from raw hard clusters
        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )

        # convert to annotation
        to_annotation = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.0,
            min_duration_off=0.0
        )
        result = to_annotation(discrete_diarization)
        result.uri = sess_name

        if self.rttm_out_dir is not None:
            assert sess_name is not None
            rttm_out = os.path.join(self.rttm_out_dir, sess_name + ".rttm")
            with open(rttm_out, "w") as f:
                f.write(result.to_rttm())
        return SimpleNamespace(speaker_diarization=result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This script performs diarization using DiariZen pipeline ",
        add_help=True,
        usage="%(prog)s [options]",
    )

    # Required paths
    parser.add_argument(
        "--in_wav_scp",
        type=str,
        required=True,
        help="Path to wav.scp."
    )
    parser.add_argument(
        "--diarizen_hub",
        type=str,
        required=True,
        help="Path to DiariZen model hub directory."
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=True,
        help="Path to pretrained embedding model."
    )

    # inference parameters
    parser.add_argument(
        "--seg_duration",
        type=int,
        default=16,
        help="Segment duration in seconds.",
    )
    parser.add_argument(
        "--segmentation_step",
        type=float,
        default=0.1,
        help="Shifting ratio during segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Input batch size for inference.",
    )
    parser.add_argument(
        "--apply_median_filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply median filtering to segmentation output.",
    )

    # clustering parameters
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="VBxClustering",
        choices=["VBxClustering", "AgglomerativeClustering"],
        help="Clustering method to use.",
    )
    parser.add_argument(
        "--min_speakers",
        type=int,
        default=1,
        help="Minimum number of speakers.",
    )
    parser.add_argument(
        "--max_speakers",
        type=int,
        default=20,
        help="Maximum number of speakers.",
    )
    parser.add_argument(
        "--ahc_criterion",
        type=str,
        default="distance",
        help="AHC criterion (for VBx).",
    )
    parser.add_argument(
        "--ahc_threshold",
        type=float,
        default=0.6,
        help="AHC threshold.",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=13,
        help="Minimum cluster size (for AHC).",
    )
    parser.add_argument(
        "--Fa",
        type=float,
        default=0.07,
        help="VBx Fa parameter.",
    )
    parser.add_argument(
        "--Fb",
        type=float,
        default=0.8,
        help="VBx Fb parameter.",
    )
    parser.add_argument(
        "--lda_dim",
        type=int,
        default=128,
        help="VBx LDA dimension.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=20,
        help="VBx maximum iterations.",
    )

    # Output
    parser.add_argument(
        "--rttm_out_dir",
        type=str,
        default=None,
        required=False,
        help="Path to output folder.",
    )

    args = parser.parse_args()
    print(args)

    inference_config = {
        "seg_duration": args.seg_duration,
        "segmentation_step": args.segmentation_step,
        "batch_size": args.batch_size,
        "apply_median_filtering": args.apply_median_filtering
    }

    clustering_config = {
        "method": args.clustering_method,
        "min_speakers": args.min_speakers,
        "max_speakers": args.max_speakers
    }
    if args.clustering_method == "AgglomerativeClustering":
        clustering_config.update({
            "ahc_threshold": args.ahc_threshold,
            "min_cluster_size": args.min_cluster_size
        })
    elif args.clustering_method == "VBxClustering":
        clustering_config.update({
            "ahc_criterion": args.ahc_criterion,
            "ahc_threshold": args.ahc_threshold,
            "Fa": args.Fa,
            "Fb": args.Fb,
            "lda_dim": args.lda_dim,
            "max_iters": args.max_iters
        })
    else:
        raise ValueError(f"Unsupported clustering method: {args.clustering_method}")

    config_parse = {
        "inference": {"args": inference_config},
        "clustering": {"args": clustering_config}
    }

    diarizen_pipeline = DiariZenPipeline(
        diarizen_hub=Path(args.diarizen_hub),
        embedding_model=args.embedding_model,
        config_parse=config_parse,
        rttm_out_dir=args.rttm_out_dir
    )

    audio_f = scp2path(args.in_wav_scp)
    for audio_file in audio_f:
        sess_name = Path(audio_file).stem.split('.')[0]
        print(f'Prosessing: {sess_name}')
        diarizen_pipeline(audio_file, sess_name=sess_name)
