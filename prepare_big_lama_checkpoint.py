#!/usr/bin/env python3
"""Extract a clean generator state dict from the verified official Big-LaMa checkpoint."""

import argparse
import hashlib
import pickle
from pathlib import Path

import torch


OFFICIAL_BIG_LAMA_CHECKPOINT_SHA256 = (
    "fccb7adffd53ec0974ee5503c3731c2c2f1e7e07856fd9228cdcc0b46fd5d423"
)


class _MetadataPlaceholder:
    """Inert target for training-only Lightning/OmegaConf metadata."""

    def __new__(cls, *_args, **_kwargs):
        return super().__new__(cls)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _RestrictedCheckpointUnpickler(pickle.Unpickler):
    _SAFE_GLOBALS = {
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch", "FloatStorage"),
        ("torch", "LongStorage"),
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "int"),
        ("typing", "Any"),
    }
    _IGNORED_TRAINING_GLOBALS = {
        ("pytorch_lightning.callbacks.model_checkpoint", "ModelCheckpoint"),
        ("omegaconf.dictconfig", "DictConfig"),
        ("omegaconf.base", "ContainerMetadata"),
        ("omegaconf.base", "Metadata"),
        ("omegaconf.nodes", "AnyNode"),
        ("omegaconf.listconfig", "ListConfig"),
    }

    def find_class(self, module, name):
        if module == "__builtin__":
            module = "builtins"
        if name == "long" and module == "builtins":
            name = "int"
        key = (module, name)
        if key in self._IGNORED_TRAINING_GLOBALS:
            return _MetadataPlaceholder
        if key not in self._SAFE_GLOBALS:
            raise pickle.UnpicklingError(
                f"Refusing unsupported checkpoint global {module}.{name}"
            )
        return super().find_class(module, name)


class _RestrictedPickleModule:
    __name__ = "restricted_big_lama_pickle"
    Unpickler = _RestrictedCheckpointUnpickler
    Pickler = pickle.Pickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="../lama_model/big-lama/models/best.ckpt",
    )
    parser.add_argument(
        "--output",
        default="../lama_model/big-lama/models/generator.pt",
    )
    parser.add_argument(
        "--expected-sha256",
        default=OFFICIAL_BIG_LAMA_CHECKPOINT_SHA256,
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    actual_hash = _sha256(checkpoint_path)
    expected_hash = str(args.expected_sha256).strip().lower()
    if expected_hash and actual_hash != expected_hash:
        raise RuntimeError(
            "Big-LaMa checkpoint SHA-256 mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        pickle_module=_RestrictedPickleModule,
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint does not contain a state_dict")
    generator_state = {
        key.removeprefix("generator."): value
        for key, value in state_dict.items()
        if key.startswith("generator.")
    }
    if not generator_state:
        raise RuntimeError("Checkpoint does not contain generator weights")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "big-lama-generator-v1",
        "source_checkpoint_sha256": actual_hash,
        "state_dict": generator_state,
    }, output_path)
    print(f"Saved {len(generator_state)} generator tensors to {output_path}")


if __name__ == "__main__":
    main()
