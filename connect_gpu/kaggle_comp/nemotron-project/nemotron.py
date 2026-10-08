# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

if "KAGGLE_KERNEL_RUN_TYPE" not in os.environ:
    raise EnvironmentError(
        "Run this notebook on Kaggle with GPU enabled. It is not intended for local execution."
    )


BASE_REQUIRED_PACKAGES = {
    "kagglehub": "kagglehub>=1.0.0",
    "datasets": "datasets>=3.0.0",
    "peft": "peft>=0.15.0",
    "accelerate": "accelerate>=1.5.0",
    "sentencepiece": "sentencepiece>=0.2.0",
    "transformers": "transformers>=4.57.3",
}

OPTIONAL_RUNTIME_PACKAGES = {
    "bitsandbytes": "bitsandbytes>=0.46.0",
}

MODEL_RUNTIME_RELEASES = {
    "causal_conv1d": {
        "repo": "Dao-AILab/causal-conv1d",
        "tag": "v1.6.1.post4",
        "filename_prefix": "causal_conv1d-1.6.1",
    },
    "mamba_ssm": {
        "repo": "state-spaces/mamba",
        "tag": "v2.3.1",
        "filename_prefix": "mamba_ssm-2.3.1",
    },
}

MINIMUM_CUDA_CAPABILITY = (7, 0)
DEFAULT_PIP_TIMEOUT_SECONDS = "30"
DEFAULT_PIP_RETRIES = "1"


def dns_available(hostname: str) -> bool:
    try:
        socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    return True


def pip_install_arguments(packages: list[str], no_build_isolation: bool = False) -> list[str]:
    arguments = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--disable-pip-version-check",
        "--retries",
        DEFAULT_PIP_RETRIES,
        "--timeout",
        DEFAULT_PIP_TIMEOUT_SECONDS,
    ]
    if no_build_isolation:
        arguments.append("--no-build-isolation")
    arguments.extend(packages)
    return arguments


def install_missing_packages(
    requirements: Dict[str, str],
    retry_with_no_build_isolation: bool = False,
) -> None:
    missing_modules = [
        module_name
        for module_name in requirements
        if importlib.util.find_spec(module_name) is None
    ]
    missing_packages = [
        requirement
        for module_name, requirement in requirements.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if not missing_packages:
        return

    if not dns_available("pypi.org"):
        raise RuntimeError(
            "Missing required Python packages but the Kaggle draft session cannot resolve PyPI. "
            f"Missing modules={missing_modules}. Turn Internet on in the Kaggle session settings, "
            "or rerun the published notebook version that has Internet enabled."
        )

    install_command = pip_install_arguments(missing_packages)
    try:
        subprocess.check_call(install_command)
    except subprocess.CalledProcessError:
        if not retry_with_no_build_isolation:
            raise

        retry_command = pip_install_arguments(missing_packages, no_build_isolation=True)
        subprocess.check_call(retry_command)


install_missing_packages(BASE_REQUIRED_PACKAGES)


# %%
import gc
import inspect
import json
import platform
import random
import re
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, cast

import kagglehub
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from IPython.display import display
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import DataCollator


assert torch.cuda.is_available(), "Enable a Kaggle GPU accelerator before running this notebook."

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")


# %%
def runtime_wheel_tags() -> Dict[str, str]:
    torch_version_match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    if torch_version_match is None:
        raise RuntimeError(f"Unsupported torch version format: {torch.__version__}")

    cuda_version = torch.version.cuda
    if not cuda_version:
        raise RuntimeError("This notebook requires a CUDA-enabled PyTorch runtime.")

    machine = platform.machine().lower()
    if sys.platform != "linux":
        raise RuntimeError(f"Unsupported platform for Nemotron runtime wheels: {sys.platform}")
    if machine in {"x86_64", "amd64"}:
        platform_tag = "linux_x86_64"
    elif machine in {"aarch64", "arm64"}:
        platform_tag = "linux_aarch64"
    else:
        raise RuntimeError(f"Unsupported machine architecture for Nemotron runtime wheels: {machine}")

    if hasattr(torch, "compiled_with_cxx11_abi"):
        abi_enabled = bool(torch.compiled_with_cxx11_abi())
    else:
        abi_enabled = bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False))

    return {
        "python_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "torch_tag": f"torch{torch_version_match.group(1)}.{torch_version_match.group(2)}",
        "cuda_tag": f"cu{cuda_version.split('.')[0]}",
        "primary_abi_tag": f"cxx11abi{'TRUE' if abi_enabled else 'FALSE'}",
        "fallback_abi_tag": f"cxx11abi{'FALSE' if abi_enabled else 'TRUE'}",
        "platform_tag": platform_tag,
    }


def runtime_wheel_candidate_names(module_name: str, wheel_tags: Dict[str, str]) -> List[str]:
    release_info = MODEL_RUNTIME_RELEASES[module_name]
    return [
        (
            f"{release_info['filename_prefix']}+{wheel_tags['cuda_tag']}{wheel_tags['torch_tag']}{abi_tag}"
            f"-{wheel_tags['python_tag']}-{wheel_tags['python_tag']}-{wheel_tags['platform_tag']}.whl"
        )
        for abi_tag in (wheel_tags["primary_abi_tag"], wheel_tags["fallback_abi_tag"])
    ]


def local_wheel_search_roots() -> List[Path]:
    roots = [
        Path.cwd(),
        Path("/kaggle/working"),
        Path("/kaggle/input"),
        Path("/root/.cache/pip"),
        Path.home() / ".cache" / "pip",
    ]
    unique_roots: List[Path] = []
    seen = set()
    for root in roots:
        normalized = str(root)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_roots.append(root)
    return unique_roots


def find_local_wheel(candidate_names: Sequence[str]) -> Optional[Path]:
    for root in local_wheel_search_roots():
        if not root.exists():
            continue
        for candidate_name in candidate_names:
            direct_path = root / candidate_name
            if direct_path.exists():
                return direct_path
            matches = list(root.rglob(candidate_name))
            if matches:
                return matches[0]
    return None


def resolve_release_wheel_url(module_name: str, wheel_tags: Dict[str, str]) -> Tuple[str, str]:
    release_info = MODEL_RUNTIME_RELEASES[module_name]
    candidate_names = runtime_wheel_candidate_names(module_name, wheel_tags)

    local_wheel = find_local_wheel(candidate_names)
    if local_wheel is not None:
        return local_wheel.name, str(local_wheel)

    if not dns_available("github.com"):
        raise RuntimeError(
            "Nemotron runtime wheels are missing locally and the Kaggle session cannot resolve github.com. "
            f"Tried={candidate_names}. Turn Internet on in the draft session, or attach a wheel dataset under /kaggle/input."
        )

    for wheel_name in candidate_names:
        wheel_url = f"https://github.com/{release_info['repo']}/releases/download/{release_info['tag']}/{wheel_name}"
        return wheel_name, wheel_url

    raise RuntimeError(
        "No compatible prebuilt wheel was found for the Nemotron runtime dependency "
        f"{module_name!r}. Tried={candidate_names}"
    )


def install_model_runtime_packages() -> None:
    missing_modules = [
        module_name for module_name in MODEL_RUNTIME_RELEASES if importlib.util.find_spec(module_name) is None
    ]
    if not missing_modules:
        return

    wheel_tags = runtime_wheel_tags()
    print(json.dumps({"runtime_package_tags": wheel_tags, "missing_modules": missing_modules}, indent=2))

    wheel_names = []
    wheel_urls = []
    for module_name in missing_modules:
        wheel_name, wheel_url = resolve_release_wheel_url(module_name, wheel_tags)
        wheel_names.append(wheel_name)
        wheel_urls.append(wheel_url)

    print(json.dumps({"runtime_wheels": wheel_names, "runtime_sources": wheel_urls}, indent=2))

    install_command = pip_install_arguments(["--no-deps", *wheel_urls])
    try:
        subprocess.check_call(install_command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to install the prebuilt Nemotron runtime wheels. "
            f"Torch={torch.__version__}, CUDA={torch.version.cuda}, wheels={wheel_names}"
        ) from exc

    importlib.invalidate_caches()
    remaining_missing = [
        module_name for module_name in MODEL_RUNTIME_RELEASES if importlib.util.find_spec(module_name) is None
    ]
    if remaining_missing:
        raise RuntimeError(f"Nemotron runtime installation completed but modules are still missing: {remaining_missing}")


install_model_runtime_packages()


# %%
@dataclass(frozen=True)
class TrainingStage:
    name: str
    competition_limit: int
    aux_limit: int
    epochs: float
    learning_rate: float
    max_seq_length: int
    gradient_accumulation_steps: int


@dataclass(frozen=True)
class RunVariant:
    name: str
    description: str
    debug: bool
    lora_rank: int
    lora_alpha_multiplier: float
    lora_target_modules: Tuple[str, ...]
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    max_valid_samples: int
    preview_max_new_tokens: int
    logging_steps: int
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    gpu_headroom_gb: float
    cpu_budget_gb: int
    allow_cpu_offload: bool
    offload_buffers: bool
    attention_implementation: str
    experts_implementation: str
    use_aux_dataset: bool
    stages: Tuple[TrainingStage, ...]

    @property
    def max_sequence_length(self) -> int:
        return max(stage.max_seq_length for stage in self.stages)

    @property
    def final_stage(self) -> TrainingStage:
        return self.stages[-1]


@dataclass(frozen=True)
class RuntimeProfile:
    gpu_count: int
    gpu_names: Tuple[str, ...]
    gpu_capabilities: Tuple[Tuple[int, int], ...]
    gpu_memory_gb: Tuple[float, ...]
    cpu_memory_gb: float


SEED = 42
COMPETITION_HANDLE = "nvidia-nemotron-model-reasoning-challenge"
MODEL_HANDLE = "metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1"
AUX_DATASET_HANDLE = "thedevastator/grade-school-math-8k-q-a"

VARIANT_ALIASES = {
    "smoke_debug": "t4x2_smoke",
    "competition_plus_gsm8k": "t4x2_balanced",
    "higher_rank": "t4x2_accuracy",
    "gpu_fast": "t4x2_fast",
    "gpu_balanced": "t4x2_balanced",
    "gpu_accuracy": "t4x2_accuracy",
    "rtxpro6000_fast": "t4x2_fast",
    "rtxpro6000_balanced": "t4x2_balanced",
    "rtxpro6000_accuracy": "t4x2_accuracy",
    "rtx_pro_6000_fast": "t4x2_fast",
    "rtx_pro_6000_balanced": "t4x2_balanced",
    "rtx_pro_6000_accuracy": "t4x2_accuracy",
}

RUN_VARIANTS = {
    "t4x2_smoke": RunVariant(
        name="t4x2_smoke",
        description="Fast sanity check on dual T4 with minimal memory and time.",
        debug=True,
        lora_rank=8,
        lora_alpha_multiplier=2.0,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_valid_samples=12,
        preview_max_new_tokens=96,
        logging_steps=2,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        gpu_headroom_gb=1.5,
        cpu_budget_gb=16,
        allow_cpu_offload=True,
        offload_buffers=False,
        attention_implementation="sdpa",
        experts_implementation="eager",
        use_aux_dataset=True,
        stages=(
            TrainingStage(
                name="smoke",
                competition_limit=32,
                aux_limit=64,
                epochs=0.15,
                learning_rate=3e-4,
                max_seq_length=768,
                gradient_accumulation_steps=4,
            ),
        ),
    ),
    "t4x2_fast": RunVariant(
        name="t4x2_fast",
        description="Best turnaround time while still using both T4s and LoRA.",
        debug=True,
        lora_rank=8,
        lora_alpha_multiplier=2.0,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_valid_samples=24,
        preview_max_new_tokens=128,
        logging_steps=5,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=0.3,
        gpu_headroom_gb=1.2,
        cpu_budget_gb=18,
        allow_cpu_offload=True,
        offload_buffers=False,
        attention_implementation="sdpa",
        experts_implementation="eager",
        use_aux_dataset=True,
        stages=(
            TrainingStage(
                name="reasoning_warmup",
                competition_limit=96,
                aux_limit=256,
                epochs=0.30,
                learning_rate=2.5e-4,
                max_seq_length=1024,
                gradient_accumulation_steps=8,
            ),
            TrainingStage(
                name="competition_focus",
                competition_limit=224,
                aux_limit=96,
                epochs=0.35,
                learning_rate=1.8e-4,
                max_seq_length=1024,
                gradient_accumulation_steps=10,
            ),
        ),
    ),
    "t4x2_balanced": RunVariant(
        name="t4x2_balanced",
        description="Default dual-T4 plan balancing memory, speed, and accuracy.",
        debug=True,
        lora_rank=16,
        lora_alpha_multiplier=2.0,
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_valid_samples=40,
        preview_max_new_tokens=160,
        logging_steps=5,
        warmup_ratio=0.04,
        weight_decay=0.01,
        max_grad_norm=0.3,
        gpu_headroom_gb=0.9,
        cpu_budget_gb=22,
        allow_cpu_offload=True,
        offload_buffers=False,
        attention_implementation="sdpa",
        experts_implementation="eager",
        use_aux_dataset=True,
        stages=(
            TrainingStage(
                name="reasoning_warmup",
                competition_limit=160,
                aux_limit=384,
                epochs=0.40,
                learning_rate=2.0e-4,
                max_seq_length=1024,
                gradient_accumulation_steps=12,
            ),
            TrainingStage(
                name="mixed_competition",
                competition_limit=320,
                aux_limit=192,
                epochs=0.55,
                learning_rate=1.6e-4,
                max_seq_length=1280,
                gradient_accumulation_steps=16,
            ),
        ),
    ),
    "t4x2_accuracy": RunVariant(
        name="t4x2_accuracy",
        description="Uses more memory and more training stages for the best score on dual T4.",
        debug=True,
        lora_rank=24,
        lora_alpha_multiplier=2.0,
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_valid_samples=64,
        preview_max_new_tokens=192,
        logging_steps=8,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=0.3,
        gpu_headroom_gb=0.7,
        cpu_budget_gb=24,
        allow_cpu_offload=True,
        offload_buffers=False,
        attention_implementation="sdpa",
        experts_implementation="eager",
        use_aux_dataset=True,
        stages=(
            TrainingStage(
                name="reasoning_warmup",
                competition_limit=192,
                aux_limit=640,
                epochs=0.45,
                learning_rate=2.0e-4,
                max_seq_length=1024,
                gradient_accumulation_steps=16,
            ),
            TrainingStage(
                name="mixed_competition",
                competition_limit=384,
                aux_limit=256,
                epochs=0.55,
                learning_rate=1.5e-4,
                max_seq_length=1280,
                gradient_accumulation_steps=20,
            ),
            TrainingStage(
                name="competition_focus",
                competition_limit=512,
                aux_limit=96,
                epochs=0.35,
                learning_rate=1.1e-4,
                max_seq_length=1536,
                gradient_accumulation_steps=24,
            ),
        ),
    ),
}

REQUESTED_VARIANT = os.environ.get("NEMOTRON_VARIANT", "gpu_balanced")
ACTIVE_VARIANT = VARIANT_ALIASES.get(REQUESTED_VARIANT, REQUESTED_VARIANT)
REQUESTED_LOAD_MODE = os.environ.get("NEMOTRON_LOAD_MODE", "auto").strip().lower()

if ACTIVE_VARIANT not in RUN_VARIANTS:
    raise KeyError(f"Unknown ACTIVE_VARIANT={REQUESTED_VARIANT!r}. Choose one of: {sorted(RUN_VARIANTS)}")

if REQUESTED_LOAD_MODE not in {"auto", "4bit", "full", "fp16", "bf16"}:
    raise KeyError(
        f"Unknown NEMOTRON_LOAD_MODE={REQUESTED_LOAD_MODE!r}. "
        "Choose one of: ['auto', '4bit', 'full', 'fp16', 'bf16']"
    )

VARIANT = RUN_VARIANTS[ACTIVE_VARIANT]
WORKING_DIR = Path("/kaggle/working")
RUN_NAME = f"nemotron_{ACTIVE_VARIANT}_seed{SEED}"
RUN_DIR = WORKING_DIR / RUN_NAME
ARTIFACT_DIR = RUN_DIR / "artifacts"
ADAPTER_DIR = ARTIFACT_DIR / "adapter"
OFFLOAD_DIR = RUN_DIR / "offload"
SUBMISSION_PATH = RUN_DIR / "submission.zip"
BF16_AVAILABLE = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
COMPUTE_DTYPE = torch.bfloat16 if BF16_AVAILABLE else torch.float16
BITSANDBYTES_AVAILABLE = importlib.util.find_spec("bitsandbytes") is not None
MAX_SEQ_LENGTH = VARIANT.max_sequence_length
SYSTEM_PROMPT = (
    "You are solving a reasoning benchmark problem. Solve it carefully and end the visible response "
    "with exactly one final answer written as \\boxed{...}."
)
QUESTION_COLUMN_CANDIDATES = (
    "question",
    "problem",
    "prompt",
    "query",
    "task",
    "instruction",
    "input",
)
ANSWER_COLUMN_CANDIDATES = (
    "answer",
    "solution",
    "target",
    "output",
    "final_answer",
    "ground_truth",
    "label",
)
FINAL_MARKER_PATTERN = re.compile(r"####\s*([^\n]+)")
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
NUMBER_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")

set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True


# %%
class RuntimeManager:
    @staticmethod
    def system_memory_gb() -> float:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            physical_pages = os.sysconf("SC_PHYS_PAGES")
            return float(page_size * physical_pages) / (1024**3)
        except (AttributeError, OSError, ValueError):
            return 30.0

    @classmethod
    def build_profile(cls) -> RuntimeProfile:
        gpu_names = []
        gpu_capabilities = []
        gpu_memory_gb = []
        for index in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(index))
            major, minor = torch.cuda.get_device_capability(index)
            gpu_capabilities.append((int(major), int(minor)))
            gpu_memory_gb.append(torch.cuda.get_device_properties(index).total_memory / (1024**3))
        return RuntimeProfile(
            gpu_count=torch.cuda.device_count(),
            gpu_names=tuple(gpu_names),
            gpu_capabilities=tuple(gpu_capabilities),
            gpu_memory_gb=tuple(gpu_memory_gb),
            cpu_memory_gb=cls.system_memory_gb(),
        )

    @staticmethod
    def ensure_supported_gpus(profile: RuntimeProfile) -> None:
        if profile.gpu_count == 0:
            raise RuntimeError("No CUDA GPUs are visible to this Kaggle session.")

        for gpu_name, capability in zip(profile.gpu_names, profile.gpu_capabilities):
            if capability < MINIMUM_CUDA_CAPABILITY:
                raise RuntimeError(
                    "Kaggle assigned an unsupported GPU for this Nemotron stack: "
                    f"{gpu_name} (sm_{capability[0]}{capability[1]}). "
                    "This notebook requires a Kaggle GPU with compute capability sm_70 or newer. "
                    "Stop the Kaggle session, start a new GPU session, and rerun from the first cell until you get "
                    "a T4, L4, A10G, or A100 instead of a P100."
                )

    @staticmethod
    def print_profile(profile: RuntimeProfile, variant: RunVariant) -> None:
        print(
            json.dumps(
                {
                    "requested_variant": REQUESTED_VARIANT,
                    "active_variant": ACTIVE_VARIANT,
                    "requested_load_mode": REQUESTED_LOAD_MODE,
                    "bitsandbytes_available": BITSANDBYTES_AVAILABLE,
                    "variant": asdict(variant),
                    "gpu_count": profile.gpu_count,
                    "gpu_names": list(profile.gpu_names),
                    "gpu_capabilities": [list(item) for item in profile.gpu_capabilities],
                    "gpu_memory_gb": [round(value, 2) for value in profile.gpu_memory_gb],
                    "cpu_memory_gb": round(profile.cpu_memory_gb, 2),
                    "allocator": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                },
                indent=2,
            )
        )

    @staticmethod
    def clear_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    @staticmethod
    def snapshot(label: str) -> None:
        print(f"\n{label}:")
        for index in range(torch.cuda.device_count()):
            allocated_gb = torch.cuda.memory_allocated(index) / (1024**3)
            reserved_gb = torch.cuda.memory_reserved(index) / (1024**3)
            print(f"  cuda:{index} allocated={allocated_gb:.2f} GB reserved={reserved_gb:.2f} GB")

    @staticmethod
    def build_gpu_max_memory_map(profile: RuntimeProfile, headroom_gb: float) -> Dict[Any, str]:
        max_memory: Dict[Any, str] = {}
        for index, total_memory_gb in enumerate(profile.gpu_memory_gb):
            usable_gb = max(8.0, total_memory_gb - headroom_gb)
            usable_mib = int(usable_gb * 1024)
            max_memory[index] = f"{usable_mib}MiB"
        return max_memory

    @staticmethod
    def build_gpu_cpu_max_memory_map(
        profile: RuntimeProfile,
        headroom_gb: float,
        cpu_budget_gb: int,
    ) -> Dict[Any, str]:
        max_memory = RuntimeManager.build_gpu_max_memory_map(profile, headroom_gb)
        cpu_budget = max(8, min(int(profile.cpu_memory_gb - 4), cpu_budget_gb))
        max_memory["cpu"] = f"{cpu_budget}GiB"
        return max_memory


def maybe_install_optional_bitsandbytes(profile: RuntimeProfile) -> bool:
    if BITSANDBYTES_AVAILABLE:
        return True

    should_try_install = REQUESTED_LOAD_MODE == "4bit" or sum(profile.gpu_memory_gb) < 70
    if not should_try_install:
        return False

    try:
        install_missing_packages(OPTIONAL_RUNTIME_PACKAGES)
    except Exception as exc:
        print(f"Optional bitsandbytes install skipped: {exc}")

    return importlib.util.find_spec("bitsandbytes") is not None


RUNTIME_PROFILE = RuntimeManager.build_profile()
RuntimeManager.ensure_supported_gpus(RUNTIME_PROFILE)
BITSANDBYTES_AVAILABLE = maybe_install_optional_bitsandbytes(RUNTIME_PROFILE)
RuntimeManager.print_profile(RUNTIME_PROFILE, VARIANT)

input_root = Path("/kaggle/input")
if input_root.exists():
    attached_inputs = sorted(child.name for child in input_root.iterdir())
    print(f"Attached Kaggle inputs: {attached_inputs}")


# %%
def handle_slug(handle: str) -> str:
    if "/" not in handle:
        return handle
    return handle.split("/")[1]


def empty_supervised_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["question", "answer", "source", "file_name"])


def resolve_competition_dir(handle: str) -> Path:
    mounted_dir = Path("/kaggle/input") / handle
    if mounted_dir.exists():
        return mounted_dir
    return Path(kagglehub.competition_download(handle))


def resolve_dataset_dir(handle: str) -> Path:
    mounted_dir = Path("/kaggle/input") / handle_slug(handle)
    if mounted_dir.exists():
        return mounted_dir
    return Path(kagglehub.dataset_download(handle))


def resolve_model_dir(handle: str) -> Path:
    input_root = Path("/kaggle/input")
    model_slug = handle_slug(handle).lower()

    if input_root.exists():
        for config_path in input_root.rglob("config.json"):
            model_dir = config_path.parent
            model_dir_text = str(model_dir).lower()
            has_weights = model_dir.joinpath("model.safetensors.index.json").exists() or any(
                model_dir.glob("model-*.safetensors")
            )
            if model_slug in model_dir_text and has_weights and model_dir.joinpath("tokenizer_config.json").exists():
                return model_dir

    return Path(kagglehub.model_download(handle))


def discover_tabular_files(root_dir: Path) -> List[Path]:
    supported_suffixes = {".csv", ".json", ".jsonl", ".parquet", ".pq", ".tsv"}
    files = []
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_suffixes:
            files.append(file_path)
    return sorted(files)


def load_table(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix == ".jsonl":
        return pd.read_json(file_path, lines=True)
    if suffix == ".json":
        payload = json.loads(file_path.read_text())
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    return pd.DataFrame(value)
        raise ValueError(f"Unsupported JSON structure in {file_path}")
    raise ValueError(f"Unsupported file type: {file_path}")


def infer_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    normalized_columns = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized_columns:
            return normalized_columns[candidate]

    for candidate in candidates:
        for normalized_name, original_name in normalized_columns.items():
            if candidate in normalized_name:
                return original_name

    return None


def to_supervised_frame(
    frame: pd.DataFrame,
    question_column: str,
    answer_column: str,
    source_name: str,
    file_name: str,
) -> pd.DataFrame:
    cleaned = frame[[question_column, answer_column]].copy()
    cleaned.columns = ["question", "answer"]
    cleaned = cleaned.dropna(subset=["question", "answer"])
    cleaned["question"] = cleaned["question"].astype(str).str.strip()
    cleaned["answer"] = cleaned["answer"].astype(str).str.strip()
    cleaned = cleaned[(cleaned["question"] != "") & (cleaned["answer"] != "")]
    cleaned["source"] = source_name
    cleaned["file_name"] = file_name
    cleaned = cleaned.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
    return cleaned


def collect_supervised_candidates(root_dir: Path, source_name: str) -> List[Tuple[Path, pd.DataFrame]]:
    candidates: List[Tuple[Path, pd.DataFrame]] = []
    for file_path in discover_tabular_files(root_dir):
        try:
            frame = load_table(file_path)
        except Exception as exc:
            print(f"Skipping {file_path.name}: {exc}")
            continue

        question_column = infer_column(frame.columns, QUESTION_COLUMN_CANDIDATES)
        answer_column = infer_column(frame.columns, ANSWER_COLUMN_CANDIDATES)
        if question_column is None or answer_column is None:
            continue

        cleaned = to_supervised_frame(
            frame=frame,
            question_column=question_column,
            answer_column=answer_column,
            source_name=source_name,
            file_name=file_path.name,
        )
        if not cleaned.empty:
            candidates.append((file_path, cleaned))

    return candidates


def select_best_candidate(
    candidates: Sequence[Tuple[Path, pd.DataFrame]],
    split: str,
    exclude_paths: Optional[Sequence[Path]] = None,
) -> Optional[Tuple[Path, pd.DataFrame]]:
    if not candidates:
        return None

    exclude_set = {path.resolve() for path in (exclude_paths or [])}
    split_weights = {
        "train": {
            "train": 8,
            "main": 3,
            "socratic": 1,
            "valid": -4,
            "val": -4,
            "dev": -4,
            "test": -6,
        },
        "valid": {
            "valid": 8,
            "val": 8,
            "dev": 8,
            "test": 7,
            "main": 1,
            "socratic": 1,
            "train": -2,
        },
    }

    scored_candidates = []
    for file_path, frame in candidates:
        if file_path.resolve() in exclude_set:
            continue

        file_name = file_path.name.lower()
        score = min(len(frame), 100000)
        for token, weight in split_weights[split].items():
            if token in file_name:
                score += weight
        scored_candidates.append((score, file_path, frame))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_path, best_frame = scored_candidates[0]
    return best_path, best_frame


def sample_frame(frame: pd.DataFrame, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def split_train_valid(frame: pd.DataFrame, valid_rows: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    bounded_valid_rows = min(valid_rows, max(1, len(shuffled) - 1))
    valid_df = shuffled.iloc[:bounded_valid_rows].copy().reset_index(drop=True)
    train_df = shuffled.iloc[bounded_valid_rows:].copy().reset_index(drop=True)
    return train_df, valid_df


def summarize_candidates(label: str, candidates: Sequence[Tuple[Path, pd.DataFrame]]) -> None:
    print(f"\n{label} candidates:")
    if not candidates:
        print("  none")
        return
    for file_path, frame in candidates:
        print(f"  {file_path.name}: rows={len(frame)} columns={list(frame.columns)}")


def summarize_frame(label: str, frame: pd.DataFrame) -> None:
    print(f"\n{label}: rows={len(frame)}")
    if "source" in frame.columns:
        print(frame["source"].value_counts().to_string())
    display(frame.head(3))


def summarize_source_mix(frame: pd.DataFrame) -> Dict[str, int]:
    if frame.empty or "source" not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame["source"].value_counts().to_dict().items()}


def extract_boxed_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""

    marker_match = FINAL_MARKER_PATTERN.search(text)
    if marker_match:
        return marker_match.group(1).strip().replace(",", "")

    matches = BOXED_PATTERN.findall(text)
    if matches:
        return matches[-1].strip()

    numeric_matches = NUMBER_PATTERN.findall(text)
    if numeric_matches:
        return numeric_matches[-1].replace(",", "").strip()

    return text.strip()


def normalize_answer_for_training(answer_text: str) -> str:
    raw_text = str(answer_text).strip()
    marker_match = FINAL_MARKER_PATTERN.search(raw_text)
    if marker_match:
        final_answer = marker_match.group(1).strip().replace(",", "")
        reasoning = raw_text[: marker_match.start()].strip()
        if reasoning:
            return f"{reasoning}\n\nTherefore the final answer is \\boxed{{{final_answer}}}."
        return f"The final answer is \\boxed{{{final_answer}}}."

    if "\\boxed{" in raw_text:
        return raw_text

    cleaned = extract_boxed_answer(raw_text)
    if cleaned:
        return f"The final answer is \\boxed{{{cleaned}}}."
    return raw_text


def maybe_parse_number(text: str) -> Optional[float]:
    cleaned = text.replace(",", "").strip()
    if cleaned.count("/") == 1 and all(part.strip("-").isdigit() for part in cleaned.split("/")):
        numerator, denominator = cleaned.split("/")
        denominator_value = float(denominator)
        if denominator_value == 0:
            return None
        return float(numerator) / denominator_value

    try:
        return float(cleaned)
    except ValueError:
        return None


def answers_match(predicted: str, target: str, tolerance: float = 1e-2) -> bool:
    normalized_prediction = extract_boxed_answer(predicted)
    normalized_target = extract_boxed_answer(target)
    if normalized_prediction == normalized_target:
        return True

    prediction_value = maybe_parse_number(normalized_prediction)
    target_value = maybe_parse_number(normalized_target)
    if prediction_value is None or target_value is None:
        return False

    if target_value == 0:
        return abs(prediction_value) <= tolerance

    relative_error = abs(prediction_value - target_value) / abs(target_value)
    return relative_error <= tolerance


def serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, float)):
            serialized[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            serialized[key] = int(value)
        elif isinstance(value, torch.Tensor):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def render_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        signature = inspect.signature(tokenizer.apply_chat_template)
        if "enable_thinking" in signature.parameters:
            kwargs["enable_thinking"] = False
        return tokenizer.apply_chat_template(messages, **kwargs)

    text_blocks = [f"{message['role'].upper()}: {message['content']}" for message in messages]
    if add_generation_prompt:
        text_blocks.append("ASSISTANT:")
    return "\n\n".join(text_blocks)


def tokenize_supervised_examples(tokenizer, frame: pd.DataFrame, max_length: int) -> Dataset:
    records = []
    for row in frame.itertuples(index=False):
        question = str(row.question)
        answer = normalize_answer_for_training(str(row.answer))
        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        full_messages = prompt_messages + [{"role": "assistant", "content": answer}]

        prompt_text = render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
        full_text = render_chat(tokenizer, full_messages, add_generation_prompt=False)

        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )

        labels = list(full_tokens["input_ids"])
        prompt_token_count = min(len(prompt_tokens["input_ids"]), len(labels))
        labels[:prompt_token_count] = [-100] * prompt_token_count

        if all(label == -100 for label in labels):
            continue

        records.append(
            {
                "input_ids": full_tokens["input_ids"],
                "attention_mask": full_tokens["attention_mask"],
                "labels": labels,
            }
        )

    return Dataset.from_list(records)


def build_generation_inputs(tokenizer, question: str, max_length: int) -> Dict[str, torch.Tensor]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt_text = render_chat(tokenizer, messages, add_generation_prompt=True)
    return tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)


# %%
class SupervisedDataBuilder:
    def __init__(
        self,
        competition_dir: Path,
        aux_dataset_dir: Path,
        variant: RunVariant,
        seed: int,
    ) -> None:
        self.competition_dir = competition_dir
        self.aux_dataset_dir = aux_dataset_dir
        self.variant = variant
        self.seed = seed

        self.competition_candidates = collect_supervised_candidates(self.competition_dir, source_name="competition")
        self.aux_candidates = (
            collect_supervised_candidates(self.aux_dataset_dir, source_name=handle_slug(AUX_DATASET_HANDLE))
            if self.variant.use_aux_dataset
            else []
        )

        self.competition_train_df, self.competition_valid_df = self._resolve_source_frames(self.competition_candidates)
        self.aux_train_df, self.aux_valid_df = self._resolve_source_frames(self.aux_candidates)
        self.valid_df = self._build_validation_frame()

    def _resolve_source_frames(
        self,
        candidates: Sequence[Tuple[Path, pd.DataFrame]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_candidate = select_best_candidate(candidates, split="train")
        valid_candidate = select_best_candidate(
            candidates,
            split="valid",
            exclude_paths=[train_candidate[0]] if train_candidate else None,
        )
        train_df = train_candidate[1].copy() if train_candidate is not None else empty_supervised_frame()
        valid_df = valid_candidate[1].copy() if valid_candidate is not None else empty_supervised_frame()
        return train_df, valid_df

    def _build_validation_frame(self) -> pd.DataFrame:
        valid_parts = []
        if not self.competition_valid_df.empty:
            valid_parts.append(sample_frame(self.competition_valid_df, self.variant.max_valid_samples, self.seed + 2))
        if self.variant.use_aux_dataset and not self.aux_valid_df.empty:
            valid_parts.append(sample_frame(self.aux_valid_df, self.variant.max_valid_samples, self.seed + 3))

        if valid_parts:
            valid_df = pd.concat(valid_parts, ignore_index=True)
            valid_df = valid_df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
            return sample_frame(valid_df, self.variant.max_valid_samples, self.seed + 4)

        reference_parts = []
        if not self.competition_train_df.empty:
            reference_parts.append(self.competition_train_df)
        if self.variant.use_aux_dataset and not self.aux_train_df.empty:
            reference_parts.append(self.aux_train_df)

        if not reference_parts:
            raise RuntimeError("No labeled training data was found in the competition files or attached Kaggle datasets.")

        reference_df = pd.concat(reference_parts, ignore_index=True)
        _, valid_df = split_train_valid(reference_df, valid_rows=self.variant.max_valid_samples, seed=self.seed + 5)
        return valid_df.reset_index(drop=True)

    def build_stage_train_frame(self, stage: TrainingStage) -> pd.DataFrame:
        train_parts = []
        if not self.competition_train_df.empty and stage.competition_limit > 0:
            train_parts.append(sample_frame(self.competition_train_df, stage.competition_limit, self.seed + len(stage.name)))
        if self.variant.use_aux_dataset and not self.aux_train_df.empty and stage.aux_limit > 0:
            train_parts.append(sample_frame(self.aux_train_df, stage.aux_limit, self.seed + len(stage.name) + 11))

        if not train_parts:
            raise RuntimeError(f"Stage {stage.name!r} has no training rows after source selection.")

        train_df = pd.concat(train_parts, ignore_index=True)
        train_df = train_df.drop_duplicates(subset=["question", "answer"]).reset_index(drop=True)
        if train_df.empty:
            raise RuntimeError(f"Stage {stage.name!r} became empty after de-duplication.")
        return train_df

    def describe(self) -> None:
        if self.variant.debug:
            summarize_candidates("competition", self.competition_candidates)
            summarize_candidates("auxiliary", self.aux_candidates)

        print(f"Competition directory: {self.competition_dir}")
        print(f"Auxiliary dataset directory: {self.aux_dataset_dir}")
        summarize_frame("Competition training pool", self.competition_train_df)
        if self.variant.use_aux_dataset:
            summarize_frame("Auxiliary training pool", self.aux_train_df)
        summarize_frame("Validation pool", self.valid_df)


# %%
competition_dir = resolve_competition_dir(COMPETITION_HANDLE)
aux_dataset_dir = resolve_dataset_dir(AUX_DATASET_HANDLE)

data_builder = SupervisedDataBuilder(
    competition_dir=competition_dir,
    aux_dataset_dir=aux_dataset_dir,
    variant=VARIANT,
    seed=SEED,
)
data_builder.describe()


# %%
class NemotronModelFactory:
    def __init__(
        self,
        model_dir: Path,
        variant: RunVariant,
        runtime_profile: RuntimeProfile,
        offload_dir: Path,
    ) -> None:
        self.model_dir = model_dir
        self.variant = variant
        self.runtime_profile = runtime_profile
        self.offload_dir = offload_dir

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _should_use_4bit_quantization(self) -> bool:
        total_gpu_memory_gb = sum(self.runtime_profile.gpu_memory_gb)
        largest_gpu_memory_gb = max(self.runtime_profile.gpu_memory_gb)

        if REQUESTED_LOAD_MODE == "4bit":
            if not BITSANDBYTES_AVAILABLE:
                raise RuntimeError(
                    "NEMOTRON_LOAD_MODE=4bit was requested, but bitsandbytes is not installed. "
                    "Turn Internet on in the Kaggle draft session or use a local bitsandbytes wheel."
                )
            return True

        if REQUESTED_LOAD_MODE in {"full", "fp16", "bf16"}:
            return False

        if BITSANDBYTES_AVAILABLE and total_gpu_memory_gb < 70:
            return True

        if not BITSANDBYTES_AVAILABLE and largest_gpu_memory_gb < 70 and total_gpu_memory_gb < 70:
            raise RuntimeError(
                "bitsandbytes is unavailable and the current GPU memory is too small for the automatic full-precision fallback. "
                "Use a larger GPU such as RTX Pro 6000/A100, or turn Internet on so bitsandbytes can be installed."
            )

        return False

    def _build_quantization_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=COMPUTE_DTYPE,
        )

    def _build_load_plans(self) -> List[Tuple[str, Dict[Any, str], bool]]:
        plans = [
            (
                "gpu_only",
                RuntimeManager.build_gpu_max_memory_map(self.runtime_profile, self.variant.gpu_headroom_gb),
                False,
            )
        ]
        if self.variant.allow_cpu_offload:
            plans.append(
                (
                    "gpu_plus_cpu_offload",
                    RuntimeManager.build_gpu_cpu_max_memory_map(
                        self.runtime_profile,
                        self.variant.gpu_headroom_gb,
                        self.variant.cpu_budget_gb,
                    ),
                    True,
                )
            )
        return plans

    def _resolve_lora_target_modules(self, model) -> List[str]:
        discovered = {name.split(".")[-1] for name, module in model.named_modules() if hasattr(module, "weight")}
        resolved = [module_name for module_name in self.variant.lora_target_modules if module_name in discovered]
        if not resolved:
            raise RuntimeError(
                "Could not match any requested LoRA target modules. "
                f"Requested={self.variant.lora_target_modules}, discovered sample={sorted(list(discovered))[:20]}"
            )
        print(f"Resolved LoRA target modules: {resolved}")
        return resolved

    def _describe_device_map(self, model) -> Dict[str, int]:
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map is None:
            return {}
        summary: Dict[str, int] = {}
        for device in hf_device_map.values():
            key = str(device)
            summary[key] = summary.get(key, 0) + 1
        print(json.dumps({"hf_device_map_summary": summary}, indent=2))
        return summary

    def load_model(self, tokenizer):
        self.offload_dir.mkdir(parents=True, exist_ok=True)
        RuntimeManager.clear_memory()
        RuntimeManager.snapshot("Before model load")

        config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        if hasattr(config, "use_cache"):
            config.use_cache = False

        use_4bit_quantization = self._should_use_4bit_quantization()
        quantization_config = self._build_quantization_config() if use_4bit_quantization else None
        print(
            json.dumps(
                {
                    "requested_load_mode": REQUESTED_LOAD_MODE,
                    "selected_load_mode": "4bit" if use_4bit_quantization else "full_precision",
                    "bitsandbytes_available": BITSANDBYTES_AVAILABLE,
                },
                indent=2,
            )
        )
        last_exception: Optional[BaseException] = None
        model = None

        for plan_name, max_memory_map, use_cpu_offload in self._build_load_plans():
            RuntimeManager.clear_memory()
            print(json.dumps({"model_load_plan": plan_name, "max_memory_map": max_memory_map}, indent=2))
            try:
                load_kwargs = {
                    "config": config,
                    "trust_remote_code": True,
                    "device_map": "auto",
                    "max_memory": max_memory_map,
                    "low_cpu_mem_usage": True,
                    "offload_folder": str(self.offload_dir) if use_cpu_offload else None,
                    "offload_state_dict": use_cpu_offload,
                    "offload_buffers": self.variant.offload_buffers if use_cpu_offload else False,
                    "dtype": COMPUTE_DTYPE,
                    "attn_implementation": self.variant.attention_implementation,
                    "experts_implementation": self.variant.experts_implementation,
                }
                if quantization_config is not None:
                    load_kwargs["quantization_config"] = quantization_config

                model = AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    **load_kwargs,
                )
                print(f"Model load plan succeeded: {plan_name}")
                break
            except torch.OutOfMemoryError as exc:
                last_exception = exc
                print(f"Model load plan failed: {plan_name}\n{exc}")
                continue
            except ImportError as exc:
                error_text = str(exc).lower()
                if any(token in error_text for token in ("mamba-ssm", "mamba_ssm", "causal-conv1d", "causal_conv1d")):
                    raise RuntimeError(
                        "Nemotron custom model code could not import its Mamba runtime. "
                        "Cell 1 installs mamba-ssm and causal-conv1d automatically. "
                        "Restart the notebook kernel and rerun from Cell 1."
                    ) from exc
                raise

        if model is None:
            raise RuntimeError(
                "Nemotron did not fit into the current Kaggle memory plan. "
                "Use the gpu_fast variant first, or restart the session if the GPUs are already fragmented."
            ) from last_exception

        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        for parameter in model.parameters():
            parameter.requires_grad = False
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if use_4bit_quantization:
            model = prepare_model_for_kbit_training(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        lora_target_modules = self._resolve_lora_target_modules(model)
        lora_config = LoraConfig(
            r=self.variant.lora_rank,
            lora_alpha=int(self.variant.lora_rank * self.variant.lora_alpha_multiplier),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        setattr(model, "_nemotron_use_4bit", use_4bit_quantization)
        model.print_trainable_parameters()
        device_map_summary = self._describe_device_map(model)
        RuntimeManager.snapshot("After model load")
        return model, device_map_summary


# %%
model_dir = resolve_model_dir(MODEL_HANDLE)
print(f"Model directory: {model_dir}")

model_factory = NemotronModelFactory(
    model_dir=model_dir,
    variant=VARIANT,
    runtime_profile=RUNTIME_PROFILE,
    offload_dir=OFFLOAD_DIR,
)
tokenizer = model_factory.load_tokenizer()
model, device_map_summary = model_factory.load_model(tokenizer)


# %%
class NemotronTrainingOrchestrator:
    def __init__(
        self,
        model,
        tokenizer,
        variant: RunVariant,
        data_builder: SupervisedDataBuilder,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant
        self.data_builder = data_builder

    def _build_training_args(self, stage: TrainingStage) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(RUN_DIR / stage.name),
            num_train_epochs=stage.epochs,
            per_device_train_batch_size=self.variant.per_device_train_batch_size,
            per_device_eval_batch_size=self.variant.per_device_eval_batch_size,
            gradient_accumulation_steps=stage.gradient_accumulation_steps,
            learning_rate=stage.learning_rate,
            warmup_ratio=self.variant.warmup_ratio,
            lr_scheduler_type="cosine",
            logging_strategy="steps",
            logging_steps=self.variant.logging_steps,
            eval_strategy="no",
            save_strategy="no",
            optim="paged_adamw_8bit" if getattr(self.model, "_nemotron_use_4bit", False) else "adamw_torch",
            weight_decay=self.variant.weight_decay,
            max_grad_norm=self.variant.max_grad_norm,
            report_to="none",
            bf16=BF16_AVAILABLE,
            fp16=not BF16_AVAILABLE,
            run_name=f"{RUN_NAME}_{stage.name}",
            log_level="info" if self.variant.debug else "warning",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )

    def run(self) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
        stage_summaries: List[Dict[str, Any]] = []
        validation_frame = sample_frame(self.data_builder.valid_df, self.variant.max_valid_samples, SEED + 90)

        for stage_index, stage in enumerate(self.variant.stages, start=1):
            stage_train_df = self.data_builder.build_stage_train_frame(stage)
            print(f"\nStage {stage_index}/{len(self.variant.stages)}: {stage.name}")
            summarize_frame(f"Stage {stage.name} training frame", stage_train_df)

            train_dataset = tokenize_supervised_examples(self.tokenizer, stage_train_df, stage.max_seq_length)
            valid_dataset = tokenize_supervised_examples(self.tokenizer, validation_frame, stage.max_seq_length)

            if len(train_dataset) == 0:
                raise RuntimeError(f"Tokenization produced zero training rows for stage {stage.name!r}.")
            if len(valid_dataset) == 0:
                valid_dataset = train_dataset.select(range(min(8, len(train_dataset))))

            print(
                json.dumps(
                    {
                        "stage": stage.name,
                        "train_rows": len(stage_train_df),
                        "tokenized_train_rows": len(train_dataset),
                        "tokenized_valid_rows": len(valid_dataset),
                        "max_seq_length": stage.max_seq_length,
                        "competition_rows": int(sum(stage_train_df["source"] == "competition")) if "source" in stage_train_df.columns else 0,
                        "aux_rows": int(sum(stage_train_df["source"] != "competition")) if "source" in stage_train_df.columns else 0,
                    },
                    indent=2,
                )
            )

            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                label_pad_token_id=-100,
                return_tensors="pt",
            )

            trainer = Trainer(
                model=self.model,
                args=self._build_training_args(stage),
                data_collator=cast(DataCollator, data_collator),
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                processing_class=self.tokenizer,
            )

            RuntimeManager.snapshot(f"Before training stage {stage.name}")
            train_result = trainer.train()
            RuntimeManager.snapshot(f"After training stage {stage.name}")

            stage_summary = {
                "stage": stage.name,
                "competition_rows": int(sum(stage_train_df["source"] == "competition")) if "source" in stage_train_df.columns else 0,
                "aux_rows": int(sum(stage_train_df["source"] != "competition")) if "source" in stage_train_df.columns else 0,
                "train_rows": len(stage_train_df),
                "valid_rows": len(validation_frame),
                "train_metrics": serialize_metrics(train_result.metrics),
            }
            stage_summaries.append(stage_summary)

            del trainer
            RuntimeManager.clear_memory()

        return stage_summaries, validation_frame


# %%
training_orchestrator = NemotronTrainingOrchestrator(
    model=model,
    tokenizer=tokenizer,
    variant=VARIANT,
    data_builder=data_builder,
)
stage_summaries, valid_df = training_orchestrator.run()
print(json.dumps(stage_summaries, indent=2))


# %%
preview_rows = []
model.eval()
inference_device = next(model.parameters()).device

for row in valid_df.head(4).itertuples(index=False):
    generation_inputs = build_generation_inputs(tokenizer, str(row.question), VARIANT.final_stage.max_seq_length)
    generation_inputs = {key: value.to(inference_device) for key, value in generation_inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **generation_inputs,
            max_new_tokens=VARIANT.preview_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = generation_inputs["input_ids"].shape[1]
    decoded_text = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()
    preview_rows.append(
        {
            "question": str(row.question)[:140],
            "target": extract_boxed_answer(str(row.answer)),
            "prediction": extract_boxed_answer(decoded_text),
            "match": answers_match(decoded_text, str(row.answer)),
            "source": getattr(row, "source", "unknown"),
        }
    )

preview_df = pd.DataFrame(preview_rows)
display(preview_df)


# %%
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
preview_df.to_csv(ARTIFACT_DIR / "preview_predictions.csv", index=False)

summary = {
    "run_name": RUN_NAME,
    "requested_variant": REQUESTED_VARIANT,
    "active_variant": ACTIVE_VARIANT,
    "variant": asdict(VARIANT),
    "runtime_profile": asdict(RUNTIME_PROFILE),
    "competition_handle": COMPETITION_HANDLE,
    "aux_dataset_handle": AUX_DATASET_HANDLE,
    "model_handle": MODEL_HANDLE,
    "max_sequence_length": MAX_SEQ_LENGTH,
    "lora_rank": VARIANT.lora_rank,
    "device_map_summary": device_map_summary,
    "validation_rows": len(valid_df),
    "validation_sources": summarize_source_mix(valid_df),
    "stage_summaries": stage_summaries,
}
(ARTIFACT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))

if SUBMISSION_PATH.exists():
    SUBMISSION_PATH.unlink()

with zipfile.ZipFile(SUBMISSION_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file_path in sorted(ADAPTER_DIR.rglob("*")):
        if file_path.is_file():
            archive.write(file_path, arcname=str(file_path.relative_to(ADAPTER_DIR)))

adapter_files = sorted(path.name for path in ADAPTER_DIR.iterdir() if path.is_file())
print(f"Adapter files: {adapter_files}")
print(f"Submission ready: {SUBMISSION_PATH}")
