"""
Core conversion functions for rosbag to lerobot conversion
"""

import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from airoa_metadata import Metadata, MetadataBase, MetadataLoader, MetadataV1_0
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from rosbags.typesys.store import Typestore

from hsr_data_converter.convert_config import ConvertConfig
from hsr_data_converter.hsr_utils import hsr_features

from .episode_loader import load_hsr_episodes
from .utils import update_episodes_jsonl


def perform_conversion(cfg: ConvertConfig, raw_dir: Path, out_dir: Path):
    """Perform the actual rosbag to lerobot conversion."""
    # Create a type store to use if the bag has no message definitions.
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS1_NOETIC)

    if cfg.conversion_type == "individual":
        return _convert_individual_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, typestore
        )
    elif cfg.conversion_type == "aggregate":
        return _convert_aggregate_rosbag_to_lerobot_format(
            cfg, raw_dir, out_dir, typestore
        )
    else:
        raise ValueError(f"Invalid conversion type: {cfg.conversion_type}")


def _convert_individual_rosbag_to_lerobot_format(
    cfg: ConvertConfig,
    raw_dir: Path,
    out_dir: Path,
    typestore: Typestore,
):
    """
    Convert HSR rosbag datasets to LeRobotDatasetV2 format
    specified by the configuration cfg
    """
    # get subdirectories
    rosbag_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())

    # check if there is no rosbag folder
    if not rosbag_dirs:
        print(f"[ERROR] No rosbag folders found in {raw_dir}")
        return None

    pipeline_git_hash = os.getenv("GIT_HASH")
    pipeline_git_branch = os.getenv("GIT_BRANCH")
    if pipeline_git_hash is None or pipeline_git_branch is None:
        print("[ERROR] Environment variables GIT_HASH and GIT_BRANCH are not defined")
        return None

    # iterate over each subdirectory
    new_data_for_episodes_jsonl = {}
    for rosbag_dir in rosbag_dirs:
        # Update FPS in features metadata
        for feature in hsr_features.values():
            if "info" in feature:
                feature["info"]["video.fps"] = cfg.fps

        # Create a new empty dataset for this episode
        dataset = LeRobotDataset.create(
            repo_id=cfg.repo_id,
            fps=cfg.fps,
            root=out_dir / rosbag_dir.name,
            robot_type=cfg.robot_type,  # specify robot type
            features=hsr_features,
            tolerance_s=0.5,
            use_videos=True,
            image_writer_processes=4,
            image_writer_threads=4,
        )
        dataset_root = dataset.root

        try:
            # time statistics
            rosbag1_process_start_time = time.time()

            # Parse meta.json for episode task
            meta_path = rosbag_dir / "meta.json"
            if not meta_path.exists():
                print(f"[WARN] meta.json missing in {rosbag_dir}, skipping.")
                shutil.rmtree(dataset_root)
                continue

            loader = MetadataLoader()
            metadata = loader.load_from_file(str(meta_path))
            if not isinstance(metadata, Metadata):
                # Convert to latest metadata format.
                metadata = Metadata.convert(metadata)

            # Load frames from rosbag
            episodes, all_tasks, all_timestamps, _ = load_hsr_episodes(
                cfg, metadata, rosbag_dir, typestore, dataset
            )

            for frames, tasks, timestamps in zip(episodes, all_tasks, all_timestamps):
                if not frames:
                    print(f"[WARN] No valid frames in {rosbag_dir}, skipping.")
                    shutil.rmtree(dataset_root)
                    continue

                # Add frames to dataset
                for frame, task, timestamp in zip(frames, tasks, timestamps):
                    dataset.add_frame(frame=frame, task=task, timestamp=timestamp)

                # get the episode index
                episode_index = dataset.episode_buffer["episode_index"]

                # NOTE: Fix timestamp shape to match the shape of 'episode_index'.
                dataset.episode_buffer["timestamp"] = np.squeeze(
                    dataset.episode_buffer["timestamp"]
                )

                # Save the episode using new API
                dataset.save_episode()

                # Add new tasks to the tasks dictionary
                # short_horizon_task = metadata.run.instructions[-1][0]
                # short_horizon_task_index = dataset.meta.get_task_index(short_horizon_task)
                # if short_horizon_task_index is None:
                #     dataset.meta.add_task(short_horizon_task)

                # update the last line of episodes.jsonl
                task_type = "PA" if cfg.separate_per_primitive else "SHT"
                task_success = bool(
                    frames[0]["success_primitive_action"][0]
                    if cfg.separate_per_primitive
                    else metadata.run.segments[-1].success
                )
                new_data_for_episodes_jsonl[episode_index] = _create_episodes_info(
                    metadata,
                    task_type,
                    task_success,
                    pipeline_git_hash,
                    pipeline_git_branch,
                )

                print(
                    f"[INFO] Elapsed time: {time.time() - rosbag1_process_start_time} [s]"
                )
                print("[INFO] UPDATE the episodes.jsonl")
                update_episodes_jsonl(
                    new_data_for_episodes_jsonl, dataset.root / "meta/episodes.jsonl"
                )

                print(f"[INFO] Conversion complete. Dataset saved to {out_dir}")
        finally:
            dataset.stop_image_writer()

    return dataset


def _convert_aggregate_rosbag_to_lerobot_format(
    cfg: ConvertConfig,
    raw_dir: Path,
    out_dir: Path,
    typestore: Typestore,
):
    """
    Convert rosbag datasets to LeRobotDataverV2 format
    """
    # get subdirectories
    rosbag_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]

    pipeline_git_hash = os.getenv("GIT_HASH")
    pipeline_git_branch = os.getenv("GIT_BRANCH")
    if pipeline_git_hash is None or pipeline_git_branch is None:
        print("[ERROR] Environment variables GIT_HASH and GIT_BRANCH are not defined")
        return None

    # update video.fps in hsr_features with provided fps
    for feature in hsr_features.values():
        if "info" in feature:
            feature["info"]["video.fps"] = cfg.fps

    dataset = LeRobotDataset.create(
        repo_id=cfg.repo_id,
        fps=cfg.fps,
        root=out_dir,
        robot_type=cfg.robot_type,
        features=hsr_features,
        tolerance_s=0.5,
        use_videos=True,
        image_writer_processes=4,
        image_writer_threads=4,
    )

    # time statistics
    time_statistics = {"min": np.inf, "max": -np.inf, "mean": 0.0, "count": 0}

    # iterate over each subdirectory
    new_data_for_episodes_jsonl = {}
    for rosbag_dir in rosbag_dirs:
        rosbag1_process_start_time = time.time()

        # load meta.json and get the list of MetaDataset
        meta_path = rosbag_dir / "meta.json"
        if not meta_path.exists():
            print(f"[WARN] meta.json missing in {rosbag_dir}, skipping.")
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_dataset_list = json.load(f)
            if isinstance(meta_dataset_list, list):
                if len(meta_dataset_list) != 1:
                    raise ValueError(
                        f"meta.json shoud have only one MetaDataset, but got {len(meta_dataset_list)}"
                    )
            else:
                meta_dataset_list = [meta_dataset_list]

        loader = MetadataLoader()
        metadata = loader.load_from_file(str(meta_path))
        if not isinstance(metadata, Metadata):
            metadata = Metadata.convert(metadata)
        print(f"metadata: {metadata}")

        # load the episode from the rosbag
        episodes, all_tasks, all_timestamps, last_timestamp = load_hsr_episodes(
            cfg, metadata, rosbag_dir, typestore, dataset
        )

        for frames, tasks, timestamps in zip(episodes, all_tasks, all_timestamps):
            if len(frames) == 0:
                print(f"[WARN] Skipping empty rosbag {rosbag_dir}.")
                continue

            # add frames to the dataset
            for frame, task, timestamp in zip(frames, tasks, timestamps):
                dataset.add_frame(frame=frame, task=task, timestamp=timestamp)

            # get the episode index
            episode_index = dataset.episode_buffer["episode_index"]

            # NOTE: Fix timestamp shape to match the shape of 'episode_index'.
            dataset.episode_buffer["timestamp"] = np.squeeze(
                dataset.episode_buffer["timestamp"]
            )

            # save the episode
            dataset.save_episode()

            # Add new tasks to the tasks dictionary
            # short_horizon_task = metadataset.run.instructions[-1][0]
            # short_horizon_task_index = dataset.meta.get_task_index(short_horizon_task)
            # if short_horizon_task_index is None:
            #     dataset.meta.add_task(short_horizon_task)

            # update the last line of episodes.jsonl
            task_type = "PA" if cfg.separate_per_primitive else "SHT"
            task_success = bool(
                frames[0]["success_primitive_action"][0]
                if cfg.separate_per_primitive
                else metadata.run.segments[-1].success
            )

            new_data_for_episodes_jsonl[episode_index] = _create_episodes_info(
                metadata,
                task_type,
                task_success,
                pipeline_git_hash,
                pipeline_git_branch,
            )

            print(
                f"[INFO] Elapsed time: {time.time() - rosbag1_process_start_time} [s]"
            )

            # update time statistics
            time_statistics["min"] = min(time_statistics["min"], last_timestamp)
            time_statistics["max"] = max(time_statistics["max"], last_timestamp)
            time_statistics["mean"] += last_timestamp
            time_statistics["count"] += 1

    # update mean time
    print(f"[INFO] Time statistics: {time_statistics}")

    # consolidate the dataset and save it
    print("[INFO] UPDATE the episodes.jsonl")
    update_episodes_jsonl(
        new_data_for_episodes_jsonl, dataset.root / "meta/episodes.jsonl"
    )
    print(f"[INFO] Dataset saved to {out_dir}")

    dataset.stop_image_writer()

    return dataset


def _create_episodes_info(
    metadata: Metadata,
    task_type: str,
    task_success: bool,
    pipeline_git_hash: str,
    pipeline_git_branch: str,
) -> dict[str, Any]:
    """Create episode information from metadata."""
    metadata_v1_0 = _convert_metadata_to_v1_0(metadata)

    instruction_texts = [
        instruction.text[0] for instruction in metadata.run.instructions
    ]

    if len(instruction_texts) > 1:
        short_horizon_task = instruction_texts[-1]
        primitive_action = instruction_texts[:-1]
    else:
        short_horizon_task = []
        primitive_action = instruction_texts

    episodes_info = {
        "bag_path": metadata_v1_0.bag_path,
        "version": metadata_v1_0.version,
        "location_name": metadata_v1_0.location_name,
        "interface": metadata_v1_0.interface,
        "git_hash": metadata_v1_0.git_hash,
        "git_branch": metadata_v1_0.git_branch,
        "interface_git_hash": metadata_v1_0.interface_git_hash,
        "interface_git_branch": metadata_v1_0.interface_git_branch,
        "pipeline_git_hash": pipeline_git_hash,
        "pipeline_git_branch": pipeline_git_branch,
        "label": metadata_v1_0.label,
        "hsr_id": metadata_v1_0.hsr_id,
        "task_type": task_type,
        "task_success": task_success,
        "short_horizon_task": short_horizon_task,
        "primitive_action": primitive_action,
        "success_short_horizon_task": metadata.run.segments[-1].success,
        "uuid": metadata.uuid,
        "context": asdict(metadata.context),
    }

    return episodes_info


def _convert_metadata_to_v1_0(metadata: Metadata) -> MetadataV1_0:
    """
    Convert from your “input format” to the desired “output format”,
    using dict.get() instead of direct indexing.
    """
    out = {}
    json_str = metadata.to_json()
    input_json = json.loads(json_str)

    # 1. bag_path
    files = input_json.get("files", [])
    first_file = files[0] if files else {}
    out["bag_path"] = first_file.get("name", "")

    # 2. hsr_id
    entities = input_json.get("context", {}).get("entities", [])
    robot: dict[str, str] = next((e for e in entities if e.get("role") == "robot"), {})
    out["hsr_id"] = robot.get("id", "")

    # 3. version
    out["version"] = "1.0"

    # 4. location_name
    location: dict[str, str] = next(
        (e for e in entities if e.get("role") == "location"), {}
    )
    out["location_name"] = location.get("name", "")

    # 5. interface
    components = input_json.get("context", {}).get("components", [])
    interface: dict[str, Any] = next(
        (c for c in components if c.get("role") == "interface"), {}
    )
    out["interface"] = interface.get("name", "")

    # 6. instructions
    instructions = input_json.get("run", {}).get("instructions", [])
    out["instructions"] = [step.get("text", []) for step in instructions]

    # 7. segments
    raw_segments = input_json.get("run", {}).get("segments", [])
    segs = []
    for seg in raw_segments:
        start = seg.get("start_time", 0.0)
        end = seg.get("end_time", 0.0)

        segs.append(
            {
                "start_time": start,
                "end_time": end,
                "instructions_index": seg.get("instruction_idx", 0),
                "has_suboptimal": not seg.get("success", False),
                "is_directed": True,
            }
        )

    # 8. find the task entity
    task_ent: dict[str, dict[str, str]] = next(
        (e for e in entities if e.get("role") == "task"), {}
    )

    template = task_ent.get("template", {})
    description = template.get("description", "")
    # check if the last segment is missing:
    if segs[0]["start_time"] >= 100000:
        ts = 0.0
    if segs[-1]["start_time"] >= segs[0]["end_time"]:
        segs.append(
            {
                "start_time": segs[0]["start_time"] + ts,
                "end_time": segs[-1]["end_time"] + ts,
                "instructions_index": len(out["instructions"]),
                "has_suboptimal": False,  # Assuming that it is false
                "is_directed": True,
            }
        )
        out["instructions"].append([description])

    out["segments"] = segs

    # 8. label
    operator: dict[str, str] = next(
        (e for e in entities if e.get("role") == "operator"), {}
    )
    out["label"] = operator.get("id", "")

    # 9. git info for data-collection component
    data_collection: dict[str, Any] = next(
        (c for c in components if c.get("role") == "data_collection"), {}
    )
    dc_source: dict[str, dict[str, str]] = data_collection.get("source", {})
    git_dc: dict[str, str] = dc_source.get("git", {})
    out["git_hash"] = git_dc.get("hash", "")
    out["git_branch"] = git_dc.get("branch", "")

    # 10. interface git info
    source: dict[str, dict[str, str]] = interface.get("source", {})
    git_if: dict[str, str] = source.get("git", {})
    out["interface_git_hash"] = git_if.get("hash", "")
    out["interface_git_branch"] = git_if.get("branch", "")

    metadata_v1_0: MetadataBase = MetadataLoader.load_from_dict(out)
    if not isinstance(metadata_v1_0, MetadataV1_0):
        raise ValueError("Failed to convert metadata to V1.0.")

    return metadata_v1_0
