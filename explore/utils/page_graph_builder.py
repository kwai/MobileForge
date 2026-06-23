"""
Page graph builder - Reuse existing code to build page navigation graph from trajectory data
"""
import os
import json
from glob import glob
from typing import Dict, List, Tuple, Set, Optional, Iterator
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
import gc
import tempfile
import pickle
import hashlib
from pathlib import Path

# Reuse existing modules
from utils.utils import load_object_from_disk, save_object_to_disk
from utils.device import UIElement

# Safe import of page analyzer
try:
    from utils.page_analyzer import page_analyzer, PageInfo
    PAGE_ANALYZER_AVAILABLE = True
except ImportError:
    print("Page analyzer module cannot be imported, will use simplified version")
    PAGE_ANALYZER_AVAILABLE = False

    # Create simplified PageInfo class
    from dataclasses import dataclass
    @dataclass
    class PageInfo:
        page_id: str
        page_type: str = "unknown_page"
        activity_name: str = ""
        confidence: float = 0.5
        ui_signature: str = ""
        screenshot_hash: str = ""
        key_features: List[str] = None
        ui_element_count: int = 0
        clickable_count: int = 0
        input_count: int = 0

        def __post_init__(self):
            if self.key_features is None:
                self.key_features = []

@dataclass
class PageTransition:
    """Page transition information"""
    from_page_id: str
    to_page_id: str
    action_type: str
    action_target: str
    success_count: int = 1
    total_count: int = 1
    avg_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.total_count, 1)

@dataclass
class PageNode:
    """Page node information"""
    page_info: PageInfo
    visit_count: int = 0
    representative_screenshot: Optional[str] = None
    common_actions: List[str] = None
    incoming_transitions: List[str] = None  # transition_ids
    outgoing_transitions: List[str] = None  # transition_ids

    def __post_init__(self):
        if self.common_actions is None:
            self.common_actions = []
        if self.incoming_transitions is None:
            self.incoming_transitions = []
        if self.outgoing_transitions is None:
            self.outgoing_transitions = []

class StreamingPageGraphBuilder:
    """Streaming page graph builder - Optimize memory usage, support large-scale trajectory processing"""

    def __init__(self, app_package: str, exploration_output_dir: str = "./exploration_output",
                 batch_size: int = 10, use_disk_cache: bool = True, use_mllm: bool = False):
        self.app_package = app_package
        self.exploration_output_dir = exploration_output_dir
        self.app_output_dir = os.path.join(exploration_output_dir, app_package)
        self.batch_size = batch_size
        self.use_disk_cache = use_disk_cache
        self.use_mllm = use_mllm  # Add parameter to control MLLM usage

        # Create temporary work directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"page_graph_{app_package}_")
        self.page_cache_dir = os.path.join(self.temp_dir, "pages")
        self.transition_cache_dir = os.path.join(self.temp_dir, "transitions")
        os.makedirs(self.page_cache_dir, exist_ok=True)
        os.makedirs(self.transition_cache_dir, exist_ok=True)

        # In-memory data structures (lightweight)
        self.page_registry: Dict[str, str] = {}  # page_id -> cache_file_path
        self.transition_registry: Dict[str, str] = {}  # transition_id -> cache_file_path
        self.page_id_mapping: Dict[str, str] = {}  # original_page_id -> merged_page_id

        # Statistics
        self.total_trajectories = 0
        self.total_steps = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def build_graph_from_trajectories(self, max_trajectories: int = None) -> Dict:
        """Streaming build page graph - Optimize memory usage"""
        print(f"Starting streaming page graph construction for application {self.app_package}...")
        print(f"Batch size: {self.batch_size}, Using disk cache: {self.use_disk_cache}")

        # Get all trajectory files
        trajectory_files = glob(os.path.join(self.app_output_dir, "**", "*.pkl.zst"), recursive=True)

        if not trajectory_files:
            print(f"No trajectory files found for application {self.app_package}")
            return {}

        if max_trajectories:
            trajectory_files = trajectory_files[:max_trajectories]

        print(f"Found {len(trajectory_files)} trajectory files")

        try:
            # Phase 1: Streaming extract page information
            print("Phase 1: Streaming extract page information...")
            self._streaming_extract_pages(trajectory_files)

            # Phase 2: Merge similar pages
            print("Phase 2: Merge similar pages...")
            self._merge_similar_pages_streaming()

            # Phase 3: Streaming build transition relationships
            print("Phase 3: Streaming build transition relationships...")
            self._streaming_extract_transitions(trajectory_files)

            # Phase 4: Generate final graph data
            print("Phase 4: Generate final graph data...")
            graph_data = self._finalize_graph_streaming()

            print(f"Streaming page graph construction completed!")
            print(f"- Total pages: {len(self.page_registry)}")
            print(f"- Total transitions: {len(self.transition_registry)}")
            print(f"- Processed trajectories: {self.total_trajectories}")
            print(f"- Processed steps: {self.total_steps}")
            print(f"- Temporary files: {self.temp_dir}")

            return graph_data

        finally:
            # Cleanup resources
            self._cleanup()

    def _streaming_extract_pages(self, trajectory_files: List[str]):
        """Stream extract page information"""
        for batch_start in range(0, len(trajectory_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(trajectory_files))
            batch_files = trajectory_files[batch_start:batch_end]

            print(f"  Processing batch {batch_start//self.batch_size + 1}: "
                  f"trajectories {batch_start+1}-{batch_end}/{len(trajectory_files)}")

            # Process current batch
            for i, traj_file in enumerate(batch_files):
                self._process_trajectory_for_pages_streaming(traj_file)

                # Periodic garbage collection
                if i % 5 == 0:
                    gc.collect()

            print(f"  Batch completed, current page count: {len(self.page_registry)}")

    def _process_trajectory_for_pages_streaming(self, trajectory_file: str):
        """Stream process single trajectory file to extract page information"""
        try:
            # Process trajectory step by step
            for step_data in self._load_trajectory_streaming(trajectory_file):
                self.total_steps += 1

                # Extract page information from current step
                page_info = self._extract_page_info_from_step(step_data)
                if page_info:
                    self._save_page_info_to_cache(page_info, step_data)

                # Immediately release step data
                del step_data

            self.total_trajectories += 1

        except Exception as e:
            print(f"Failed to process trajectory file {trajectory_file}: {e}")

    def _load_trajectory_streaming(self, trajectory_file: str) -> Iterator[Dict]:
        """Stream load trajectory data, returning each step progressively"""
        try:
            # Load complete trajectory
            trajectory_data = load_object_from_disk(trajectory_file)

            if not trajectory_data or not isinstance(trajectory_data, list):
                return

            # Yield each step progressively instead of returning all at once
            for step_data in trajectory_data:
                yield step_data

            # Immediately release trajectory data
            del trajectory_data
            gc.collect()

        except Exception as e:
            print(f"Failed to load trajectory file {trajectory_file}: {e}")

    def _extract_page_info_from_step(self, step_data: Dict) -> Optional[PageInfo]:
        """Extract page information from step, immediately release screenshot memory"""
        screenshot = self._extract_screenshot(step_data)
        ui_elements = self._extract_ui_elements(step_data)
        activity_name = self._extract_activity_name(step_data)

        if screenshot is not None and ui_elements:
            try:
                if PAGE_ANALYZER_AVAILABLE and self.use_mllm:
                    # Use page analyzer for classification (requires MLLM call)
                    page_info = page_analyzer.classify_page_type(
                        screenshot, ui_elements, activity_name, self.usage
                    )
                else:
                    # Use simplified heuristic classification
                    page_info = self._simple_page_classification(screenshot, ui_elements, activity_name)

                # Immediately release screenshot memory
                del screenshot

                return page_info
            except Exception as e:
                print(f"Page classification failed: {e}")
                # Use simplified version as fallback
                page_info = self._simple_page_classification(screenshot, ui_elements, activity_name)
                del screenshot
                return page_info

        return None

    def _simple_page_classification(self, screenshot: np.ndarray, ui_elements: List[UIElement], activity_name: str) -> PageInfo:
        """Simplified page classification method"""
        import hashlib

        # Generate page ID
        ui_signature = f"total_{len(ui_elements)}_click_{sum(1 for ui in ui_elements if getattr(ui, 'is_clickable', False))}_input_{sum(1 for ui in ui_elements if ui.class_name and 'edit' in ui.class_name.lower())}"

        # Optimize screenshot hash calculation: use shape and few pixel points instead of entire image
        if screenshot.size > 0:
            # Use image shape, mean value and several sampling points to generate hash
            shape_str = f"{screenshot.shape[0]}x{screenshot.shape[1]}"
            if len(screenshot.shape) == 3:
                shape_str += f"x{screenshot.shape[2]}"

            # Sample a few pixel points instead of entire image
            sample_points = []
            h, w = screenshot.shape[:2]
            for y in [0, h//4, h//2, 3*h//4, h-1]:
                for x in [0, w//4, w//2, 3*w//4, w-1]:
                    if len(screenshot.shape) == 3:
                        sample_points.append(str(screenshot[y, x, 0]))  # Only take first channel
                    else:
                        sample_points.append(str(screenshot[y, x]))

            hash_input = f"{shape_str}_{'.'.join(sample_points[:10])}"  # Only use first 10 sampling points
        else:
            hash_input = "empty_screenshot"

        screenshot_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]
        page_id = f"simple_{activity_name.split('.')[-1] if activity_name else 'unknown'}_{screenshot_hash}"

        # Simple page type inference
        page_type = "custom_page"
        if activity_name:
            activity_lower = activity_name.lower()
            if "main" in activity_lower or "home" in activity_lower:
                page_type = "main_page"
            elif "setting" in activity_lower:
                page_type = "settings_page"
            elif "login" in activity_lower or "auth" in activity_lower:
                page_type = "login_page"
            elif "list" in activity_lower:
                page_type = "list_page"

        return PageInfo(
            page_id=page_id,
            page_type=page_type,
            activity_name=activity_name or "unknown",
            confidence=0.6,
            ui_signature=ui_signature,
            screenshot_hash=screenshot_hash,
            key_features=[f"Contains {len(ui_elements)} UI elements"],
            ui_element_count=len(ui_elements),
            clickable_count=sum(1 for ui in ui_elements if getattr(ui, 'is_clickable', False)),
            input_count=sum(1 for ui in ui_elements if ui.class_name and 'edit' in ui.class_name.lower())
        )

    def _save_page_info_to_cache(self, page_info: PageInfo, step_data: Dict):
        """Save page information to disk cache"""
        page_id = page_info.page_id

        if page_id in self.page_registry:
            # Update existing page information
            cached_page = self._load_page_from_cache(page_id)
            if cached_page:
                cached_page.visit_count += 1
                self._save_page_to_cache(page_id, cached_page)
        else:
            # Create new page node
            page_node = PageNode(
                page_info=page_info,
                visit_count=1,
                representative_screenshot=self._save_representative_screenshot_compressed(step_data, page_id),
                common_actions=[]
            )

            # Extract and save action information
            action_info = self._extract_action_info(step_data)
            if action_info:
                page_node.common_actions.append(action_info)

            self._save_page_to_cache(page_id, page_node)

    def _save_page_to_cache(self, page_id: str, page_node: PageNode):
        """Save page node to disk cache"""
        if self.use_disk_cache:
            cache_file = os.path.join(self.page_cache_dir, f"{page_id}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(page_node, f)
            self.page_registry[page_id] = cache_file
        else:
            # If not using disk cache, store in memory (higher risk)
            if not hasattr(self, '_memory_pages'):
                self._memory_pages = {}
            self._memory_pages[page_id] = page_node

    def _load_page_from_cache(self, page_id: str) -> Optional[PageNode]:
        """Load page node from cache"""
        if self.use_disk_cache:
            cache_file = self.page_registry.get(page_id)
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Failed to load page cache {page_id}: {e}")
        else:
            return getattr(self, '_memory_pages', {}).get(page_id)
        return None

    def _save_representative_screenshot_compressed(self, step_data: Dict, page_id: str) -> str:
        """Save compressed representative screenshot"""
        screenshot = self._extract_screenshot(step_data)
        if screenshot is None:
            return ""

        try:
            from PIL import Image

            # Create screenshot save directory
            screenshot_dir = os.path.join(self.app_output_dir, "page_screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)

            # Compress screenshot to save storage space
            if isinstance(screenshot, np.ndarray):
                if screenshot.dtype != np.uint8:
                    screenshot = (screenshot * 255).astype(np.uint8)

                # Compress to smaller size
                pil_image = Image.fromarray(screenshot)
                compressed_image = pil_image.resize((540, 960), Image.Resampling.LANCZOS)

                screenshot_path = os.path.join(screenshot_dir, f"{page_id}.jpg")
                compressed_image.save(screenshot_path, "JPEG", quality=80, optimize=True)

                # Immediately release image memory
                del pil_image, compressed_image, screenshot

                return screenshot_path

        except Exception as e:
            print(f"Failed to save compressed screenshot: {e}")

        return ""

    def _merge_similar_pages_streaming(self):
        """Stream merge similar pages"""
        print("  Starting page similarity analysis...")

        page_ids = list(self.page_registry.keys())
        merged_count = 0

        # Process page similarity calculation in batches to avoid memory explosion
        for i in range(len(page_ids)):
            page_id1 = page_ids[i]
            if page_id1 in self.page_id_mapping:
                continue  # Already merged

            page1 = self._load_page_from_cache(page_id1)
            if not page1:
                continue

            # Compare with subsequent pages
            for j in range(i + 1, len(page_ids)):
                page_id2 = page_ids[j]
                if page_id2 in self.page_id_mapping:
                    continue

                page2 = self._load_page_from_cache(page_id2)
                if not page2:
                    continue

                # Fast similarity check
                similarity = self._calculate_page_similarity_fast(page1.page_info, page2.page_info)

                if similarity > 0.85:
                    # Merge pages
                    page1.visit_count += page2.visit_count
                    page1.common_actions.extend(page2.common_actions)
                    page1.common_actions = list(set(page1.common_actions))  # Remove duplicates

                    # Update mapping relationship
                    self.page_id_mapping[page_id2] = page_id1

                    # Delete merged page cache
                    self._remove_page_from_cache(page_id2)

                    merged_count += 1
                    print(f"    Merged page: {page_id2} -> {page_id1} (similarity: {similarity:.2f})")

            # Update merged page
            self._save_page_to_cache(page_id1, page1)

            # Periodic garbage collection
            if i % 20 == 0:
                gc.collect()

        print(f"  Page merging completed, merged {merged_count} similar pages")

    def _remove_page_from_cache(self, page_id: str):
        """Remove page from cache"""
        if page_id in self.page_registry:
            cache_file = self.page_registry[page_id]
            if os.path.exists(cache_file):
                os.remove(cache_file)
            del self.page_registry[page_id]

    def _streaming_extract_transitions(self, trajectory_files: List[str]):
        """Stream extract transition relationships"""
        for batch_start in range(0, len(trajectory_files), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(trajectory_files))
            batch_files = trajectory_files[batch_start:batch_end]

            print(f"  Analyzing transition batch {batch_start//self.batch_size + 1}: "
                  f"trajectories {batch_start+1}-{batch_end}/{len(trajectory_files)}")

            for traj_file in batch_files:
                self._process_trajectory_for_transitions_streaming(traj_file)

            # Clean memory after batch completion
            gc.collect()

        print(f"  Transition extraction completed, total transitions: {len(self.transition_registry)}")

    def _process_trajectory_for_transitions_streaming(self, trajectory_file: str):
        """Stream process transition relationships"""
        try:
            page_sequence = []

            # Stream process trajectory, only keep page ID sequence
            for step_data in self._load_trajectory_streaming(trajectory_file):
                page_info = self._extract_page_info_from_step(step_data)
                if page_info:
                    # Find merged page ID
                    merged_page_id = self.page_id_mapping.get(page_info.page_id, page_info.page_id)
                    page_sequence.append((merged_page_id, self._extract_action_info(step_data)))

            # Analyze page transitions
            for i in range(len(page_sequence) - 1):
                from_page_id, from_action = page_sequence[i]
                to_page_id, _ = page_sequence[i + 1]

                if from_page_id != to_page_id:
                    self._add_transition_to_cache(from_page_id, to_page_id, from_action)

        except Exception as e:
            print(f"Failed to process transition relationships {trajectory_file}: {e}")

    def _add_transition_to_cache(self, from_page_id: str, to_page_id: str, action_info: str):
        """Add transition to cache"""
        transition_id = f"{from_page_id}_to_{to_page_id}"

        # Load or create transition
        transition = self._load_transition_from_cache(transition_id)
        if transition:
            transition.total_count += 1
            transition.success_count += 1
        else:
            action_type = action_info.split('_')[0] if action_info else "unknown"
            transition = PageTransition(
                from_page_id=from_page_id,
                to_page_id=to_page_id,
                action_type=action_type,
                action_target=action_info or "unknown",
                success_count=1,
                total_count=1
            )

        self._save_transition_to_cache(transition_id, transition)

    def _save_transition_to_cache(self, transition_id: str, transition: PageTransition):
        """Save transition to cache"""
        if self.use_disk_cache:
            cache_file = os.path.join(self.transition_cache_dir, f"{transition_id}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(transition, f)
            self.transition_registry[transition_id] = cache_file
        else:
            if not hasattr(self, '_memory_transitions'):
                self._memory_transitions = {}
            self._memory_transitions[transition_id] = transition

    def _load_transition_from_cache(self, transition_id: str) -> Optional[PageTransition]:
        """Load transition from cache"""
        if self.use_disk_cache:
            cache_file = self.transition_registry.get(transition_id)
            if cache_file and os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except Exception as e:
                    print(f"Failed to load transition cache {transition_id}: {e}")
        else:
            return getattr(self, '_memory_transitions', {}).get(transition_id)
        return None

    def _finalize_graph_streaming(self) -> Dict:
        """Stream generate final graph data"""
        print("  Generating final graph data...")

        # Collect all page information
        pages_data = {}
        total_visits = 0

        for page_id in self.page_registry:
            if page_id not in self.page_id_mapping:  # Only process unmerged pages
                page_node = self._load_page_from_cache(page_id)
                if page_node:
                    pages_data[page_id] = {
                        **asdict(page_node.page_info),
                        "visit_count": page_node.visit_count,
                        "representative_screenshot": page_node.representative_screenshot,
                        "common_actions": page_node.common_actions,
                        "incoming_transitions": [],
                        "outgoing_transitions": []
                    }
                    total_visits += page_node.visit_count

        # Collect all transition information
        transitions_data = {}
        success_rates = []

        for transition_id in self.transition_registry:
            transition = self._load_transition_from_cache(transition_id)
            if transition:
                transitions_data[transition_id] = asdict(transition)
                success_rates.append(transition.success_rate)

                # Update page transition relationships
                if transition.from_page_id in pages_data:
                    pages_data[transition.from_page_id]["outgoing_transitions"].append(transition_id)
                if transition.to_page_id in pages_data:
                    pages_data[transition.to_page_id]["incoming_transitions"].append(transition_id)

        # Calculate statistics
        avg_success_rate = np.mean(success_rates) if success_rates else 0
        sorted_pages = sorted(pages_data.items(), key=lambda x: x[1]["visit_count"], reverse=True)
        key_pages = [page_id for page_id, _ in sorted_pages[:min(5, len(sorted_pages))]]

        return {
            "app_package": self.app_package,
            "build_timestamp": time.time(),
            "statistics": {
                "total_pages": len(pages_data),
                "total_transitions": len(transitions_data),
                "total_visits": total_visits,
                "avg_success_rate": avg_success_rate,
                "processed_trajectories": self.total_trajectories,
                "processed_steps": self.total_steps,
                "key_pages": key_pages,
                "memory_optimized": True,
                "batch_size": self.batch_size
            },
            "pages": pages_data,
            "transitions": transitions_data,
            "usage": self.usage
        }

    def _cleanup(self):
        """Clean up temporary files and memory"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"  Cleaned temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"Failed to clean temporary files: {e}")

        # Clean memory cache
        if hasattr(self, '_memory_pages'):
            del self._memory_pages
        if hasattr(self, '_memory_transitions'):
            del self._memory_transitions

        gc.collect()

    # Reuse existing helper methods
    def _extract_screenshot(self, step_data: Dict) -> Optional[np.ndarray]:
        """Extract screenshot"""
        screenshot_keys = [
            'after_screenshot_with_som', 'before_screenshot_with_som',
            'after_screenshot', 'before_screenshot', 'raw_screenshot'
        ]

        for key in screenshot_keys:
            if key in step_data and step_data[key] is not None:
                screenshot = step_data[key]
                if isinstance(screenshot, np.ndarray) and screenshot.size > 0:
                    return screenshot
        return None

    def _extract_ui_elements(self, step_data: Dict) -> List[UIElement]:
        """Extract UI elements"""
        if 'ui_elements' not in step_data:
            return []

        ui_elements_raw = step_data['ui_elements']
        if not ui_elements_raw:
            return []

        ui_elements = []
        for ui_data in ui_elements_raw:
            if isinstance(ui_data, dict):
                ui_elements.append(UIElement(**ui_data))
            elif isinstance(ui_data, UIElement):
                ui_elements.append(ui_data)

        return ui_elements

    def _extract_activity_name(self, step_data: Dict) -> Optional[str]:
        """Extract Activity name"""
        activity_keys = ['activity', 'activity_name', 'current_activity']
        for key in activity_keys:
            if key in step_data and step_data[key]:
                activity = step_data[key]
                if isinstance(activity, list) and activity:
                    return activity[0]
                elif isinstance(activity, str):
                    return activity
        return None

    def _extract_action_info(self, step_data: Dict) -> Optional[str]:
        """Extract action information"""
        if 'converted_action' in step_data:
            action = step_data['converted_action']
            if isinstance(action, dict):
                action_type = action.get('action_type', 'unknown')
                if action_type in ['click', 'long_press']:
                    index = action.get('index', 'unknown')
                    return f"{action_type}_{index}"
                elif action_type == 'scroll':
                    direction = action.get('direction', 'unknown')
                    return f"scroll_{direction}"
                elif action_type == 'input_text':
                    return "input_text"
                else:
                    return action_type
        return None

    def _calculate_page_similarity_fast(self, page1: PageInfo, page2: PageInfo) -> float:
        """Fast calculate page similarity"""
        if page1.activity_name == page2.activity_name and page1.page_type == page2.page_type:
            ui_diff = abs(page1.ui_element_count - page2.ui_element_count)
            if ui_diff <= 2:
                return 0.9
            elif ui_diff <= 5:
                return 0.7
            else:
                return 0.4
        elif page1.page_type == page2.page_type:
            return 0.5
        else:
            return 0.1

    def save_graph(self, graph_data: Dict, output_file: str = None):
        """Save page graph data"""
        if output_file is None:
            output_dir = os.path.join(".", "page_graphs")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{self.app_package}_page_graph.json")

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        print(f"Page graph data saved to: {output_file}")
        return output_file

import time

# Class alias for stable imports
PageGraphBuilder = StreamingPageGraphBuilder