"""
Page Analyzer - Reuse existing code to implement page classification and similarity calculation
"""
import hashlib
import json
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from dataclasses import dataclass, asdict

# Reuse existing modules
from utils.utils import openai_request, pil_to_webp_base64
from utils.device import UIElement, _generate_ui_elements_description_list
from utils.prompt_templates import PAGE_TYPE_CLASSIFIER, PAGE_SIMILARITY_CALCULATOR
from evaluate_existing_trajectories import robust_extract_json

@dataclass
class PageInfo:
    """Page information data class"""
    page_id: str
    page_type: str
    activity_name: str
    confidence: float
    ui_signature: str
    screenshot_hash: str
    key_features: List[str]
    ui_element_count: int
    clickable_count: int
    input_count: int

class PageAnalyzer:
    """Page Analyzer - Reuse existing code to implement page classification and graph building"""

    # Reuse existing page type definitions
    PAGE_TYPES = {
        "main_page", "settings_page", "profile_page", "login_page",
        "search_page", "list_page", "detail_page", "edit_page",
        "help_page", "about_page", "custom_page"
    }

    def __init__(self):
        self.page_cache = {}  # Cache page classification results
        self.similarity_cache = {}  # Cache similarity calculation results

    def extract_page_features(self, ui_elements: List[UIElement], activity_name: str = None) -> Dict:
        """Extract page features - Reuse existing UI element processing logic"""
        if not ui_elements:
            return {
                "ui_element_count": 0,
                "clickable_count": 0,
                "input_count": 0,
                "ui_signature": "empty"
            }

        # Reuse existing UI element description generation logic
        logical_screen_size = (1080, 1920)  # Default screen size
        ui_elements_desc = _generate_ui_elements_description_list(ui_elements, logical_screen_size)

        # Count UI element features
        visible_elements = [e for e in ui_elements if e.is_visible]
        clickable_count = sum(1 for e in visible_elements if e.is_clickable)
        input_count = sum(1 for e in visible_elements if e._maybe_is_editable())

        # Generate UI signature
        ui_signature = f"total_{len(visible_elements)}_click_{clickable_count}_input_{input_count}"
        if activity_name:
            ui_signature += f"_activity_{activity_name.split('.')[-1]}"

        return {
            "ui_element_count": len(visible_elements),
            "clickable_count": clickable_count,
            "input_count": input_count,
            "ui_signature": ui_signature,
            "ui_elements_desc": ui_elements_desc
        }

    def classify_page_type(self, screenshot: np.ndarray, ui_elements: List[UIElement],
                          activity_name: str = None, usage: Dict[str, int] = None) -> PageInfo:
        """Page type classification - Reuse existing MLLM calling logic"""
        if usage is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0}

        # Extract features
        features = self.extract_page_features(ui_elements, activity_name)

        # Generate screenshot hash
        screenshot_hash = self._generate_screenshot_hash(screenshot)

        # Check cache
        cache_key = f"{screenshot_hash}_{features['ui_signature']}"
        if cache_key in self.page_cache:
            return self.page_cache[cache_key]

        # Try heuristic classification first
        heuristic_result = self._heuristic_classify(activity_name, features)
        if heuristic_result["confidence"] > 0.8:
            page_info = PageInfo(
                page_id=cache_key,
                page_type=heuristic_result["page_type"],
                activity_name=activity_name or "unknown",
                confidence=heuristic_result["confidence"],
                ui_signature=features["ui_signature"],
                screenshot_hash=screenshot_hash,
                key_features=heuristic_result["key_features"],
                ui_element_count=features["ui_element_count"],
                clickable_count=features["clickable_count"],
                input_count=features["input_count"]
            )
            self.page_cache[cache_key] = page_info
            return page_info

        # Use MLLM classification - Reuse existing openai_request logic
        try:
            prompt = PAGE_TYPE_CLASSIFIER.format(
                activity_name=activity_name or "unknown",
                ui_elements_desc=features["ui_elements_desc"] or "No visible UI elements"
            )

            # Reuse existing screenshot processing logic
            if isinstance(screenshot, np.ndarray):
                if screenshot.dtype != np.uint8:
                    screenshot = (screenshot * 255).astype(np.uint8)
                pil_image = Image.fromarray(screenshot)
            else:
                pil_image = screenshot

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{pil_to_webp_base64(pil_image)}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            response = openai_request(messages=messages, timeout=30, usage=usage, max_tokens=200)

            # Reuse existing JSON parsing logic
            result = robust_extract_json(response)

            if result and isinstance(result, dict) and "page_type" in result:
                page_info = PageInfo(
                    page_id=cache_key,
                    page_type=result.get("page_type", "custom_page"),
                    activity_name=activity_name or "unknown",
                    confidence=result.get("confidence", 0.5),
                    ui_signature=features["ui_signature"],
                    screenshot_hash=screenshot_hash,
                    key_features=result.get("key_features", []),
                    ui_element_count=features["ui_element_count"],
                    clickable_count=features["clickable_count"],
                    input_count=features["input_count"]
                )
            else:
                # Fallback classification
                page_info = self._fallback_classify(features, activity_name, cache_key, screenshot_hash)

        except Exception as e:
            print(f"MLLM page classification failed: {e}")
            # Fallback classification
            page_info = self._fallback_classify(features, activity_name, cache_key, screenshot_hash)

        self.page_cache[cache_key] = page_info
        return page_info

    def calculate_page_similarity(self, screenshot1: np.ndarray, screenshot2: np.ndarray,
                                page_info1: PageInfo, page_info2: PageInfo,
                                usage: Dict[str, int] = None) -> Dict:
        """Calculate page similarity - Reuse existing MLLM calling logic"""
        if usage is None:
            usage = {"prompt_tokens": 0, "completion_tokens": 0}

        # Quick check: identical screenshots
        if page_info1.screenshot_hash == page_info2.screenshot_hash:
            return {
                "similarity_score": 1.0,
                "is_same_page": True,
                "reasoning": "Screenshots are identical",
                "key_differences": []
            }

        # Cache check
        cache_key = f"{page_info1.page_id}_{page_info2.page_id}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        # Heuristic similarity check
        heuristic_sim = self._heuristic_similarity(page_info1, page_info2)
        if heuristic_sim["confidence"] > 0.9:
            self.similarity_cache[cache_key] = heuristic_sim
            return heuristic_sim

        # Use MLLM to calculate similarity
        try:
            prompt = PAGE_SIMILARITY_CALCULATOR

            # Prepare images
            images = []
            for screenshot in [screenshot1, screenshot2]:
                if isinstance(screenshot, np.ndarray):
                    if screenshot.dtype != np.uint8:
                        screenshot = (screenshot * 255).astype(np.uint8)
                    pil_image = Image.fromarray(screenshot)
                else:
                    pil_image = screenshot
                images.append(pil_image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{pil_to_webp_base64(images[0])}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/webp;base64,{pil_to_webp_base64(images[1])}",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ]

            response = openai_request(messages=messages, timeout=30, usage=usage, max_tokens=200)
            result = robust_extract_json(response)

            if result and isinstance(result, dict) and "similarity_score" in result:
                similarity_result = {
                    "similarity_score": result.get("similarity_score", 0.0),
                    "is_same_page": result.get("is_same_page", False),
                    "reasoning": result.get("reasoning", "MLLM analysis result"),
                    "key_differences": result.get("key_differences", [])
                }
            else:
                similarity_result = heuristic_sim

        except Exception as e:
            print(f"MLLM similarity calculation failed: {e}")
            similarity_result = heuristic_sim

        self.similarity_cache[cache_key] = similarity_result
        return similarity_result

    def _generate_screenshot_hash(self, screenshot: np.ndarray) -> str:
        """Generate screenshot hash"""
        if isinstance(screenshot, np.ndarray):
            # Compress screenshot for hash calculation
            small_img = Image.fromarray(screenshot.astype(np.uint8)).resize((64, 64))
            img_bytes = np.array(small_img).tobytes()
            return hashlib.md5(img_bytes).hexdigest()[:16]
        return "unknown"

    def _heuristic_classify(self, activity_name: str, features: Dict) -> Dict:
        """Heuristic page classification"""
        if not activity_name:
            return {"page_type": "custom_page", "confidence": 0.3, "key_features": []}

        activity_lower = activity_name.lower()

        # Heuristic rules based on Activity name
        type_keywords = {
            "main_page": ["main", "home", "dashboard", "launcher"],
            "settings_page": ["setting", "preference", "config", "option"],
            "profile_page": ["profile", "account", "user", "personal"],
            "login_page": ["login", "auth", "signin", "register"],
            "search_page": ["search", "find", "query"],
            "help_page": ["help", "support", "guide", "about"],
        }

        for page_type, keywords in type_keywords.items():
            for keyword in keywords:
                if keyword in activity_lower:
                    return {
                        "page_type": page_type,
                        "confidence": 0.85,
                        "key_features": [f"Activity name contains '{keyword}'"]
                    }

        # Classification based on UI features
        if features["input_count"] > 2:
            return {
                "page_type": "edit_page",
                "confidence": 0.7,
                "key_features": [f"Contains {features['input_count']} input fields"]
            }
        elif features["clickable_count"] > 10:
            return {
                "page_type": "list_page",
                "confidence": 0.6,
                "key_features": [f"Contains {features['clickable_count']} clickable elements"]
            }

        return {"page_type": "custom_page", "confidence": 0.4, "key_features": []}

    def _heuristic_similarity(self, page1: PageInfo, page2: PageInfo) -> Dict:
        """Heuristic similarity calculation"""
        # Same Activity name and similar UI signature
        if (page1.activity_name == page2.activity_name and
            page1.page_type == page2.page_type and
            abs(page1.ui_element_count - page2.ui_element_count) <= 2):
            return {
                "similarity_score": 0.95,
                "is_same_page": True,
                "reasoning": "Same Activity name and page type, similar UI element count",
                "key_differences": [],
                "confidence": 0.95
            }

        # Different page types
        if page1.page_type != page2.page_type:
            return {
                "similarity_score": 0.1,
                "is_same_page": False,
                "reasoning": f"Different page types: {page1.page_type} vs {page2.page_type}",
                "key_differences": ["Different page types"],
                "confidence": 0.9
            }

        return {
            "similarity_score": 0.5,
            "is_same_page": False,
            "reasoning": "Requires further analysis",
            "key_differences": [],
            "confidence": 0.3
        }

    def _fallback_classify(self, features: Dict, activity_name: str, cache_key: str, screenshot_hash: str) -> PageInfo:
        """Fallback classification"""
        return PageInfo(
            page_id=cache_key,
            page_type="custom_page",
            activity_name=activity_name or "unknown",
            confidence=0.3,
            ui_signature=features["ui_signature"],
            screenshot_hash=screenshot_hash,
            key_features=["Fallback classification"],
            ui_element_count=features["ui_element_count"],
            clickable_count=features["clickable_count"],
            input_count=features["input_count"]
        )

# Global page analyzer instance
page_analyzer = PageAnalyzer()