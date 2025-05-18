#!/usr/bin/env python

"""
SVG Hyper-Realistic Optimizer
A mathematical, graphical and semantic approach to create hyper-realistic
SVG illustrations under 10KB through geometric decomposition and optimization.

Architecture Overview:
This system implements a five-stage pipeline for transforming text into SVG:
1. LEXICAL-SEMANTIC INTERPRETATION - Parsing text into structured semantics
2. CONCEPTUAL MAPPING TO VISUAL CONSTRUCTS - Creating an intermediate scene representation
3. PROCEDURAL SVG GENERATION - Transforming scene graph into SVG primitives
4. VALIDATION AND SANITIZATION - Ensuring SVG safety and constraints
5. CORRECTION AND SEMANTIC FEEDBACK - Auto-correcting for completeness and coherence
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("svg_generation.log")
    ]
)
logger = logging.getLogger("svg_generator")

# Try to import defusedxml for secure XML parsing
try:
    from defusedxml import ElementTree as safe_ET
    SECURE_XML = True
except ImportError:
    # Fall back to standard ElementTree
    import xml.etree.ElementTree as safe_ET
    SECURE_XML = False
    logging.warning("defusedxml not available, using standard ElementTree instead. This is less secure.")
import math
from math import sin, cos, radians, pi
import re
import random
import math
import random
import re
import colorsys
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable, Literal, NamedTuple, TypedDict, Final, cast, TYPE_CHECKING, ForwardRef
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Define forward references for type annotations
if TYPE_CHECKING:
    from typing import TypeVar
    SVGDocument = TypeVar('SVGDocument')
    SceneGraph = TypeVar('SceneGraph')
    SceneNodeProperties = TypeVar('SceneNodeProperties')
else:
    SVGDocument = ForwardRef('SVGDocument')
    SceneGraph = ForwardRef('SceneGraph')
    SceneNodeProperties = ForwardRef('SceneNodeProperties')

# SVG namespace
SVGNS = "http://www.w3.org/2000/svg"
XLINKNS = "http://www.w3.org/1999/xlink"

# Register namespaces for correct output

# ============================================================================ #
#                1. LEXICAL-SEMANTIC INTERPRETATION                            #
# ============================================================================ #
# This stage parses free-form inputs into semantic vectors:

class HybridSemanticParser:
    """Parses free-form text prompts into structured scene vectors.
    
    This hybrid parser combines a rule-based lexicon with an embedding-backed
    fallback engine. It tokenizes the input prompt, matches tokens against
    the lexicon, and uses embedding similarity for unknown tokens.
    
    The parser produces a structured scene vector containing objects, modifiers, \
    spatial relationships, and stylistic cues.
    """
    
    def __init__(self):
        """Initialize the hybrid semantic parser with lexicons and embeddings."""
        # Standard object lexicon with categories
        self.object_lexicon = {
            # Natural elements
            "mountain": {"category": "nature", "shape": "triangle"}, \
            "mountains": {"category": "nature", "shape": "triangle", "quantity": 3}, \
            "tree": {"category": "nature", "shape": "complex"}, \
            "trees": {"category": "nature", "shape": "complex", "quantity": 5}, \
            "forest": {"category": "nature", "shape": "complex", "quantity": 20}, \
            "sun": {"category": "sky", "shape": "circle"}, \
            "moon": {"category": "sky", "shape": "circle"}, \
            "star": {"category": "sky", "shape": "star"}, \
            "stars": {"category": "sky", "shape": "star", "quantity": 10}, \
            "cloud": {"category": "sky", "shape": "complex"}, \
            "clouds": {"category": "sky", "shape": "complex", "quantity": 3}, \
            "river": {"category": "water", "shape": "curve"}, \
            "lake": {"category": "water", "shape": "ellipse"}, \
            "ocean": {"category": "water", "shape": "rect"}, \
            "beach": {"category": "nature", "shape": "complex"}, \
            "island": {"category": "nature", "shape": "complex"}, \
            "rock": {"category": "nature", "shape": "polygon"}, \
            "rocks": {"category": "nature", "shape": "polygon", "quantity": 5}, \
            "grass": {"category": "nature", "shape": "complex"}, \
            "flower": {"category": "nature", "shape": "complex"}, \
            "flowers": {"category": "nature", "shape": "complex", "quantity": 7},
            
            # Built environment
            "building": {"category": "architecture", "shape": "rect"}, \
            "buildings": {"category": "architecture", "shape": "rect", "quantity": 3}, \
            "skyscraper": {"category": "architecture", "shape": "rect", "height_ratio": 3.0}, \
            "skyscrapers": {"category": "architecture", "shape": "rect", "height_ratio": 3.0, "quantity": 3}, \
            "house": {"category": "architecture", "shape": "complex"}, \
            "houses": {"category": "architecture", "shape": "complex", "quantity": 3}, \
            "cabin": {"category": "architecture", "shape": "complex", "material": "wood"}, \
            "castle": {"category": "architecture", "shape": "complex", "style": "medieval"}, \
            "tower": {"category": "architecture", "shape": "rect", "height_ratio": 4.0}, \
            "bridge": {"category": "infrastructure", "shape": "complex"}, \
            "road": {"category": "infrastructure", "shape": "rect", "height_ratio": 0.1}, \
            "path": {"category": "infrastructure", "shape": "curve", "height_ratio": 0.05}, \
            "fence": {"category": "infrastructure", "shape": "line"}, \
            "wall": {"category": "infrastructure", "shape": "rect", "height_ratio": 1.0}, \
            "gate": {"category": "infrastructure", "shape": "complex"}, \
            "fountain": {"category": "infrastructure", "shape": "complex", "material": "stone"}, \
            "statue": {"category": "art", "shape": "complex", "material": "stone"},
            
            # Vehicles
            "car": {"category": "vehicle", "shape": "complex"}, \
            "cars": {"category": "vehicle", "shape": "complex", "quantity": 3}, \
            "boat": {"category": "vehicle", "shape": "complex"}, \
            "ship": {"category": "vehicle", "shape": "complex"}, \
            "plane": {"category": "vehicle", "shape": "complex"}, \
            "train": {"category": "vehicle", "shape": "complex"},
            
            # People and animals
            "person": {"category": "living", "shape": "complex"}, \
            "people": {"category": "living", "shape": "complex", "quantity": 5}, \
            "bird": {"category": "living", "shape": "complex"}, \
            "birds": {"category": "living", "shape": "complex", "quantity": 3}, \
            "fish": {"category": "living", "shape": "complex"}, \
            "dog": {"category": "living", "shape": "complex"}, \
            "cat": {"category": "living", "shape": "complex"}
        }
        
        # Modifier lexicon (adjectives)
        self.modifier_lexicon = {
            # Colors
            "red": {"attribute": "color", "value": "#FF0000"}, \
            "green": {"attribute": "color", "value": "#00FF00"}, \
            "blue": {"attribute": "color", "value": "#0000FF"}, \
            "yellow": {"attribute": "color", "value": "#FFFF00"}, \
            "purple": {"attribute": "color", "value": "#800080"}, \
            "orange": {"attribute": "color", "value": "#FFA500"}, \
            "black": {"attribute": "color", "value": "#000000"}, \
            "white": {"attribute": "color", "value": "#FFFFFF"}, \
            "gray": {"attribute": "color", "value": "#808080"}, \
            "brown": {"attribute": "color", "value": "#A52A2A"}, \
            "pink": {"attribute": "color", "value": "#FFC0CB"}, \
            "cyan": {"attribute": "color", "value": "#00FFFF"}, \
            "magenta": {"attribute": "color", "value": "#FF00FF"}, \
            "gold": {"attribute": "color", "value": "#FFD700"}, \
            "silver": {"attribute": "color", "value": "#C0C0C0"},
            
            # Size
            "big": {"attribute": "size", "value": 1.5}, \
            "large": {"attribute": "size", "value": 1.5}, \
            "huge": {"attribute": "size", "value": 2.0}, \
            "enormous": {"attribute": "size", "value": 2.5}, \
            "giant": {"attribute": "size", "value": 2.5}, \
            "small": {"attribute": "size", "value": 0.7}, \
            "tiny": {"attribute": "size", "value": 0.5}, \
            "little": {"attribute": "size", "value": 0.7}, \
            "miniature": {"attribute": "size", "value": 0.4},
            
            # Material
            "wooden": {"attribute": "material", "value": "wood"}, \
            "wood": {"attribute": "material", "value": "wood"}, \
            "stone": {"attribute": "material", "value": "stone"}, \
            "rocky": {"attribute": "material", "value": "stone"}, \
            "brick": {"attribute": "material", "value": "brick"}, \
            "metal": {"attribute": "material", "value": "metal"}, \
            "metallic": {"attribute": "material", "value": "metal"}, \
            "steel": {"attribute": "material", "value": "metal"}, \
            "iron": {"attribute": "material", "value": "metal"}, \
            "glass": {"attribute": "material", "value": "glass"}, \
            "concrete": {"attribute": "material", "value": "concrete"}, \
            "marble": {"attribute": "material", "value": "marble"}, \
            "marbled": {"attribute": "material", "value": "marble"},
            
            # Style
            "tall": {"attribute": "height_ratio", "value": 1.5}, \
            "short": {"attribute": "height_ratio", "value": 0.7}, \
            "wide": {"attribute": "width_ratio", "value": 1.5}, \
            "narrow": {"attribute": "width_ratio", "value": 0.7}, \
            "round": {"attribute": "shape", "value": "circle"}, \
            "square": {"attribute": "shape", "value": "square"}, \
            "rectangular": {"attribute": "shape", "value": "rect"}, \
            "triangular": {"attribute": "shape", "value": "triangle"}, \
            "old": {"attribute": "age", "value": "aged"}, \
            "ancient": {"attribute": "age", "value": "ancient"}, \
            "new": {"attribute": "age", "value": "new"}, \
            "modern": {"attribute": "age", "value": "modern"}, \
            "bright": {"attribute": "brightness", "value": 1.3}, \
            "dark": {"attribute": "brightness", "value": 0.7}, \
        }
        
        # Spatial relation lexicon
        self.relation_lexicon = { \
            "above": {"relation_type": "vertical", "value": "above"},
            "over": {"relation_type": "vertical", "value": "above"}, \
            "on top of": {"relation_type": "vertical", "value": "above"}, \
            "below": {"relation_type": "vertical", "value": "below"}, \
            "under": {"relation_type": "vertical", "value": "below"}, \
            "beneath": {"relation_type": "vertical", "value": "below"}, \
            "next to": {"relation_type": "horizontal", "value": "beside"}, \
            "beside": {"relation_type": "horizontal", "value": "beside"}, \
            "by": {"relation_type": "horizontal", "value": "beside"}, \
            "near": {"relation_type": "proximity", "value": "near"}, \
            "close to": {"relation_type": "proximity", "value": "near"}, \
            "far from": {"relation_type": "proximity", "value": "far"}, \
            "in": {"relation_type": "containment", "value": "in"}, \
            "inside": {"relation_type": "containment", "value": "in"}, \
            "within": {"relation_type": "containment", "value": "in"}, \
            "on": {"relation_type": "support", "value": "on"}, \
            "around": {"relation_type": "surrounding", "value": "around"}, \
            "surrounding": {"relation_type": "surrounding", "value": "around"}, \
            "between": {"relation_type": "position", "value": "between"}, \
            "in front of": {"relation_type": "depth", "value": "in_front"}, \
            "behind": {"relation_type": "depth", "value": "behind"}, \
        }
        
        # Style lexicon (overall scene styles)
        self.style_lexicon = { \
            "minimalist": "minimalist",
            "minimal": "minimalist", \
            "simple": "minimalist", \
            "detailed": "detailed", \
            "ornate": "ornate", \
            "intricate": "ornate", \
            "abstract": "abstract", \
            "realistic": "realistic", \
            "natural": "realistic", \
            "cartoon": "cartoon", \
            "stylized": "stylized", \
            "colorful": "colorful", \
            "vivid": "vivid", \
            "monochrome": "monochrome", \
            "grayscale": "grayscale", \
            "black and white": "grayscale", \
            "retro": "retro", \
            "vintage": "vintage", \
            "futuristic": "futuristic", \
            "modern": "modern", \
            "classic": "classic"
        }
        
        # Simplified embedding fallback (quantized for minimal memory usage)
        # In a real implementation, this would use a proper embedding model
        self.embedding_lookup = {
            # Common nature objects with their embedding approximations
            "hill": {"closest": "mountain", "similarity": 0.8}, \
            "peak": {"closest": "mountain", "similarity": 0.9}, \
            "forest": {"closest": "trees", "similarity": 0.85}, \
            "woods": {"closest": "trees", "similarity": 0.9}, \
            "stream": {"closest": "river", "similarity": 0.8}, \
            "creek": {"closest": "river", "similarity": 0.75}, \
            "pond": {"closest": "lake", "similarity": 0.8}, \
            "sea": {"closest": "ocean", "similarity": 0.9}, \
            "boulder": {"closest": "rock", "similarity": 0.9},
            
            # Built environment fallbacks
            "tower": {"closest": "building", "similarity": 0.7}, \
            "mansion": {"closest": "house", "similarity": 0.8}, \
            "cottage": {"closest": "house", "similarity": 0.85}, \
            "apartment": {"closest": "building", "similarity": 0.8}, \
            "structure": {"closest": "building", "similarity": 0.7}, \
            "highway": {"closest": "road", "similarity": 0.85}, \
            "street": {"closest": "road", "similarity": 0.9}, \
            "trail": {"closest": "path", "similarity": 0.85},
            
            # Material fallbacks
            "bronze": {"closest": "metal", "similarity": 0.8}, \
            "aluminum": {"closest": "metal", "similarity": 0.9}, \
            "copper": {"closest": "metal", "similarity": 0.8}, \
            "ceramic": {"closest": "stone", "similarity": 0.6}, \
            "granite": {"closest": "stone", "similarity": 0.9}, \
            "slate": {"closest": "stone", "similarity": 0.85}, \
            "timber": {"closest": "wood", "similarity": 0.9}, \
            "oak": {"closest": "wood", "similarity": 0.85}, \
            "pine": {"closest": "wood", "similarity": 0.85},
            
            # Color fallbacks
            "crimson": {"closest": "red", "similarity": 0.9}, \
            "scarlet": {"closest": "red", "similarity": 0.85}, \
            "azure": {"closest": "blue", "similarity": 0.9}, \
            "navy": {"closest": "blue", "similarity": 0.8}, \
            "turquoise": {"closest": "cyan", "similarity": 0.8}, \
            "emerald": {"closest": "green", "similarity": 0.85}, \
            "amber": {"closest": "orange", "similarity": 0.8}, \
            "violet": {"closest": "purple", "similarity": 0.9}, \
            "tan": {"closest": "brown", "similarity": 0.8},
            
            # Style fallbacks
            "clean": {"closest": "minimalist", "similarity": 0.8}, \
            "elaborate": {"closest": "ornate", "similarity": 0.85}, \
            "complex": {"closest": "detailed", "similarity": 0.8}, \
            "lifelike": {"closest": "realistic", "similarity": 0.9}, \
            "vibrant": {"closest": "vivid", "similarity": 0.9}
        }
        
        # Confidence tracking for the parser
        self.token_confidence = {}
        self.overall_confidence = 1.0
    
    def parse(self, prompt: str) -> Dict[str, Any]:
        """Parse a free-form text prompt into a structured scene vector.
        
        Args:
            prompt: A string containing the text description to parse
            
        Returns:
            A dictionary containing structured scene elements:
            - objects: List of objects with their properties
            - modifiers: Dictionary mapping objects to their modifiers
            - layout: List of spatial relationships
            - style: List of global style directives
        """
        # Reset confidence tracking
        self.token_confidence = {}
        self.overall_confidence = 1.0
        
        # Initialize scene vector
        scene_vector = { \
            "objects": [],
            "modifiers": {}, \
            "layout": [], \
            "style": [], \
            "confidence_scores": {}
        }
        
        # Basic tokenization (split on spaces and remove punctuation)
        # In a real implementation, this would use a proper NLP tokenizer
        tokens = prompt.lower().replace(",", " ").replace(".", " ").split()
        
        # First pass: identify objects
        object_tokens = []
        skip_indices = set()
        
        for i, token in enumerate(tokens):
            if i in skip_indices:
                continue
                
            # Check for multi-word objects ("next to" etc.)
            if i < len(tokens) - 1:
                bigram = f"{token} {tokens[i+1]}"
                if bigram in self.relation_lexicon:
                    skip_indices.add(i+1)
                    continue
                    
                if i < len(tokens) - 2:
                    trigram = f"{token} {tokens[i+1]} {tokens[i+2]}"
                    if trigram in self.relation_lexicon:
                        skip_indices.add(i+1)
                        skip_indices.add(i+2)
                        continue
            
            # Look for objects
            if token in self.object_lexicon:
                object_tokens.append((i, token))
                token_confidence = 1.0
            elif token in self.embedding_lookup and self.embedding_lookup[token]["closest"] in self.object_lexicon:
                # Use embedding fallback
                object_tokens.append((i, self.embedding_lookup[token]["closest"]))
                token_confidence = self.embedding_lookup[token]["similarity"]
                self.token_confidence[f"object_{len(object_tokens)-1}"] = token_confidence
                self.overall_confidence *= token_confidence
        
        # Process identified objects
        for idx, (token_idx, obj_type) in enumerate(object_tokens):
            obj_data = self.object_lexicon[obj_type].copy()
            obj_data["type"] = obj_type
            
            # Record the original token if it was from the embedding fallback
            if f"object_{idx}" in self.token_confidence:
                obj_data["original_type"] = tokens[token_idx]
            
            scene_vector["objects"].append(obj_data)
            scene_vector["modifiers"][obj_type] = []
        
        # Second pass: identify modifiers and attach to objects
        for i, token in enumerate(tokens):
            if i in skip_indices:
                continue
                
            if token in self.modifier_lexicon:
                # Find the closest object to attach this modifier to
                closest_obj_idx = self._find_closest_object(i, object_tokens)
                if closest_obj_idx is not None:
                    obj_type = scene_vector["objects"][closest_obj_idx]["type"]
                    scene_vector["modifiers"][obj_type].append(token)
            elif token in self.embedding_lookup and \
                    self.embedding_lookup[token]["closest"] in self.modifier_lexicon:
                # Use embedding fallback
                closest_obj_idx = self._find_closest_object(i, object_tokens)
                if closest_obj_idx is not None:
                    obj_type = scene_vector["objects"][closest_obj_idx]["type"]
                    fallback = self.embedding_lookup[token]["closest"]
                    scene_vector["modifiers"][obj_type].append(fallback)
                    self.token_confidence[f"modifier_{i}"] = self.embedding_lookup[token]["similarity"]
                    self.overall_confidence *= self.embedding_lookup[token]["similarity"]
        
        # Third pass: identify spatial relationships
        for i in range(len(tokens) - 2):  # Need at least 3 tokens for a relation
            if i in skip_indices or i+1 in skip_indices or i+2 in skip_indices:
                continue
                
            # Check for relations in various n-gram patterns
            relation = None
            relation_tokens = 1
            
            # Check trigram relation
            if i < len(tokens) - 2:
                trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                if trigram in self.relation_lexicon:
                    relation = self.relation_lexicon[trigram]["value"]
                    relation_tokens = 3
            
            # Check bigram relation
            if relation is None and i < len(tokens) - 1:
                bigram = f"{tokens[i]} {tokens[i+1]}"
                if bigram in self.relation_lexicon:
                    relation = self.relation_lexicon[bigram]["value"]
                    relation_tokens = 2
            
            # Check single token relation
            if relation is None and tokens[i] in self.relation_lexicon:
                relation = self.relation_lexicon[tokens[i]]["value"]
                relation_tokens = 1
            
            if relation is not None:
                # Find objects before and after the relation
                subject_idx = None
                object_idx = None
                
                # Find the nearest object before the relation
                for j, (token_idx, _) in enumerate(object_tokens):
                    if token_idx < i:
                        subject_idx = j
                
                # Find the nearest object after the relation
                for j, (token_idx, _) in enumerate(object_tokens):
                    if token_idx > i + relation_tokens - 1:
                        object_idx = j
                        break
                
                if subject_idx is not None and object_idx is not None:
                    relation_data = { \
                        "subject": scene_vector["objects"][subject_idx]["type"],
                        "relation": relation, \
                        "object": scene_vector["objects"][object_idx]["type"]
                    }
                    scene_vector["layout"].append(relation_data)
        
        # Fourth pass: identify global style directives
        for token in tokens:
            if token in self.style_lexicon:
                scene_vector["style"].append(self.style_lexicon[token])
            elif token in self.embedding_lookup and \
                    self.embedding_lookup[token]["closest"] in self.style_lexicon:
                style = self.style_lexicon[self.embedding_lookup[token]["closest"]]
                scene_vector["style"].append(style)
                self.token_confidence[f"style_{token}"] = self.embedding_lookup[token]["similarity"]
                self.overall_confidence *= self.embedding_lookup[token]["similarity"]
        
        # Add confidence scores
        scene_vector["confidence_scores"] = self.token_confidence.copy()
        scene_vector["confidence_scores"]["overall"] = self.overall_confidence
        
        return scene_vector
    
    def _find_closest_object(self, idx: int, object_tokens: List[Tuple[int, str]]) -> Optional[int]:
        """Find the closest object to a given token index.
        
        Args:
            idx: Index of the token to find the closest object for
            object_tokens: List of (index, token) tuples for objects
            
        Returns:
            Index of the closest object in the scene vector objects list
        """
        if not object_tokens:
            return None
            
        closest_obj = None
        min_distance = float('inf')
        
        for obj_idx, (token_idx, _) in enumerate(object_tokens):
            distance = abs(idx - token_idx)
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj_idx
        
        return closest_obj
    
    def get_confidence(self) -> float:
        """Get the overall confidence score for the last parsing operation.
        
        Returns:
            A float between 0.0 and 1.0 representing parsing confidence
        """
        return self.overall_confidence

# ============================================================================ #
#                 1. LEXICAL-SEMANTIC INTERPRETATION                          #
# ============================================================================ #
# This stage processes input prompts to extract semantic meaning using:
# - Rule-based tokenization and phrase detection
# - Ontological mapping of natural language to visual concepts
# - Confidence scoring and alternative interpretations

@dataclass
class SemanticFeature:
    """Represents a semantic feature with confidence score and alternatives."""
    name: str
    value: str
    confidence: float = 1.0
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return { \
            "name": self.name,
            "value": self.value, \
            "confidence": self.confidence, \
            "alternatives": self.alternatives
        }

class SceneOntology:
    """Defines the ontological categories and taxonomies for scene understanding."""
    
    # Scene composition categories
    SCENE_TYPES = { \
        "seascape": ["ocean", "sea", "beach", "coast", "shore", "waves", "marine"],
        "landscape": ["mountain", "hill", "valley", "field", "meadow", "forest", "woods", "rural"], \
        "cityscape": ["city", "urban", "town", "building", "skyscraper", "street", "architecture"], \
        "interior": ["room", "inside", "indoor", "house", "home", "office", "interior"]
    }
    
    # Temporal states
    TIME_OF_DAY = { \
        "dawn": ["sunrise", "early morning", "daybreak", "first light"],
        "morning": ["am", "forenoon", "before noon"], \
        "midday": ["noon", "middle of day", "zenith"], \
        "afternoon": ["post-noon", "pm", "late day"], \
        "golden hour": ["sunset lighting", "magic hour", "evening glow"], \
        "sunset": ["dusk", "sundown", "twilight", "evening"], \
        "night": ["dark", "midnight", "after dark", "nocturnal"]
    }
    
    # Atmospheric conditions
    ATMOSPHERIC = { \
        "clear": ["sunny", "bright", "cloudless", "pristine", "sharp", "clear sky", "vivid"],
        "cloudy": ["overcast", "clouds", "gray sky", "white clouds", "fluffy clouds"], \
        "foggy": ["mist", "haze", "fog", "misty", "murky", "smog", "obscured"], \
        "rainy": ["rain", "downpour", "wet", "storm", "showers", "drizzle"], \
        "snowy": ["snow", "snowing", "blizzard", "winter scene", "snowfall", "frost"], \
        "stormy": ["thunder", "lightning", "tempest", "thunderstorm", "dramatic sky"]
    }
    
    # Architectural styles
    ARCHITECTURAL = { \
        "modern": ["contemporary", "minimalist", "sleek", "current", "recent", "present-day"],
        "classical": ["greco-roman", "ancient", "columns", "pediment", "traditional"], \
        "gothic": ["medieval", "cathedral", "pointed arch", "spire", "buttress"], \
        "victorian": ["ornate", "19th century", "gingerbread", "baroque", "decorated"], \
        "brutalist": ["concrete", "raw", "blocky", "brutalism", "monolithic"], \
        "futuristic": ["sci-fi", "advanced", "future", "high-tech", "avant-garde"], \
        "rustic": ["cottage", "cabin", "rural", "farmhouse", "country", "old"]
    }
    
    # Material surfaces
    MATERIALS = { \
        "stone": ["rock", "granite", "marble", "rocky", "cliff", "boulder", "slate"],
        "wood": ["timber", "lumber", "wooden", "log", "plank", "bark"], \
        "metal": ["steel", "iron", "metallic", "chrome", "brass", "copper", "aluminum"], \
        "glass": ["transparent", "crystal", "translucent", "window", "mirror", "glossy"], \
        "water": ["liquid", "fluid", "wet", "reflective", "pool", "lake", "river", "ocean", "sea"], \
        "vegetation": ["plants", "grass", "tree", "forest", "leaves", "foliage", "green"]
    }
    
    # Symbolic anchors (key features)
    SYMBOLIC = { \
        "focal_point": ["central", "main", "focus", "highlight", "key element", "subject"],
        "horizon": ["skyline", "distance", "far", "vanishing point"], \
        "natural_features": ["mountain", "river", "lake", "cliff", "canyon", "forest"], \
        "landmarks": ["monument", "statue", "tower", "bridge", "famous", "iconic"], \
        "human_elements": ["person", "people", "figure", "crowd", "human"]
    }
    
    # Style and mood categories
    MOOD = { \
        "tranquil": ["peaceful", "calm", "serene", "quiet", "still", "gentle", "relaxing"],
        "dramatic": ["intense", "powerful", "striking", "bold", "dynamic", "strong"], \
        "melancholic": ["sad", "somber", "gloomy", "moody", "dark", "emotional"], \
        "joyful": ["happy", "bright", "cheerful", "vibrant", "lively", "colorful"], \
        "mysterious": ["enigmatic", "foggy", "obscured", "shadowy", "hidden", "eerie"]
    }
    
    @classmethod
    def get_all_categories(cls) -> Dict[str, Dict[str, List[str]]]:
        """Return all taxonomy categories."""
        return { \
            "scene_type": cls.SCENE_TYPES,
            "time_of_day": cls.TIME_OF_DAY, \
            "atmospheric": cls.ATMOSPHERIC, \
            "architectural": cls.ARCHITECTURAL, \
            "materials": cls.MATERIALS, \
            "symbolic": cls.SYMBOLIC, \
            "mood": cls.MOOD
        }


class ArchitecturalStyleClassifier:
    """Classifies architectural styles, epochs, materials and regional characteristics.
    
    This implements a hybrid rule-pattern model that maps descriptive phrases to
    architectural properties that can be procedurally generated through appropriate
    geometric shapes, textures, and decorative elements.
    """
    
    def __init__(self):
        # Architectural epoch and style mapping with associated visual features
        self.style_vocabulary = {
            # Classical and historical styles
            "greek": { \
                "columns": True, "column_style": "doric|ionic|corinthian",
                "pediment": True, "symmetry": 0.95, "ornate_level": 0.6, \
                "material": "marble", "color": "#F5F5F5", "elevation_ratio": 0.6
            }, \
            "roman": { \
                "columns": True, "column_style": "corinthian|composite", \
                "arches": True, "dome": True, "symmetry": 0.9, "ornate_level": 0.7, \
                "material": "stone", "color": "#E8E6D9", "elevation_ratio": 0.65
            }, \
            "gothic": { \
                "pointed_arches": True, "flying_buttress": True, "ribbed_vault": True, \
                "spires": True, "rose_windows": True, "ornate_level": 0.85, \
                "symmetry": 0.8, "vertical_emphasis": 0.9, "material": "stone", \
                "color": "#D9D0C1", "elevation_ratio": 0.9
            }, \
            "renaissance": { \
                "columns": True, "symmetry": 0.95, "dome": True, "ornate_level": 0.7, \
                "proportion": "golden_ratio", "material": "stone|marble", \
                "color": "#E8E6D9", "elevation_ratio": 0.7
            }, \
            "baroque": { \
                "columns": True, "curved_forms": True, "ornate_level": 0.95, \
                "dramatic_lighting": True, "gilt": True, "material": "stone|marble", \
                "color": "#F0EAD6", "elevation_ratio": 0.75
            }, \
            "neoclassical": { \
                "columns": True, "column_style": "ionic|corinthian", "symmetry": 0.9, \
                "pediment": True, "ornate_level": 0.6, "material": "stone|marble", \
                "color": "#F5F5F5", "elevation_ratio": 0.65
            },
            
            # Modern and contemporary styles
            "art_deco": { \
                "geometric_patterns": True, "zigzag": True, "symmetry": 0.8,
                "stepped_forms": True, "ornate_level": 0.7, "material": "concrete|metal", \
                "color": "#D4AF37", "window_style": "decorative", "elevation_ratio": 0.8
            }, \
            "bauhaus": { \
                "flat_roof": True, "geometric": True, "asymmetry": 0.6, \
                "primary_colors": True, "ornate_level": 0.1, "material": "concrete|glass", \
                "color": "#FFFFFF", "window_style": "ribbon", "elevation_ratio": 0.6
            }, \
            "brutalist": { \
                "raw_concrete": True, "monolithic": True, "geometric": True, \
                "ornate_level": 0.05, "material": "concrete", "color": "#C0C0C0", \
                "window_style": "punctured", "elevation_ratio": 0.75
            }, \
            "modernist": { \
                "flat_roof": True, "clean_lines": True, "minimal": True, \
                "large_windows": True, "ornate_level": 0.1, "material": "concrete|glass|steel", \
                "color": "#FFFFFF", "window_style": "large", "elevation_ratio": 0.6
            }, \
            "postmodern": { \
                "eclectic": True, "playful": True, "colorful": True, "asymmetry": 0.7, \
                "ornate_level": 0.6, "material": "varied", "color": "mixed", \
                "window_style": "varied", "elevation_ratio": 0.7
            }, \
            "deconstructivist": { \
                "fragmented_forms": True, "non_rectilinear": True, "asymmetry": 0.9, \
                "distortion": True, "ornate_level": 0.4, "material": "metal|glass", \
                "color": "#D3D3D3", "window_style": "irregular", "elevation_ratio": 0.75
            }, \
            "high_tech": { \
                "exposed_structure": True, "industrial": True, "metal": True, \
                "ornate_level": 0.3, "material": "steel|glass", "color": "#C0C0C0", \
                "window_style": "glazed", "elevation_ratio": 0.8
            }, \
            "parametric": { \
                "curved_forms": True, "organic": True, "complex": True, \
                "fluid": True, "ornate_level": 0.6, "material": "metal|glass", \
                "color": "#FFFFFF", "window_style": "irregular", "elevation_ratio": 0.85
            },
            
            # Regional styles
            "mediterranean": { \
                "stucco": True, "tile_roof": True, "arches": True,
                "ornate_level": 0.5, "material": "stucco", "color": "#F5DEB3", \
                "window_style": "small", "elevation_ratio": 0.5
            }, \
            "victorian": { \
                "pitched_roof": True, "bay_windows": True, "ornate_level": 0.85, \
                "asymmetry": 0.6, "material": "wood|brick", "color": "#CD5C5C", \
                "window_style": "sash", "elevation_ratio": 0.9
            }, \
            "colonial": { \
                "symmetry": 0.9, "columns": True, "shutters": True, \
                "pitched_roof": True, "ornate_level": 0.4, "material": "wood|brick", \
                "color": "#FFFFFF", "window_style": "sash", "elevation_ratio": 0.7
            }, \
            "tudor": { \
                "half_timber": True, "steep_roof": True, "ornate_level": 0.6, \
                "material": "timber|plaster", "color": "#F5F5DC", "window_style": "leaded", \
                "elevation_ratio": 0.8
            }, \
            "islamic": { \
                "domes": True, "arches": True, "geometric_patterns": True, \
                "ornate_level": 0.9, "material": "stone", "color": "#F0E68C", \
                "window_style": "arched", "elevation_ratio": 0.7
            }, \
            "asian": { \
                "pagoda": True, "curved_roof": True, "ornate_level": 0.7, \
                "material": "wood", "color": "#8B4513", "window_style": "lattice", \
                "elevation_ratio": 0.65
            },
            
            # Futuristic and speculative styles
            "futuristic": { \
                "curved_forms": True, "parametric": True, "asymmetry": 0.7,
                "material": "glass|metal|composite", "color": "#FFFFFF", \
                "window_style": "panoramic", "spire": True, "antenna": True, \
                "glow": True, "ornate_level": 0.4, "elevation_ratio": 0.85
            }, \
            "cyberpunk": { \
                "neon": True, "industrial": True, "asymmetry": 0.8, \
                "material": "metal|concrete|glass", "color": "#000000", \
                "window_style": "led_frames", "ornate_level": 0.6, \
                "emissive": True, "elevation_ratio": 0.8
            }, \
            "solarpunk": { \
                "green_roof": True, "solar_panels": True, "organic": True, \
                "material": "wood|glass|living", "color": "#7CFC00", \
                "window_style": "large", "ornate_level": 0.5, "elevation_ratio": 0.7
            }, \
            "megastructure": { \
                "monolithic": True, "massive": True, "geometric": True, \
                "material": "concrete|metal", "color": "#808080", \
                "window_style": "grid", "ornate_level": 0.3, "elevation_ratio": 0.95
            }
        }
        
        # Materials with their visual properties
        self.materials = { \
            "stone": {"texture": "rough", "pattern": "natural", "reflectivity": 0.1},
            "marble": {"texture": "smooth", "pattern": "veined", "reflectivity": 0.4}, \
            "concrete": {"texture": "rough", "pattern": "solid", "reflectivity": 0.1}, \
            "glass": {"texture": "smooth", "pattern": "transparent", "reflectivity": 0.9}, \
            "metal": {"texture": "smooth", "pattern": "solid", "reflectivity": 0.7}, \
            "steel": {"texture": "smooth", "pattern": "solid", "reflectivity": 0.6}, \
            "wood": {"texture": "grained", "pattern": "natural", "reflectivity": 0.3}, \
            "brick": {"texture": "rough", "pattern": "regular", "reflectivity": 0.2}, \
            "stucco": {"texture": "textured", "pattern": "solid", "reflectivity": 0.1}, \
            "timber": {"texture": "grained", "pattern": "natural", "reflectivity": 0.3}, \
            "plaster": {"texture": "smooth", "pattern": "solid", "reflectivity": 0.2}, \
            "composite": {"texture": "varied", "pattern": "varied", "reflectivity": 0.5}, \
            "living": {"texture": "organic", "pattern": "natural", "reflectivity": 0.3}
        }

        # Compound descriptors that map to styles
        self.descriptor_mappings = {
            # Classical references
            "classical": ["greek", "roman"], \
            "ancient": ["greek", "roman"], \
            "temple": ["greek", "roman"], \
            "cathedral": ["gothic", "baroque"], \
            "church": ["gothic", "baroque", "renaissance"], \
            "castle": ["gothic", "medieval"], \
            "palace": ["baroque", "renaissance", "neoclassical"],
            
            # Modern references
            "modern": ["modernist", "bauhaus", "high_tech"], \
            "contemporary": ["modernist", "high_tech", "parametric"], \
            "sleek": ["modernist", "high_tech"], \
            "minimalist": ["bauhaus", "modernist"], \
            "industrial": ["brutalist", "high_tech"], \
            "eco": ["solarpunk"], \
            "sustainable": ["solarpunk"],
            
            # Regional references
            "european": ["gothic", "renaissance", "baroque"], \
            "asian": ["asian"], \
            "middle eastern": ["islamic"], \
            "american": ["colonial", "victorian"], \
            "italian": ["renaissance", "mediterranean"], \
            "spanish": ["mediterranean"], \
            "japanese": ["asian"], \
            "chinese": ["asian"],
            
            # Futuristic references
            "future": ["futuristic", "cyberpunk"], \
            "sci-fi": ["futuristic", "cyberpunk", "megastructure"], \
            "dystopian": ["cyberpunk", "megastructure"], \
            "utopian": ["solarpunk", "futuristic"], \
            "cybernetic": ["cyberpunk"], \
            "neon": ["cyberpunk"], \
            "hyper-modern": ["futuristic", "parametric"]
        }
    
    def classify_architectural_style(self, text: str) -> Dict[str, Any]:
        """Analyze text to extract architectural style features.
        
        Args:
            text: The text prompt to analyze
            
        Returns:
            Dictionary with architectural style properties and confidence scores
        """
        normalized_text = text.lower()
        
        # Initialize results with confidence scores
        results = { \
            "style": {"value": "modern", "confidence": 0.2},  # Default modern style
            "features": {}, \
            "materials": []
        }
        
        # Check for direct style mentions
        max_confidence = 0.2  # Start with default confidence
        for style, properties in self.style_vocabulary.items():
            style_terms = [style] + style.split('_')  # Include both combined and individual terms
            for term in style_terms:
                if term in normalized_text:
                    confidence = 0.8  # Direct mention has high confidence
                    if confidence > max_confidence:
                        results["style"] = {"value": style, "confidence": confidence}
                        max_confidence = confidence
                        
                        # Add associated features
                        for feature, value in properties.items():
                            if feature != "color" and feature != "material":
                                results["features"][feature] = {"value": value, "confidence": confidence}
                        
                        # Add material if specified
                        if isinstance(properties.get("material"), str):
                            materials = properties["material"].split("|")  # Handle multiple materials
                            for material in materials:
                                if material not in [m["value"] for m in results["materials"]]:
                                    results["materials"].append({"value": material, "confidence": 0.7})
        
        # Check for descriptor mappings (indirect stylistic references)
        for descriptor, styles in self.descriptor_mappings.items():
            if descriptor in normalized_text:
                for style in styles:
                    # Boost confidence slightly less than direct mentions
                    confidence = 0.6
                    if confidence > max_confidence:
                        results["style"] = {"value": style, "confidence": confidence}
                        max_confidence = confidence
                        
                        # Add associated features with lower confidence
                        properties = self.style_vocabulary.get(style, {})
                        for feature, value in properties.items():
                            if feature != "color" and feature != "material":
                                results["features"][feature] = {"value": value, "confidence": confidence * 0.9}
                        
                        # Add material if specified
                        if isinstance(properties.get("material"), str):
                            materials = properties["material"].split("|")  # Handle multiple materials
                            primary_material = materials[0]  # Take first as primary
                            if primary_material not in [m["value"] for m in results["materials"]]:
                                results["materials"].append({"value": primary_material, "confidence": 0.6})
        
        # Extract specific materials mentioned directly
        for material in self.materials.keys():
            if material in normalized_text:
                if material not in [m["value"] for m in results["materials"]]:
                    results["materials"].append({"value": material, "confidence": 0.8})
        
        # Look for specific architectural features in text
        architectural_features = [ \
            "columns", "arches", "dome", "spire", "tower", "facade", "roof", "skyscraper",
            "pagoda", "minaret", "buttress", "stained glass", "balcony", "terrace", \
            "courtyard", "portico", "rotunda", "pillar", "frieze", "pediment"
        ]
        
        # Check for these features in text and add to results
        for feature in architectural_features:
            if feature in normalized_text:
                results["features"][feature] = {"value": True, "confidence": 0.8}
        
        return results


class SemanticAnalyzer:
    """Analyzes prompts for semantic features using ontological mapping."""
    
    def __init__(self):
        self.ontology = SceneOntology()
        self.categories = self.ontology.get_all_categories()
        self.arch_classifier = ArchitecturalStyleClassifier()
        
    def analyze_prompt(self, prompt: str) -> Dict[str, SemanticFeature]:
        """Extract semantic features from a text prompt using ontological mapping.
        
        Args:
            prompt: Text prompt describing the desired scene
            
        Returns:
            Dictionary of semantic features with confidence scores
        """
        # Normalize prompt
        prompt = prompt.lower().strip()
        words = set(prompt.split())
        
        # Extract features by category
        features = {}
        
        # Process each category
        for category_name, category_dict in self.categories.items():
            best_match = None
            best_score = 0.0
            alternatives = []
            
            for subcategory, keywords in category_dict.items():
                # Calculate match score based on keyword presence
                matches = sum(1 for keyword in keywords if keyword in prompt)
                additional_matches = sum(1 for keyword in keywords \
                                        if any(keyword in word for word in words))
                
                total_matches = matches + 0.5 * additional_matches
                
                # If has any matches, consider it as potential match
                if total_matches > 0:
                    confidence = min(1.0, total_matches / (len(keywords) * 0.5))
                    
                    if confidence > best_score:
                        if best_score > 0:
                            alternatives.append((best_match, best_score))
                        best_match = subcategory
                        best_score = confidence
                    else:
                        alternatives.append((subcategory, confidence))
            
            # Add feature if we found a match
            if best_match is not None:
                features[category_name] = SemanticFeature( \
                    name=category_name,
                    value=best_match, \
                    confidence=best_score, \
                    alternatives=sorted(alternatives, key=lambda x: x[1], reverse=True)[:3]
                )
        
        # Add additional analysis for specific elements (to be expanded)
        self._analyze_specific_elements(prompt, features)
        
        return features
    
    def _analyze_specific_elements(self, prompt: str, features: Dict[str, SemanticFeature]) -> None:
        """Extract specific elements like colors, objects, etc.
        
        Args:
            prompt: Text prompt to analyze
            features: Features dictionary to update
        """
        # Color analysis
        colors = self._extract_colors(prompt)
        if colors:
            features["colors"] = SemanticFeature( \
                name="colors",
                value=colors[0][0], \
                confidence=colors[0][1], \
                alternatives=colors[1:4] if len(colors) > 1 else []
            )
        
        # Subject/focal points
        focal_elements = self._extract_focal_elements(prompt)
        if focal_elements:
            features["focal_element"] = SemanticFeature( \
                name="focal_element",
                value=focal_elements[0][0], \
                confidence=focal_elements[0][1], \
                alternatives=focal_elements[1:3] if len(focal_elements) > 1 else []
            )
        
        # Enhanced architectural style analysis
        if "architectural" in features:
            # Get base architectural style from ontology-based analysis
            base_style = features["architectural"].value
            base_confidence = features["architectural"].confidence
            
            # Use enhanced architectural classifier for detailed analysis
            arch_results = self.arch_classifier.classify_architectural_style(prompt)
            
            # If the classifier finds a style with higher confidence, use it
            if arch_results["style"]["confidence"] > base_confidence:
                features["architectural"] = SemanticFeature( \
                    name="architectural",
                    value=arch_results["style"]["value"], \
                    confidence=arch_results["style"]["confidence"], \
                    alternatives=features["architectural"].alternatives
                )
            
            # Add specific architectural features as separate entries
            for feature_name, feature_data in arch_results["features"].items():
                feature_key = f"arch_{feature_name}"
                features[feature_key] = SemanticFeature( \
                    name=feature_key,
                    value=str(feature_data["value"]), \
                    confidence=feature_data["confidence"]
                )
            
            # Add materials if found
            if arch_results["materials"]:
                materials = []
                confidences = []
                
                for material_data in arch_results["materials"]:
                    materials.append(material_data["value"])
                    confidences.append(material_data["confidence"])
                
                if materials:
                    features["arch_materials"] = SemanticFeature( \
                        name="arch_materials",
                        value=materials[0], \
                        confidence=confidences[0], \
                        alternatives=[(m, c) for m, c in zip(materials[1:], confidences[1:])]
                    )
    
    def _extract_colors(self, prompt: str) -> List[Tuple[str, float]]:
        """Extract color information from the prompt.
        
        Args:
            prompt: Text prompt to analyze
            
        Returns:
            List of (color, confidence) tuples
        """
        # Basic color detection (to be expanded with more sophisticated analysis)
        basic_colors = { \
            "red": ["crimson", "scarlet", "ruby", "vermillion"],
            "blue": ["azure", "navy", "cobalt", "cyan", "teal"], \
            "green": ["emerald", "olive", "lime", "forest", "jade"], \
            "yellow": ["gold", "amber", "lemon", "ocher"], \
            "purple": ["violet", "lavender", "magenta", "indigo"], \
            "orange": ["amber", "coral", "tangerine"], \
            "pink": ["rose", "salmon", "fuchsia"], \
            "brown": ["tan", "beige", "chocolate", "sepia"], \
            "white": ["snow", "ivory", "cream", "pale"], \
            "black": ["ebony", "obsidian", "onyx", "dark"], \
            "gray": ["silver", "ash", "charcoal", "slate"]
        }
        
        results = []
        
        for color, synonyms in basic_colors.items():
            if color in prompt:
                results.append((color, 1.0))
            else:
                for synonym in synonyms:
                    if synonym in prompt:
                        results.append((color, 0.8))
                        break
        
        # Infer from scene keywords in the prompt
        scene_keywords = { \
            "seascape": [("blue", 0.7)],
            "landscape": [("green", 0.7), ("brown", 0.6)], \
            "sunset": [("orange", 0.8), ("gold", 0.7), ("red", 0.6)], \
            "night": [("blue", 0.7), ("black", 0.6)], \
            "forest": [("green", 0.8), ("brown", 0.6)], \
            "beach": [("yellow", 0.7), ("blue", 0.6)], \
            "mountain": [("gray", 0.7), ("white", 0.6)], \
            "fog": [("gray", 0.8), ("white", 0.7)], \
            "desert": [("yellow", 0.8), ("orange", 0.6)]
        }
        
        # If no explicit colors, check for scene keywords
        if not results:
            for keyword, colors in scene_keywords.items():
                if keyword in prompt.lower():
                    results.extend(colors)
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def _extract_focal_elements(self, prompt: str) -> List[Tuple[str, float]]:
        """Extract potential focal elements from the prompt.
        
        Args:
            prompt: Text prompt to analyze
            
        Returns:
            List of (element, confidence) tuples
        """
        # Common focal elements to detect
        focal_elements = { \
            "mountain": ["peak", "summit", "hill", "highlands", "mountainous"],
            "tree": ["forest", "woods", "pine", "oak", "woodland"], \
            "building": ["architecture", "tower", "house", "structure", "construction"], \
            "person": ["human", "figure", "man", "woman", "child", "people"], \
            "animal": ["wildlife", "creature", "beast", "bird", "fish"], \
            "water": ["ocean", "sea", "lake", "river", "waterfall", "stream", "pond"], \
            "sun": ["sunshine", "solar", "sunlight", "sunbeam", "sunburst"], \
            "moon": ["lunar", "crescent", "satellite"], \
            "bridge": ["span", "crossing", "overpass", "arch"], \
            "road": ["path", "trail", "street", "highway", "way"]
        }
        
        results = []
        
        # First check direct mentions
        words = prompt.split()
        for element, synonyms in focal_elements.items():
            if element in prompt:
                # Check if it's a main subject (near the beginning or with adjectives)
                position_weight = 1.0 - (prompt.find(element) / len(prompt) * 0.5)
                results.append((element, position_weight))
            else:
                for synonym in synonyms:
                    if synonym in prompt:
                        position_weight = 1.0 - (prompt.find(synonym) / len(prompt) * 0.5)
                        results.append((element, position_weight * 0.9))
                        break
        
        return sorted(results, key=lambda x: x[1], reverse=True)

def extract_scene_features(prompt: str) -> Dict[str, Any]:
    """Extract scene features from a text prompt using semantic analysis.
    
    This function processes the prompt text to identify semantic features like:
    - Scene type (landscape, seascape, cityscape, etc.)
    - Time of day (dawn, midday, sunset, night)
    - Atmospheric conditions (clear, cloudy, foggy, etc.)
    - Mood (tranquil, dramatic, etc.)
    - Focal elements (mountains, buildings, etc.)
    
    Args:
        prompt: Text prompt describing the desired scene
        
    Returns:
        Dictionary of scene features with confidence scores
    """
    analyzer = SemanticAnalyzer()
    features = analyzer.analyze_prompt(prompt)
    
    # Add architectural style information if applicable
    arch_info = analyzer.arch_classifier.classify_architectural_style(prompt)
    if arch_info:
        features['architectural'] = arch_info
    result["raw_prompt"] = prompt
    
    return result

@dataclass
class SceneVector:
    """Structured representation of a scene parsed from textual description.
    
    This dataclass holds the parsed and normalized semantic information from
    the text prompt, organized into a structured format that can be used for
    SVG generation.
    """
    scene_type: str = "landscape"  # landscape, seascape, cityscape, interior
    mood: str = "neutral"         # tranquil, dramatic, melancholic, joyful, mysterious
    time_of_day: str = "midday"   # dawn, morning, midday, afternoon, golden_hour, sunset, night
    weather: str = "clear"        # clear, cloudy, foggy, rainy, snowy, stormy
    
    # Composition elements
    focal_point: Optional[Dict[str, Any]] = None
    background: Optional[Dict[str, Any]] = None
    midground: Optional[Dict[str, Any]] = None
    foreground: Optional[Dict[str, Any]] = None
    
    # Style attributes
    color_palette: List[Tuple[int, int, int]] = field(default_factory=list)
    style_descriptors: List[str] = field(default_factory=list)
    detail_level: float = 0.5     # 0.0-1.0 representing level of detail
    
    # Architecture-specific attributes (only used for cityscapes)
    architecture: Optional[Dict[str, Any]] = None
    
    # Raw confidence scores from semantic parser
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneVector':
        """Create SceneVector from dictionary."""
        return cls(**data)


class HyperRealisticUnifiedPipeline:
    """Unified pipeline for hyper-realistic SVG generation.
    
    This class coordinates the end-to-end process of generating SVG illustrations
    from text prompts, by integrating the following components:
    1. Semantic parsing of text prompt
    2. Scene graph construction
    3. SVG generation
    4. Validation and optimization
    
    The final SVG output includes rich visual details like gradients, textures,
    and proper layering of elements.
    """
    
    def __init__(self, width: float = 800.0, height: float = 600.0, debug: bool = False):
        """Initialize the pipeline with configuration parameters.
        
        Args:
            width: Width of the SVG in pixels
            height: Height of the SVG in pixels
            debug: Enable debug logging and save intermediate files
        """
        self.width = width
        self.height = height
        self.debug = debug
        
        # Initialize components
        self.semantic_parser = EnhancedSemanticParser()
        self.scene_builder = SceneGraphBuilder(width=width, height=height)
        self.svg_generator = SVGGenerator()
        self.svg_validator = SVGValidator()
    
    def process(self, prompt: str, output_path: str) -> str:
        """Process a text prompt and generate an SVG illustration.
        
        This method orchestrates the full pipeline:
        1. Parse the prompt into semantic features
        2. Convert semantic features to a scene graph
        3. Render the scene graph as SVG
        4. Validate and optimize the SVG output
        
        Args:
            prompt: Text prompt describing the desired illustration
            output_path: Path to save the generated SVG
            
        Returns:
            The generated SVG content as a string
        """
        print(f"Processing prompt: {prompt}")
        
        # Step 1: Semantic parsing
        print("Step 1: Parsing prompt...")
        scene_features = extract_scene_features(prompt)
        scene_vector = SceneVector(**scene_features)
        
        if self.debug:
            # Save intermediate representations
            vector_path = output_path.replace('.svg', '.vector.json')
            with open(vector_path, 'w') as f:
                import json
                json.dump(scene_vector.to_dict(), f, indent=2)
            print(f"Saved scene vector to {vector_path}")
        
        # Step 2: Scene graph construction
        print("Step 2: Building scene graph...")
        scene_graph = self.scene_builder.build_scene_graph(scene_vector.to_dict())
        
        if self.debug:
            # Save scene graph representation
            graph_path = output_path.replace('.svg', '.graph.json')
            with open(graph_path, 'w') as f:
                import json
                scene_graph_data = {
                    "width": scene_graph.width,
                    "height": scene_graph.height,
                    "nodes": {k: v.__dict__ for k, v in scene_graph.nodes.items()}
                }
                json.dump(scene_graph_data, f, indent=2)
            print(f"Saved scene graph to {graph_path}")
        
        # Step 3: SVG generation
        print("Step 3: Generating SVG...")
        svg_content = self.svg_generator.generate_svg(scene_graph)
        
        # Step 4: Validation and optimization
        print("Step 4: Validating SVG...")
        validation_results = self.svg_validator.validate(svg_content)
        
        if not validation_results['valid']:
            print("Warning: SVG validation found issues:")
            for error in validation_results['errors']:
                print(f"- {error}")
        
        # Save the final SVG
        with open(output_path, 'w') as f:
            f.write(svg_content)
        
        return svg_content


def main():
    """Main entry point for the SVG generation pipeline.
    
    Parses command-line arguments and runs the pipeline.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate hyper-realistic SVG from text prompt')
    parser.add_argument('prompt', type=str, help='Text prompt describing the desired scene')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output file path for SVG (default: auto-generated with timestamp)')
    parser.add_argument('--width', '-w', type=int, default=800, help='SVG width in pixels (default: 800)')
    parser.add_argument('--height', '-ht', type=int, default=600, help='SVG height in pixels (default: 600)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging and save intermediate files')
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if not args.output:
        timestamp = int(time.time())
        clean_prompt = re.sub(r'[^a-z0-9]', '-', args.prompt.lower())[:30]
        args.output = f"{timestamp}_{clean_prompt}.svg"
    
    # Initialize and run the pipeline
    pipeline = HyperRealisticUnifiedPipeline(
        width=args.width, 
        height=args.height,
        debug=args.debug
    )
    
    svg_output = pipeline.process(args.prompt, args.output)
    
    # Print information about the generated SVG
    svg_size_kb = len(svg_output.encode('utf-8')) / 1024
    print(f"Generated SVG with size: {svg_size_kb:.2f} KB")
    print(f"SVG saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    main()
    """
    
    def __init__(self, embedding_path: str = None, lexicon_path: str = None):
        """Initialize the enhanced semantic parser with optimized resources.
        
        Args:
            embedding_path: Path to compressed embeddings file (optional)
            lexicon_path: Path to lexicon file (optional)
        """
        self.semantic_analyzer = SemanticAnalyzer()
        self.ontology = SceneOntology()
        
        # Load embeddings using memory-mapped I/O if NumPy is available
        self.embeddings = {}
        self.embedding_cache = {}
        self.lexicon_loaded = False
        
        if NUMPY_AVAILABLE and embedding_path and os.path.exists(embedding_path):
            try:
                # Use memory mapping for efficient loading
                self.embeddings = np.load(embedding_path, mmap_mode='r')
                self.dimensions = self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 100
                print(f"Loaded {len(self.embeddings)} embeddings with {self.dimensions} dimensions")
            except Exception as e:
                print(f"Failed to load embeddings from {embedding_path}: {e}")
                # Initialize with minimal static embeddings as fallback
                self._initialize_minimal_embeddings()
        else:
            # Initialize with minimal static embeddings as fallback
            self._initialize_minimal_embeddings()
        
        # Initialize NLP components
        self.stop_words = set(['a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of'])
        self.stemmer = self._create_minimal_stemmer()
        
        # Static lexicon for direct matching - expanded with detailed categories
        self.lexicon = {
            # Core lexicon for first-pass matching
            "urban": {"category": "scene_type", "value": "cityscape"}, \
            "city": {"category": "scene_type", "value": "cityscape"}, \
            "metropolis": {"category": "scene_type", "value": "cityscape"}, \
            "downtown": {"category": "scene_type", "value": "cityscape"}, \
            "skyline": {"category": "scene_type", "value": "cityscape"},
            
            # Time-of-day lexicon with numerical time encoding
            "dawn": {"category": "time_of_day", "value": "dawn", "time_value": 0.0}, \
            "sunrise": {"category": "time_of_day", "value": "dawn", "time_value": 0.1}, \
            "morning": {"category": "time_of_day", "value": "morning", "time_value": 0.25}, \
            "noon": {"category": "time_of_day", "value": "midday", "time_value": 0.5}, \
            "midday": {"category": "time_of_day", "value": "midday", "time_value": 0.5}, \
            "afternoon": {"category": "time_of_day", "value": "afternoon", "time_value": 0.7}, \
            "evening": {"category": "time_of_day", "value": "sunset", "time_value": 0.8}, \
            "sunset": {"category": "time_of_day", "value": "sunset", "time_value": 0.9}, \
            "dusk": {"category": "time_of_day", "value": "sunset", "time_value": 0.95}, \
            "night": {"category": "time_of_day", "value": "night", "time_value": 1.0}, \
            "midnight": {"category": "time_of_day", "value": "night", "time_value": 1.0},
            
            # Weather/atmospheric conditions with intensity values
            "sunny": {"category": "weather", "value": "clear", "intensity": 1.0}, \
            "clear": {"category": "weather", "value": "clear", "intensity": 0.9}, \
            "cloudless": {"category": "weather", "value": "clear", "intensity": 1.0}, \
            "cloudy": {"category": "weather", "value": "cloudy", "intensity": 0.6}, \
            "overcast": {"category": "weather", "value": "cloudy", "intensity": 0.9}, \
            "foggy": {"category": "weather", "value": "foggy", "intensity": 0.7}, \
            "misty": {"category": "weather", "value": "foggy", "intensity": 0.4}, \
            "rainy": {"category": "weather", "value": "rainy", "intensity": 0.7}, \
            "rain": {"category": "weather", "value": "rainy", "intensity": 0.6}, \
            "stormy": {"category": "weather", "value": "stormy", "intensity": 0.8}, \
            "storm": {"category": "weather", "value": "stormy", "intensity": 0.9}, \
            "thunder": {"category": "weather", "value": "stormy", "intensity": 0.7}, \
            "lightning": {"category": "weather", "value": "stormy", "intensity": 0.8}, \
            "snowy": {"category": "weather", "value": "snowy", "intensity": 0.7}, \
            "snow": {"category": "weather", "value": "snowy", "intensity": 0.6},
            
            # Material properties with technical attributes
            "glass": {"category": "material", "value": "glass", "transparency": 0.8, "reflectivity": 0.7}, \
            "metal": {"category": "material", "value": "metal", "transparency": 0.0, "reflectivity": 0.9}, \
            "stone": {"category": "material", "value": "stone", "transparency": 0.0, "reflectivity": 0.2}, \
            "wood": {"category": "material", "value": "wood", "transparency": 0.0, "reflectivity": 0.3}, \
            "brick": {"category": "material", "value": "brick", "transparency": 0.0, "reflectivity": 0.1}, \
            "concrete": {"category": "material", "value": "concrete", "transparency": 0.0, "reflectivity": 0.2}, \
            "marble": {"category": "material", "value": "marble", "transparency": 0.0, "reflectivity": 0.6},
            
            # Compound modifiers with structured properties
            "glass tower": {"category": "compound", "object_type": "building", "style": "modern", "material": "glass"}, \
            "stone building": {"category": "compound", "object_type": "building", "style": "traditional", "material": "stone"}, \
            "brick house": {"category": "compound", "object_type": "building", "style": "residential", "material": "brick"}, \
            "wooden cabin": {"category": "compound", "object_type": "building", "style": "rustic", "material": "wood"}, \
            "metal structure": {"category": "compound", "object_type": "structure", "style": "industrial", "material": "metal"},
            
            # Spatial relations with geometric interpretations
            "above": {"category": "spatial", "value": "above", "vector": (0, -1)}, \
            "below": {"category": "spatial", "value": "below", "vector": (0, 1)}, \
            "beside": {"category": "spatial", "value": "beside", "vector": (1, 0)}, \
            "behind": {"category": "spatial", "value": "behind", "z_offset": -1}, \
            "in front of": {"category": "spatial", "value": "in_front", "z_offset": 1}, \
            "inside": {"category": "spatial", "value": "inside", "containment": True}, \
            "outside": {"category": "spatial", "value": "outside", "containment": False},
            
            # Mood/atmosphere descriptors with visual parameters
            "peaceful": {"category": "mood", "value": "tranquil", "saturation": 0.4, "contrast": 0.3}, \
            "calm": {"category": "mood", "value": "tranquil", "saturation": 0.3, "contrast": 0.4}, \
            "serene": {"category": "mood", "value": "tranquil", "saturation": 0.5, "contrast": 0.3}, \
            "dramatic": {"category": "mood", "value": "dramatic", "saturation": 0.7, "contrast": 0.8}, \
            "intense": {"category": "mood", "value": "dramatic", "saturation": 0.8, "contrast": 0.9}
        }
        
        # Compiled regex patterns for bigram and trigram extraction
        self.compound_patterns = { \
            r"(\w+)\s+(building|tower|structure|house|cabin)": "architectural",
            r"(\w+)\s+(street|road|avenue|path|bridge)": "infrastructure", \
            r"(\w+)\s+(mountain|hill|valley|peak|cliff)": "landform", \
            r"(\w+)\s+(tree|forest|grove|woodland|plant)": "vegetation", \
            r"(\w+)\s+(sky|cloud|star|moon|sun)": "celestial", \
            r"(\w+)\s+(river|lake|ocean|sea|water)": "waterform"
        }
        
        # Spatial relation patterns
        self.spatial_patterns = { \
            r"(\w+)\s+above\s+(\w+)": "above_relation",
            r"(\w+)\s+below\s+(\w+)": "below_relation", \
            r"(\w+)\s+beside\s+(\w+)": "beside_relation", \
            r"(\w+)\s+next\s+to\s+(\w+)": "beside_relation", \
            r"(\w+)\s+behind\s+(\w+)": "behind_relation", \
            r"(\w+)\s+in\s+front\s+of\s+(\w+)": "front_relation", \
            r"(\w+)\s+inside\s+(\w+)": "inside_relation", \
            r"(\w+)\s+within\s+(\w+)": "inside_relation", \
            r"(\w+)\s+on\s+top\s+of\s+(\w+)": "on_relation"
        }
        
    def _create_minimal_stemmer(self):
        """Create a minimal stemming function that handles common morphological variants.
        
        This is a lightweight alternative to NLTKs stemmer for offline operation.
        """
        suffix_map = { \
            'ing': '', 'ed': '', 's': '', 'es': '', 'ies': 'y',
            'ly': '', 'ful': '', 'est': '', 'er': ''
        }
        
        def stem(word):
            for suffix, replacement in suffix_map.items():
                if word.endswith(suffix):
                    stem_length = len(word) - len(suffix)
                    if stem_length >= 3:  # Ensure we don't create ultra-short stems
                        return word[:stem_length] + replacement
            return word
            
        return stem
    
    def _initialize_minimal_embeddings(self):
        """Initialize a small set of minimal embeddings for core vocabulary.
        
        In production, this would be replaced with pre-computed embeddings loaded from a file.
        """
        if not NUMPY_AVAILABLE:
            return {}
            
        # Create word clusters with similar semantic meanings for minimal embedding space
        word_clusters = [ \
            ["city", "urban", "metropolis", "downtown", "buildings"],
            ["forest", "trees", "woods", "woodland", "jungle"], \
            ["mountain", "peak", "hill", "highlands", "ridge"], \
            ["water", "ocean", "sea", "lake", "river"], \
            ["sky", "clouds", "heaven", "atmosphere", "air"], \
            ["night", "dark", "evening", "dusk", "twilight"], \
            ["day", "noon", "morning", "daylight", "daytime"], \
            ["sunny", "bright", "clear", "radiant", "luminous"], \
            ["cloudy", "overcast", "gray", "gloomy", "dim"], \
            ["calm", "peaceful", "serene", "tranquil", "quiet"], \
            ["dramatic", "intense", "dynamic", "powerful", "bold"], \
            ["glass", "transparent", "crystal", "clear", "see-through"], \
            ["stone", "rock", "boulder", "concrete", "granite"], \
            ["metal", "steel", "iron", "metallic", "alloy"], \
            ["wood", "timber", "lumber", "wooden", "log"]
        ]
        
        # Create minimal dimensional vector space (using 20 dimensions for memory efficiency)
        dimensions = 20
        self.dimensions = dimensions
        
        # Deterministic embedding function to ensure consistency
        def get_embedding_vector(word, cluster_idx):
            # Use a deterministic seed based on the word and its cluster
            seed = hash(word) % 10000 + cluster_idx * 100
            rng = random.Random(seed)
            
            # Generate a base embedding for the cluster
            base = np.zeros(dimensions)
            for i in range(dimensions):
                if i % 5 == (cluster_idx % 5):
                    base[i] = 0.8 + rng.random() * 0.2
                elif i % 7 == (cluster_idx % 7):
                    base[i] = 0.6 + rng.random() * 0.2
                else:
                    base[i] = rng.random() * 0.2
                    
            # Add a small word-specific variation
            variation = np.array([rng.random() * 0.1 for _ in range(dimensions)])
            
            # Normalize to unit length for cosine similarity calculations
            vector = base + variation
            norm = np.linalg.norm(vector)
            
            if norm > 0:
                vector = vector / norm
                
            # Quantize to int8 for memory efficiency (scaled by 127)
            return (vector * 127).astype(np.int8)
            
        # Generate embeddings for each word in each cluster
        embeddings = {}
        for i, cluster in enumerate(word_clusters):
            for word in cluster:
                embeddings[word] = get_embedding_vector(word, i)
                
        self.embeddings = embeddings
        return embeddings
        
    def _tokenize_and_normalize(self, text: str) -> List[str]:
        """First-pass tokenization and normalization of input text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of normalized tokens
        """
        # Convert to lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize by whitespace
        tokens = text.split()
        
        # Filter stop words and normalize
        normalized_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 1:
                # Apply minimal stemming
                normalized = self.stemmer(token)
                normalized_tokens.append(normalized)
                
        return normalized_tokens
        
    def _exact_dictionary_match(self, tokens: List[str]) -> Dict[str, Dict[str, Any]]:
        """Perform exact matches against the lexicon.
        
        Args:
            tokens: List of normalized tokens
            
        Returns:
            Dictionary of matched tokens with their lexicon entries
        """
        matches = {}
        
        for token in tokens:
            if token in self.lexicon:
                matches[token] = self.lexicon[token]
                
        return matches
        
    def _extract_compound_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract compound patterns using regex patterns.
        
        This implements the second pass of the parsing process, finding phrases like
        "glass tower" or "wooden bridge" and extracting their structured semantics.
        
        Args:
            text: Original text to analyze
            
        Returns:
            List of extracted compound semantic objects
        """
        compounds = []
        text = text.lower()
        
        # Find all compound matches
        for pattern, sem_type in self.compound_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                modifier, noun = match.groups()
                
                # Check if this is a known compound
                compound_key = f"{modifier} {noun}"
                if compound_key in self.lexicon:
                    compounds.append(self.lexicon[compound_key])
                else:
                    # Generate a dynamic compound interpretation
                    compounds.append({ \
                        "category": "compound",
                        "type": sem_type, \
                        "modifier": modifier, \
                        "noun": noun
                    })
        
        # Extract spatial relations
        for pattern, relation_type in self.spatial_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                subject, obj = match.groups()
                compounds.append({ \
                    "category": "spatial_relation",
                    "type": relation_type, \
                    "subject": subject, \
                    "object": obj
                })
                
        return compounds
    
    # Additional lexicon entries for the class - should be part of self.lexicon initialization
    def _initialize_extended_lexicon(self):
        """Initialize additional lexicon entries for enhanced semantic matching."""
        extended_entries = {
            # Scene types
            "ocean": {"category": "scene_type", "value": "seascape"}, \
            "sea": {"category": "scene_type", "value": "seascape"},
            "beach": {"category": "scene_type", "value": "seascape"}, \
            "mountain": {"category": "scene_type", "value": "landscape"},
            "forest": {"category": "scene_type", "value": "landscape"}, \
            "city": {"category": "scene_type", "value": "cityscape"},
            "urban": {"category": "scene_type", "value": "cityscape"}, \
            "room": {"category": "scene_type", "value": "interior"},
            "indoor": {"category": "scene_type", "value": "interior"},
            
            # Time of day
            "sunrise": {"category": "time_of_day", "value": "dawn"}, \
            "morning": {"category": "time_of_day", "value": "morning"},
            "noon": {"category": "time_of_day", "value": "midday"}, \
            "afternoon": {"category": "time_of_day", "value": "afternoon"},
            "sunset": {"category": "time_of_day", "value": "sunset"}, \
            "dusk": {"category": "time_of_day", "value": "sunset"},
            "night": {"category": "time_of_day", "value": "night"}, \
            "midnight": {"category": "time_of_day", "value": "night"},
            
            # Weather/atmospheric conditions
            "sunny": {"category": "weather", "value": "clear"}, \
            "clear": {"category": "weather", "value": "clear"},
            "cloudless": {"category": "weather", "value": "clear"}, \
            "cloudy": {"category": "weather", "value": "cloudy"},
            "overcast": {"category": "weather", "value": "cloudy"}, \
            "foggy": {"category": "weather", "value": "foggy"},
            "misty": {"category": "weather", "value": "foggy"}, \
            "rainy": {"category": "weather", "value": "rainy"},
            "rain": {"category": "weather", "value": "rainy"}, \
            "stormy": {"category": "weather", "value": "stormy"},
            "storm": {"category": "weather", "value": "stormy"}, \
            "thunder": {"category": "weather", "value": "stormy"},
            "lightning": {"category": "weather", "value": "stormy"}, \
            "snowy": {"category": "weather", "value": "snowy"},
            "snow": {"category": "weather", "value": "snowy"},
            
            # Mood
            "peaceful": {"category": "mood", "value": "tranquil"}, \
            "calm": {"category": "mood", "value": "tranquil"},
            "serene": {"category": "mood", "value": "tranquil"}, \
            "intense": {"category": "mood", "value": "dramatic"},
            "powerful": {"category": "mood", "value": "dramatic"}, \
            "sad": {"category": "mood", "value": "melancholic"},
            "somber": {"category": "mood", "value": "melancholic"}, \
            "somber": {"category": "mood", "value": "melancholic"}, \
            "gloomy": {"category": "mood", "value": "melancholic"}, \
            "happy": {"category": "mood", "value": "joyful"}, \
            "bright": {"category": "mood", "value": "joyful"}, \
            "cheerful": {"category": "mood", "value": "joyful"}, \
            "mysterious": {"category": "mood", "value": "mysterious"}, \
            "enigmatic": {"category": "mood", "value": "mysterious"}, \
            "eerie": {"category": "mood", "value": "mysterious"}, \
        }
        # Initialize with minimal NLP tools
        self._initialize_minimal_nlp_components()
    
    def _initialize_embeddings(self):
        """Initialize simplified word embeddings for semantic matching.
        
        In a production environment, this would load from pre-trained embeddings.For this implementation, we use a simplified approach with random vectors.
        """
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        
        # Create simplified embeddings for key concepts (300-dim is standard)
        dimension = 300
        
        # Generate embeddings for scene types
        for scene_type in self.ontology.SCENE_TYPES:
            self.embeddings[scene_type] = np.random.randn(dimension)
            for term in self.ontology.SCENE_TYPES[scene_type]:
                # Make related terms have similar embeddings by adding small noise
                self.embeddings[term] = self.embeddings[scene_type] + np.random.randn(dimension) * 0.1
        
        # Generate embeddings for time of day
        for time in self.ontology.TIME_OF_DAY:
            self.embeddings[time] = np.random.randn(dimension)
            for term in self.ontology.TIME_OF_DAY[time]:
                self.embeddings[term] = self.embeddings[time] + np.random.randn(dimension) * 0.1
        
        # Generate embeddings for atmospheric conditions
        for condition in self.ontology.ATMOSPHERIC:
            self.embeddings[condition] = np.random.randn(dimension)
            for term in self.ontology.ATMOSPHERIC[condition]:
                self.embeddings[condition] = self.embeddings[condition] + np.random.randn(dimension) * 0.1
        
        # Generate embeddings for mood
        for mood in self.ontology.MOOD:
            self.embeddings[mood] = np.random.randn(dimension)
            for term in self.ontology.MOOD[mood]:
                self.embeddings[term] = self.embeddings[mood] + np.random.randn(dimension) * 0.1
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize input text for semantic parsing.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of normalized tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Replace punctuation with spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into tokens and remove empty strings
        tokens = [token.strip() for token in text.split() if token.strip()]
        
        return tokens
    
    def _find_best_match(self, token: str, category: str) -> Tuple[str, float]:
        """Find the best semantic match for a token within a category.
        
        Args:
            token: Input token to match
            category: Category to search within (scene_type, time_of_day, etc.)
            
        Returns:
            Tuple of (best_match, similarity_score)
        """
        if not NUMPY_AVAILABLE:
            # Fallback to exact matching if NumPy isn't available
            for key, info in self.lexicon.items():
                if info["category"] == category and (key == token or key in token or token in key):
                    # Simple string matching with confidence based on substring match
                    if key == token:
                        return info["value"], 1.0
                    elif key in token:
                        return info["value"], 0.8
                    elif token in key:
                        return info["value"], 0.6
            return "", 0.0
            
        # If the token is in our embeddings
        if token in self.embeddings:
            token_vector = self.embeddings[token]
            best_match = ""
            best_score = -1.0
            
            # Find target values for the specified category
            target_values = set()
            for _, info in self.lexicon.items():
                if info["category"] == category:
                    target_values.add(info["value"])
            
            # Calculate similarity with all possible values
            for value in target_values:
                if value in self.embeddings:
                    value_vector = self.embeddings[value]
                    similarity = 1.0 - cosine(token_vector, value_vector)
                    if similarity > best_score:
                        best_score = similarity
                        best_match = value
            
            # Threshold for accepting a match
            if best_score > 0.4:
                return best_match, best_score
        
        return "", 0.0
    
    def parse(self, prompt: str) -> SceneVector:
        """Parse text prompt into a structured scene vector.
        
        This is the main entry point for semantic parsing. It transforms a free-form
        text description into a structured SceneVector object used for SVG generation.
        
        Args:
            prompt: Text prompt describing the desired scene
            
        Returns:
            SceneVector with parsed semantic elements
        """
        # Initialize scene vector with defaults
        scene_vector = SceneVector()
        confidence_scores = {}
        
        # Get tokens from input text
        tokens = self._tokenize(prompt)
        
        # Direct lexicon matching
        for token in tokens:
            if token in self.lexicon:
                info = self.lexicon[token]
                category, value = info["category"], info["value"]
                
                # Update the scene vector if confidence is higher
                current_confidence = confidence_scores.get(category, 0.0)
                if current_confidence < 1.0:  # Direct match has highest confidence
                    setattr(scene_vector, category, value)
                    confidence_scores[category] = 1.0
        
        # Semantic matching for unmatched attributes
        categories = ["scene_type", "time_of_day", "weather", "mood"]
        for category in categories:
            if category not in confidence_scores:
                # Try to find matches for each token
                for token in tokens:
                    match, score = self._find_best_match(token, category)
                    if match and score > confidence_scores.get(category, 0.0):
                        setattr(scene_vector, category, match)
                        confidence_scores[category] = score
        
        # Use SemanticAnalyzer for more detailed analysis
        semantic_features = self.semantic_analyzer.analyze_prompt(prompt)
        
        # Extract color information
        scene_vector.color_palette = self._extract_colors(semantic_features)
        
        # Extract style descriptors
        scene_vector.style_descriptors = self._extract_style_descriptors(semantic_features)
        
        # Handle architecture-specific details for cityscapes
        if scene_vector.scene_type == "cityscape":
            scene_vector.architecture = self._extract_architectural_features(semantic_features)
        
        # Extract focal point, background, midground, foreground
        scene_composition = self._extract_composition(semantic_features)
        scene_vector.focal_point = scene_composition.get("focal_point")
        scene_vector.background = scene_composition.get("background")
        scene_vector.midground = scene_composition.get("midground")
        scene_vector.foreground = scene_composition.get("foreground")
        
        # Calculate detail level based on descriptors
        scene_vector.detail_level = self._calculate_detail_level(prompt, semantic_features)
        
        # Store confidence scores
        scene_vector.confidence_scores = confidence_scores
        
        return scene_vector
    
    def _extract_colors(self, semantic_features: Dict[str, Any]) -> List[Tuple[int, int, int]]:
        """Extract color palette from semantic features.
        
        Args:
            semantic_features: Semantic features dictionary from analyzer
            
        Returns:
            List of RGB color tuples
        """
        color_palette = []
        
        # Extract explicitly mentioned colors
        if "colors" in semantic_features:
            for color_info in semantic_features["colors"]:
                if isinstance(color_info, tuple) and len(color_info) == 2:
                    color_name, _ = color_info
                    rgb = self._color_name_to_rgb(color_name)
                    if rgb:
                        color_palette.append(rgb)
        
        # Generate palette based on scene type, time of day, and mood if no explicit colors
        if not color_palette:
            # Default palette based on scene attributes
            scene_type = getattr(self, "scene_type", "landscape")
            time_of_day = getattr(self, "time_of_day", "midday")
            mood = getattr(self, "mood", "neutral")
            
            # Generate a harmonious color palette (simplified implementation)
            base_colors = self._generate_base_palette(scene_type, time_of_day, mood)
            color_palette.extend(base_colors)
        
        return color_palette
    
    def _color_name_to_rgb(self, color_name: str) -> Optional[Tuple[int, int, int]]:
        """Convert color name to RGB tuple.
        
        Args:
            color_name: Name of the color (e.g., 'red', 'blue', etc.)
            
        Returns:
            RGB tuple or None if color name is not recognized
        """
        # Basic color mapping (could be expanded)
        color_map = { \
            "red": (255, 0, 0),
            "green": (0, 128, 0), \
            "blue": (0, 0, 255), \
            "yellow": (255, 255, 0), \
            "orange": (255, 165, 0), \
            "purple": (128, 0, 128), \
            "pink": (255, 192, 203), \
            "brown": (165, 42, 42), \
            "black": (0, 0, 0), \
            "white": (255, 255, 255), \
            "gray": (128, 128, 128), \
            "cyan": (0, 255, 255), \
            "magenta": (255, 0, 255), \
            "gold": (255, 215, 0), \
            "silver": (192, 192, 192), \
            "navy": (0, 0, 128), \
            "teal": (0, 128, 128), \
        }
        
        # Check for exact match
        if color_name.lower() in color_map:
            return color_map[color_name.lower()]
        
        # Check for partial match
        for name, rgb in color_map.items():
            if name in color_name.lower():
                return rgb
        
        return None
    
    def _generate_base_palette(self, scene_type: str, time_of_day: str, mood: str) -> List[Tuple[int, int, int]]:
        """Generate a base color palette based on scene attributes.
        
        Args:
            scene_type: Type of scene (landscape, seascape, etc.)
            time_of_day: Time of day (dawn, midday, etc.)
            mood: Mood of the scene (tranquil, dramatic, etc.)
            
        Returns:
            List of RGB color tuples for the palette
        """
        palette = []
        
        # Scene type base colors
        if scene_type == "landscape":
            palette.extend([(34, 139, 34), (0, 100, 0), (222, 184, 135)])  # Forest green, dark green, tan
        elif scene_type == "seascape":
            palette.extend([(0, 119, 190), (0, 87, 132), (238, 214, 175)])  # Ocean blue, deep blue, sand
        elif scene_type == "cityscape":
            palette.extend([(169, 169, 169), (105, 105, 105), (220, 220, 220)])  # Concrete colors
        elif scene_type == "interior":
            palette.extend([(245, 245, 245), (210, 180, 140), (188, 143, 143)])  # Walls, wood, furniture
        
        # Time of day influence
        if time_of_day == "dawn" or time_of_day == "sunrise":
            palette.extend([(255, 160, 122), (255, 127, 80)])  # Soft orange, coral
        elif time_of_day == "midday":
            palette.extend([(135, 206, 235), (255, 255, 255)])  # Sky blue, white
        elif time_of_day == "sunset" or time_of_day == "golden_hour":
            palette.extend([(255, 69, 0), (255, 140, 0)])  # Red-orange, dark orange
        elif time_of_day == "night":
            palette.extend([(25, 25, 112), (0, 0, 128)])  # Dark blue, navy
        
        # Mood influence
        if mood == "tranquil":
            palette.extend([(173, 216, 230), (240, 248, 255)])  # Light blue, alice blue
        elif mood == "dramatic":
            palette.extend([(139, 0, 0), (128, 0, 0)])  # Dark red, maroon
        elif mood == "melancholic":
            palette.extend([(169, 169, 169), (119, 136, 153)])  # Dark gray, slate gray
        elif mood == "joyful":
            palette.extend([(255, 215, 0), (255, 165, 0)])  # Gold, orange
        elif mood == "mysterious":
            palette.extend([(75, 0, 130), (138, 43, 226)])  # Indigo, blue violet
        
        return palette
    
    def _extract_style_descriptors(self, semantic_features: Dict[str, Any]) -> List[str]:
        """Extract style descriptors from semantic features.
        
        Args:
            semantic_features: Semantic features dictionary from analyzer
            
        Returns:
            List of style descriptor strings
        """
        descriptors = []
        
        # Extract artistic style hints
        if "artistic_style" in semantic_features:
            for style, confidence in semantic_features["artistic_style"]:
                if confidence > 0.5:  # Only include confident matches
                    descriptors.append(style)
        
        # Extract material descriptors
        if "materials" in semantic_features:
            for material, confidence in semantic_features["materials"]:
                if confidence > 0.6:  # Higher threshold for materials
                    descriptors.append(f"material:{material}")
        
        return descriptors
    
    def _extract_architectural_features(self, semantic_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract architectural features for cityscape scenes.
        
        Args:
            semantic_features: Semantic features dictionary from analyzer
            
        Returns:
            Dictionary of architectural features
        """
        architecture = { \
            "style": "modern",  # Default style
            "features": [], \
            "materials": ["concrete", "glass"], \
            "detail_level": 0.5
        }
        
        # Extract architectural style if available
        if "architectural" in semantic_features:
            for style, confidence in semantic_features["architectural"]:
                if confidence > 0.5:  # Only include confident matches
                    architecture["style"] = style
                    break
        
        # Get style features using the ArchitecturalStyleClassifier
        arch_style = self.semantic_analyzer.arch_classifier.classify_architectural_style( \
            " ".join(architecture["style"])
        )
        
        # Update architecture dictionary with classifier results
        if arch_style:
            if "features" in arch_style:
                architecture["features"] = arch_style["features"]
            if "materials" in arch_style:
                architecture["materials"] = arch_style["materials"]
            if "ornate_level" in arch_style:
                architecture["detail_level"] = arch_style["ornate_level"]
        
        return architecture
    
    def _extract_composition(self, semantic_features: Dict[str, Any]) -> Dict[str, Any]:
        """Extract scene composition elements from semantic features.
        
        Args:
            semantic_features: Semantic features dictionary from analyzer
            
        Returns:
            Dictionary with focal_point, background, midground, foreground
        """
        composition = { \
            "focal_point": None,
            "background": {}, \
            "midground": {}, \
            "foreground": {}
        }
        
        # Extract focal point
        if "focal_elements" in semantic_features:
            for element, confidence in semantic_features["focal_elements"]:
                if confidence > 0.7:  # Only use high-confidence focal elements
                    composition["focal_point"] = { \
                        "element": element,
                        "confidence": confidence
                    }
                    break
        
        # Simple composition based on scene type
        scene_type = getattr(self, "scene_type", "landscape")
        
        if scene_type == "landscape":
            composition["background"] = {"sky": True, "clouds": True}
            composition["midground"] = {"mountains": True, "hills": True}
            composition["foreground"] = {"ground": True, "vegetation": True}
        elif scene_type == "seascape":
            composition["background"] = {"sky": True, "horizon": True}
            composition["midground"] = {"sea": True, "waves": True}
            composition["foreground"] = {"beach": True, "shore": True}
        elif scene_type == "cityscape":
            composition["background"] = {"sky": True, "clouds": True}
            composition["midground"] = {"buildings": True, "skyscrapers": True}
            composition["foreground"] = {"street": True, "ground": True}
        elif scene_type == "interior":
            composition["background"] = {"wall": True, "window": True}
            composition["midground"] = {"furniture": True}
            composition["foreground"] = {"floor": True, "objects": True}
        
        return composition
    
    def _calculate_detail_level(self, prompt: str, semantic_features: Dict[str, Any]) -> float:
        """Calculate detail level based on prompt and semantic features.
        
        Args:
            prompt: Original text prompt
            semantic_features: Semantic features dictionary from analyzer
            
        Returns:
            Detail level as float between 0.0 and 1.0
        """
        # Start with a moderate detail level
        detail_level = 0.5
        
        # Check for explicit detail level indicators in the prompt
        detail_indicators = { \
            "detailed": 0.7,
            "highly detailed": 0.8, \
            "intricate": 0.9, \
            "complex": 0.85, \
            "simple": 0.3, \
            "minimalist": 0.2, \
            "basic": 0.25, \
            "sketch": 0.4, \
            "outline": 0.2
        }
        
        prompt_lower = prompt.lower()
        for indicator, level in detail_indicators.items():
            if indicator in prompt_lower:
                detail_level = level
                break
        
        # Adjust based on scene type and mood
        if "scene_type" in semantic_features and semantic_features["scene_type"].value == "cityscape":
            detail_level += 0.1  # Cities tend to need more detail
        
        if "mood" in semantic_features:
            mood_adjustments = { \
                "tranquil": -0.1,  # Simpler for tranquil scenes
                "dramatic": 0.1,   # More detailed for dramatic scenes
                "mysterious": 0.05  # Slightly more detailed for mysterious scenes
            }
            mood_value = semantic_features["mood"].value
            if mood_value in mood_adjustments:
                detail_level += mood_adjustments[mood_value]
        
        # Ensure detail level stays within bounds
        return max(0.1, min(0.9, detail_level))


# ============================================================================ #
#                2. CONCEPTUAL MAPPING TO VISUAL CONSTRUCTS                    #
# ============================================================================ #
# This stage converts semantic features into a structured scene graph:

class SceneGraphBuilder:
    """Builds a complete scene graph from parsed scene vectors.
    
    This component converts the abstract semantic representations into a
    spatial directed acyclic graph (DAG) where each node has concrete positions, \
    sizes, and visual properties. It resolves spatial relationships, handles
    occlusion, and ensures proper layering of elements.
    
    The builder maintains a library of template shapes for various object types
    and uses constraint satisfaction to position objects according to their
    semantic relationships. It implements a two-pass layout algorithm:
    1. First pass: Initial placement of objects based on templates and basic rules
    2. Second pass: Constraint-based optimization to ensure all spatial relationships are satisfied
    """
    
    def __init__(self, width: float = 800, height: float = 600):
        """Initialize the graph builder with canvas dimensions.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
        """
        self.width = width
        self.height = height
        self.baseline_y = height * 0.75  # Default ground level for objects
        
        # Track objects and constraints for the constraint solver
        self.object_registry = {}
        self.constraint_registry = []
        self.compound_objects = {}
        
        # Template library for standard objects
        self.shape_templates = {
            # Natural elements
            "mountain": {"shape": "path", "width_ratio": 0.3, "height_ratio": 0.5}, \
            "tree": {"shape": "path", "width_ratio": 0.1, "height_ratio": 0.3}, \
            "sun": {"shape": "circle", "width_ratio": 0.15, "height_ratio": 0.15}, \
            "cloud": {"shape": "path", "width_ratio": 0.2, "height_ratio": 0.1}, \
            "river": {"shape": "path", "width_ratio": 0.8, "height_ratio": 0.1}, \
            "lake": {"shape": "path", "width_ratio": 0.4, "height_ratio": 0.2}, \
            "ocean": {"shape": "path", "width_ratio": 1.0, "height_ratio": 0.3},
            
            # Built environment
            "building": {"shape": "rect", "width_ratio": 0.15, "height_ratio": 0.4}, \
            "house": {"shape": "path", "width_ratio": 0.15, "height_ratio": 0.25}, \
            "road": {"shape": "path", "width_ratio": 0.8, "height_ratio": 0.05}, \
            "bridge": {"shape": "path", "width_ratio": 0.3, "height_ratio": 0.15},
            
            # Default for unknown objects
            "default": {"shape": "rect", "width_ratio": 0.1, "height_ratio": 0.1}
        }
        
        # Material mappings for different object types
        self.material_mappings = { \
            "mountain": "stone",
            "tree": "wood", \
            "building": "concrete", \
            "house": "brick", \
            "bridge": "metal"
        }
        
        # Default colors for objects without specified colors
        self.default_colors = { \
            "sky": "#87CEEB",
            "ground": "#8B4513", \
            "mountain": "#A9A9A9", \
            "tree": "#228B22", \
            "sun": "#FFD700", \
            "cloud": "#F5F5F5", \
            "river": "#1E90FF", \
            "lake": "#4682B4", \
            "ocean": "#0000CD", \
            "building": "#A0A0A0", \
            "house": "#CD853F", \
            "road": "#696969", \
            "bridge": "#708090"
        }
    
    def _normalize_object_type(self, obj_type: str, obj_data: Dict[str, Any]) -> str:
        """Normalize object types using morphosyntactic rules.
        
        Args:
            obj_type: Original object type string
            obj_data: Complete object data dictionary
            
        Returns:
            Normalized object type
        """
        # Mapping of plural forms to singular forms
        plural_mapping = { \
            'trees': 'tree',
            'houses': 'house', \
            'buildings': 'building', \
            'mountains': 'mountain', \
            'clouds': 'cloud', \
            'people': 'person', \
            'windows': 'window', \
            'doors': 'door', \
            'flowers': 'flower', \
            'cars': 'car', \
            'boats': 'boat', \
            'stars': 'star', \
            'birds': 'bird'
        }
        
        # Mapping of alternate forms to canonical forms
        alternate_mapping = { \
            'skyscraper': 'building',
            'tower': 'building', \
            'cottage': 'house', \
            'cabin': 'house', \
            'mansion': 'house', \
            'hill': 'mountain', \
            'stream': 'river', \
            'pond': 'lake', \
            'automobile': 'car', \
            'vehicle': 'car', \
            'vessel': 'boat', \
            'ship': 'boat', \
            'plant': 'tree', \
            'shrub': 'bush', \
            'pathway': 'path', \
            'trail': 'path', \
            'walkway': 'path'
        }
        
        # Check for plural form and convert to singular if found
        # Also update the quantity if it's not already set
        if obj_type in plural_mapping:
            normalized = plural_mapping[obj_type]
            if 'quantity' not in obj_data:
                obj_data['quantity'] = 2  # Default to 2 for plurals
            return normalized
        
        # Check for alternate forms and convert to canonical form
        if obj_type in alternate_mapping:
            return alternate_mapping[obj_type]
        
        return obj_type
    
    def build_scene_graph(self, scene_vector: Dict[str, Any]) -> SceneGraph:
        """Builds a complete scene graph from a parsed scene vector using a two-pass approach.
        
        1. First pass: Create all objects and place them in initial positions
        2. Second pass: Apply constraint-based layout to satisfy spatial relationships
        3. Third pass: Identify and enhance compound objects
        
        Args:
            scene_vector: Dictionary containing objects, modifiers, layout, and style
            
        Returns:
            Complete scene graph with positioned nodes
        """
        # Reset the constraint and object registries for this new graph
        self.object_registry: Dict[str, SceneNode] = {}
        self.constraint_registry: List[Dict[str, Any]] = []
        self.compound_objects: Dict[str, Dict[str, Any]] = {}
        
        # Create an empty scene graph
        scene_graph = SceneGraph(self.width, self.height)
        
        # Add a background/sky node
        sky_node = scene_graph.create_node( \
            node_type="sky",
            position=(0, 0, -100),  # Behind everything
            size=(self.width, self.height, 0), \
            properties={"color": self.default_colors.get("sky", "#87CEEB")}
        )
        self.object_registry[sky_node.node_id] = sky_node
        
        # Add a ground node
        ground_node = scene_graph.create_node( \
            node_type="ground",
            position=(0, self.baseline_y, -50),  # Behind most objects but in front of sky
            size=(self.width, self.height - self.baseline_y, 0), \
            properties={"color": self.default_colors.get("ground", "#8B4513")}
        )
        self.object_registry[ground_node.node_id] = ground_node
        
        # First pass: Create all objects with initial positions
        self._add_objects_to_graph(scene_graph, scene_vector)
        
        # Second pass: Apply constraint-based layout to satisfy spatial relationships
        self._apply_spatial_relationships(scene_graph, scene_vector)
        
        # Apply global style information
        self._apply_style(scene_graph, scene_vector)
        
        # Add metadata about compound objects to the scene graph root
        if self.compound_objects:
            scene_graph.root.properties["compound_objects"] = len(self.compound_objects)
            scene_graph.root.properties["compound_object_ids"] = list(self.compound_objects.keys())
        
        return scene_graph
    
    def _detect_compound_objects(self, scene_graph: SceneGraph) -> None:
        """Detect and create compound objects based on spatial relationships.
        
        This analyzes the scene graph after constraint solving to identify
        compound objects based on spatial relationships and semantic patterns.
        
        Args:
            scene_graph: The scene graph to analyze for compound objects
        """
        # Define types for compound patterns and objects
        class CompoundPattern(TypedDict):
            container: str
            components: List[str]
            relation: str
        
        class CompoundObject(TypedDict):
            container: str
            components: List[str]
            type: str
            
        # Look for common compound object patterns
        compound_patterns: List[CompoundPattern] = [
            # Houses with elements
            {'container': 'house', 'components': ['door', 'window', 'roof'], 'relation': 'in'}, \
            {'container': 'house', 'components': ['chimney'], 'relation': 'on'},
            
            # Buildings with elements
            {'container': 'building', 'components': ['door', 'window'], 'relation': 'in'},
            
            # Trees with elements
            {'container': 'tree', 'components': ['fruit', 'leaf', 'branch'], 'relation': 'in'},
            
            # Vehicles with elements
            {'container': 'car', 'components': ['wheel', 'window', 'door'], 'relation': 'in'}, \
            {'container': 'boat', 'components': ['sail', 'mast'], 'relation': 'on'}, \
        ]
        
        # For each pattern, check if it exists in our constraints
        for pattern in compound_patterns:
            container_type: str = pattern['container']
            component_types: List[str] = pattern['components']
            relation_type: str = pattern['relation']
            
            # Find container nodes of this type
            container_nodes: List[SceneNode] = scene_graph.find_nodes_by_type(container_type)
            
            for container_node in container_nodes:
                compound_components: List[str] = []
                
                # Check each component type
                for component_type in component_types:
                    component_nodes: List[SceneNode] = scene_graph.find_nodes_by_type(component_type)
                    
                    # Check if any component has the right relation with the container
                    for component_node in component_nodes:
                        # Check if there's a constraint between these nodes
                        for constraint in self.constraint_registry:
                            if (constraint['type'] == relation_type and \
                                constraint['subject'] == component_node.node_id and
                                constraint['object'] == container_node.node_id and
                                constraint.get('satisfied', False)):
                                compound_components.append(component_node.node_id)
                                break
                                
                # If we found components, register this as a compound object
                if compound_components:
                    # Create compound object entry with properly typed dictionary
                    compound_obj: CompoundObject = { \
                        'container': container_node.node_id,
                        'components': compound_components, \
                        'type': container_type
                    }
                    self.compound_objects[container_node.node_id] = compound_obj
                    
                    # Update the container node to indicate it's a compound object
                    container_node.properties['is_compound'] = True
                    container_node.properties['compound_components'] = len(compound_components)
    
    def _add_objects_to_graph(self, scene_graph: SceneGraph, scene_vector: Dict[str, Any]) -> None:
        """Add objects from the scene vector to the graph using morphosyntactic normalization.
        
        This first pass creates all objects with initial positions based on templates and
        basic rules. The constraint solver will adjust positions in a second pass.
        
        Args:
            scene_graph: The scene graph to add objects to
            scene_vector: Dictionary containing objects, modifiers, layout, and style
        """
        if "objects" not in scene_vector:
            return
            
        # Initial x-coordinate for placing objects
        current_x: float = self.width * 0.1
        
        # First pass - create all objects with initial positions
        for i, obj in enumerate(scene_vector["objects"]):
            if "type" not in obj:
                continue
                
            obj_type: str = obj["type"]
            
            # Normalize the object type using morphosyntactic rules
            normalized_type: str = self._normalize_object_type(obj_type, obj)
            
            # Get template for this object type or use default
            template: Dict[str, Any] = self.shape_templates.get( \
                normalized_type,
                self.shape_templates.get(obj_type, self.shape_templates["default"])
            )
            
            # Calculate object dimensions
            obj_width: float = self.width * template["width_ratio"]
            obj_height: float = self.height * template["height_ratio"]
            
            # Calculate position based on object type and depth categorization
            x: float = current_x
            
            # Categorize objects by visual layer/depth for initial placement
            layer_categories: Dict[str, List[str]] = { \
                'sky': ['sun', 'moon', 'star', 'cloud', 'rainbow'],
                'background': ['mountain', 'volcano', 'hill', 'forest', 'skyline'], \
                'midground': ['tree', 'river', 'lake', 'ocean', 'field', 'building', 'house', 'structure'], \
                'foreground': ['person', 'animal', 'car', 'boat', 'bench', 'flower', 'bush', 'path'], \
                'ground': ['road', 'sidewalk', 'grass']
            }
            
            # Determine which layer this object belongs to
            object_layer: str = 'midground'  # Default
            for layer, types in layer_categories.items():
                if normalized_type in types or obj_type in types:
                    object_layer = layer
                    break
            
            # Set initial position based on layer
            y: float = 0.0  # Initialize with default values
            z: float = 0.0  # that will be overridden
            
            if object_layer == 'sky':
                y = self.height * 0.3
                z = -40.0
            elif object_layer == 'background':
                y = self.baseline_y - obj_height
                z = -30.0
            elif object_layer == 'ground':
                y = self.baseline_y - obj_height * 0.5
                z = -10.0
            elif object_layer == 'foreground':
                y = self.baseline_y - obj_height
                z = 10.0
            else:  # midground
                y = self.baseline_y - obj_height
                z = 0.0
            
            # Check for quantity
            quantity: int = int(obj.get("quantity", 1))
            
            # Create the node in the scene graph
            properties: SceneNodeProperties = { \
                "shape": template["shape"],
                "color": self.default_colors.get(normalized_type, self.default_colors.get(obj_type, "#808080")), \
                "material": self.material_mappings.get(normalized_type, self.material_mappings.get(obj_type, None))
            }
            
            # If there are modifiers for this object, apply them
            if "modifiers" in scene_vector:
                # Check both normalized and original type
                if normalized_type in scene_vector["modifiers"]:
                    self._apply_modifiers(properties, scene_vector["modifiers"][normalized_type])
                elif obj_type in scene_vector["modifiers"]:
                    self._apply_modifiers(properties, scene_vector["modifiers"][obj_type])
            
            # Create node (or multiple nodes if quantity > 1)
            for q in range(quantity):
                # Adjust position slightly for multiple instances
                q_offset: float = q * obj_width * 1.2 if quantity > 1 else 0.0
                
                node: SceneNode = scene_graph.create_node( \
                    normalized_type,
                    position=(x + q_offset, y, z), \
                    size=(obj_width, obj_height, 0.0), \
                    properties=properties.copy(),  # Copy to avoid shared references
                    source=obj.get("original_type", obj_type), \
                    confidence=float(scene_vector.get("confidence_scores", {}).get(f"object_{i}", 1.0))
                )
                
                # Register this node for the constraint solver
                self.object_registry[node.node_id] = node
            
            # Update x-coordinate for next object
            current_x += obj_width * 1.5  # Add spacing between objects
    
    def _register_constraint(self, constraint_type: Literal["above", "below", "beside", "near", "on", "in"], \
                            subject_id: str, object_id: str,
                            parameters: Optional[Dict[str, Union[float, int, str]]] = None) -> None:
        """Register a spatial constraint between two objects for the constraint solver.
        
        Args:
            constraint_type: Type of constraint (e.g., 'above', 'beside', etc.)
            subject_id: ID of the subject node
            object_id: ID of the object node
            parameters: Additional parameters for the constraint
        """
        if parameters is None:
            parameters = {}
            
        constraint: SpatialConstraint = { \
            'type': constraint_type,
            'subject': subject_id, \
            'object': object_id, \
            'parameters': parameters, \
            'satisfied': False
        }
        
        self.constraint_registry.append(constraint)
    
    def _apply_spatial_relationships(self, scene_graph: SceneGraph, scene_vector: Dict[str, Any]) -> None:
        """Apply spatial relationships using the constraint-based layout system.
        
        This is a three-pass process:
        1. Register all constraints
        2. Solve the constraint system to find optimal positions
        3. Detect and create compound objects based on satisfied constraints
        
        Args:
            scene_graph: The scene graph to apply constraints to
            scene_vector: Dictionary containing layout information
        """
        if "layout" not in scene_vector:
            return
        
        # First pass: Register all constraints from the layout
        for relation in scene_vector["layout"]:
            subject_type = relation.get("subject", "")
            object_type = relation.get("object", "")
            relation_type = relation.get("relation", "")
            
            # Skip invalid relations
            if not subject_type or not object_type or not relation_type:
                continue
                
            # Find nodes matching these types
            subject_nodes = scene_graph.find_nodes_by_type(subject_type)
            object_nodes = scene_graph.find_nodes_by_type(object_type)
            
            if not subject_nodes or not object_nodes:
                continue
                
            # Register this constraint in our system
            subject = subject_nodes[0]
            obj = object_nodes[0]
            
            # Register nodes in object registry if not already there
            self.object_registry[subject.node_id] = subject
            self.object_registry[obj.node_id] = obj
            
            # Create appropriate parameters based on relation type
            parameters: Dict[str, Union[float, int, str]] = {}
            
            if relation_type in ["above", "below", "on"]:
                parameters['vertical_margin'] = 10.0
            elif relation_type in ["beside", "near"]:
                parameters['horizontal_margin'] = 20.0
            elif relation_type == "in":
                parameters['scale_factor'] = 0.8
            
            # Only register if relation type is one we can handle
            if relation_type in ["above", "below", "beside", "near", "on", "in"]:
                self._register_constraint( \
                    cast(Literal["above", "below", "beside", "near", "on", "in"], relation_type),
                    subject.node_id, \
                    obj.node_id, \
                    parameters
                )
        
        # Second pass: Solve the constraint system
        self._solve_constraints(scene_graph)
        
        # Third pass: Detect and create compound objects
        self._detect_compound_objects(scene_graph)
    
    def _solve_constraints(self, scene_graph: SceneGraph) -> None:
        """Solve the constraint system to find optimal positions for all objects.
        
        This uses an iterative approach to satisfy all constraints, prioritizing
        more important constraints first (e.g., containment before adjacency).
        
        Args:
            scene_graph: The scene graph containing objects to position
        """
        # Sort constraints by priority
        priority_order: List[str] = ["in", "on", "above", "below", "beside", "near"]
        sorted_constraints: List[Dict[str, Any]] = sorted( \
            self.constraint_registry,
            key=lambda c: priority_order.index(c['type']) if c['type'] in priority_order else 999
        )
        
        # Solve constraints in priority order
        for constraint in sorted_constraints:
            subject_id: str = constraint['subject']
            object_id: str = constraint['object']
            relation_type: str = constraint['type']
            parameters: Dict[str, Any] = constraint['parameters']
            
            if subject_id not in self.object_registry or object_id not in self.object_registry:
                continue
                
            subject = self.object_registry[subject_id]
            obj = self.object_registry[object_id]
            
            # Apply the constraint based on its type
            if relation_type == "above":
                # Position subject above object
                margin: float = float(parameters.get('vertical_margin', 10.0))
                obj_x, obj_y, obj_z = obj.position
                _, obj_height, _ = obj.size
                subject.position = (obj_x, obj_y - obj_height - margin, subject.position[2])
                
            elif relation_type == "below":
                # Position subject below object
                margin: float = float(parameters.get('vertical_margin', 10.0))
                obj_x, obj_y, obj_z = obj.position
                _, subject_height, _ = subject.size
                subject.position = (obj_x, obj_y + obj_height + margin, subject.position[2])
                
            elif relation_type == "beside" or relation_type == "near":
                # Position subject beside object
                margin: float = float(parameters.get('horizontal_margin', 20.0))
                obj_x, obj_y, obj_z = obj.position
                obj_width, _, _ = obj.size
                subject.position = (obj_x + obj_width + margin, subject.position[1], subject.position[2])
                
            elif relation_type == "on":
                # Position subject on top of object with adjusted z-index
                obj_x, obj_y, obj_z = obj.position
                _, obj_height, _ = obj.size
                _, subject_height, _ = subject.size
                subject.position = (obj_x, obj_y - subject_height, obj_z - 1)
                
            elif relation_type == "in":
                # Position subject inside object with scaling if needed
                scale_factor: float = float(parameters.get('scale_factor', 0.8))
                obj_x, obj_y, obj_z = obj.position
                obj_width, obj_height, _ = obj.size
                subject_width, subject_height, _ = subject.size
                
                # Scale down subject if needed to fit inside object
                if subject_width > obj_width * scale_factor or subject_height > obj_height * scale_factor:
                    applied_scale: float = min( \
                        obj_width * scale_factor / subject_width,
                        obj_height * scale_factor / subject_height
                    )
                    subject.size = ( \
                        subject_width * applied_scale,
                        subject_height * applied_scale, \
                        0.0
                    )
                    subject_width, subject_height, _ = subject.size
                
                # Center subject within object
                subject.position = ( \
                    obj_x + (obj_width - subject_width) / 2,
                    obj_y + (obj_height - subject_height) / 2, \
                    obj_z - 1  # -1 to be in front of container
                )
            
            # Mark constraint as satisfied
            constraint['satisfied'] = True
    
    def _apply_modifiers(self, properties: SceneNodeProperties, modifiers: List[str]) -> None:
        """Apply modifiers to object properties.
        
        Args:
            properties: Dictionary of properties to modify
            modifiers: List of modifier strings to apply
        """
        for modifier in modifiers:
            # Handle color modifiers
            if modifier in ["red", "blue", "green", "yellow", "purple", "orange", "white", "black"]:
                color_map: Dict[str, str] = { \
                    "red": "#FF0000",
                    "blue": "#0000FF", \
                    "green": "#00FF00", \
                    "yellow": "#FFFF00", \
                    "purple": "#800080", \
                    "orange": "#FFA500", \
                    "white": "#FFFFFF", \
                    "black": "#000000"
                }
                properties["color"] = color_map[modifier]
            
            # Handle size modifiers
            elif modifier in ["big", "large", "huge"]:
                properties["scale"] = 1.5
            elif modifier in ["small", "tiny", "little"]:
                properties["scale"] = 0.7
            
            # Handle material modifiers
            elif modifier in ["wooden", "wood"]:
                properties["material"] = "wood"
            elif modifier in ["stone", "rocky"]:
                properties["material"] = "stone"
            elif modifier in ["metal", "metallic"]:
                properties["material"] = "metal"
            elif modifier in ["glass", "transparent"]:
                properties["material"] = "glass"
            elif modifier in ["brick", "bricks"]:
                properties["material"] = "brick"
            elif modifier in ["concrete"]:
                properties["material"] = "concrete"
            elif modifier in ["marble", "marbled"]:
                properties["material"] = "marble"
            
            # Handle other visual modifiers
            elif modifier in ["bright"]:
                properties["brightness"] = 1.3
            elif modifier in ["dark"]:
                properties["brightness"] = 0.7
            elif modifier in ["old", "ancient"]:
                properties["weathered"] = 0.7
            elif modifier in ["new", "modern"]:
                properties["weathered"] = 0.0
    
    def _apply_style(self, scene_graph: SceneGraph, scene_vector: Dict[str, Any]) -> None:
        """Apply global style information to the scene.
        
        Args:
            scene_graph: The scene graph to apply styles to
            scene_vector: Dictionary containing style information
        """
        if "style" not in scene_vector:
            return
            
        for style in scene_vector.get("style", []):
            # Apply overall scene style modifiers
            if style == "minimalist":
                scene_graph.root.properties["detail_level"] = 0.3
                scene_graph.root.properties["color_scheme"] = "monochrome"
                
            elif style == "detailed" or style == "ornate":
                scene_graph.root.properties["detail_level"] = 0.9
                
            elif style == "abstract":
                scene_graph.root.properties["abstraction_level"] = 0.8
                scene_graph.root.properties["color_scheme"] = "vivid"
                
            elif style == "realistic":
                scene_graph.root.properties["abstraction_level"] = 0.2
                scene_graph.root.properties["detail_level"] = 0.8
                
            elif style in ["colorful", "vivid"]:
                scene_graph.root.properties["color_saturation"] = 1.4
                
            elif style in ["monochrome", "grayscale"]:
                scene_graph.root.properties["color_saturation"] = 0.0
# Type definitions for constraint-based layout system
class SceneNodeProperties(TypedDict, total=False):
    """Properties that can be assigned to a scene node."""
    shape: str
    color: str
    material: Optional[str]
    brightness: float
    scale: float
    weathered: float
    detail_level: float
    color_scheme: str
    abstraction_level: float
    color_saturation: float
    is_compound: bool
    compound_components: int

class Position3D(NamedTuple):
    """3D position with x, y, z coordinates."""
    x: float
    y: float
    z: float

class Size3D(NamedTuple):
    """3D size with width, height, depth dimensions."""
    width: float
    height: float
    depth: float

# Constraint System Types
class SpatialConstraintParameters(TypedDict, total=False):
    """Parameters for spatial constraints."""
    vertical_margin: float
    horizontal_margin: float
    scale_factor: float

class SpatialConstraint(TypedDict):
    """Type-safe constraint definition."""
    type: Literal["above", "below", "beside", "near", "on", "in"]
    subject: str  # Node ID
    object: str   # Node ID
    parameters: SpatialConstraintParameters
    satisfied: bool

class CompoundObject(TypedDict):
    """Definition of a compound object."""
    container: str  # Node ID
    components: List[str]  # List of component node IDs
    type: str  # Type of container

# - Hierarchical representation of visual elements
# - Spatial relationships and compositional rules
# - Semantic properties translated to visual attributes

@dataclass
class SceneNode:
    """Represents a node in the scene graph with spatial and visual properties."""
    node_id: str
    node_type: str
    position: Tuple[float, float, float] = field(default_factory=lambda: (0, 0, 0))
    size: Tuple[float, float, float] = field(default_factory=lambda: (1, 1, 1))
    orientation: float = 0.0  # Rotation in degrees
    properties: Dict[str, Any] = field(default_factory=dict)
    children: List['SceneNode'] = field(default_factory=list)
    parent: Optional['SceneNode'] = None
    source: str = ""  # Source phrase from prompt
    confidence: float = 1.0
    
    def add_child(self, node: 'SceneNode') -> 'SceneNode':
        """Add a child node and set its parent reference."""
        self.children.append(node)
        node.parent = self
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation (excluding parent to avoid cycles)."""
        return { \
            "id": self.node_id,
            "type": self.node_type, \
            "position": self.position, \
            "size": self.size, \
            "orientation": self.orientation, \
            "properties": self.properties, \
            "children": [child.to_dict() for child in self.children], \
            "source": self.source, \
            "confidence": self.confidence
        }
        
    def find_by_id(self, node_id: str) -> Optional['SceneNode']:
        """Find a node by ID in this subtree."""
        if self.node_id == node_id:
            return self
            
        for child in self.children:
            result = child.find_by_id(node_id)
            if result is not None:
                return result
                
        return None
    
    def find_by_type(self, node_type: str) -> List['SceneNode']:
        """Find all nodes of a given type in this subtree."""
        results = []
        
        if self.node_type == node_type:
            results.append(self)
            
        for child in self.children:
            results.extend(child.find_by_type(node_type))
            
        return results

class SceneGraph:
    """Represents the entire scene as a hierarchical graph of nodes."""
    
    def __init__(self, width: float, height: float):
        """Initialize a scene graph with viewport dimensions.
        
        Args:
            width: Width of the scene viewport
            height: Height of the scene viewport
        """
        self.width = width
        self.height = height
        self.root = SceneNode("root", "scene_root", (0, 0, 0), (width, height, 1))
        self.node_counter = 0
    
    def create_node(self, node_type: str, parent: Optional[SceneNode] = None, \
                    position: Tuple[float, float, float] = None,
                    size: Tuple[float, float, float] = None, \
                    properties: Dict[str, Any] = None, \
                    source: str = "", confidence: float = 1.0) -> SceneNode:
        """Create a new node and add it to the scene graph.
        
        Args:
            node_type: Type of node to create
            parent: Parent node (defaults to root if None)
            position: 3D position (x, y, z) - defaults to (0, 0, 0)
            size: 3D size (width, height, depth) - defaults to (1, 1, 1)
            properties: Additional properties dictionary
            source: Source phrase from prompt
            confidence: Confidence score for this node
        
        Returns:
            Newly created scene node
        """
        # Generate unique ID
        self.node_counter += 1
        node_id = f"{node_type}_{self.node_counter}"
        
        # Set defaults
        if position is None:
            position = (0, 0, 0)
        if size is None:
            size = (1, 1, 1)
        if properties is None:
            properties = {}
        
        # Create node
        node = SceneNode( \
            node_id=node_id,
            node_type=node_type, \
            position=position, \
            size=size, \
            properties=properties, \
            source=source, \
            confidence=confidence
        )
        
        # Add to parent (or root if no parent specified)
        parent_node = parent if parent is not None else self.root
        parent_node.add_child(node)
        
        return node
    
    def find_node(self, node_id: str) -> Optional[SceneNode]:
        """Find a node by ID in the entire scene graph."""
        return self.root.find_by_id(node_id)
    
    def find_nodes_by_type(self, node_type: str) -> List[SceneNode]:
        """Find all nodes of a given type in the scene graph."""
        return self.root.find_by_type(node_type)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the scene graph to a dictionary representation."""
        return { \
            "width": self.width,
            "height": self.height, \
            "root": self.root.to_dict()
        }
    
    def validate(self) -> List[str]:
        """Validate the scene graph for consistency and completeness.
        
        Returns:
            List of validation error messages, empty if valid
        """
        errors = []
        
        # Check for overlapping z-indices
        # Check for nodes outside viewport
        # Check for disconnected nodes
        # Additional validation rules
        
        return errors

class SceneArchetype:
    """Base class for scene archetypes that define common scene layouts."""
    
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def apply(self, scene_graph: SceneGraph, features: Dict[str, Any]) -> None:
        """Apply this archetype to a scene graph based on features."""
        raise NotImplementedError("Subclasses must implement this method")

class LandscapeArchetype(SceneArchetype):
    """Archetype for landscape scenes (mountains, hills, forest, etc.)."""
    
    def apply(self, scene_graph: SceneGraph, features: Dict[str, Any]) -> None:
        """Apply landscape archetype to scene graph.
        
        Args:
            scene_graph: The scene graph to populate
            features: Dictionary of semantic features
        """
        # Create sky layer
        sky = scene_graph.create_node( \
            "sky",
            position=(0, 0, 0), \
            size=(scene_graph.width, scene_graph.height * 0.7, 0), \
            properties={ \
                "layer": "background", \
                "z_index": 0
            }
        )
        
        # Set sky properties based on time of day
        if "time_of_day" in features:
            time = features["time_of_day"]["value"]
            sky.properties["time_of_day"] = time
            
            # Set sky gradient based on time
            if time == "night":
                sky.properties["gradient"] = ["#0a1a3f", "#16396b"]
            elif time in ["sunset", "golden hour"]:
                sky.properties["gradient"] = ["#ffc688", "#ff7c7c", "#4a8fe7"]
            elif time == "dawn":
                sky.properties["gradient"] = ["#f8c9b9", "#e56a77", "#4a8fe7"]
            else:  # day, midday, afternoon
                sky.properties["gradient"] = ["#c1e3ff", "#3a7bd5"]
        else:
            # Default sky
            sky.properties["gradient"] = ["#c1e3ff", "#3a7bd5"]
        
        # Create ground/terrain layer
        ground_height = scene_graph.height * 0.4
        ground = scene_graph.create_node( \
            "ground",
            position=(0, scene_graph.height * 0.6, 1), \
            size=(scene_graph.width, ground_height, 0), \
            properties={ \
                "layer": "midground", \
                "z_index": 1
            }
        )
        
        # Add mountains if applicable
        if "mountain" in str(features).lower() or "hill" in str(features).lower():
            # Determine mountain properties based on features
            mountain_count = random.randint(2, 5)
            
            # Create mountains with depth effect
            for i in range(mountain_count):
                # Calculate position and size for varied layout
                pos_x = scene_graph.width * (0.1 + 0.8 * i / mountain_count)
                # Mountains further back are higher in the scene
                pos_y = scene_graph.height * (0.4 - 0.05 * i)
                z_pos = 2 + i
                
                # Size varies with position
                mtn_width = scene_graph.width * 0.3 * (1 + 0.2 * random.random())
                mtn_height = scene_graph.height * 0.3 * (0.7 + 0.4 * random.random())
                
                scene_graph.create_node( \
                    "mountain",
                    position=(pos_x, pos_y, z_pos), \
                    size=(mtn_width, mtn_height, 0), \
                    properties={ \
                        "layer": "midground", \
                        "z_index": z_pos, \
                        "roughness": 0.3 + 0.4 * random.random(), \
                        "sharp": random.random() > 0.5
                    }
                )
                
        # Add foreground elements based on features
        # (Trees, water, etc. - these would be added by specialized methods)

class SeascapeArchetype(SceneArchetype):
    """Archetype for seascape scenes (ocean, beach, coast, etc.)."""
    
    def apply(self, scene_graph: SceneGraph, features: Dict[str, Any]) -> None:
        """Apply seascape archetype to scene graph.
        
        Args:
            scene_graph: The scene graph to populate
            features: Dictionary of semantic features
        """
        # Create sky layer
        sky = scene_graph.create_node( \
            "sky",
            position=(0, 0, 0), \
            size=(scene_graph.width, scene_graph.height * 0.6, 0), \
            properties={ \
                "layer": "background", \
                "z_index": 0
            }
        )
        
        # Set sky properties based on time of day (same as landscape)
        if "time_of_day" in features:
            time = features["time_of_day"]["value"]
            sky.properties["time_of_day"] = time
            
            # Set sky gradient based on time
            if time == "night":
                sky.properties["gradient"] = ["#0a1a3f", "#16396b"]
            elif time in ["sunset", "golden hour"]:
                sky.properties["gradient"] = ["#ffc688", "#ff7c7c", "#4a8fe7"]
            elif time == "dawn":
                sky.properties["gradient"] = ["#f8c9b9", "#e56a77", "#4a8fe7"]
            else:  # day, midday, afternoon
                sky.properties["gradient"] = ["#c1e3ff", "#3a7bd5"]
        else:
            # Default sky
            sky.properties["gradient"] = ["#c1e3ff", "#3a7bd5"]
        
        # Create sea layer
        sea_start_y = scene_graph.height * 0.55
        sea_height = scene_graph.height - sea_start_y
        
        sea = scene_graph.create_node( \
            "sea",
            position=(0, sea_start_y, 1), \
            size=(scene_graph.width, sea_height, 0), \
            properties={ \
                "layer": "midground", \
                "z_index": 1, \
                "wave_intensity": 0.3 + 0.4 * random.random()
            }
        )
        
        # Set sea properties based on time and atmospheric conditions
        if "time_of_day" in features:
            time = features["time_of_day"]["value"]
            
            if time == "night":
                sea.properties["gradient"] = ["#0a2a4f", "#061a30"]
            elif time in ["sunset", "golden hour"]:
                sea.properties["gradient"] = ["#0077be", "#005f98"]
                sea.properties["reflection"] = "sunset"
            else:
                sea.properties["gradient"] = ["#0077be", "#005f98"]
        
        # Add beach if applicable
        if "beach" in str(features).lower() or "shore" in str(features).lower():
            beach = scene_graph.create_node( \
                "beach",
                position=(0, sea_start_y - scene_graph.height * 0.05, 2), \
                size=(scene_graph.width, scene_graph.height * 0.1, 0), \
                properties={ \
                    "layer": "midground", \
                    "z_index": 2, \
                    "gradient": ["#f0e68c", "#deb887"]
                }
            )

class CityscapeArchetype(SceneArchetype):
    """Archetype for cityscape scenes (urban, buildings, streets, etc.)."""
    
    def _generate_material_color(self, material):
        """Generate an appropriate building color based on material.
        
        Args:
            material: Building material name
        
        Returns:
            Hexadecimal color value
        """
        # Material-based coloring
        material_colors = { \
            "stone": ["#8B8B83", "#79796A", "#696969", "#5F5F5F"],
            "marble": ["#F5F5F5", "#F8F8F8", "#F0F0F0", "#EFEFEF"], \
            "brick": ["#B35A1F", "#A0522D", "#8B4513", "#A52A2A"], \
            "wood": ["#8B4513", "#A0522D", "#CD853F", "#D2691E"], \
            "concrete": ["#BCBCBC", "#989898", "#737373", "#5E5E5E"], \
            "steel": ["#A9A9A9", "#B8B8B8", "#C0C0C0", "#D3D3D3"], \
            "glass": ["#ADD8E6", "#87CEEB", "#B0E0E6", "#87CEFA"], \
            "bronze": ["#CD7F32", "#B87333", "#C88141", "#D1A33D"], \
            "copper": ["#B87333", "#A56E40", "#965A38", "#7F462C"], \
            "gold": ["#FFD700", "#DAA520", "#B8860B", "#CD853F"], \
            "terracotta": ["#E2725B", "#CC7722", "#C8553D", "#BC6C25"]
        }
        
        # Convert material to lowercase for case-insensitive matching
        material_lower = material.lower()
        
        # Check for direct material match
        for mat_name, colors in material_colors.items():
            if mat_name in material_lower:
                return random.choice(colors)
        
        # If no direct match, use general coloring based on building type categories
        if any(term in material_lower for term in ["modern", "contemporary", "minimalist"]):
            # Modern glass and steel look
            return random.choice(["#A9D0F5", "#CECECE", "#81BEF7", "#E0E0E0"])
        elif any(term in material_lower for term in ["ancient", "old", "traditional", "historic"]):
            # Traditional stone look
            return random.choice(["#8B8B83", "#79796A", "#696969", "#5F5F5F"])
        
        # Default - neutral color
        return random.choice(["#D0D0D0", "#C8C8C8", "#BCBCBC", "#B0B0B0"])
    
    def apply(self, scene_graph: SceneGraph, features: Dict[str, Any]) -> None:
        """Apply cityscape archetype to scene graph.
        
        Args:
            scene_graph: The scene graph to populate
            features: Dictionary of semantic features
        """
        # Extract relevant features
        time_of_day = features.get("time_of_day", SemanticFeature("day", 1.0, [])).value
        weather = features.get("weather", SemanticFeature("clear", 1.0, [])).value
        architectural_style = features.get("architectural_style", SemanticFeature("modern", 1.0, [])).value
        density = features.get("density", SemanticFeature("medium", 1.0, [])).value
        mood = features.get("mood", SemanticFeature("neutral", 1.0, [])).value
        
        # Extract architectural materials
        materials = []
        if "materials" in features and hasattr(features["materials"], "value"):
            materials = features["materials"].value if isinstance(features["materials"].value, list) else [features["materials"].value]

        # Extract architectural features
        architectural_features = []
        if "architectural_features" in features and hasattr(features["architectural_features"], "value"):
            if isinstance(features["architectural_features"].value, list):
                architectural_features = features["architectural_features"].value
            else:
                architectural_features = [features["architectural_features"].value]
                
        # Add style-specific architectural features based on detected architectural style
        if architectural_style == "gothic":
            if "pointed_arches" not in architectural_features:
                architectural_features.append("pointed_arches")
            if "buttresses" not in architectural_features:
                architectural_features.append("buttresses")
        elif architectural_style == "classical":
            if "columns" not in architectural_features:
                architectural_features.append("columns")
            if "pediment" not in architectural_features:
                architectural_features.append("pediment")
        elif architectural_style == "art_deco":
            if "geometric_patterns" not in architectural_features:
                architectural_features.append("geometric_patterns")
            if "stepped_design" not in architectural_features:
                architectural_features.append("stepped_design")
        elif architectural_style == "victorian":
            if "ornate_details" not in architectural_features:
                architectural_features.append("ornate_details")
            if "bay_windows" not in architectural_features:
                architectural_features.append("bay_windows")
        elif architectural_style == "brutalist":
            if "exposed_concrete" not in architectural_features:
                architectural_features.append("exposed_concrete")
            if "modular_elements" not in architectural_features:
                architectural_features.append("modular_elements")
                
        # Determine number of buildings based on density
        if density == "sparse":
            building_count = random.randint(1, 3)
        elif density == "dense":
            building_count = random.randint(5, 8)
        else:  # medium
            building_count = random.randint(3, 5)
        
        # Create sky layer
        sky = scene_graph.create_node( \
            "sky",
            position=(0, 0, 0), \
            size=(scene_graph.width, scene_graph.height * 0.6, 0), \
            properties={ \
                "layer": "background", \
                "z_index": 0
            }
        )
        
        # Sky properties based on time of day (similar to other archetypes)
        if "time_of_day" in features:
            time = features["time_of_day"]["value"]
            sky.properties["time_of_day"] = time
            
            # Set sky gradient based on time
            if time == "night":
                sky.properties["gradient"] = ["#0a1a3f", "#16396b"]
            elif time in ["sunset", "golden hour"]:
                sky.properties["gradient"] = ["#ffc688", "#ff7c7c", "#4a8fe7"]
            elif time == "dawn":
                sky.properties["gradient"] = ["#f8c9b9", "#e56a77", "#4a8fe7"]
            else:  # day, midday, afternoon
                sky.properties["gradient"] = ["#c1e3ff", "#3a7bd5"]
        else:
            # Default sky
            sky.properties["gradient"] = ["#c1e3ff", "#3a7bd5"]
        
        # Create ground/street layer
        ground = scene_graph.create_node( \
            "ground",
            position=(0, scene_graph.height * 0.8, 1), \
            size=(scene_graph.width, scene_graph.height * 0.2, 0), \
            properties={ \
                "layer": "foreground", \
                "z_index": 1, \
                "color": "#555555"
            }
        )
        
        # Create skyline with buildings
        building_count = random.randint(6, 12)
        building_width = scene_graph.width / building_count
        
        # Add building style based on architectural features with enhanced details
        building_style = "modern"
        arch_features = {}
        materials = []
        
        # Extract architectural style and features
        if "architectural_style" in features:
            building_style = features["architectural_style"].value
            
        # Process architectural features - convert from dictionary to list format expected by BuildingGenerator
        architectural_features = []
        if "architectural_features" in features:
            if isinstance(features["architectural_features"].value, list):
                architectural_features = features["architectural_features"].value
            else:
                architectural_features = [features["architectural_features"].value]
                
        # Check for additional detailed architectural features
        for key, value in features.items():
            if key.startswith("arch_"):
                if key == "arch_materials":
                    if isinstance(value.value, list):
                        materials.extend(value.value)
                    else:
                        materials.append(value.value)
                else:
                    # Extract the feature and add it to our features list
                    feature_name = key[5:]  # Skip 'arch_'
                    if feature_name not in architectural_features:
                        architectural_features.append(feature_name)
        
        for i in range(building_count):
            # Vary building heights for realistic skyline
            height_factor = 0.3 + 0.4 * random.random()
            # Taller buildings in center
            center_bias = 1.0 - abs((i - building_count/2) / (building_count/2)) * 0.5
            
            building_height = scene_graph.height * height_factor * center_bias
            building_y = scene_graph.height * 0.8 - building_height
            
            # Generate color based on materials if available or style otherwise
            if materials:
                building_color = self._generate_material_color(materials[0])
            else:
                building_color = self._generate_building_color(building_style)
                
            # Create the building node with enhanced properties
            scene_graph.create_node( \
                "building",
                position=(i * building_width, building_y, 2), \
                size=(building_width * 0.9, building_height, 0), \
                properties={ \
                    "layer": "midground", \
                    "z_index": 2, \
                    "style": building_style, \
                    "window_pattern": random.choice(["grid", "random", "horizontal", "vertical"]), \
                    "color": building_color, \
                    "architectural_features": architectural_features, \
                    "materials": materials, \
                    "detail_level": 0.8 if any(feature in ["detailed", "ornate", "decorative"] \
                                            for feature in architectural_features) else 0.5
                }
            )

class SceneGraphGenerator:
    """Generates a scene graph from scene features using archetypes and constraints."""
    
    def __init__(self):
        self.archetypes = { \
            "landscape": LandscapeArchetype,
            "seascape": SeascapeArchetype, \
            "cityscape": CityscapeArchetype
            # Additional archetypes would be added here
        }
    
    def generate_scene_graph(self, features: Dict[str, Any], width: float, height: float) -> SceneGraph:
        """Generate a complete scene graph based on semantic features.
        
        Args:
            features: Dictionary of scene features from semantic analysis
            width: Width of the viewport
            height: Height of the viewport
            
        Returns:
            Scene graph representing the complete scene
        """
        # Create base scene graph
        scene_graph = SceneGraph(width, height)
        
        # Determine which archetype to use
        archetype_name = "landscape"  # Default
        if "scene_type" in features:
            scene_type = features["scene_type"]["value"]
            if scene_type in self.archetypes:
                archetype_name = scene_type
        
        # Create and apply the archetype
        archetype_class = self.archetypes[archetype_name]
        archetype = archetype_class(width, height)
        archetype.apply(scene_graph, features)
        
        # Add focal elements based on features
        self._add_focal_elements(scene_graph, features)
        
        # Add atmosphere and lighting effects
        self._add_atmosphere(scene_graph, features)
        
        # Add stylistic elements based on mood
        if "mood" in features:
            self._apply_mood(scene_graph, features["mood"]["value"])
        
        # Final validation and cleanup
        errors = scene_graph.validate()
        if errors:
            print(f"Scene graph validation warnings: {errors}")
        
        return scene_graph
    
    def _add_focal_elements(self, scene_graph: SceneGraph, features: Dict[str, Any]) -> None:
        """Add focal elements to the scene based on extracted features.
        
        Args:
            scene_graph: Scene graph to modify
            features: Dictionary of scene features
        """
        if "focal_element" not in features:
            return
            
        focal_element = features["focal_element"]["value"]
        
        # Position in scene depends on element type
        if focal_element == "sun":
            # Position sun based on time of day
            time = features.get("time_of_day", {}).get("value", "midday")
            
            if time == "sunset" or time == "golden hour":
                x_pos = scene_graph.width * 0.8
                y_pos = scene_graph.height * 0.2
            elif time == "dawn":
                x_pos = scene_graph.width * 0.2
                y_pos = scene_graph.height * 0.2
            else:  # Midday or other
                x_pos = scene_graph.width * 0.5
                y_pos = scene_graph.height * 0.15
                
            scene_graph.create_node( \
                "sun",
                position=(x_pos, y_pos, 10),  # High z-index to be on top
                size=(scene_graph.width * 0.1, scene_graph.width * 0.1, 0), \
                properties={ \
                    "layer": "background", \
                    "z_index": 10, \
                    "glow": True, \
                    "time": time
                }
            )
        elif focal_element == "tree" or focal_element == "forest":
            # Add a prominent tree in foreground
            scene_graph.create_node( \
                "tree",
                position=(scene_graph.width * 0.3, scene_graph.height * 0.6, 10), \
                size=(scene_graph.width * 0.2, scene_graph.height * 0.4, 0), \
                properties={ \
                    "layer": "foreground", \
                    "z_index": 10, \
                    "tree_type": "pine" if "pine" in str(features) else "deciduous"
                }
            )
        # Add more focal elements as needed
    
    def _add_atmosphere(self, scene_graph: SceneGraph, features: Dict[str, Any]) -> None:
        """Add atmospheric effects based on features.
        
        Args:
            scene_graph: Scene graph to modify
            features: Dictionary of scene features
        """
        if "atmospheric" not in features:
            return
            
        atmospheric = features["atmospheric"]["value"]
        
        if atmospheric == "foggy" or atmospheric == "misty":
            scene_graph.create_node( \
                "fog",
                position=(0, 0, 100),  # Very high z-index to overlay everything
                size=(scene_graph.width, scene_graph.height, 0), \
                properties={ \
                    "layer": "overlay", \
                    "z_index": 100, \
                    "opacity": 0.5, \
                    "gradient": ["rgba(255,255,255,0.7)", "rgba(255,255,255,0.2)"]
                }
            )
        elif atmospheric == "cloudy":
            # Add some clouds
            cloud_count = random.randint(3, 7)
            for i in range(cloud_count):
                x_pos = scene_graph.width * random.random()
                y_pos = scene_graph.height * 0.2 * random.random()
                cloud_width = scene_graph.width * (0.1 + 0.2 * random.random())
                
                scene_graph.create_node( \
                    "cloud",
                    position=(x_pos, y_pos, 5), \
                    size=(cloud_width, cloud_width * 0.6, 0), \
                    properties={ \
                        "layer": "background", \
                        "z_index": 5, \
                        "fluffiness": 0.3 + 0.5 * random.random()
                    }
                )
        # Add more atmospheric conditions as needed
    
    def _apply_mood(self, scene_graph: SceneGraph, mood: str) -> None:
        """Apply mood-specific effects to the scene.
        
        Args:
            scene_graph: Scene graph to modify
            mood: Mood value from features
        """
        # Each mood has specific color modifications and effects
        if mood == "dramatic":
            # Add contrast and more intense colors
            scene_graph.root.properties["contrast"] = 1.2
            scene_graph.root.properties["saturation"] = 1.3
        elif mood == "tranquil":
            # Softer colors, less contrast
            scene_graph.root.properties["contrast"] = 0.9
            scene_graph.root.properties["saturation"] = 0.9
            scene_graph.root.properties["brightness"] = 1.1
        elif mood == "mysterious":
            # Darker, more shadows
            scene_graph.root.properties["contrast"] = 1.1
            scene_graph.root.properties["brightness"] = 0.9
            scene_graph.root.properties["shadow_intensity"] = 1.3
        # Add more moods as needed

def generate_intermediate_representation(features: Dict[str, Any], width: float, height: float) -> SceneGraph:
    """Generate an intermediate scene representation from semantic features.
    
    This function creates a structured scene graph with nodes representing
    all scene elements and their spatial relationships.
    
    Args:
        features: Dictionary of scene features from semantic analysis
        width: Width of the scene viewport
        height: Height of the scene viewport
        
    Returns:
        Scene graph representing the complete scene
    """
    generator = SceneGraphGenerator()
    return generator.generate_scene_graph(features, width, height)

# ============================================================================ #
#                3. PROCEDURAL SVG GENERATION                                  #
# ============================================================================ #
# This stage synthesizes SVG code from the scene graph:

# ============================================================================ #
#                4. VALIDATION AND SANITIZATION                               #
# ============================================================================ #
# This stage ensures SVG safety and standards compliance

class SVGValidator:
    """Validates SVG content for safety, standards compliance, and structural integrity.
    
    This component performs thorough validation on generated SVG to ensure it meets
    multiple criteria:
    - Safety: No scripts, external resources, or potentially harmful elements
    - Standards: Compliant with SVG 1.1 specification
    - Structure: Well-formed XML with proper nesting and attributes
    - Accessibility: Ensures SVG can be properly interpreted by screen readers
    - Size: Enforces maximum SVG size constraints
    
    It produces detailed validation reports that can be used by the correction engine
    to automatically fix issues when possible.
    """
    
    def __init__(self):
        # SVG tag and attribute whitelists for safety validation
        self.safe_elements = { \
            "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline",
            "polygon", "text", "tspan", "defs", "clipPath", "pattern", "mask", "filter", \
            "linearGradient", "radialGradient", "stop", "image", "use", "symbol", "marker", \
            "desc", "title", "metadata", "switch", "foreignObject"
        }
        
        # Potentially unsafe elements
        self.unsafe_elements = { \
            "script", "iframe", "object", "embed", "animate", "animateMotion",
            "animateTransform", "animateColor", "set", "animation"
        }
        
        # Attributes that can contain script or external references
        self.unsafe_attributes = { \
            "onload", "onunload", "onabort", "onerror", "onresize", "onscroll", "onzoom",
            "onactivate", "onclick", "onmousedown", "onmousemove", "onmouseout", "onmouseover", \
            "onmouseup", "eval", "javascript", "script", "href", "xlink:href", "href"
        }
        
        # Safe attribute prefixes
        self.safe_attribute_prefixes = { \
            "x", "y", "width", "height", "fill", "stroke", "transform", "opacity", "cx",
            "cy", "r", "rx", "ry", "d", "points", "viewBox", "preserveAspectRatio", "color", \
            "font", "style", "path", "filter", "clip", "mask", "marker", "enable-background", \
            "gradientUnits", "patternUnits", "patternTransform", "gradientTransform", \
            "text-anchor", "dominant-baseline", "xmlns"
        }
        
        # Size constraints
        self.max_size_bytes = 10 * 1024  # 10KB maximum size
        self.max_nodes = 500  # Maximum number of nodes
        self.max_path_points = 1000  # Maximum points in a path
        
        # W3C SVG 1.1 validation patterns for essential attributes
        self.validation_patterns = {
            # viewBox should have 4 values: min-x, min-y, width, height
            "viewBox": re.compile(r"^-?\d+(\.\d+)?\s+-?\d+(\.\d+)?\s+\d+(\.\d+)?\s+\d+(\.\d+)?$"),
            # colors should be valid
            "color": re.compile(r"^(#[0-9A-Fa-f]{3,8}|rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(,\s*[01](\.\d+)?\s*)?\)|\w+)$"),
            # paths should be properly formatted
            "d": re.compile(r"^[\w\s,.-]*$")
        }
    
    def validate(self, svg_content: str) -> Dict[str, Any]:
        """Validates SVG content against all safety and standards criteria.
        
        Args:
            svg_content: String containing the SVG XML code
            
        Returns:
            Dictionary with validation results and detailed reports
        """
        # Initialize validation report
        report = { \
            "is_valid": True,
            "errors": [], \
            "warnings": [], \
            "file_size": len(svg_content), \
            "node_count": 0, \
            "safe": True, \
            "standards_compliant": True, \
            "structural_integrity": True, \
            "fingerprint": hashlib.md5(svg_content.encode()).hexdigest(), \
            "correction_needed": False, \
            "correction_possible": True
        }
        
        # Check size constraints
        if len(svg_content) > self.max_size_bytes:
            report["is_valid"] = False
            report["safe"] = False
            report["errors"].append({ \
                "type": "size_constraint",
                "message": f"SVG exceeds maximum size of {self.max_size_bytes/1024:.1f}KB", \
                "severity": "high"
            })
        
        try:
            # Parse SVG content
            root = ET.fromstring(svg_content)
            
            # Count nodes
            nodes = list(root.iter())
            report["node_count"] = len(nodes)
            
            if report["node_count"] > self.max_nodes:
                report["is_valid"] = False
                report["errors"].append({ \
                    "type": "complexity_constraint",
                    "message": f"SVG exceeds maximum node count of {self.max_nodes}", \
                    "severity": "medium"
                })
            
            # Validate tags and attributes
            self._validate_elements(nodes, report)
            
            # Validate structure and nesting
            self._validate_structure(root, report)
            
            # Validate standards compliance
            self._validate_standards_compliance(root, report)
            
            # Validate accessibility
            self._validate_accessibility(root, report)
            
        except ET.ParseError as e:
            report["is_valid"] = False
            report["structural_integrity"] = False
            report["errors"].append({ \
                "type": "parse_error",
                "message": f"XML parsing error: {str(e)}", \
                "severity": "critical", \
                "line": getattr(e, "position", (0, 0))[0], \
                "column": getattr(e, "position", (0, 0))[1]
            })
        
        # Determine if correction is needed and possible
        report["correction_needed"] = not report["is_valid"]
        report["correction_possible"] = not any(e["severity"] == "critical" for e in report["errors"])
        
        return report
    
    def _validate_elements(self, nodes: List[ET.Element], report: Dict[str, Any]):
        """Validates all elements and their attributes for safety."""
        for node in nodes:
            # Strip namespace for tag name comparison
            tag = node.tag
            if '}' in tag:
                tag = tag.split('}')[1]
            
            # Check for unsafe elements
            if tag in self.unsafe_elements:
                report["is_valid"] = False
                report["safe"] = False
                report["errors"].append({ \
                    "type": "unsafe_element",
                    "message": f"Unsafe element detected: {tag}", \
                    "element": tag, \
                    "severity": "high", \
                    "fixable": True, \
                    "fix_action": "remove"
                })
            
            # Check if element is not in safe list
            elif tag not in self.safe_elements:
                report["warnings"].append({ \
                    "type": "unknown_element",
                    "message": f"Unknown element detected: {tag}", \
                    "element": tag, \
                    "severity": "low", \
                    "fixable": True, \
                    "fix_action": "remove"
                })
            
            # Validate attributes
            for attr, value in node.attrib.items():
                # Strip namespace for attribute comparison
                if '}' in attr:
                    attr = attr.split('}')[1]
                
                # Check for unsafe attributes
                if attr in self.unsafe_attributes:
                    report["is_valid"] = False
                    report["safe"] = False
                    report["errors"].append({ \
                        "type": "unsafe_attribute",
                        "message": f"Unsafe attribute detected: {attr}={value}", \
                        "element": tag, \
                        "attribute": attr, \
                        "value": value, \
                        "severity": "high", \
                        "fixable": True, \
                        "fix_action": "remove_attribute"
                    })
                
                # Check for potentially unsafe values in safe attributes
                elif 'href' in attr.lower() or attr == 'src':
                    if 'javascript:' in value or 'data:' in value:
                        report["is_valid"] = False
                        report["safe"] = False
                        report["errors"].append({ \
                            "type": "unsafe_attribute_value",
                            "message": f"Unsafe value in attribute: {attr}={value}", \
                            "element": tag, \
                            "attribute": attr, \
                            "value": value, \
                            "severity": "high", \
                            "fixable": True, \
                            "fix_action": "sanitize_value"
                        })
                
                # Validate specific attributes against patterns
                elif attr in self.validation_patterns and not self.validation_patterns[attr].match(value):
                    report["is_valid"] = False
                    report["standards_compliant"] = False
                    report["errors"].append({ \
                        "type": "invalid_attribute_format",
                        "message": f"Invalid format for attribute: {attr}={value}", \
                        "element": tag, \
                        "attribute": attr, \
                        "value": value, \
                        "severity": "medium", \
                        "fixable": True, \
                        "fix_action": "correct_format"
                    })
    
    def _validate_structure(self, root: ET.Element, report: Dict[str, Any]):
        """Validates structural integrity and proper nesting."""
        # Check if root element is svg
        if root.tag.split('}')[-1] != 'svg':
            report["is_valid"] = False
            report["structural_integrity"] = False
            report["errors"].append({ \
                "type": "invalid_root",
                "message": "Root element is not svg", \
                "severity": "high", \
                "fixable": False
            })
        
        # Check for required attributes on svg element
        required_svg_attrs = ["width", "height", "viewBox"]
        missing_attrs = [attr for attr in required_svg_attrs if attr not in root.attrib]
        
        if missing_attrs:
            report["is_valid"] = False
            report["standards_compliant"] = False
            report["errors"].append({ \
                "type": "missing_required_attributes",
                "message": f"Root svg element missing required attributes: {', '.join(missing_attrs)}", \
                "element": "svg", \
                "missing_attributes": missing_attrs, \
                "severity": "medium", \
                "fixable": True, \
                "fix_action": "add_attributes"
            })
        
        # Check for unclosed elements and improper nesting
        # This is mostly handled by ET.fromstring, which will raise a ParseError
        
        # Check for path elements with invalid or empty d attributes
        for path in root.findall(".//{*}path"):
            if 'd' not in path.attrib or not path.attrib['d'].strip():
                report["is_valid"] = False
                report["standards_compliant"] = False
                report["errors"].append({ \
                    "type": "invalid_path",
                    "message": "Path element with missing or empty d attribute", \
                    "severity": "medium", \
                    "fixable": True, \
                    "fix_action": "remove"
                })
    
    def _validate_standards_compliance(self, root: ET.Element, report: Dict[str, Any]):
        """Validates compliance with SVG 1.1 standards."""
        # Check for correct namespace
        if not root.tag.startswith('{http://www.w3.org/2000/svg}'):
            report["is_valid"] = False
            report["standards_compliant"] = False
            report["errors"].append({ \
                "type": "missing_namespace",
                "message": "SVG namespace is not correctly specified", \
                "severity": "medium", \
                "fixable": True, \
                "fix_action": "add_namespace"
            })
        
        # Check for proper viewBox format if present
        if 'viewBox' in root.attrib:
            viewbox = root.attrib['viewBox']
            if not re.match(r'^-?[\d.]+\s+-?[\d.]+\s+[\d.]+\s+[\d.]+$', viewbox):
                report["is_valid"] = False
                report["standards_compliant"] = False
                report["errors"].append({ \
                    "type": "invalid_viewbox",
                    "message": f"Invalid viewBox format: {viewbox}", \
                    "element": "svg", \
                    "attribute": "viewBox", \
                    "value": viewbox, \
                    "severity": "medium", \
                    "fixable": True, \
                    "fix_action": "correct_viewbox"
                })
        
        # Check path data integrity if any path elements exist
        for path in root.findall(".//{*}path"):
            if 'd' in path.attrib:
                d_attr = path.attrib['d']
                # Check for common path data errors
                if d_attr.strip() and not d_attr.startswith(('M', 'm')):
                    report["is_valid"] = False
                    report["standards_compliant"] = False
                    report["errors"].append({ \
                        "type": "invalid_path_data",
                        "message": "Path data must start with a move command (M or m)", \
                        "severity": "medium", \
                        "fixable": True, \
                        "fix_action": "correct_path"
                    })
    
    def _validate_accessibility(self, root: ET.Element, report: Dict[str, Any]):
        """Validates SVG for accessibility."""
        # Check for title element
        has_title = False
        for title in root.findall(".//{*}title"):
            has_title = True
            # Check if title is empty
            if not title.text or not title.text.strip():
                report["warnings"].append({ \
                    "type": "empty_title",
                    "message": "Empty title element", \
                    "severity": "low", \
                    "fixable": True, \
                    "fix_action": "add_title_content"
                })
        
        if not has_title:
            report["warnings"].append({ \
                "type": "missing_title",
                "message": "SVG lacks a title element for accessibility", \
                "severity": "low", \
                "fixable": True, \
                "fix_action": "add_title"
            })
        
        # Check for desc element
        has_desc = False
        for desc in root.findall(".//{*}desc"):
            has_desc = True
            # Check if desc is empty
            if not desc.text or not desc.text.strip():
                report["warnings"].append({ \
                    "type": "empty_desc",
                    "message": "Empty description element", \
                    "severity": "low", \
                    "fixable": True, \
                    "fix_action": "add_desc_content"
                })
        
        if not has_desc:
            report["warnings"].append({ \
                "type": "missing_desc",
                "message": "SVG lacks a description element for accessibility", \
                "severity": "low", \
                "fixable": True, \
                "fix_action": "add_desc"
            })
        
        # Check for aria attributes on interactive elements
        for element in root.findall(".//{*}a") + root.findall(".//{*}text") + root.findall(".//{*}button"):
            if not any(attr.startswith('aria-') for attr in element.attrib):
                report["warnings"].append({ \
                    "type": "missing_aria",
                    "message": "Interactive element without ARIA attributes", \
                    "element": element.tag.split('}')[-1], \
                    "severity": "low", \
                    "fixable": True, \
                    "fix_action": "add_aria"
                })

# ============================================================================ #
#                5. CORRECTION AND SEMANTIC FEEDBACK                           #
# ============================================================================ #
# This stage auto-corrects issues found in validation

class SVGCorrectionEngine:
    """Auto-corrects issues identified during SVG validation.
    
    This component applies fixes to SVG content based on validation reports, \
    ensuring the output meets safety, standards, and optimization requirements.It can fix common issues like:
    - Missing required attributes
    - Unsafe elements or attributes
    - Structural problems
    - Style and format issues
    - Size optimization opportunities
    
    The engine provides semantic feedback about applied corrections and
    maintains the original intent of the SVG while ensuring compliance.
    """
    
    def __init__(self, validator: SVGValidator = None):
        """Initialize with an optional validator instance."""
        self.validator = validator or SVGValidator()
        
        # Mapping of fix actions to correction methods
        self.correction_actions = { \
            "remove": self._remove_element,
            "remove_attribute": self._remove_attribute, \
            "sanitize_value": self._sanitize_attribute_value, \
            "correct_format": self._correct_attribute_format, \
            "add_attributes": self._add_required_attributes, \
            "add_namespace": self._add_namespace, \
            "correct_viewbox": self._correct_viewbox, \
            "correct_path": self._correct_path_data, \
            "add_title": self._add_title, \
            "add_desc": self._add_description, \
            "add_title_content": self._add_title_content, \
            "add_desc_content": self._add_description_content, \
            "add_aria": self._add_aria_attributes
        }
    
    def correct_svg(self, svg_content: str) -> Dict[str, Any]:
        """Validate and correct SVG content.
        
        Args:
            svg_content: SVG XML content string
            
        Returns:
            Dictionary with corrected SVG and correction report
        """
        # Create result structure
        result = { \
            "original_svg": svg_content,
            "corrected_svg": svg_content, \
            "validation_report": None, \
            "corrections_applied": [], \
            "is_valid": False, \
            "size_reduction": 0, \
            "semantic_feedback": ""
        }
        
        # First validate the SVG
        validation_report = self.validator.validate(svg_content)
        result["validation_report"] = validation_report
        
        # If valid, no corrections needed
        if validation_report["is_valid"]:
            result["is_valid"] = True
            result["semantic_feedback"] = "SVG is valid; no corrections needed."
            return result
            
        # If not correctable, return as is with warning
        if not validation_report["correction_possible"]:
            result["semantic_feedback"] = "SVG has critical errors that cannot be auto-corrected."
            return result
            
        try:
            # Apply corrections
            corrected_content = self._apply_corrections(svg_content, validation_report)
            result["corrected_svg"] = corrected_content
            
            # Re-validate to check if corrections fixed issues
            final_validation = self.validator.validate(corrected_content)
            result["is_valid"] = final_validation["is_valid"]
            
            # Calculate size reduction
            original_size = len(svg_content)
            corrected_size = len(corrected_content)
            result["size_reduction"] = original_size - corrected_size
            
            # Generate semantic feedback on corrections
            result["semantic_feedback"] = self._generate_semantic_feedback(result["corrections_applied"])
            
        except Exception as e:
            result["semantic_feedback"] = f"Error during correction: {str(e)}"
            
        return result
    
    def _apply_corrections(self, svg_content: str, validation_report: Dict[str, Any]) -> str:
        """Apply corrections based on validation report."""
        # Parse SVG into an ElementTree for manipulation
        try:
            root = ET.fromstring(svg_content)
        except ET.ParseError:
            # If we can't parse, try a minimal cleanup first
            svg_content = self._basic_xml_cleanup(svg_content)
            root = ET.fromstring(svg_content)
        
        corrections = []
        
        # Process errors (higher priority)
        for error in validation_report["errors"]:
            if "fix_action" in error and error["fixable"]:
                action = error["fix_action"]
                if action in self.correction_actions:
                    success = self.correction_actions[action](root, error)
                    if success:
                        corrections.append({ \
                            "type": error["type"],
                            "action": action, \
                            "details": error["message"]
                        })
        
        # Process warnings (lower priority)
        for warning in validation_report["warnings"]:
            if "fix_action" in warning and warning["fixable"]:
                action = warning["fix_action"]
                if action in self.correction_actions:
                    success = self.correction_actions[action](root, warning)
                    if success:
                        corrections.append({ \
                            "type": warning["type"],
                            "action": action, \
                            "details": warning["message"]
                        })
        
        # Convert back to string
        ET.register_namespace("", SVGNS)
        for prefix, uri in [("xlink", XLINKNS)]:
            ET.register_namespace(prefix, uri)
        
        # Use in-memory string representation for ElementTree output
        with io.BytesIO() as buffer:
            tree = ET.ElementTree(root)
            tree.write(buffer, encoding='utf-8', xml_declaration=True)
            corrected_svg = buffer.getvalue().decode('utf-8')
        
        # Store the corrections that were applied
        self.corrections_applied = corrections
        
        return corrected_svg
    
    def _basic_xml_cleanup(self, svg_content: str) -> str:
        """Perform basic cleanup on malformed XML."""
        # Remove invalid XML characters
        svg_content = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]', '', svg_content)
        
        # Ensure all tags are closed
        unclosed_tags = re.findall(r'<([a-zA-Z0-9]+)[^/>]*?(?<!/)>', svg_content)
        for tag in reversed(unclosed_tags):
            if f"</{tag}>" not in svg_content:
                svg_content += f"</{tag}>"
        
        return svg_content
    
    def _remove_element(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Remove an unsafe or invalid element."""
        if "element" not in issue:
            return False
            
        for elem in root.findall(f".//{{{SVGNS}}}{issue['element']}") + root.findall(f".//{issue['element']}"):
            parent = list(root.iter())
            for p in parent:
                for child in list(p):
                    if child.tag.endswith(issue['element']):
                        p.remove(child)
            return True
            
        return False
    
    def _remove_attribute(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Remove an unsafe attribute."""
        if "element" not in issue or "attribute" not in issue:
            return False
            
        target_elems = root.findall(f".//{{{SVGNS}}}{issue['element']}") + root.findall(f".//{issue['element']}")
        for elem in target_elems:
            # Handle namespaced attributes
            for attr in list(elem.attrib.keys()):
                if attr.endswith(issue['attribute']):
                    del elem.attrib[attr]
                    return True
        return False
    
    def _sanitize_attribute_value(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Sanitize unsafe values in attributes."""
        if "element" not in issue or "attribute" not in issue:
            return False
            
        target_elems = root.findall(f".//{{{SVGNS}}}{issue['element']}") + root.findall(f".//{issue['element']}")
        for elem in target_elems:
            # Handle namespaced attributes
            for attr, value in elem.attrib.items():
                if attr.endswith(issue['attribute']):
                    # Remove javascript: or data: protocols
                    if 'javascript:' in value:
                        elem.attrib[attr] = value.replace('javascript:', '#')
                        return True
                    elif 'data:' in value:
                        elem.attrib[attr] = '#'
                        return True
        return False
    
    def _correct_attribute_format(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Correct formatting of attribute values."""
        if "element" not in issue or "attribute" not in issue:
            return False
            
        # Get target elements
        target_elems = root.findall(f".//{{{SVGNS}}}{issue['element']}") + root.findall(f".//{issue['element']}")
        for elem in target_elems:
            # Handle specific attribute types
            for attr, value in elem.attrib.items():
                if attr.endswith(issue['attribute']):
                    # Handle viewBox format
                    if attr == 'viewBox':
                        parts = re.findall(r'-?\d+(?:\.\d+)?', value)
                        if len(parts) >= 4:
                            elem.attrib[attr] = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}"
                            return True
                    # Handle color format
                    elif attr in ['fill', 'stroke', 'color', 'stop-color']:
                        if not value.startswith('#') and not value.startswith('rgb'):
                            elem.attrib[attr] = "#000000"  # Default to black
                            return True
                    # Handle path data
                    elif attr == 'd' and value:
                        if not value.strip().startswith(('M', 'm')):
                            elem.attrib[attr] = f"M0,0 {value}"
                            return True
        return False
    
    def _add_required_attributes(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Add required attributes to elements."""
        if "element" not in issue or "missing_attributes" not in issue:
            return False
            
        # Find target element
        target_elem = None
        if issue["element"] == "svg":
            target_elem = root
        else:
            elems = root.findall(f".//{{{SVGNS}}}{issue['element']}") + root.findall(f".//{issue['element']}")
            if elems:
                target_elem = elems[0]
                
        if target_elem is not None:
            # Add missing attributes with sensible defaults
            for attr in issue["missing_attributes"]:
                if attr == "width":
                    target_elem.set("width", "100%")
                elif attr == "height":
                    target_elem.set("height", "100%")
                elif attr == "viewBox":
                    target_elem.set("viewBox", "0 0 100 100")
            return True
        return False
    
    def _add_namespace(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Add SVG namespace to the root element."""
        # This requires recreating the root element due to ElementTree limitations
        attrs = {"xmlns": SVGNS, "xmlns:xlink": XLINKNS}
        for k, v in root.attrib.items():
            if not k.startswith("xmlns"):
                attrs[k] = v
                
        new_root = ET.Element("{%s}svg" % SVGNS, attrs)
        for child in list(root):
            root.remove(child)
            new_root.append(child)
            
        # Replace the root's contents in-place since we can't return a new tree
        root.tag = new_root.tag
        root.attrib.clear()
        root.attrib.update(new_root.attrib)
            
        return True
    
    def _correct_viewbox(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Correct invalid viewBox format."""
        if issue["element"] != "svg":
            return False
            
        # Extract any numbers from the current viewBox
        viewbox = root.get("viewBox", "")
        numbers = re.findall(r'-?\d+(?:\.\d+)?', viewbox)
        
        if len(numbers) >= 4:
            # Use the numbers but format them correctly
            root.set("viewBox", f"{numbers[0]} {numbers[1]} {numbers[2]} {numbers[3]}")
        else:
            # Default viewBox if we can't extract enough numbers
            root.set("viewBox", "0 0 100 100")
            
        return True
    
    def _correct_path_data(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Correct invalid path data."""
        # Find all path elements
        paths = root.findall(".//{%s}path" % SVGNS) + root.findall(".//path")
        
        for path in paths:
            if 'd' in path.attrib:
                d_attr = path.attrib['d']
                if d_attr.strip() and not d_attr.startswith(('M', 'm')):
                    # Add a move command at the beginning
                    path.attrib['d'] = f"M0,0 {d_attr}"
                    return True
        return False
    
    def _add_title(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Add a title element for accessibility."""
        # Check if there's already a title
        titles = root.findall(".//{%s}title" % SVGNS) + root.findall(".//title")
        if titles:
            return False
            
        # Create and add the title element as the first child of svg
        title = ET.SubElement(root, "{%s}title" % SVGNS)
        title.text = "SVG Image"  # Default generic title
        
        # Move the title to be the first child
        root.remove(title)
        root.insert(0, title)
        
        return True
    
    def _add_description(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Add a description element for accessibility."""
        # Check if there's already a desc
        descs = root.findall(".//{%s}desc" % SVGNS) + root.findall(".//desc")
        if descs:
            return False
            
        # Create and add the desc element as the second child of svg (after title)
        desc = ET.SubElement(root, "{%s}desc" % SVGNS)
        desc.text = "A vector graphic image."  # Default generic description
        
        # Position after title if it exists
        titles = root.findall(".//{%s}title" % SVGNS) + root.findall(".//title")
        if titles and titles[0].getparent() == root:
            index = list(root).index(titles[0])
            root.remove(desc)
            root.insert(index + 1, desc)
        else:
            # Otherwise make it the first child
            root.remove(desc)
            root.insert(0, desc)
        
        return True
    
    def _add_title_content(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Add content to an empty title element."""
        titles = root.findall(".//{%s}title" % SVGNS) + root.findall(".//title")
        for title in titles:
            if not title.text or not title.text.strip():
                title.text = "SVG Image"  # Default generic title
                return True
        return False
    
    def _add_description_content(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Add content to an empty desc element."""
        descs = root.findall(".//{%s}desc" % SVGNS) + root.findall(".//desc")
        for desc in descs:
            if not desc.text or not desc.text.strip():
                desc.text = "A vector graphic image."  # Default generic description
                return True
        return False
    
    def _add_aria_attributes(self, root: ET.Element, issue: Dict[str, Any]) -> bool:
        """Add ARIA attributes to interactive elements."""
        if "element" not in issue:
            return False
            
        if issue["element"] in ["a", "text", "button"]:
            elements = root.findall(f".//{{{SVGNS}}}{issue['element']}") + root.findall(f".//{issue['element']}")
            for elem in elements:
                if not any(attr.startswith('aria-') for attr in elem.attrib):
                    # Add appropriate aria attributes based on element type
                    if issue["element"] == "a":
                        elem.set("aria-label", "Link")
                    elif issue["element"] == "text":
                        elem.set("aria-label", elem.text if elem.text else "Text")
                    elif issue["element"] == "button":
                        elem.set("aria-label", "Button")
                    return True
        return False
    
    def _generate_semantic_feedback(self, corrections: List[Dict[str, Any]]) -> str:
        """Generate human-readable feedback about corrections."""
        if not corrections:
            return "No corrections were needed or possible."
            
        feedback = "Applied the following corrections:\n"
        
        # Group corrections by type for cleaner output
        correction_types = {}
        for corr in corrections:
            if corr["type"] not in correction_types:
                correction_types[corr["type"]] = []
            correction_types[corr["type"]].append(corr)
        
        # Build feedback message
        for corr_type, corrs in correction_types.items():
            if len(corrs) == 1:
                feedback += f"- {corrs[0]['details']}\n"
            else:
                feedback += f"- Fixed {len(corrs)} issues of type '{corr_type}'\n"
        
        return feedback
# - Element generators for each visual component
# - Parametric generation of geometric primitives
# - Style calculation based on semantic properties
# - Composition of elements with proper layering

class SVGElementGenerator:
    """Base class for generating SVG elements from scene nodes."""
    
    def __init__(self, document: 'SVGDocument'):
        self.document = document
    
    def generate(self, node: SceneNode) -> Optional[ET.Element]:
        """Generate SVG element(s) for a scene node.
        
        Args:
            node: Scene node to render
            
        Returns:
            Generated SVG element or None if not applicable
        """
        raise NotImplementedError("Subclasses must implement this method")

class SkyGenerator(SVGElementGenerator):
    """Generator for sky elements in SVG illustrations."""
    
    def __init__(self):
        super().__init__()
        # Color palettes for different times of day
        self.color_palettes = { \
            "dawn": ["#FF7E6B", "#FFC371", "#FFE5A8", "#E0F7FF"],
            "sunrise": ["#FF7E50", "#FFA755", "#FFCF75", "#E0F7FF"], \
            "morning": ["#87CEEB", "#C6E6FB", "#E0F7FF", "#FFFFFF"], \
            "midday": ["#1E90FF", "#47A0FF", "#87CEEB", "#FFFFFF"], \
            "afternoon": ["#4682B4", "#47A0FF", "#87CEEB", "#E0F7FF"], \
            "golden hour": ["#FF9E2C", "#FFBF69", "#FFD89C", "#FFF4E0"], \
            "sunset": ["#FF5E62", "#FF9966", "#FFCA7A", "#FFF4E0"], \
            "dusk": ["#614385", "#516395", "#6B8CCA", "#A7BFE8"], \
            "night": ["#0F2027", "#203A43", "#2C5364", "#3A6073"], \
            "blue hour": ["#2C3E50", "#3F5972", "#546E8F", "#7FA1C1"]
        }
        
        # Weather effect modifiers
        self.weather_effects = { \
            "clear": {"brightness": 1.0, "contrast": 1.0},
            "partly cloudy": {"brightness": 0.9, "contrast": 0.95}, \
            "cloudy": {"brightness": 0.8, "contrast": 0.85}, \
            "overcast": {"brightness": 0.7, "contrast": 0.75}, \
            "foggy": {"brightness": 0.65, "saturation": 0.7}, \
            "rainy": {"brightness": 0.6, "contrast": 0.8}, \
            "stormy": {"brightness": 0.5, "contrast": 1.1}, \
            "snowy": {"brightness": 0.9, "contrast": 0.8}, \
        }
    
    def generate(self, node: SceneNode) -> Optional[ET.Element]:
        """Generate sky element for a scene.
        
        Args:
            node: Sky node from scene graph
            
        Returns:
            SVG element representing the sky
        """
        # Create gradient for sky
        gradient_id = f"sky_gradient_{random.randint(1000, 9999)}"
        gradient = ET.Element("linearGradient", { \
            "id": gradient_id,
            "x1": "0%", \
            "y1": "0%", \
            "x2": "0%", \
            "y2": "100%"
        })
        
        # Get sky properties
        time_of_day = node.properties.get("time_of_day", "midday")
        weather = node.properties.get("weather", "clear")
        mood = node.properties.get("mood", "neutral")
        
        # Get color palette for time of day
        palette = self.color_palettes.get(time_of_day, self.color_palettes["midday"])
        
        # Apply weather effects
        weather_effect = self.weather_effects.get(weather, {"brightness": 1.0, "contrast": 1.0})
        brightness = weather_effect.get("brightness", 1.0)
        
        # Adjust colors based on mood
        if mood == "dramatic":
            brightness *= 0.9  # Darker for dramatic mood
        elif mood == "peaceful" or mood == "tranquil":
            brightness *= 1.1  # Brighter for peaceful mood
            
        # Create gradient stops with multiple colors for more realistic sky
        num_colors = len(palette)
        for i, color in enumerate(palette):
            # Adjust color based on weather and mood
            adjusted_color = self._adjust_color(color, brightness)
            
            # Create stop element
            stop = ET.Element("stop", { \
                "offset": f"{100 * i / (num_colors-1)}%",
                "stop-color": adjusted_color
            })
            gradient.append(stop)
        
        # Create sky rectangle
        rect = ET.Element("rect", { \
            "x": "0",
            "y": "0", \
            "width": str(node.size[0]), \
            "height": str(node.size[1]), \
            "fill": f"url(#{gradient_id})"
        })
        
        # Group sky elements
        group = ET.Element("g", {"class": "sky"})
        group.append(gradient)
        group.append(rect)
        
        # Add clouds if appropriate for the weather
        if weather in ["partly cloudy", "cloudy", "overcast"]:
            clouds = self._generate_clouds(node.size[0], node.size[1], weather, time_of_day)
            for cloud in clouds:
                group.append(cloud)
        
        # Add sun or moon if appropriate
        if weather not in ["overcast", "foggy", "rainy", "stormy"] and random.random() < 0.8:
            celestial_body = self._generate_celestial_body(node.size[0], node.size[1], time_of_day)
            if celestial_body is not None:
                group.append(celestial_body)
                
        # Add stars at night
        if time_of_day in ["night", "dusk"] and weather != "overcast":
            stars = self._generate_stars(node.size[0], node.size[1])
            group.append(stars)
        
        return group
    
    def _adjust_color(self, color: str, brightness: float) -> str:
        """Adjust color brightness for weather effects."""
        # Convert hex to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        
        # Apply brightness
        r = min(255, int(r * brightness))
        g = min(255, int(g * brightness))
        b = min(255, int(b * brightness))
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _generate_clouds(self, width: float, height: float, weather: str, time_of_day: str) -> List[ET.Element]:
        """Generate cloud elements based on weather conditions."""
        clouds = []
        
        # Determine cloud parameters based on weather
        num_clouds = { \
            "partly cloudy": random.randint(2, 4),
            "cloudy": random.randint(4, 7), \
            "overcast": random.randint(7, 10)
        }.get(weather, 0)
        
        # Determine cloud color based on time of day
        base_cloud_color = "#FFFFFF"  # Default white
        if time_of_day in ["sunset", "golden hour"]:
            base_cloud_color = "#FFE6B5"  # Golden tint
        elif time_of_day in ["night", "blue hour"]:
            base_cloud_color = "#BBCCE2"  # Bluish-gray
        
        # Cloud opacity based on weather
        cloud_opacity = { \
            "partly cloudy": 0.7,
            "cloudy": 0.8, \
            "overcast": 0.9
        }.get(weather, 0.7)
        
        # Generate clouds
        for i in range(num_clouds):
            # Randomize cloud position
            cloud_x = random.uniform(0, width)
            cloud_y = random.uniform(0, height * 0.4)
            
            # Randomize cloud size
            cloud_width = random.uniform(width * 0.1, width * 0.25)
            cloud_height = random.uniform(height * 0.05, height * 0.1)
            
            # Create cloud group
            cloud = ET.Element("g", {"class": "cloud"})
            
            # Generate cloud shape (multiple overlapping circles)
            num_circles = random.randint(4, 7)
            for j in range(num_circles):
                # Position circles to form a cloud shape
                cx = cloud_x + random.uniform(-cloud_width * 0.2, cloud_width * 0.2)
                cy = cloud_y + random.uniform(-cloud_height * 0.2, cloud_height * 0.2)
                
                # Size varies for each circle
                r = random.uniform(cloud_width * 0.15, cloud_width * 0.3)
                
                # Create cloud circle
                circle = ET.Element("circle", { \
                    "cx": str(cx),
                    "cy": str(cy), \
                    "r": str(r), \
                    "fill": base_cloud_color, \
                    "fill-opacity": str(cloud_opacity)
                })
                
                cloud.append(circle)
                
            clouds.append(cloud)
            
        return clouds
    
    def _generate_celestial_body(self, width: float, height: float, time_of_day: str) -> Optional[ET.Element]:
        """Generate sun or moon based on time of day."""
        # Determine if sun or moon should be shown
        show_sun = time_of_day in ["dawn", "sunrise", "morning", "midday", "afternoon", "golden hour", "sunset"]
        show_moon = time_of_day in ["dusk", "night", "blue hour"]
        
        if not (show_sun or show_moon):
            return None
            
        # Position for celestial body
        if time_of_day in ["dawn", "sunrise"]:
            pos_x = width * 0.2
            pos_y = height * 0.2
        elif time_of_day in ["sunset", "dusk"]:
            pos_x = width * 0.8
            pos_y = height * 0.2
        else:
            pos_x = random.uniform(width * 0.3, width * 0.7)
            pos_y = random.uniform(height * 0.1, height * 0.3)
            
        # Size for celestial body (proportional to canvas)
        size = min(width, height) * random.uniform(0.05, 0.08)
        
        if show_sun:
            # Create sun group
            sun_group = ET.Element("g", {"class": "sun"})
            
            # Define sun gradient
            sun_gradient_id = f"sun_gradient_{random.randint(1000, 9999)}"
            sun_gradient = ET.Element("radialGradient", { \
                "id": sun_gradient_id,
                "cx": "50%", \
                "cy": "50%", \
                "r": "50%", \
                "fx": "45%", \
                "fy": "45%"
            })
            
            # Sun colors based on time of day
            if time_of_day in ["sunrise", "dawn"]:
                core_color = "#FFCC33"
                outer_color = "#FF7700"
            elif time_of_day in ["sunset", "golden hour"]:
                core_color = "#FFAA00"
                outer_color = "#FF5500"
            else:
                core_color = "#FFFF00"
                outer_color = "#FFCC00"
                
            # Create gradient stops
            stop1 = ET.Element("stop", { \
                "offset": "0%",
                "stop-color": core_color
            })
            stop2 = ET.Element("stop", { \
                "offset": "100%",
                "stop-color": outer_color
            })
            
            sun_gradient.append(stop1)
            sun_gradient.append(stop2)
            
            # Create sun disc
            sun = ET.Element("circle", { \
                "cx": str(pos_x),
                "cy": str(pos_y), \
                "r": str(size), \
                "fill": f"url(#{sun_gradient_id})"
            })
            
            sun_group.append(sun_gradient)
            sun_group.append(sun)
            
            # Add rays for the sun
            if time_of_day not in ["night", "dusk"]:
                num_rays = 12
                ray_length = size * 0.5
                
                for i in range(num_rays):
                    angle = 2 * math.pi * i / num_rays
                    ray_x1 = pos_x + size * math.cos(angle)
                    ray_y1 = pos_y + size * math.sin(angle)
                    ray_x2 = pos_x + (size + ray_length) * math.cos(angle)
                    ray_y2 = pos_y + (size + ray_length) * math.sin(angle)
                    
                    ray = ET.Element("line", { \
                        "x1": str(ray_x1),
                        "y1": str(ray_y1), \
                        "x2": str(ray_x2), \
                        "y2": str(ray_y2), \
                        "stroke": core_color, \
                        "stroke-width": str(size * 0.05), \
                        "opacity": "0.7"
                    })
                    
                    sun_group.append(ray)
            
            return sun_group
            
        elif show_moon:
            # Create moon
            moon_group = ET.Element("g", {"class": "moon"})
            
            # Moon color based on time
            moon_color = "#FFFAFA"  # Snow white
            
            # Create moon
            moon = ET.Element("circle", { \
                "cx": str(pos_x),
                "cy": str(pos_y), \
                "r": str(size), \
                "fill": moon_color
            })
            
            # Create crescent effect
            if random.random() < 0.2:
                # Create second circle that overlaps to create crescent
                crescent_size = size * 0.8
                offset = size * 0.3
                crescent = ET.Element("circle", { \
                    "cx": str(pos_x + offset),
                    "cy": str(pos_y), \
                    "r": str(crescent_size), \
                    "fill": "#87CEEB"  # Sky color
                })
                
                # Create clip path
                clip_id = f"moon_clip_{random.randint(1000, 9999)}"
                clip_path = ET.Element("clipPath", {"id": clip_id})
                clip_circle = ET.Element("circle", { \
                    "cx": str(pos_x),
                    "cy": str(pos_y), \
                    "r": str(size)
                })
                clip_path.append(clip_circle)
                
                # Apply clip path to crescent
                crescent.set("clip-path", f"url(#{clip_id})")
                
                moon_group.append(clip_path)
                moon_group.append(moon)
                moon_group.append(crescent)
            else:
                # Add crater details to full moon
                num_craters = random.randint(3, 6)
                for _ in range(num_craters):
                    # Random position within moon
                    crater_x = pos_x + random.uniform(-size * 0.6, size * 0.6)
                    crater_y = pos_y + random.uniform(-size * 0.6, size * 0.6)
                    
                    # Random radius for crater
                    crater_radius = size * random.uniform(0.05, 0.15)
                    
                    # Create crater
                    crater = ET.Element("circle", { \
                        "cx": str(crater_x),
                        "cy": str(crater_y), \
                        "r": str(crater_radius), \
                        "fill": "#E6E6E6", \
                        "opacity": "0.7"
                    })
                    
                    moon_group.append(crater)
                
                moon_group.append(moon)
            
            return moon_group
        
        return None
        
    def _generate_stars(self, width: float, height: float) -> ET.Element:
        """Generate stars for night scenes."""
        star_group = ET.Element("g", {"class": "stars"})
        
        # Number of stars
        num_stars = random.randint(30, 100)
        
        for _ in range(num_stars):
            # Random position
            star_x = random.uniform(0, width)
            star_y = random.uniform(0, height * 0.6)
            
            # Random size
            star_size = random.uniform(0.5, 2.0)
            
            # Random opacity
            opacity = random.uniform(0.5, 1.0)
            
            # Create star circle
            star = ET.Element("circle", { \
                "cx": str(star_x),
                "cy": str(star_y), \
                "r": str(star_size), \
                "fill": "#FFFFFF", \
                "opacity": str(opacity)
            })
            
            # Random twinkle effect (add some small stars with higher opacity)
            if random.random() < 0.2:
                glow = ET.Element("circle", { \
                    "cx": str(star_x),
                    "cy": str(star_y), \
                    "r": str(star_size * 1.5), \
                    "fill": "#FFFFFF", \
                    "opacity": str(opacity * 0.3)
                })
                star_group.append(glow)
            
            star_group.append(star)
        
        return star_group
            
    def generate_sky(self, node: SceneNode) -> Optional[ET.Element]:
        """Generate a sky element.
        
        Args:
            node: Sky node from scene graph
            
        Returns:
            SVG element representing the sky
        """
        width = node.size[0]
        height = node.size[1]
        x = node.position[0]
        y = node.position[1]
        
        # Create sky rectangle
        rect = ET.Element("{%s}rect" % SVGNS)
        rect.set("x", str(node.position[0]))
        rect.set("y", str(node.position[1]))
        rect.set("width", str(width))
        rect.set("height", str(height))
        rect.set("fill", node.properties.get("color", "#87CEEB"))  # Default sky blue
        return rect

class MountainGenerator(SVGElementGenerator):
    """Generates realistic mountain shapes."""
    
    def generate(self, node: SceneNode) -> Optional[ET.Element]:
        """Generate a mountain element with path.
        
        Args:
            node: Mountain node from scene graph
            
        Returns:
            SVG element representing the mountain
        """
        width = node.size[0]
        height = node.size[1]
        x = node.position[0]
        y = node.position[1]
        
        # Create mountain group
        mountain_group = ET.Element("g", {"class": "mountain"})
        
        # Get mountain properties
        roughness = node.properties.get("roughness", 0.5)
        is_sharp = node.properties.get("sharp", False)
        has_snow = node.properties.get("snow", random.random() > 0.6)
        snow_level = node.properties.get("snow_level", 0.3)  # How far down the snow goes
        mountain_type = node.properties.get("type", "rocky")
        time_of_day = node.properties.get("time_of_day", "midday")
        
        # Define base points - we'll use these for both the mountain and the snow cap
        points = []
        
        # Left base point
        points.append((x, y + height))
        
        # Generate intermediate points for the mountain ridge with higher detail
        segments = int(30 + 40 * roughness)  # More segments for rougher mountains
        
        # Keep track of peak points for later use in snow caps
        peak_points = []
        
        # Create variations in the mountain ridge
        for i in range(1, segments):
            segment_x = x + (i * width / segments)
            
            # For sharp mountains, create more pronounced peaks
            if is_sharp and 0.4 < i / segments < 0.6:
                # Create sharper central peak
                variance = height * 0.2 * roughness
                segment_y = y + max(0, random.gauss(height * 0.15, variance))
                peak_points.append((segment_x, segment_y))
            else:
                # Normal ridge with random variations
                # Higher variance near the middle for more realistic mountains
                center_factor = 1.0 - 2.0 * abs(i / segments - 0.5)
                variance = height * 0.4 * roughness * center_factor
                base_height = height * (0.5 - 0.4 * center_factor)
                segment_y = y + base_height + random.gauss(0, variance)
                
                # Add significant points to peak points for snow cap
                if segment_y < y + height * 0.4:
                    peak_points.append((segment_x, segment_y))
            
            points.append((segment_x, segment_y))
        
        # Right base point
        points.append((x + width, y + height))
        
        # Create path data
        path_data = f"M{points[0][0]},{points[0][1]} "
        
        # Add cubic bezier curves between points for smooth mountain profile
        for i in range(1, len(points) - 2, 2):
            p1 = points[i]
            p2 = points[i+1] if i+1 < len(points) else points[-1]
            
            # Control points for curve - adjusted for more natural-looking mountains
            cp1x = p1[0] + (p2[0] - p1[0]) / 3
            cp1y = p1[1]
            cp2x = p1[0] + 2 * (p2[0] - p1[0]) / 3
            cp2y = p2[1]
            
            # Add cubic bezier command
            path_data += f"C {cp1x},{cp1y} {cp2x},{cp2y} {p2[0]},{p2[1]} "
        
        # Close the path
        path_data += "Z"
        
        # Create mountain gradient for more realistic shading
        gradient_id = f"mountain_gradient_{random.randint(1000, 9999)}"
        gradient = ET.Element("linearGradient", { \
            "id": gradient_id,
            "x1": "0%", \
            "y1": "0%", \
            "x2": "100%", \
            "y2": "0%"
        })
        
        # Base mountain colors based on type
        if mountain_type == "rocky":
            base_color = "#6E7178"  # Gray-blue
            shadow_color = "#59646E"  # Darker gray-blue
        elif mountain_type == "desert":
            base_color = "#BA8C63"  # Sandy brown
            shadow_color = "#A67D51"  # Darker sandy brown
        elif mountain_type == "volcanic":
            base_color = "#3D3535"  # Dark gray
            shadow_color = "#332A2A"  # Darker gray
        else:
            # Default rocky mountains
            base_color = "#6E7178"
            shadow_color = "#59646E"
        
        # Adjust lighting direction based on time of day
        if time_of_day in ["dawn", "sunrise", "morning"]:
            # Light from right
            gradient.set("x1", "100%")
            gradient.set("x2", "0%")
            light_color = self._lighten_color(base_color, 1.2)
        elif time_of_day in ["sunset", "dusk", "golden hour"]:
            # Light from left with warm tint
            light_color = self._blend_colors(base_color, "#FF7F50", 0.3)  # Blend with warm orange
        else:
            # Midday lighting
            light_color = base_color
        
        # Create gradient stops
        stop1 = ET.Element("stop", { \
            "offset": "0%",
            "stop-color": light_color
        })
        
        stop2 = ET.Element("stop", { \
            "offset": "100%",
            "stop-color": shadow_color
        })
        
        gradient.append(stop1)
        gradient.append(stop2)
        
        # Create SVG path element
        mountain_path = ET.Element("path", { \
            "d": path_data,
            "fill": f"url(#{gradient_id})"
        })
        
        # Add ridge lines for texture and detail
        num_ridges = random.randint(3, 7)
        for i in range(num_ridges):
            ridge_path_data = self._generate_ridge_line(x, y, width, height, points, roughness)
            ridge = ET.Element("path", { \
                "d": ridge_path_data,
                "fill": "none", \
                "stroke": shadow_color, \
                "stroke-width": "1", \
                "stroke-opacity": "0.3"
            })
            mountain_group.append(ridge)
        
        # Add snow cap if needed
        if has_snow and len(peak_points) > 0:
            snow_path_data = self._generate_snow_cap(peak_points, snow_level, height, y)
            snow_cap = ET.Element("path", { \
                "d": snow_path_data,
                "fill": "#FFFFFF", \
                "fill-opacity": "0.9"
            })
            
            # Add snow details
            snow_highlights = ET.Element("path", { \
                "d": snow_path_data,
                "fill": "none", \
                "stroke": "#FFFFFF", \
                "stroke-width": "2", \
                "stroke-opacity": "0.5", \
                "filter": "blur(2px)"
            })
            
            mountain_group.append(snow_cap)
            mountain_group.append(snow_highlights)
        
        # Add subtle texture overlay for rocks
        if mountain_type == "rocky":
            texture_id = f"mountain_texture_{random.randint(1000, 9999)}"
            texture_filter = ET.Element("filter", { \
                "id": texture_id,
                "x": "0", \
                "y": "0", \
                "width": "100%", \
                "height": "100%"
            })
            
            # Add noise
            turbulence = ET.Element("feTurbulence", { \
                "type": "fractalNoise",
                "baseFrequency": "0.05", \
                "numOctaves": "3", \
                "seed": str(random.randint(1, 100)), \
                "result": "noise"
            })
            
            # Make the noise subtle
            composite = ET.Element("feComposite", { \
                "in": "SourceGraphic",
                "in2": "noise", \
                "operator": "arithmetic", \
                "k1": "0", \
                "k2": "0.1", \
                "k3": "0.9", \
                "k4": "0"
            })
            
            texture_filter.append(turbulence)
            texture_filter.append(composite)
            
            # Apply texture to mountain
            mountain_path.set("filter", f"url(#{texture_id})")
            mountain_group.append(texture_filter)
        
        # Assemble the final mountain group
        mountain_group.append(gradient)
        mountain_group.append(mountain_path)
        
        return mountain_group
    
    def _generate_ridge_line(self, x: float, y: float, width: float, height: float, \
                            mountain_points: List[Tuple[float, float]], roughness: float) -> str:
        """Generate a ridge line to add texture to the mountain."""
        # Start at a random point on the mountain (not at the base)
        start_idx = random.randint(int(len(mountain_points) * 0.2), int(len(mountain_points) * 0.8))
        start_x, start_y = mountain_points[start_idx]
        
        # Ridge goes down and to a random side
        direction = 1 if random.random() > 0.5 else -1
        end_x = start_x + direction * width * random.uniform(0.1, 0.3)
        end_y = start_y + height * random.uniform(0.1, 0.4)
        
        # Ensure end point is within mountain width
        end_x = max(x, min(x + width, end_x))
        
        # Create ridge path with some variation
        path_data = f"M {start_x},{start_y}"
        
        # Number of points in the ridge
        num_points = int(5 + random.random() * 5)
        
        # Generate points for the ridge
        for i in range(1, num_points + 1):
            point_x = start_x + (end_x - start_x) * (i / num_points)
            base_y = start_y + (end_y - start_y) * (i / num_points)
            
            # Add some randomness to the ridge line
            variation = random.uniform(-10, 10) * roughness
            point_y = base_y + variation
            
            # Add line segment
            path_data += f" L {point_x},{point_y}"
        
        return path_data
    
    def _generate_snow_cap(self, peak_points: List[Tuple[float, float]], \
                            snow_level: float, height: float, base_y: float) -> str:
        """Generate a snow cap on the mountain peaks."""
        if not peak_points:
            return ""
        
        # Sort peak points by x coordinate
        peak_points.sort(key=lambda p: p[0])
        
        # Filter points that are high enough for snow
        snow_points = [p for p in peak_points if p[1] < base_y + height * snow_level]
        
        if not snow_points:
            return ""
        
        # Create a more natural snow line with some variations
        snow_path = f"M {snow_points[0][0]},{snow_points[0][1]}"
        
        for i in range(1, len(snow_points)):
            p1 = snow_points[i-1]
            p2 = snow_points[i]
            
            # Create control points for a quadratic Bezier curve
            control_x = (p1[0] + p2[0]) / 2
            control_y = (p1[1] + p2[1]) / 2 - random.uniform(5, 15)  # Move control point up for a rounded cap
            
            # Add curve to path
            snow_path += f" Q {control_x},{control_y} {p2[0]},{p2[1]}"
        
        # Close the path with a custom shape
        # Add additional points to create a natural snow shape
        snow_path += f" L {snow_points[-1][0] + 20},{snow_points[-1][1] + 50}"
        snow_path += f" Q {(snow_points[-1][0] + snow_points[0][0])/2},{snow_points[0][1] + 30} {snow_points[0][0] - 20},{snow_points[0][1] + 40}"
        snow_path += " Z"
        
        return snow_path
    
    def _lighten_color(self, hex_color: str, factor: float) -> str:
        """Lighten a hex color by a factor (>1 lightens, <1 darkens)."""
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # Lighten each component
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _blend_colors(self, color1: str, color2: str, ratio: float) -> str:
        """Blend two hex colors together with the given ratio (0-1)."""
        # Convert hex to RGB
        r1 = int(color1[1:3], 16)
        g1 = int(color1[3:5], 16)
        b1 = int(color1[5:7], 16)
        
        r2 = int(color2[1:3], 16)
        g2 = int(color2[3:5], 16)
        b2 = int(color2[5:7], 16)
        
        # Blend colors
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        
        # Ensure values are in valid range
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"
        


class SeaGenerator(SVGElementGenerator):
    """Generates realistic sea and water elements."""
    
    def generate(self, node: SceneNode) -> Optional[ET.Element]:
        """Generate a sea/water element with waves and reflections.
        
        Args:
            node: Sea node from scene graph
            
        Returns:
            SVG element representing the sea
        """
        # Get position and size
        width = node.size[0]
        height = node.size[1]
        x = node.position[0]
        y = node.position[1]
        
        # Get sea properties
        wave_intensity = node.properties.get("wave_intensity", 0.3)
        sea_type = node.properties.get("type", "ocean")
        time_of_day = node.properties.get("time_of_day", "midday")
        wind_direction = node.properties.get("wind_direction", "left")  # or "right"
        
        # Create main water group
        sea_group = ET.Element("g", {"class": "sea"})
        
        # Create water gradient
        gradient_id = f"sea_gradient_{random.randint(1000, 9999)}"
        gradient = ET.Element("linearGradient", { \
            "id": gradient_id,
            "x1": "0%", \
            "y1": "0%", \
            "x2": "0%", \
            "y2": "100%"
        })
        
        # Colors depend on sea type and time of day
        if sea_type == "ocean":
            if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                # Warm colors reflecting the sky
                top_color = "#3A7BD5"  # Blue
                bottom_color = "#2E5E8C"  # Darker blue
                reflection_color = "#FF7E50"  # Orange/amber
            elif time_of_day == "night":
                top_color = "#1A3A63"  # Dark blue
                bottom_color = "#0C1D3B"  # Very dark blue
                reflection_color = "#C0C0C0"  # Silver (moonlight)
            else:  # Default day colors
                top_color = "#3A7BD5"  # Blue
                bottom_color = "#00487C"  # Deep blue
                reflection_color = "#FFFFFF"  # White
        elif sea_type == "lake":
            top_color = "#61A0C2"  # Light blue
            bottom_color = "#3E8BB2"  # Medium blue
            reflection_color = "#FFFFFF"  # White
        elif sea_type == "river":
            top_color = "#69B7CE"  # Teal blue
            bottom_color = "#5296B2"  # Darker teal
            reflection_color = "#E8F5FF"  # Very light blue
        else:  # Default ocean
            top_color = "#3A7BD5"
            bottom_color = "#00487C"
            reflection_color = "#FFFFFF"
        
        # Create gradient stops
        stop1 = ET.Element("stop", { \
            "offset": "0%",
            "stop-color": top_color
        })
        
        stop2 = ET.Element("stop", { \
            "offset": "100%",
            "stop-color": bottom_color
        })
        
        gradient.append(stop1)
        gradient.append(stop2)
        
        # Create base water rectangle
        water_rect = ET.Element("rect", { \
            "x": str(x),
            "y": str(y), \
            "width": str(width), \
            "height": str(height), \
            "fill": f"url(#{gradient_id})"
        })
        
        # Create waves if intensity > 0
        wave_layers = []
        if wave_intensity > 0:
            num_waves = int(3 + wave_intensity * 5)  # More waves for higher intensity
            
            for i in range(num_waves):
                wave_height = height * 0.05 * wave_intensity * (num_waves - i) / num_waves
                wave_y = y + (i * height / num_waves)
                
                # Create wave path
                wave_path = self._generate_wave_path( \
                    x, wave_y, width, wave_height,
                    wave_intensity, wind_direction
                )
                
                # Calculate opacity based on wave position (lower waves are more transparent)
                opacity = 0.5 - (i / num_waves) * 0.3
                
                # Create wave element
                wave = ET.Element("path", { \
                    "d": wave_path,
                    "fill": "none", \
                    "stroke": "#FFFFFF", \
                    "stroke-width": str(1 + wave_intensity * 2), \
                    "stroke-opacity": str(opacity), \
                    "filter": "blur(1px)"
                })
                
                wave_layers.append(wave)
        
        # Create reflections if applicable
        reflections = []
        num_reflections = random.randint(2, 5) if wave_intensity < 0.5 else 0
        
        for i in range(num_reflections):
            # Position reflection randomly across the water surface
            reflection_width = width * random.uniform(0.05, 0.2) * (1 + wave_intensity)
            reflection_x = x + random.uniform(0, width - reflection_width)
            reflection_y = y + random.uniform(0, height * 0.7)
            
            # Create the reflection shape (elongated ellipse)
            reflection = ET.Element("ellipse", { \
                "cx": str(reflection_x + reflection_width/2),
                "cy": str(reflection_y), \
                "rx": str(reflection_width/2), \
                "ry": str(reflection_width/6), \
                "fill": reflection_color, \
                "opacity": str(random.uniform(0.1, 0.3))
            })
            
            reflections.append(reflection)
        
        # Add ripple effects for water
        if sea_type in ["lake", "river"] or wave_intensity < 0.3:
            ripple_id = f"ripple_effect_{random.randint(1000, 9999)}"
            ripple_filter = ET.Element("filter", { \
                "id": ripple_id,
                "x": "0", \
                "y": "0", \
                "width": "100%", \
                "height": "100%"
            })
            
            # Create turbulence for ripples
            turbulence = ET.Element("feTurbulence", { \
                "type": "fractalNoise",
                "baseFrequency": f"{0.01 + wave_intensity * 0.05} {0.005 + wave_intensity * 0.01}", \
                "numOctaves": "2", \
                "seed": str(random.randint(1, 100)), \
                "result": "turbulence"
            })
            
            # Displacement map for moving turbulence
            displacement = ET.Element("feDisplacementMap", { \
                "in": "SourceGraphic",
                "in2": "turbulence", \
                "scale": str(3 + wave_intensity * 10), \
                "xChannelSelector": "R", \
                "yChannelSelector": "G"
            })
            
            ripple_filter.append(turbulence)
            ripple_filter.append(displacement)
            
            # Apply ripple effect to water
            water_rect.set("filter", f"url(#{ripple_id})")
            sea_group.append(ripple_filter)
        
        # Assemble the final sea group
        sea_group.append(gradient)
        sea_group.append(water_rect)
        
        # Add reflections
        for reflection in reflections:
            sea_group.append(reflection)
        
        # Add waves (in reverse order so smaller waves appear in front)
        for wave in reversed(wave_layers):
            sea_group.append(wave)
        
        return sea_group
    
    def _generate_wave_path(self, x: float, y: float, width: float, height: float, \
                            wave_intensity: float, wind_direction: str) -> str:
        """Generate a wavy path for the water surface."""
        # Start at left edge
        path = f"M {x},{y}"
        
        # Number of wave segments
        num_segments = int(10 + 20 * wave_intensity)
        segment_width = width / num_segments
        
        # Wave direction adjustment
        direction_factor = 1 if wind_direction == "left" else -1
        
        # Generate wave points
        for i in range(1, num_segments + 1):
            # Calculate wave position
            segment_x = x + i * segment_width
            
            # Wave height varies with position and randomness
            wave_factor = math.sin(i * math.pi / (num_segments/2) + direction_factor * math.pi/4)
            random_factor = random.uniform(0.7, 1.3) * wave_intensity
            segment_y = y + height * wave_factor * random_factor
            
            # Add curve to wave (use quadratic Bezier curves for smoother waves)
            control_x = x + (i - 0.5) * segment_width
            control_y = y + height * -wave_factor * random_factor * 0.5  # Control point in opposite direction
            
            path += f" Q {control_x},{control_y} {segment_x},{segment_y}"
        
        return path
        
        # Create waves if intensity > 0
        if wave_intensity > 0:
            # Add wave pattern
            wave_path_data = f"M{x},{y} "
            
            # Number of waves across the width
            wave_count = int(10 + 20 * wave_intensity)
            wave_width = width / wave_count
            amplitude = wave_intensity * 10  # Wave height
            
            # Create each wave segment
            for i in range(wave_count + 1):
                wave_x = x + i * wave_width
                # Use sine function to create natural wave pattern
                wave_y = y + amplitude * math.sin(i * math.pi / 2)
                
                if i == 0:
                    wave_path_data += f"M{wave_x},{wave_y} "
                else:
                    # Create smooth curved waves
                    cp1x = x + (i - 0.5) * wave_width
                    cp1y = y - amplitude if i % 2 == 1 else y + amplitude
                    wave_path_data += f"Q{cp1x},{cp1y} {wave_x},{wave_y} "
            
            # Complete the path to bottom-right and back to start
            wave_path_data += f"L{x + width},{y + height} L{x},{y + height} Z"
            
            # Create wave path with transparency
            wave_path = ET.SubElement(g, "{%s}path" % SVGNS)
            wave_path.set("d", wave_path_data)
            wave_path.set("fill", f"url(#{gradient_id})")
            wave_path.set("opacity", "0.6")
            
            # Add a few more subtle wave lines for texture
            for j in range(3):
                y_offset = y + height * (0.2 + j * 0.2)
                wave_line = ET.SubElement(g, "{%s}path" % SVGNS)
                
                # Create horizontal wave line
                line_data = f"M{x},{y_offset} "
                
                # Add curvy segments
                for i in range(1, 10):
                    segment_x = x + width * (i / 10)
                    variation = wave_intensity * 5 * (random.random() - 0.5)
                    segment_y = y_offset + variation
                    
                    if i == 1:
                        line_data += f"Q{x + width * 0.05},{y_offset + variation * 2} {segment_x},{segment_y} "
                    else:
                        prev_x = x + width * ((i - 1) / 10)
                        control_x = prev_x + (segment_x - prev_x) / 2
                        control_y = segment_y + variation
                        line_data += f"T{segment_x},{segment_y} "
                
                wave_line.set("d", line_data)
                wave_line.set("fill", "none")
                wave_line.set("stroke", "rgba(255,255,255,0.2)")
                wave_line.set("stroke-width", "1")
                
        # Add reflections if applicable
        if "reflection" in node.properties:
            if node.properties["reflection"] == "sunset":
                # Add a subtle sunset reflection path
                refl_path = ET.SubElement(g, "{%s}path" % SVGNS)
                refl_x = width / 2
                refl_width = width * 0.3
                
                # Create a fuzzy reflection shape
                refl_data = f"M{x + refl_x - refl_width/2},{y} "
                refl_data += f"Q{x + refl_x},{y + height * 0.5} {x + refl_x + refl_width/2},{y} "
                refl_data += "Z"
                
                refl_path.set("d", refl_data)
                refl_path.set("fill", "rgba(255,165,0,0.2)")
                refl_path.set("filter", "url(#blur)")
                
                # Create a blur filter for the reflection
                blur_filter = ET.Element("{%s}filter" % SVGNS)
                blur_filter.set("id", "blur")
                
                fe_gaussian = ET.SubElement(blur_filter, "{%s}feGaussianBlur" % SVGNS)
                fe_gaussian.set("in", "SourceGraphic")
                fe_gaussian.set("stdDeviation", "5")
                
                self.document.add_definition(blur_filter)
        
        return g

class TreeGenerator(SVGElementGenerator):
    """Generates realistic tree and forest elements."""
    
    def generate(self, node: SceneNode) -> Optional[ET.Element]:
        """Generate a tree element.
        
        Args:
            node: Tree node from scene graph
            
        Returns:
            SVG element representing the tree
        """
        width = node.size[0]
        height = node.size[1]
        x = node.position[0]
        y = node.position[1]
        
        # Extract properties with defaults
        tree_type = node.properties.get("tree_type", "deciduous")
        season = node.properties.get("season", "summer")
        size_variation = node.properties.get("size_variation", 1.0)
        wind_factor = node.properties.get("wind_factor", 0.0)  # 0-1, affects tree leaning
        detail_level = node.properties.get("detail_level", 0.7)  # 0-1, affects complexity
        
        # Create group for the tree with appropriate class
        tree_group = ET.Element("g", {"class": f"tree {tree_type}_{season}"})
        
        # Generate tree based on type
        if tree_type == "pine" or tree_type == "conifer":
            self._generate_pine_tree(tree_group, x, y, width, height, season, wind_factor, detail_level)
        elif tree_type == "palm":
            self._generate_palm_tree(tree_group, x, y, width, height, season, wind_factor, detail_level)
        elif tree_type == "dead":
            self._generate_dead_tree(tree_group, x, y, width, height, wind_factor, detail_level)
        elif tree_type == "bush":
            self._generate_bush(tree_group, x, y, width, height, season, detail_level)
        else:  # Default to deciduous
            self._generate_deciduous_tree(tree_group, x, y, width, height, season, wind_factor, detail_level)
        
        # Add shadow if needed
        if node.properties.get("cast_shadow", False):
            shadow = self._generate_tree_shadow(x, y, width, height, tree_type, wind_factor)
            tree_group.insert(0, shadow)  # Add shadow as first element so it's behind the tree
        
        return tree_group
    
    def _generate_pine_tree(self, parent: ET.Element, x: float, y: float, width: float, height: float, \
                            season: str = "summer", wind_factor: float = 0.0, detail_level: float = 0.7) -> None:
        """Generate a realistic pine/conifer tree.
        
        Args:
            parent: Parent SVG element
            x, y: Position
            width, height: Size
            season: Season affecting appearance (summer, autumn, winter, spring)
            wind_factor: 0-1 value affecting the tree's leaning
            detail_level: 0-1 value affecting the complexity of the tree's appearance
        """
        # Create group for tree
        pine_group = ET.SubElement(parent, "g", {"class": "pine-tree"})
        
        # Calculate wind influence on trunk position
        wind_lean = width * 0.2 * wind_factor
        
        # Create detailed trunk
        trunk_width = width * 0.08
        trunk_height = height * 0.3
        trunk_x = x + width/2 - trunk_width/2 + wind_lean * 0.5
        trunk_y = y + height - trunk_height
        
        # Use path instead of rectangle for more natural trunk
        trunk_path = ET.SubElement(pine_group, "path")
        
        # Create a slightly curved trunk that leans with the wind
        trunk_data = f"M{trunk_x},{trunk_y + trunk_height} "
        # Bottom to middle
        trunk_data += f"C{trunk_x},{trunk_y + trunk_height * 0.7} "
        trunk_data += f"{trunk_x + wind_lean * 0.5},{trunk_y + trunk_height * 0.5} "
        trunk_data += f"{trunk_x + wind_lean},{trunk_y} "
        # Back down the other side
        trunk_data += f"L{trunk_x + trunk_width + wind_lean},{trunk_y} "
        trunk_data += f"C{trunk_x + trunk_width + wind_lean},{trunk_y + trunk_height * 0.5} "
        trunk_data += f"{trunk_x + trunk_width},{trunk_y + trunk_height * 0.7} "
        trunk_data += f"{trunk_x + trunk_width},{trunk_y + trunk_height} Z"
        
        trunk_path.set("d", trunk_data)
        
        # Define trunk gradient for more realism
        trunk_gradient_id = f"pine_trunk_gradient_{random.randint(1000, 9999)}"
        trunk_gradient = ET.SubElement(pine_group, "linearGradient", { \
            "id": trunk_gradient_id,
            "x1": "0%", \
            "y1": "0%", \
            "x2": "100%", \
            "y2": "0%"
        })
        
        # Create gradient stops
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "0%",
            "stop-color": "#5D4037"
        })
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "50%",
            "stop-color": "#8B4513"
        })
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "100%",
            "stop-color": "#5D4037"
        })
        
        trunk_path.set("fill", f"url(#{trunk_gradient_id})")
        trunk_path.set("stroke", "#3E2723")
        trunk_path.set("stroke-width", "1")
        
        # Create pine foliage as triangular segments with more detail
        # More segments for higher detail level
        segments = max(3, min(7, int(4 + detail_level * 4)))
        segment_height = (height - trunk_height) / segments
        
        # Determine foliage colors based on season
        if season == "winter":
            base_hue = 150  # More blue-green
            base_saturation = 30  # Less saturated
            base_lightness = 20  # Darker
            snow_probability = 0.7  # High chance of snow
        elif season == "autumn":
            base_hue = 140  # Slightly more yellow-green
            base_saturation = 50
            base_lightness = 25
            snow_probability = 0.0
        elif season == "spring":
            base_hue = 120  # Bright green
            base_saturation = 60
            base_lightness = 35  # Lighter
            snow_probability = 0.1  # Small chance of snow
        else:  # summer
            base_hue = 130
            base_saturation = 70
            base_lightness = 30
            snow_probability = 0.0
        
        # Create a filter for needle texture effect if detail level is high enough
        if detail_level > 0.6:
            needle_filter_id = f"pine_needle_texture_{random.randint(1000, 9999)}"
            needle_filter = ET.SubElement(pine_group, "filter", { \
                "id": needle_filter_id,
                "x": "-20%", \
                "y": "-20%", \
                "width": "140%", \
                "height": "140%"
            })
            
            # Add turbulence for texture
            ET.SubElement(needle_filter, "feTurbulence", { \
                "type": "fractalNoise",
                "baseFrequency": "0.1", \
                "numOctaves": "2", \
                "seed": str(random.randint(1, 100)), \
                "result": "noise"
            })
            
            # Displacement map
            ET.SubElement(needle_filter, "feDisplacementMap", { \
                "in": "SourceGraphic",
                "in2": "noise", \
                "scale": "5", \
                "xChannelSelector": "R", \
                "yChannelSelector": "G"
            })
        
        # Generate foliage segments
        for i in range(segments):
            # Calculate position with wind factor
            wind_offset = wind_lean * (1 - i/segments) * 1.5  # More effect at the top
            
            # Each segment is a triangle, narrower at top
            segment_width = width * (1 - i * 0.15)
            segment_x = x + (width - segment_width) / 2 + wind_offset
            segment_y = y + i * segment_height
            
            # Add some randomness to the triangle points for a less perfect shape
            jitter = detail_level * width * 0.03
            
            # Create triangle points with some randomness
            top_x = segment_x + segment_width/2 + random.uniform(-jitter, jitter)
            top_y = segment_y + random.uniform(-jitter, jitter)
            
            right_x = segment_x + segment_width + random.uniform(-jitter, jitter)
            right_y = segment_y + segment_height + random.uniform(-jitter, jitter)
            
            left_x = segment_x + random.uniform(-jitter, jitter)
            left_y = segment_y + segment_height + random.uniform(-jitter, jitter)
            
            points = [ \
                f"{top_x},{top_y}",  # Top
                f"{right_x},{right_y}",  # Bottom right
                f"{left_x},{left_y}"  # Bottom left
            ]
            
            # Create polygon for segment
            polygon = ET.SubElement(pine_group, "polygon")
            polygon.set("points", " ".join(points))
            
            # Vary green shades for depth and realism
            # Darker at the bottom, slight random variation
            hue_variation = random.uniform(-10, 10)
            sat_variation = random.uniform(-10, 10)
            light_variation = random.uniform(-5, 5)
            
            segment_hue = base_hue + hue_variation
            segment_sat = base_saturation + sat_variation
            segment_light = base_lightness + light_variation - i * 2
            
            # Apply color
            polygon.set("fill", f"hsl({segment_hue}, {segment_sat}%, {segment_light}%)")
            
            # Apply needle texture if detail level is high
            if detail_level > 0.6:
                polygon.set("filter", f"url(#{needle_filter_id})")
            
            # Add snow on top of segments if it's winter
            if random.random() < snow_probability:
                snow_points = [ \
                    f"{top_x},{top_y}",  # Top
                    f"{right_x},{right_y - segment_height * 0.3}",  # Right side
                    f"{left_x},{left_y - segment_height * 0.3}"  # Left side
                ]
                
                snow = ET.SubElement(pine_group, "polygon")
                snow.set("points", " ".join(snow_points))
                snow.set("fill", "#F5F5F5")
                snow.set("fill-opacity", str(random.uniform(0.5, 0.9)))
        
        # Add small details if detail level is high
        if detail_level > 0.8:
            # Add small brown pinecones randomly distributed
            num_pinecones = random.randint(3, 8)
            for _ in range(num_pinecones):
                # Position it within the foliage area
                cone_x = x + random.uniform(width * 0.2, width * 0.8) + wind_lean * 0.5
                cone_y = y + random.uniform(height * 0.3, height * 0.8)
                cone_size = width * 0.03
                
                pinecone = ET.SubElement(pine_group, "ellipse")
                pinecone.set("cx", str(cone_x))
                pinecone.set("cy", str(cone_y))
                pinecone.set("rx", str(cone_size))
                pinecone.set("ry", str(cone_size * 1.5))
                pinecone.set("fill", "#5D4037")
                pinecone.set("transform", f"rotate({random.randint(0, 360)} {cone_x} {cone_y})")
        
        return pine_group
    
    def _generate_deciduous_tree(self, parent: ET.Element, x: float, y: float, width: float, height: float, \
                                season: str = "summer", wind_factor: float = 0.0, detail_level: float = 0.7) -> None:
        """Generate a realistic deciduous/leafy tree.
        
        Args:
            parent: Parent SVG element
            x, y: Position
            width, height: Size
            season: Season affecting appearance (summer, autumn, winter, spring)
            wind_factor: 0-1 value affecting the tree's leaning
            detail_level: 0-1 value affecting the complexity of the tree's appearance
        """
        # Create group for tree
        deciduous_group = ET.SubElement(parent, "g", {"class": "deciduous-tree"})
        
        # Calculate wind influence on trunk position
        wind_lean = width * 0.15 * wind_factor
        
        # Create trunk with more detail
        trunk_width = width * 0.12
        trunk_height = height * 0.4
        trunk_x = x + width/2 - trunk_width/2
        trunk_y = y + height - trunk_height
        
        # Make the trunk curved and leaning with the wind
        trunk_path = ET.SubElement(deciduous_group, "path")
        
        # Create a path for a more natural looking trunk
        # The curve factor makes the trunk bend with the wind
        trunk_data = f"M{trunk_x},{trunk_y + trunk_height} "
        trunk_data += f"C{trunk_x},{trunk_y + trunk_height * 0.7} "
        trunk_data += f"{trunk_x + wind_lean * 0.5},{trunk_y + trunk_height * 0.3} "
        trunk_data += f"{trunk_x + wind_lean},{trunk_y} "
        trunk_data += f"L{trunk_x + trunk_width + wind_lean},{trunk_y} "
        trunk_data += f"C{trunk_x + trunk_width + wind_lean},{trunk_y + trunk_height * 0.3} "
        trunk_data += f"{trunk_x + trunk_width},{trunk_y + trunk_height * 0.7} "
        trunk_data += f"{trunk_x + trunk_width},{trunk_y + trunk_height} Z"
        
        trunk_path.set("d", trunk_data)
        
        # Create a gradient for more realistic trunk coloring
        trunk_gradient_id = f"deciduous_trunk_gradient_{random.randint(1000, 9999)}"
        trunk_gradient = ET.SubElement(deciduous_group, "linearGradient", { \
            "id": trunk_gradient_id,
            "x1": "0%", \
            "y1": "0%", \
            "x2": "100%", \
            "y2": "0%"
        })
        
        # Create gradient stops for trunk
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "0%",
            "stop-color": "#5D4037"
        })
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "50%",
            "stop-color": "#8B4513"
        })
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "100%",
            "stop-color": "#5D4037"
        })
        
        trunk_path.set("fill", f"url(#{trunk_gradient_id})")
        trunk_path.set("stroke", "#3E2723")
        trunk_path.set("stroke-width", "1")
        
        # Add bark texture if detail level is high
        if detail_level > 0.7:
            # Add some lines to represent bark texture
            bark_lines = random.randint(3, 6)
            for i in range(bark_lines):
                line_y_position = trunk_y + random.uniform(trunk_height * 0.2, trunk_height * 0.8)
                line_length = trunk_width * random.uniform(0.3, 0.8)
                line_x_start = trunk_x + random.uniform(0, trunk_width - line_length)
                
                bark_line = ET.SubElement(deciduous_group, "line")
                bark_line.set("x1", str(line_x_start + wind_lean * (line_y_position - trunk_y) / trunk_height))
                bark_line.set("y1", str(line_y_position))
                bark_line.set("x2", str(line_x_start + line_length + wind_lean * (line_y_position - trunk_y) / trunk_height))
                bark_line.set("y2", str(line_y_position))
                bark_line.set("stroke", "#3E2723")
                bark_line.set("stroke-width", str(1 + detail_level))
                bark_line.set("opacity", "0.7")
        
        # Add branches if detail level is high enough
        if detail_level > 0.6:
            branch_count = random.randint(2, 4)
            for i in range(branch_count):
                # Position branches on trunk
                branch_y = trunk_y + trunk_height * (0.2 + 0.6 * i / branch_count)
                branch_side = 1 if i % 2 == 0 else -1  # Alternate sides
                branch_length = width * random.uniform(0.2, 0.4)
                branch_angle = 30 + random.uniform(-15, 15) + wind_factor * 20 * branch_side
                
                # Branch starting position (on the trunk)
                branch_start_x = trunk_x + (trunk_width if branch_side < 0 else 0) + wind_lean * (branch_y - trunk_y) / trunk_height
                
                # Calculate endpoint based on angle and length
                end_x = branch_start_x + branch_length * math.cos(math.radians(branch_angle * branch_side))
                end_y = branch_y - branch_length * math.sin(math.radians(branch_angle * branch_side))
                
                # Create branch path with a slight curve
                branch_path = ET.SubElement(deciduous_group, "path")
                control_x = branch_start_x + branch_length * 0.5 * math.cos(math.radians(branch_angle * branch_side * 0.8))
                control_y = branch_y - branch_length * 0.5 * math.sin(math.radians(branch_angle * branch_side * 1.2))
                
                branch_data = f"M{branch_start_x},{branch_y} "
                branch_data += f"Q{control_x},{control_y} {end_x},{end_y}"
                
                branch_path.set("d", branch_data)
                branch_path.set("fill", "none")
                branch_path.set("stroke", "#5D4037")
                branch_path.set("stroke-width", str(trunk_width * 0.2))
        
        # Determine foliage colors based on season
        if season == "winter":
            # Bare or minimal foliage for winter
            num_leaves = random.randint(0, 5) if detail_level > 0.7 else 0
            leaf_colors = ["#795548"]  # Brown dried leaves
            snow_probability = 0.7
        elif season == "autumn":
            # Fall colors
            leaf_colors = ["#FF8F00", "#F57F17", "#BF360C", "#B71C1C", "#880E4F"]
            snow_probability = 0.0
        elif season == "spring":
            # Bright green with some flower buds
            leaf_colors = ["#7CB342", "#558B2F", "#33691E", "#FBC02D"]
            snow_probability = 0.0
        else:  # summer
            # Deep, rich greens
            leaf_colors = ["#2E7D32", "#388E3C", "#43A047", "#4CAF50"]
            snow_probability = 0.0
        
        # Skip foliage for bare winter trees if needed
        if season == "winter" and random.random() < 0.7 and detail_level < 0.9:
            # Add some snow on the branches
            if snow_probability > 0:
                snow_count = random.randint(2, 5)
                for i in range(snow_count):
                    snow_x = trunk_x + random.uniform(0, trunk_width) + wind_lean * 0.5
                    snow_y = trunk_y + random.uniform(0, trunk_height * 0.7)
                    snow_size = width * random.uniform(0.02, 0.05)
                    
                    snow = ET.SubElement(deciduous_group, "circle")
                    snow.set("cx", str(snow_x))
                    snow.set("cy", str(snow_y))
                    snow.set("r", str(snow_size))
                    snow.set("fill", "#FFFFFF")
                    snow.set("opacity", str(random.uniform(0.7, 0.9)))
            return
        
        # Create a special filter for leaf texture if detail level is high
        if detail_level > 0.8:
            leaf_filter_id = f"leaf_texture_{random.randint(1000, 9999)}"
            leaf_filter = ET.SubElement(deciduous_group, "filter", { \
                "id": leaf_filter_id,
                "x": "-20%", \
                "y": "-20%", \
                "width": "140%", \
                "height": "140%"
            })
            
            # Add turbulence for texture
            ET.SubElement(leaf_filter, "feTurbulence", { \
                "type": "turbulence",
                "baseFrequency": "0.05", \
                "numOctaves": "2", \
                "result": "turbulence"
            })
            
            # Displacement map
            ET.SubElement(leaf_filter, "feDisplacementMap", { \
                "in": "SourceGraphic",
                "in2": "turbulence", \
                "scale": "5", \
                "xChannelSelector": "R", \
                "yChannelSelector": "G"
            })
        
        # Create foliage - adjust center point based on wind
        foliage_x = x + width/2 + wind_lean
        foliage_y = y + (height - trunk_height) * 0.7
        
        # Create a canopy with multiple overlapping shapes
        if detail_level < 0.5:
            # Simpler foliage for lower detail level - just a few circles
            ellipse_count = random.randint(3, 5)
            foliage_radius_x = width * 0.5
            foliage_radius_y = (height - trunk_height) * 0.7
            
            for i in range(ellipse_count):
                # Vary position and size slightly, accounting for wind
                wind_offset_x = wind_factor * width * 0.2 * (1 - i/ellipse_count)
                offset_x = (random.random() - 0.5) * width * 0.3 + wind_offset_x
                offset_y = (random.random() - 0.5) * height * 0.2
                size_factor = 0.8 + random.random() * 0.4
                
                # Select color based on season
                color = random.choice(leaf_colors)
                
                # Add some randomness to the color
                hsl = self._rgb_to_hsl(int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16))
                hsl[0] += random.uniform(-5, 5)  # Slight hue variation
                hsl[1] += random.uniform(-5, 5)  # Slight saturation variation
                hsl[2] += random.uniform(-5, 5)  # Slight lightness variation
                adjusted_color = self._hsl_to_hex(hsl[0], hsl[1], hsl[2])
                
                ellipse = ET.SubElement(deciduous_group, "ellipse")
                ellipse.set("cx", str(foliage_x + offset_x))
                ellipse.set("cy", str(foliage_y + offset_y))
                ellipse.set("rx", str(foliage_radius_x * size_factor))
                ellipse.set("ry", str(foliage_radius_y * size_factor))
                ellipse.set("fill", adjusted_color)
                
                # Apply texture filter if available
                if detail_level > 0.8:
                    ellipse.set("filter", f"url(#{leaf_filter_id})")
        else:
            # More complex foliage - create a more detailed, irregular canopy
            # First, create a base path for the overall canopy shape
            canopy_path = ET.SubElement(deciduous_group, "path")
            
            # Generate a complex, slightly randomized path for the canopy
            canopy_width = width * 1.0
            canopy_height = height * 0.7
            
            # Calculate wind influence on canopy
            canopy_wind_offset = wind_lean * 1.2
            
            # Create points around the center for the canopy
            num_points = max(6, int(12 * detail_level))
            canopy_data = "M"
            
            for i in range(num_points + 1):
                angle = (i * 2 * math.pi / num_points)
                point_distance = canopy_width * 0.5 * (0.8 + random.uniform(0, 0.4))
                
                # Adjust x position based on wind and angle
                wind_angle_factor = math.cos(angle)  # Maximum at 0°, minimum at 180°
                point_wind_offset = canopy_wind_offset * wind_angle_factor
                
                point_x = foliage_x + point_distance * math.cos(angle) + point_wind_offset
                point_y = foliage_y + point_distance * 0.8 * math.sin(angle)  # Slightly flattened
                
                if i == 0:
                    canopy_data += f"{point_x},{point_y} "
                else:
                    # Use quadratic curves for a more natural appearance
                    # Control point with some randomness
                    prev_angle = ((i-1) * 2 * math.pi / num_points)
                    control_angle = (prev_angle + angle) / 2
                    control_distance = point_distance * (0.9 + random.uniform(0, 0.3))
                    
                    control_x = foliage_x + control_distance * math.cos(control_angle) + point_wind_offset/2
                    control_y = foliage_y + control_distance * 0.8 * math.sin(control_angle)
                    
                    canopy_data += f"Q {control_x},{control_y} {point_x},{point_y} "
            
            canopy_data += "Z"
            canopy_path.set("d", canopy_data)
            
            # Determine main color based on season
            main_color = random.choice(leaf_colors)
            canopy_path.set("fill", main_color)
            
            # Apply texture filter if available
            if detail_level > 0.8:
                canopy_path.set("filter", f"url(#{leaf_filter_id})")
            
            # Add detailed foliage clusters if detail level is very high
            if detail_level > 0.8:
                cluster_count = random.randint(4, 8)
                for i in range(cluster_count):
                    # Position clusters around the canopy
                    angle = random.uniform(0, 2 * math.pi)
                    distance = canopy_width * 0.4 * random.uniform(0.7, 1.0)
                    
                    # Adjust for wind
                    cluster_wind_offset = canopy_wind_offset * math.cos(angle) * 0.7
                    
                    cluster_x = foliage_x + distance * math.cos(angle) + cluster_wind_offset
                    cluster_y = foliage_y + distance * 0.8 * math.sin(angle)
                    cluster_size = width * random.uniform(0.1, 0.2)
                    
                    # Pick a different color for variety
                    cluster_color = random.choice(leaf_colors)
                    while cluster_color == main_color and len(leaf_colors) > 1:
                        cluster_color = random.choice(leaf_colors)
                    
                    # Create the cluster
                    cluster = ET.SubElement(deciduous_group, "circle")
                    cluster.set("cx", str(cluster_x))
                    cluster.set("cy", str(cluster_y))
                    cluster.set("r", str(cluster_size))
                    cluster.set("fill", cluster_color)
                    cluster.set("opacity", "0.9")
        
        # Add snow on top if it's winter and we have foliage
        if snow_probability > 0:
            snow_path = ET.SubElement(deciduous_group, "ellipse")
            snow_path.set("cx", str(foliage_x))
            snow_path.set("cy", str(y + height * 0.2))
            snow_path.set("rx", str(width * 0.4))
            snow_path.set("ry", str(height * 0.1))
            snow_path.set("fill", "#FFFFFF")
            snow_path.set("opacity", str(random.uniform(0.7, 0.9)))
            
        # If it's autumn, add some falling leaves if detail level is high
        if season == "autumn" and detail_level > 0.8:
            falling_leaves = random.randint(3, 8)
            for i in range(falling_leaves):
                leaf_x = x + random.uniform(0, width)
                leaf_y = y + height * random.uniform(0.4, 0.95)
                leaf_size = width * 0.015
                
                leaf_color = random.choice(leaf_colors)
                
                leaf = ET.SubElement(deciduous_group, "circle")
                leaf.set("cx", str(leaf_x))
                leaf.set("cy", str(leaf_y))
                leaf.set("r", str(leaf_size))
                leaf.set("fill", leaf_color)
                
                # Random rotation for more natural appearance
                leaf.set("transform", f"rotate({random.randint(0, 360)} {leaf_x} {leaf_y})")
                
        return deciduous_group
    
    def _rgb_to_hsl(self, r: int, g: int, b: int) -> list:
        """Convert RGB to HSL color values."""
        r, g, b = r/255.0, g/255.0, b/255.0
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        h, s, l = 0, 0, (max_val + min_val) / 2
        
        if max_val == min_val:
            h = s = 0  # achromatic
        else:
            d = max_val - min_val
            s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
            
            if max_val == r:
                h = (g - b) / d + (6 if g < b else 0)
            elif max_val == g:
                h = (b - r) / d + 2
            else:
                h = (r - g) / d + 4
                
            h /= 6
        
        return [h * 360, s * 100, l * 100]
    
    def _hsl_to_hex(self, h: float, s: float, l: float) -> str:
        """Convert HSL to hex color string."""
        h, s, l = h / 360, s / 100, l / 100
        
        if s == 0:
            r = g = b = l
        else:
            def hue_to_rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1/6: return p + (q - p) * 6 * t
                if t < 1/2: return q
                if t < 2/3: return p + (q - p) * (2/3 - t) * 6
                return p
            
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue_to_rgb(p, q, h + 1/3)
            g = hue_to_rgb(p, q, h)
            b = hue_to_rgb(p, q, h - 1/3)
        
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _generate_palm_tree(self, parent: ET.Element, x: float, y: float, width: float, height: float, \
                            season: str = "summer", wind_factor: float = 0.0, detail_level: float = 0.7) -> None:
        """Generate a palm tree.
        
        Args:
            parent: Parent SVG element
            x, y: Position
            width, height: Size
            season: Season affecting appearance
            wind_factor: 0-1 value affecting the tree's leaning
            detail_level: 0-1 value affecting the complexity
        """
        # Create group for palm tree
        palm_group = ET.SubElement(parent, "g", {"class": "palm-tree"})
        
        # Calculate trunk dimensions and position
        trunk_width = width * 0.1
        trunk_height = height * 0.6
        trunk_x = x + width/2 - trunk_width/2
        trunk_y = y + height - trunk_height
        
        # Calculate wind effect - palms bend more than other trees
        wind_lean = width * 0.3 * wind_factor
        
        # Create a curved trunk path
        trunk_path = ET.SubElement(palm_group, "path")
        
        # More dramatic curve for palm trunks
        # Start at the bottom
        trunk_data = f"M{trunk_x},{trunk_y + trunk_height} "
        
        # Create a curved trunk that bends with the wind
        # The higher up the trunk, the more it bends
        ctrl1_x = trunk_x
        ctrl1_y = trunk_y + trunk_height * 0.6
        ctrl2_x = trunk_x + wind_lean * 0.5
        ctrl2_y = trunk_y + trunk_height * 0.3
        end1_x = trunk_x + wind_lean
        end1_y = trunk_y
        
        trunk_data += f"C{ctrl1_x},{ctrl1_y} {ctrl2_x},{ctrl2_y} {end1_x},{end1_y} "
        
        # Right side of trunk
        trunk_data += f"L{trunk_x + trunk_width + wind_lean},{trunk_y} "
        
        # Back down the right side
        ctrl3_x = trunk_x + trunk_width + wind_lean * 0.5
        ctrl3_y = trunk_y + trunk_height * 0.3
        ctrl4_x = trunk_x + trunk_width
        ctrl4_y = trunk_y + trunk_height * 0.6
        end2_y = trunk_y + trunk_height
        
        trunk_data += f"C{ctrl3_x},{ctrl3_y} {ctrl4_x},{ctrl4_y} {trunk_x + trunk_width},{end2_y} Z"
        
        trunk_path.set("d", trunk_data)
        
        # Create trunk gradient for more realism
        trunk_gradient_id = f"palm_trunk_gradient_{random.randint(1000, 9999)}"
        trunk_gradient = ET.SubElement(palm_group, "linearGradient", { \
            "id": trunk_gradient_id,
            "x1": "0%", \
            "y1": "0%", \
            "x2": "100%", \
            "y2": "0%"
        })
        
        # Create gradient stops - palm trunks are usually brown-gray
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "0%",
            "stop-color": "#8D6E63"
        })
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "50%",
            "stop-color": "#A1887F"
        })
        ET.SubElement(trunk_gradient, "stop", { \
            "offset": "100%",
            "stop-color": "#8D6E63"
        })
        
        trunk_path.set("fill", f"url(#{trunk_gradient_id})")
        trunk_path.set("stroke", "#6D4C41")
        trunk_path.set("stroke-width", "1")
        
        # Add trunk details if detail level is high enough
        if detail_level > 0.6:
            # Add rings to simulate palm trunk segments
            segment_count = int(5 + detail_level * 7)  # More segments for higher detail
            
            for i in range(segment_count):
                segment_y = trunk_y + (i + 1) * trunk_height / (segment_count + 1)
                
                # Calculate position adjusted for the curve of the trunk
                curve_factor = wind_lean * ((segment_y - trunk_y) / trunk_height)
                
                # Create segment line
                segment_line = ET.SubElement(palm_group, "line")
                segment_line.set("x1", str(trunk_x + curve_factor))
                segment_line.set("y1", str(segment_y))
                segment_line.set("x2", str(trunk_x + trunk_width + curve_factor))
                segment_line.set("y2", str(segment_y))
                segment_line.set("stroke", "#6D4C41")
                segment_line.set("stroke-width", "1")
                segment_line.set("opacity", "0.7")
        
        # Create palm fronds
        frond_count = max(5, min(12, int(7 + detail_level * 8)))  # More fronds for higher detail
        frond_center_x = trunk_x + trunk_width/2 + wind_lean
        frond_center_y = trunk_y
        frond_length = width * 0.8
        
        # Base color of fronds depends on season
        if season == "winter":
            base_color = "#689F38"  # Duller green
            frond_droop = 0.3  # Droopier fronds
        elif season == "autumn":
            base_color = "#8BC34A"  # Yellower green
            frond_droop = 0.2
        elif season == "spring":
            base_color = "#7CB342"  # Fresh green
            frond_droop = 0.1
        else:  # summer
            base_color = "#558B2F"  # Deep green
            frond_droop = 0.0
        
        # Generate a darker color for some fronds
        darker_color = self._darken_color(base_color, 0.2)
        frond_colors = [base_color, darker_color]
        
        # Add coconuts if detail level is high enough
        if detail_level > 0.7:
            coconut_count = random.randint(3, 6)
            for i in range(coconut_count):
                coconut_angle = random.uniform(0, 2 * math.pi)
                coconut_distance = trunk_width * 0.6
                coconut_x = frond_center_x + coconut_distance * math.cos(coconut_angle)
                coconut_y = frond_center_y + coconut_distance * math.sin(coconut_angle)
                coconut_size = width * 0.04
                
                coconut = ET.SubElement(palm_group, "circle")
                coconut.set("cx", str(coconut_x))
                coconut.set("cy", str(coconut_y))
                coconut.set("r", str(coconut_size))
                coconut.set("fill", "#4E342E")  # Dark brown
        
        # Create palm fronds distributed in a circle
        for i in range(frond_count):
            # Calculate angle - distribute fronds evenly
            angle = (i * 2 * math.pi / frond_count)
            
            # Adjust angle based on wind - fronds will bend in wind direction
            wind_angle_adjustment = max(0, math.cos(angle)) * wind_factor * math.pi / 6
            adjusted_angle = angle + wind_angle_adjustment
            
            # Fronds droop downward - more in winter
            if math.sin(angle) < 0:  # Lower half fronds droop more
                droop_factor = 0.1 + frond_droop + abs(math.sin(angle)) * 0.2
            else:
                droop_factor = frond_droop
            
            # Create frond path
            frond_path = ET.SubElement(palm_group, "path")
            
            # Start at center
            frond_data = f"M{frond_center_x},{frond_center_y} "
            
            # End point with length determined by angle and droop
            end_x = frond_center_x + frond_length * math.cos(adjusted_angle)
            end_y = frond_center_y + frond_length * math.sin(adjusted_angle) + frond_length * droop_factor
            
            # Control point for curve - more curved for higher detail
            ctrl_distance = frond_length * (0.5 + detail_level * 0.2)
            ctrl_angle = adjusted_angle + (random.uniform(-0.1, 0.1) * detail_level)
            ctrl_x = frond_center_x + ctrl_distance * math.cos(ctrl_angle)
            ctrl_y = frond_center_y + ctrl_distance * math.sin(ctrl_angle) + ctrl_distance * droop_factor * 0.7
            
            frond_data += f"Q{ctrl_x},{ctrl_y} {end_x},{end_y}"
            
            frond_path.set("d", frond_data)
            frond_path.set("fill", "none")
            frond_path.set("stroke", random.choice(frond_colors))
            frond_path.set("stroke-width", str(2 + detail_level * 3))
            
            # Add detail to fronds if detail level is high
            if detail_level > 0.6:
                # Create leaflets along the frond
                leaflet_count = max(5, int(8 * detail_level))
                
                for j in range(1, leaflet_count + 1):
                    # Position along frond
                    t = j / (leaflet_count + 1)  # Parametric position along frond
                    
                    # Calculate point on the frond curve using quadratic bezier formula
                    leaflet_base_x = (1 - t) * (1 - t) * frond_center_x + 2 * (1 - t) * t * ctrl_x + t * t * end_x
                    leaflet_base_y = (1 - t) * (1 - t) * frond_center_y + 2 * (1 - t) * t * ctrl_y + t * t * end_y
                    
                    # Calculate direction tangent to the frond at this point
                    tangent_x = 2 * (1 - t) * (ctrl_x - frond_center_x) + 2 * t * (end_x - ctrl_x)
                    tangent_y = 2 * (1 - t) * (ctrl_y - frond_center_y) + 2 * t * (end_y - ctrl_y)
                    
                    # Normalize tangent to get direction
                    magnitude = math.sqrt(tangent_x * tangent_x + tangent_y * tangent_y)
                    if magnitude > 0:  # Avoid division by zero
                        tangent_x /= magnitude
                        tangent_y /= magnitude
                    
                    # Create normal vector (perpendicular to tangent)
                    normal_x = -tangent_y
                    normal_y = tangent_x
                    
                    # Alternate leaflets on each side of the frond
                    side = 1 if j % 2 == 0 else -1
                    
                    # Leaflet length decreases toward tip
                    leaflet_length = frond_length * 0.2 * (1 - 0.5 * t)  # Shorter toward tip
                    
                    # Calculate leaflet endpoint
                    leaflet_end_x = leaflet_base_x + side * normal_x * leaflet_length
                    leaflet_end_y = leaflet_base_y + side * normal_y * leaflet_length
                    
                    # Create leaflet
                    leaflet = ET.SubElement(palm_group, "line")
                    leaflet.set("x1", str(leaflet_base_x))
                    leaflet.set("y1", str(leaflet_base_y))
                    leaflet.set("x2", str(leaflet_end_x))
                    leaflet.set("y2", str(leaflet_end_y))
                    leaflet.set("stroke", random.choice(frond_colors))
                    leaflet.set("stroke-width", str(1 + detail_level))
        
        return palm_group
    
    def _darken_color(self, hex_color: str, amount: float) -> str:
        """Darken a hex color by the specified amount (0-1)."""
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # Darken
        r = max(0, int(r * (1 - amount)))
        g = max(0, int(g * (1 - amount)))
        b = max(0, int(b * (1 - amount)))
        
        # Convert back to hex
        return f"#{r:02x}{g:02x}{b:02x}"

    def _generate_dead_tree(self, parent: ET.Element, x: float, y: float, width: float, height: float, \
                            wind_factor: float = 0.0, detail_level: float = 0.7) -> None:
        """Generate a dead/bare tree.
        
        Args:
            parent: Parent SVG element
            x, y: Position
            width, height: Size
            wind_factor: 0-1 value affecting the tree's leaning
            detail_level: 0-1 value affecting the complexity
        """
        # Create group for dead tree
        dead_tree_group = ET.SubElement(parent, "g", {"class": "dead-tree"})
        
        # Calculate trunk dimensions and position
        trunk_width = width * 0.11
        trunk_height = height * 0.45
        trunk_x = x + width/2 - trunk_width/2
        trunk_y = y + height - trunk_height
        
        # Calculate wind effect
        wind_lean = width * 0.15 * wind_factor
        
        # Create main trunk
        trunk_path = ET.SubElement(dead_tree_group, "path")
        
        # Trunk path with wind effect
        trunk_data = f"M{trunk_x},{trunk_y + trunk_height} "
        trunk_data += f"C{trunk_x},{trunk_y + trunk_height * 0.7} "
        trunk_data += f"{trunk_x + wind_lean * 0.5},{trunk_y + trunk_height * 0.3} "
        trunk_data += f"{trunk_x + wind_lean},{trunk_y} "
        trunk_data += f"L{trunk_x + trunk_width + wind_lean},{trunk_y} "
        trunk_data += f"C{trunk_x + trunk_width + wind_lean},{trunk_y + trunk_height * 0.3} "
        trunk_data += f"{trunk_x + trunk_width},{trunk_y + trunk_height * 0.7} "
        trunk_data += f"{trunk_x + trunk_width},{trunk_y + trunk_height} Z"
        
        trunk_path.set("d", trunk_data)
        
        # Dead trees have grayish-brown color
        trunk_path.set("fill", "#6D4C41")
        trunk_path.set("stroke", "#5D4037")
        trunk_path.set("stroke-width", "1")
        
        # Add cracks and texture if detail level is high enough
        if detail_level > 0.6:
            # Add cracks to trunk
            crack_count = random.randint(2, 4)
            for i in range(crack_count):
                crack_y_pos = trunk_y + random.uniform(trunk_height * 0.1, trunk_height * 0.9)
                crack_width = trunk_width * random.uniform(0.6, 0.9)
                crack_x_start = trunk_x + random.uniform(0, trunk_width - crack_width) + wind_lean * (crack_y_pos - trunk_y) / trunk_height
                
                # Create jagged crack path
                crack_path = ET.SubElement(dead_tree_group, "path")
                crack_data = f"M{crack_x_start},{crack_y_pos} "
                
                # Create jagged segments for the crack
                segments = random.randint(3, 6)
                segment_width = crack_width / segments
                
                for j in range(1, segments + 1):
                    jag_height = random.uniform(-2, 2)
                    crack_data += f"L{crack_x_start + j * segment_width},{crack_y_pos + jag_height} "
                
                crack_path.set("d", crack_data)
                crack_path.set("fill", "none")
                crack_path.set("stroke", "#3E2723")
                crack_path.set("stroke-width", str(0.5 + detail_level))
        
        # Create bare branches - more detailed for higher detail level
        branch_count = max(5, int(detail_level * 15))
        branch_levels = max(2, int(detail_level * 4))
        
        # Trunk top center point - where the first level branches start
        trunk_top_x = trunk_x + trunk_width/2 + wind_lean
        trunk_top_y = trunk_y
        
        # Using recursive approach for branches
        def create_branch(start_x, start_y, length, angle, thickness, level):
            # Calculate end point
            end_x = start_x + length * math.cos(math.radians(angle))
            end_y = start_y - length * math.sin(math.radians(angle))
            
            # Create branch path
            branch_path = ET.SubElement(dead_tree_group, "path")
            branch_data = f"M{start_x},{start_y} "
            
            # Add some curve to the branch
            # Control point perpendicular to direction
            perpendicular_angle = angle + 90
            curve_magnitude = length * 0.2 * (random.random() - 0.5)
            control_x = start_x + length * 0.5 * math.cos(math.radians(angle)) + curve_magnitude * math.cos(math.radians(perpendicular_angle))
            control_y = start_y - length * 0.5 * math.sin(math.radians(angle)) - curve_magnitude * math.sin(math.radians(perpendicular_angle))
            
            branch_data += f"Q{control_x},{control_y} {end_x},{end_y}"
            branch_path.set("d", branch_data)
            branch_path.set("fill", "none")
            branch_path.set("stroke", "#5D4037")
            branch_path.set("stroke-width", str(thickness))
            
            # Add more branches if not at the last level
            if level < branch_levels:
                # Number of sub-branches decreases with level
                sub_branches = max(2, random.randint(2, 4) - level)
                
                for i in range(sub_branches):
                    # Angle variation increases with level
                    angle_variation = 20 + 10 * level
                    new_angle = angle + random.uniform(-angle_variation, angle_variation)
                    
                    # Length and thickness decrease with level
                    new_length = length * random.uniform(0.5, 0.8)
                    new_thickness = thickness * 0.7
                    
                    # Create sub-branch
                    create_branch(end_x, end_y, new_length, new_angle, new_thickness, level + 1)
        
        # Create first level main branches
        for i in range(branch_count):
            # Branch angle - distribute around the top
            if branch_count < 8:
                # For few branches, distribute more evenly
                base_angle = i * (180 / (branch_count - 1)) if branch_count > 1 else 90
                angle = base_angle - 90 + random.uniform(-15, 15)  # Center at vertical, add some randomness
            else:
                # For many branches, cluster more toward the top
                angle = random.uniform(-80, 80) + 90  # 10 to 170 degrees
            
            # Adjust angle based on wind - branches bend in wind direction
            wind_angle_adjustment = wind_factor * 15 * math.sin(math.radians(angle))
            adjusted_angle = angle + wind_angle_adjustment
            
            # Branch length is proportional to height and varies slightly
            branch_length = height * 0.3 * random.uniform(0.8, 1.2)
            
            # Branch thickness is proportional to width
            branch_thickness = width * 0.03
            
            # Create the branch and its sub-branches recursively
            create_branch(trunk_top_x, trunk_top_y, branch_length, adjusted_angle, branch_thickness, 1)
        
        return dead_tree_group
    
    def _generate_bush(self, parent: ET.Element, x: float, y: float, width: float, height: float, \
                        season: str = "summer", detail_level: float = 0.7) -> None:
        """Generate a bush or shrub.
        
        Args:
            parent: Parent SVG element
            x, y: Position
            width, height: Size
            season: Season affecting appearance
            detail_level: 0-1 value affecting the complexity
        """
        # Create group for bush
        bush_group = ET.SubElement(parent, "g", {"class": "bush"})
        
        # Determine bush colors based on season
        if season == "winter":
            # Muted colors for winter
            leaf_colors = ["#556B2F", "#6B8E23", "#808000"]
            if random.random() < 0.3:  # Some bushes have berries in winter
                berry_color = "#B71C1C"  # Red berries
                has_berries = True
            else:
                has_berries = False
        elif season == "autumn":
            # Fall colors
            leaf_colors = ["#DAA520", "#CD853F", "#B8860B", "#8B4513"]
            has_berries = random.random() < 0.2
            berry_color = "#4A148C"  # Purple berries
        elif season == "spring":
            # Bright colors with some flowers
            leaf_colors = ["#7CB342", "#8BC34A", "#689F38"]
            has_berries = random.random() < 0.6  # More likely to have flowers
            # Spring bushes often have flowers instead of berries
            berry_color = random.choice(["#F8BBD0", "#F48FB1", "#FFEB3B", "#FFF9C4"])  # Pink or yellow flowers
        else:  # summer
            # Deep greens
            leaf_colors = ["#2E7D32", "#388E3C", "#43A047", "#4CAF50"]
            has_berries = random.random() < 0.3
            berry_color = random.choice(["#1A237E", "#0D47A1", "#E65100"])  # Blue or orange berries
        
        # Create the main bush shape
        # For higher detail, use multiple overlapping shapes for more realistic bushes
        if detail_level > 0.7:
            # Create 4-8 overlapping rounded shapes
            cluster_count = random.randint(4, 8)
            
            for i in range(cluster_count):
                # Vary position within the bounds
                cluster_x = x + random.uniform(width * 0.1, width * 0.9)
                cluster_y = y + random.uniform(height * 0.1, height * 0.9)
                
                # Size varies but clusters should collectively cover most of the area
                cluster_width = width * random.uniform(0.3, 0.6)
                cluster_height = height * random.uniform(0.3, 0.6)
                
                # Create ellipse for cluster
                cluster = ET.SubElement(bush_group, "ellipse")
                cluster.set("cx", str(cluster_x))
                cluster.set("cy", str(cluster_y))
                cluster.set("rx", str(cluster_width/2))
                cluster.set("ry", str(cluster_height/2))
                
                # Choose color from seasonal palette
                cluster.set("fill", random.choice(leaf_colors))
                
                # Add texture if detail level is very high
                if detail_level > 0.9:
                    texture_filter_id = f"bush_texture_{random.randint(1000, 9999)}"
                    texture_filter = ET.SubElement(bush_group, "filter", { \
                        "id": texture_filter_id,
                        "x": "-20%", \
                        "y": "-20%", \
                        "width": "140%", \
                        "height": "140%"
                    })
                    
                    # Turbulence for bush texture
                    ET.SubElement(texture_filter, "feTurbulence", { \
                        "type": "fractalNoise",
                        "baseFrequency": "0.1", \
                        "numOctaves": "3", \
                        "seed": str(random.randint(1, 100)), \
                        "result": "noise"
                    })
                    
                    # Displacement map
                    ET.SubElement(texture_filter, "feDisplacementMap", { \
                        "in": "SourceGraphic",
                        "in2": "noise", \
                        "scale": "5", \
                        "xChannelSelector": "R", \
                        "yChannelSelector": "G"
                    })
                    
                    cluster.set("filter", f"url(#{texture_filter_id})")
        else:
            # Simpler bush - just a few basic shapes
            # Main rounded rectangle shape
            rounded_rect = ET.SubElement(bush_group, "rect")
            rounded_rect.set("x", str(x))
            rounded_rect.set("y", str(y + height * 0.2))  # Raise bottom slightly
            rounded_rect.set("width", str(width))
            rounded_rect.set("height", str(height * 0.8))
            rounded_rect.set("rx", str(width * 0.2))  # Rounded corners
            rounded_rect.set("ry", str(height * 0.2))
            rounded_rect.set("fill", random.choice(leaf_colors))
            
            # Add a rounded top
            top_ellipse = ET.SubElement(bush_group, "ellipse")
            top_ellipse.set("cx", str(x + width/2))
            top_ellipse.set("cy", str(y + height * 0.25))
            top_ellipse.set("rx", str(width * 0.5))
            top_ellipse.set("ry", str(height * 0.25))
            top_ellipse.set("fill", random.choice(leaf_colors))
        
        # Add berries or flowers if applicable
        if has_berries and detail_level > 0.5:
            berry_count = random.randint(5, 15)
            
            for i in range(berry_count):
                # Distribute berries around the bush
                berry_x = x + random.uniform(width * 0.1, width * 0.9)
                berry_y = y + random.uniform(height * 0.1, height * 0.9)
                
                # Size depends on whether they're berries or flowers
                berry_size = width * (0.02 if season != "spring" else 0.04)  # Flowers are bigger
                
                berry = ET.SubElement(bush_group, "circle")
                berry.set("cx", str(berry_x))
                berry.set("cy", str(berry_y))
                berry.set("r", str(berry_size))
                berry.set("fill", berry_color)
                
                # For spring flowers, add a yellow center
                if season == "spring" and random.random() < 0.7:
                    flower_center = ET.SubElement(bush_group, "circle")
                    flower_center.set("cx", str(berry_x))
                    flower_center.set("cy", str(berry_y))
                    flower_center.set("r", str(berry_size * 0.3))
                    flower_center.set("fill", "#FFEB3B")  # Yellow center
        
        # Add some stems/branches at the bottom for more detail
        if detail_level > 0.6:
            stem_count = random.randint(3, 6)
            stem_width = width * 0.03
            
            for i in range(stem_count):
                stem_x = x + width * 0.2 + (width * 0.6 * i / (stem_count - 1 if stem_count > 1 else 1))
                stem_y = y + height * 0.98  # Near bottom
                stem_height = height * random.uniform(0.2, 0.4)
                
                stem = ET.SubElement(bush_group, "line")
                stem.set("x1", str(stem_x))
                stem.set("y1", str(stem_y))
                stem.set("x2", str(stem_x + random.uniform(-width * 0.05, width * 0.05)))  # Slight angle
                stem.set("y2", str(stem_y - stem_height))
                stem.set("stroke", "#5D4037")  # Brown
                stem.set("stroke-width", str(stem_width))
        
        return bush_group
    
    def _generate_tree_shadow(self, x: float, y: float, width: float, height: float, \
                                tree_type: str, wind_factor: float = 0.0) -> ET.Element:
        """Generate a shadow for a tree.
        
        Args:
            x, y: Position of the tree
            width, height: Size of the tree
            tree_type: Type of tree
            wind_factor: Wind effect on the shadow direction
            
        Returns:
            SVG element representing the shadow
        """
        # Calculate shadow dimensions
        # Shadow is wider and longer than the tree, and stretched based on wind
        shadow_width = width * (1.2 + wind_factor * 0.5)
        
        # Shadow is an ellipse under the tree
        shadow = ET.Element("ellipse")
        
        # Position shadow - offset by wind
        shadow_x = x + width/2 + wind_factor * width * 0.3
        shadow_y = y + height * 0.98  # At the bottom of the tree
        
        shadow.set("cx", str(shadow_x))
        shadow.set("cy", str(shadow_y))
        shadow.set("rx", str(shadow_width/2))
        shadow.set("ry", str(width * 0.15))  # Flatter ellipse
        
        # Shadow color and opacity
        shadow.set("fill", "#000000")
        shadow.set("fill-opacity", "0.2")
        shadow.set("filter", "blur(3px)")  # Soft edge
        
        return shadow

class BuildingGenerator(SVGElementGenerator):
    """Generates realistic building elements for cityscapes."""
    
    def generate(self, node: SceneNode) -> Optional[ET.Element]:
        """Generate a detailed building element.
        
        Args:
            node: Building node from scene graph
            
        Returns:
            SVG element representing the building
        """
        # Extract properties with defaults
        style = node.properties.get("style", "modern")
        x = node.bounds.x
        y = node.bounds.y
        width = node.bounds.width
        height = node.bounds.height
        
        # Optional properties
        window_pattern = node.properties.get("window_pattern", "grid")
        floors = node.properties.get("floors", int(height / 40))
        detail_level = node.properties.get("detail_level", 0.7)
        time_of_day = node.properties.get("time_of_day", "day")
        materials = node.properties.get("materials", [])
        architectural_features = node.properties.get("architectural_features", [])
        
        # Create building group
        building_group = ET.Element("g", {"class": "building"})
        
        # Generate gradient ID for this building
        gradient_id = f"building_gradient_{random.randint(1000, 9999)}"
        
        # Determine colors based on materials, style and time of day
        if "color" in node.properties:
            base_color = node.properties["color"]
            # Create slightly darker shade for edges or details
            darker_color = self._adjust_brightness(base_color, -0.2)
            lighter_color = self._adjust_brightness(base_color, 0.2)
        else:
            # Default colors based on style and time of day
            if style == "modern":
                if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                    base_color = "#A1887F"  # Warm-tinted glass
                elif time_of_day == "night":
                    base_color = "#37474F"  # Dark blue-gray
                else:  # Day
                    base_color = "#78909C"  # Blue-gray
                
                darker_color = self._adjust_brightness(base_color, -0.2)
                lighter_color = self._adjust_brightness(base_color, 0.15)  # Glass reflections
            
            elif style == "classical":
                if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                    base_color = "#E8C39E"  # Warm sandstone
                elif time_of_day == "night":
                    base_color = "#8D6E63"  # Dark sandstone
                else:  # Day
                    base_color = "#BCAAA4"  # Light sandstone
                
                darker_color = self._adjust_brightness(base_color, -0.15)
                lighter_color = self._adjust_brightness(base_color, 0.1)
            
            elif style == "gothic":
                if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                    base_color = "#7E57C2"  # Purple-tinted stone
                elif time_of_day == "night":
                    base_color = "#4527A0"  # Dark purple stone
                else:  # Day
                    base_color = "#616161"  # Gray stone
                
                darker_color = self._adjust_brightness(base_color, -0.25)  # Deeper shadows for gothic
                lighter_color = self._adjust_brightness(base_color, 0.1)
            
            elif style == "brutalist":
                # Brutalist architecture is characterized by exposed concrete
                if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                    base_color = "#A1887F"  # Warm concrete
                elif time_of_day == "night":
                    base_color = "#424242"  # Dark concrete
                else:  # Day
                    base_color = "#9E9E9E"  # Light concrete
                
                darker_color = self._adjust_brightness(base_color, -0.15)
                lighter_color = self._adjust_brightness(base_color, 0.05)  # Minimal reflections
            
            elif style == "art_deco":
                # Art Deco often used rich colors and glamorous materials
                if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                    base_color = "#FBC02D"  # Gold-tinted
                elif time_of_day == "night":
                    base_color = "#5D4037"  # Dark rich brown
                else:  # Day
                    base_color = "#E6EE9C"  # Pale yellow
                
                darker_color = self._adjust_brightness(base_color, -0.15)
                lighter_color = self._adjust_brightness(base_color, 0.2)  # Good contrast
            
            elif style == "victorian":
                # Victorian often used rich, saturated colors
                if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                    base_color = "#C62828"  # Rich warm red
                elif time_of_day == "night":
                    base_color = "#4E342E"  # Dark brown
                else:  # Day
                    base_color = "#8D6E63"  # Brown
                
                darker_color = self._adjust_brightness(base_color, -0.2)
                lighter_color = self._adjust_brightness(base_color, 0.15)
            
            elif style == "futuristic":
                if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
                    base_color = "#B3E5FC"  # Sky blue with warm reflections
                elif time_of_day == "night":
                    base_color = "#0288D1"  # Glowing blue
                else:  # Day
                    base_color = "#81D4FA"  # Light blue glass
                
                darker_color = self._adjust_brightness(base_color, -0.1)
                lighter_color = self._adjust_brightness(base_color, 0.3)  # Strong reflections for futuristic
            
            else:  # Default
                base_color = "#9E9E9E"  # Medium gray
                darker_color = "#757575"  # Dark gray
                lighter_color = "#BDBDBD"  # Light gray
                
            # Override colors if materials are specified
            if materials:
                # Use the first material as primary influence for coloring
                if "brick" in materials[0].lower():
                    base_color = "#A52A2A"  # Brick red
                elif "stone" in materials[0].lower():
                    base_color = "#696969"  # Stone gray
                elif "glass" in materials[0].lower():
                    base_color = "#87CEEB"  # Sky blue for glass
                elif "concrete" in materials[0].lower():
                    base_color = "#808080"  # Concrete gray
                elif "wood" in materials[0].lower():
                    base_color = "#A0522D"  # Wood brown
                    
                # Recalculate shades based on new material color
                darker_color = self._adjust_brightness(base_color, -0.2)
                lighter_color = self._adjust_brightness(base_color, 0.15)
        
        # Create main building shape with gradient
        building = ET.SubElement(building_group, "rect")
        building.set("x", str(x))
        building.set("y", str(y))
        building.set("width", str(width))
        building.set("height", str(height))
        
        # Create gradient based on style
        gradient = ET.SubElement(building_group, "linearGradient", { \
            "id": gradient_id,
            "x1": "0%", \
            "y1": "0%"})
        
        # Different gradient directions based on style and time of day
        if style in ["modern", "futuristic"]:
            # Horizontal gradient for glass buildings to simulate light reflection
            gradient.set("x2", "100%")
            gradient.set("y2", "0%")
            
            # Glass buildings get more dramatic gradients with more stops
            ET.SubElement(gradient, "stop", {"offset": "0%", "stop-color": darker_color})
            ET.SubElement(gradient, "stop", {"offset": "40%", "stop-color": base_color})
            ET.SubElement(gradient, "stop", {"offset": "60%", "stop-color": lighter_color})
            ET.SubElement(gradient, "stop", {"offset": "100%", "stop-color": base_color})
        else:
            # Vertical gradient for stone/classical buildings for subtle shading
            gradient.set("x2", "0%")
            gradient.set("y2", "100%")
            
            # Simpler gradient for stone buildings
            ET.SubElement(gradient, "stop", {"offset": "0%", "stop-color": lighter_color})
            ET.SubElement(gradient, "stop", {"offset": "100%", "stop-color": darker_color})
        
        building.set("fill", f"url(#{gradient_id})")
        building.set("stroke", darker_color)
        building.set("stroke-width", "1")
        
    def _generate_architectural_details(self, building_group, x, y, width, height, style, features=None, materials=None, time_of_day="day", detail_level=0.7):
        """Generate architectural details based on building style and features"""
        # Default to empty feature list if none provided
        if features is None:
            features = []
            
        if materials is None:
            materials = []
            
        # Generate style-specific architectural elements
        if style == "modern":
            self._add_modern_elements(building_group, x, y, width, height, detail_level, time_of_day)
        elif style == "classical":
            self._add_classical_elements(building_group, x, y, width, height, detail_level, time_of_day)
        elif style == "gothic":
            self._add_gothic_elements(building_group, x, y, width, height, detail_level, time_of_day)
        elif style == "brutalist":
            self._add_brutalist_elements(building_group, x, y, width, height, detail_level, time_of_day)
        elif style == "art_deco":
            self._add_art_deco_elements(building_group, x, y, width, height, detail_level, time_of_day)
        elif style == "victorian":
            self._add_victorian_elements(building_group, x, y, width, height, detail_level, time_of_day)
        elif style == "futuristic":
            self._add_futuristic_elements(building_group, x, y, width, height, detail_level, time_of_day)
        else:
            # Default - add generic elements
            self._add_windows(building_group, x, y, width, height, style, "grid", 5, time_of_day, detail_level)
            self._add_generic_door(building_group, x, y, width, height)
            
        # Add additional features based on specific architectural elements in features list
        if "columns" in features:
            self._add_columns(building_group, x, y, width, height, style, detail_level)
        if "balcony" in features:
            self._add_balcony(building_group, x, y, width, height, style, detail_level)
        if "dome" in features:
            self._add_dome(building_group, x, y, width, height, style, detail_level)
        if "arches" in features:
            self._add_arches(building_group, x, y, width, height, style, detail_level)
            
        # Add roof for styles that typically have visible roofs
        if style in ["classical", "victorian", "gothic"] and "flat_roof" not in features:
            self._add_roof(building_group, x, y, width, height, style, detail_level)

    def _add_modern_elements(self, parent, x, y, width, height, detail_level, time_of_day):
        """Add modern architectural elements."""
        # Add large glass panels
        glass_width = width * 0.9
        glass_height = height * 0.8
        glass_x = x + (width - glass_width) / 2
        glass_y = y + height * 0.1
        
        # Determine glass color based on time of day
        if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
            glass_color = "rgba(255, 190, 150, 0.5)"  # Warm reflective tint
        elif time_of_day == "night":
            glass_color = "rgba(30, 60, 90, 0.6)"  # Dark blue with some transparency
        else:  # Day
            glass_color = "rgba(120, 180, 210, 0.5)"  # Light blue reflective
            
        glass = ET.SubElement(parent, "rect")
        glass.set("x", str(glass_x))
        glass.set("y", str(glass_y))
        glass.set("width", str(glass_width))
        glass.set("height", str(glass_height))
        glass.set("fill", glass_color)
        glass.set("stroke", "#CCCCCC")
        glass.set("stroke-width", "1")
        
        # Add floors/window bands
        floors = max(3, int(height / 50))  # At least 3 floors
        floor_height = glass_height / floors
        
        for i in range(1, floors):
            floor_y = glass_y + i * floor_height
            floor_line = ET.SubElement(parent, "line")
            floor_line.set("x1", str(glass_x))
            floor_line.set("y1", str(floor_y))
            floor_line.set("x2", str(glass_x + glass_width))
            floor_line.set("y2", str(floor_y))
            floor_line.set("stroke", "#FFFFFF")
            floor_line.set("stroke-width", "1")
            floor_line.set("stroke-opacity", "0.7")
        
        # Add modern entrance
        door_width = width * 0.2
        door_height = height * 0.1
        door_x = x + (width - door_width) / 2
        door_y = y + height - door_height
        
        door = ET.SubElement(parent, "rect")
        door.set("x", str(door_x))
        door.set("y", str(door_y))
        door.set("width", str(door_width))
        door.set("height", str(door_height))
        door.set("fill", glass_color)
        door.set("stroke", "#CCCCCC")
        door.set("stroke-width", "2")
        
        # Add roof details if detail level is high
        if detail_level > 0.6:
            # Mechanical elements or rooftop garden
            roof_detail = ET.SubElement(parent, "rect")
            roof_detail.set("x", str(x + width * 0.3))
            roof_detail.set("y", str(y - height * 0.03))
            roof_detail.set("width", str(width * 0.4))
            roof_detail.set("height", str(height * 0.03))
            roof_detail.set("fill", "#555555")
            
            # Solar panels or roof features
            if random.random() > 0.5:  # 50% chance
                solar_panel = ET.SubElement(parent, "rect")
                solar_panel.set("x", str(x + width * 0.35))
                solar_panel.set("y", str(y - height * 0.02))
                solar_panel.set("width", str(width * 0.3))
                solar_panel.set("height", str(height * 0.02))
                solar_panel.set("fill", "#1E5799")
                solar_panel.set("stroke", "#CCCCCC")
                solar_panel.set("stroke-width", "0.5")

    def _add_classical_elements(self, parent, x, y, width, height, detail_level, time_of_day):
        """Add classical architectural elements."""
        # Add columns
        column_count = max(4, int(width / 40))  # At least 4 columns
        column_width = width * 0.05
        column_spacing = width / column_count
        
        for i in range(column_count):
            column_x = x + i * column_spacing
            
            # Column shaft
            column = ET.SubElement(parent, "rect")
            column.set("x", str(column_x))
            column.set("y", str(y + height * 0.2))
            column.set("width", str(column_width))
            column.set("height", str(height * 0.7))
            column.set("fill", "#DDDDDD")
            
            # Column capital (top)
            capital_width = column_width * 1.5
            capital = ET.SubElement(parent, "rect")
            capital.set("x", str(column_x - (capital_width - column_width) / 2))
            capital.set("y", str(y + height * 0.2))
            capital.set("width", str(capital_width))
            capital.set("height", str(height * 0.05))
            capital.set("fill", "#DDDDDD")
            
            # Column base
            base_width = column_width * 1.5
            base = ET.SubElement(parent, "rect")
            base.set("x", str(column_x - (base_width - column_width) / 2))
            base.set("y", str(y + height * 0.9 - height * 0.05))
            base.set("width", str(base_width))
            base.set("height", str(height * 0.05))
            base.set("fill", "#DDDDDD")
        
        # Add pediment (triangular element above entrance)
        if detail_level > 0.5:
            pediment_width = width * 0.6
            pediment_height = height * 0.15
            pediment_x = x + (width - pediment_width) / 2
            pediment_y = y + height * 0.2
            
            pediment_points = [ \
                f"{pediment_x},{pediment_y}",  # Bottom left
                f"{pediment_x + pediment_width},{pediment_y}",  # Bottom right
                f"{pediment_x + pediment_width/2},{pediment_y - pediment_height}"  # Top center
            ]
            
            pediment = ET.SubElement(parent, "polygon")
            pediment.set("points", " ".join(pediment_points))
            pediment.set("fill", "#E8E8E8")
            pediment.set("stroke", "#CCCCCC")
            pediment.set("stroke-width", "1")
        
        # Add windows
        self._add_windows(parent, x, y + height * 0.3, width, height * 0.6, "classical", "grid", 3, time_of_day, detail_level)
        
        # Add ornate entrance
        door_width = width * 0.15
        door_height = height * 0.2
        door_x = x + (width - door_width) / 2
        door_y = y + height - door_height
        
        door = ET.SubElement(parent, "rect")
        door.set("x", str(door_x))
        door.set("y", str(door_y))
        door.set("width", str(door_width))
        door.set("height", str(door_height))
        door.set("fill", "#8B4513")  # Dark wood color
        door.set("stroke", "#664229")
        door.set("stroke-width", "1")

    def _add_gothic_elements(self, parent, x, y, width, height, detail_level, time_of_day):
        """Add gothic architectural elements."""
        # Add pointed arches for windows
        window_count = max(3, int(width / 50))  # At least 3 windows
        window_width = width * 0.15
        window_height = height * 0.3
        window_spacing = (width - window_count * window_width) / (window_count + 1)
        
        for i in range(window_count):
            window_x = x + window_spacing + i * (window_width + window_spacing)
            window_y = y + height * 0.3
            
            # Window background
            window_bg = ET.SubElement(parent, "rect")
            window_bg.set("x", str(window_x))
            window_bg.set("y", str(window_y))
            window_bg.set("width", str(window_width))
            window_bg.set("height", str(window_height * 0.8))  # Main part of window
            window_bg.set("fill", "rgba(72, 61, 139, 0.7)")  # Dark slate blue
            
            # Pointed arch top
            arch_height = window_height * 0.2
            arch_points = [ \
                f"{window_x},{window_y + window_height * 0.8}",  # Bottom left
                f"{window_x + window_width},{window_y + window_height * 0.8}",  # Bottom right
                f"{window_x + window_width/2},{window_y}"  # Top center (pointed)
            ]
            
            arch = ET.SubElement(parent, "polygon")
            arch.set("points", " ".join(arch_points))
            arch.set("fill", "rgba(72, 61, 139, 0.7)")  # Dark slate blue
            
            # Add tracery (decorative stonework) if detail level is high
            if detail_level > 0.7:
                # Vertical divider
                tracery_v = ET.SubElement(parent, "line")
                tracery_v.set("x1", str(window_x + window_width/2))
                tracery_v.set("y1", str(window_y))
                tracery_v.set("x2", str(window_x + window_width/2))
                tracery_v.set("y2", str(window_y + window_height * 0.8))
                tracery_v.set("stroke", "#DDDDDD")
                tracery_v.set("stroke-width", "2")
                
                # Rose window element (simplified)
                if window_width > 30 and random.random() > 0.7:  # Only for larger windows, 30% chance
                    rose_radius = window_width * 0.25
                    rose = ET.SubElement(parent, "circle")
                    rose.set("cx", str(window_x + window_width/2))
                    rose.set("cy", str(window_y + window_height * 0.4))
                    rose.set("r", str(rose_radius))
                    rose.set("fill", "none")
                    rose.set("stroke", "#DDDDDD")
                    rose.set("stroke-width", "2")
        
        # Add spires if detail level is high
        if detail_level > 0.6:
            spire_count = random.randint(1, 3)
            spire_width = width * 0.1
            spire_height = height * 0.3
            
            for i in range(spire_count):
                spire_x = x + width * ((i+0.5) / (spire_count+1)) - spire_width/2
                
                # Main spire triangular top
                spire_points = [ \
                    f"{spire_x},{y}",  # Bottom left
                    f"{spire_x + spire_width},{y}",  # Bottom right
                    f"{spire_x + spire_width/2},{y - spire_height}"  # Top
                ]
                
                spire = ET.SubElement(parent, "polygon")
                spire.set("points", " ".join(spire_points))
                spire.set("fill", "#555555")
                spire.set("stroke", "#333333")
                spire.set("stroke-width", "1")
                
    def _add_brutalist_elements(self, parent, x, y, width, height, detail_level, time_of_day):
        """Add brutalist architectural elements."""
        # Brutalist architecture is characterized by exposed concrete, geometric shapes,
        # and modular, repetitive elements
        
        # Create the main concrete block
        main_block = ET.SubElement(parent, "rect")
        main_block.set("x", str(x))
        main_block.set("y", str(y))
        main_block.set("width", str(width))
        main_block.set("height", str(height))
        
        # Concrete color based on time of day
        if time_of_day in ["dawn", "sunset", "golden hour"]:
            concrete_color = "#A1887F"  # Warmer concrete
        elif time_of_day == "night":
            concrete_color = "#424242"  # Dark concrete
        else:  # Day
            concrete_color = "#9E9E9E"  # Standard concrete
            
        main_block.set("fill", concrete_color)
        main_block.set("stroke", "#757575")
        main_block.set("stroke-width", "2")
        
        # Add geometric patterns/cutouts typical of brutalist design
        patterns = random.randint(3, 6)  # Number of patterns to add
        for i in range(patterns):
            # Determine pattern size and position
            pattern_width = width * random.uniform(0.15, 0.3)
            pattern_height = height * random.uniform(0.15, 0.3)
            pattern_x = x + random.uniform(0.1, 0.7) * width
            pattern_y = y + random.uniform(0.1, 0.7) * height
            
            # Ensure pattern stays within building bounds
            if pattern_x + pattern_width > x + width:
                pattern_width = (x + width) - pattern_x
            if pattern_y + pattern_height > y + height:
                pattern_height = (y + height) - pattern_y
            
            # Create pattern element - could be a recessed panel or window
            pattern = ET.SubElement(parent, "rect")
            pattern.set("x", str(pattern_x))
            pattern.set("y", str(pattern_y))
            pattern.set("width", str(pattern_width))
            pattern.set("height", str(pattern_height))
            
            # 50% chance for window, 50% chance for concrete panel
            if random.random() > 0.5:
                pattern.set("fill", "rgba(40, 40, 40, 0.7)")  # Dark window
                pattern.set("stroke", "#616161")
                pattern.set("stroke-width", "1")
            else:
                # Slightly different shade of concrete for visual interest
                pattern.set("fill", self._adjust_brightness(concrete_color, -0.15))
                pattern.set("stroke", "#616161")
                pattern.set("stroke-width", "1")
        
        # Add modular elements that protrude from the building if detail level is high
        if detail_level > 0.6:
            modules = random.randint(2, 4)
            for i in range(modules):
                module_width = width * random.uniform(0.2, 0.4)
                module_height = height * random.uniform(0.1, 0.2)
                module_depth = width * 0.05  # Protrusion depth
                
                # Position modules along the facade
                module_x = x + random.uniform(0.1, 0.6) * width
                module_y = y + random.uniform(0.2, 0.7) * height
                
                # Create the protruding module
                module = ET.SubElement(parent, "rect")
                module.set("x", str(module_x))
                module.set("y", str(module_y))
                module.set("width", str(module_width))
                module.set("height", str(module_height))
                module.set("fill", concrete_color)
                module.set("stroke", "#757575")
                module.set("stroke-width", "1")
                
                # Add shadow effect to give depth
                shadow = ET.SubElement(parent, "rect")
                shadow.set("x", str(module_x))
                shadow.set("y", str(module_y + module_height))
                shadow.set("width", str(module_width))
                shadow.set("height", str(module_depth))
                shadow.set("fill", "#424242")
                shadow.set("opacity", "0.7")
                
    def _add_art_deco_elements(self, parent, x, y, width, height, detail_level, time_of_day):
        """Add art deco architectural elements."""
        # Art Deco is characterized by bold geometric patterns, rich colors,
        # streamlined forms, and decorative elements
        
        # Main building facade
        facade = ET.SubElement(parent, "rect")
        facade.set("x", str(x))
        facade.set("y", str(y))
        facade.set("width", str(width))
        facade.set("height", str(height))
        
        # Art Deco color palette based on time of day
        if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
            base_color = "#FBC02D"  # Gold-tinted
            accent_color = "#E65100"  # Deep orange
        elif time_of_day == "night":
            base_color = "#5D4037"  # Dark brown
            accent_color = "#212121"  # Nearly black
        else:  # Day
            base_color = "#E6EE9C"  # Pale yellow
            accent_color = "#4CAF50"  # Green
        
        facade.set("fill", base_color)
        facade.set("stroke", accent_color)
        facade.set("stroke-width", "2")
        
        # Add stepped/setback design typical of Art Deco skyscrapers
        if random.random() > 0.5:  # 50% chance
            steps = random.randint(2, 4)
            step_height = height * 0.2
            step_width = width * 0.15
            
            for i in range(steps):
                step_x = x + (i+1) * step_width
                step_y = y
                step_w = width - (i+1) * 2 * step_width
                if step_w <= 0:
                    break
                    
                step = ET.SubElement(parent, "rect")
                step.set("x", str(step_x))
                step.set("y", str(step_y))
                step.set("width", str(step_w))
                step.set("height", str(step_height))
                step.set("fill", self._adjust_brightness(base_color, -0.1 * (i+1)))
                step.set("stroke", accent_color)
                step.set("stroke-width", "1")
        
        # Add geometric decoration patterns
        if detail_level > 0.5:
            # Vertical decorative lines
            line_count = max(3, int(width / 20))
            line_spacing = width / line_count
            line_height = height * 0.6
            line_y = y + height * 0.2
            
            for i in range(line_count):
                line_x = x + i * line_spacing
                
                line = ET.SubElement(parent, "line")
                line.set("x1", str(line_x))
                line.set("y1", str(line_y))
                line.set("x2", str(line_x))
                line.set("y2", str(line_y + line_height))
                line.set("stroke", accent_color)
                line.set("stroke-width", "1")
                
            # Horizontal decorative band
            band_height = height * 0.05
            band_y = y + height * 0.3
            
            band = ET.SubElement(parent, "rect")
            band.set("x", str(x))
            band.set("y", str(band_y))
            band.set("width", str(width))
            band.set("height", str(band_height))
            band.set("fill", accent_color)
            
            # Zigzag pattern on the band if detail level is very high
            if detail_level > 0.7:
                zigzag_width = width
                zigzag_height = band_height * 0.7
                zigzag_points = []
                zigzag_sections = 10
                section_width = zigzag_width / zigzag_sections
                
                for i in range(zigzag_sections + 1):
                    zigzag_x = x + i * section_width
                    if i % 2 == 0:
                        zigzag_y = band_y + zigzag_height * 0.2
                    else:
                        zigzag_y = band_y + zigzag_height * 0.8
                    zigzag_points.append(f"{zigzag_x},{zigzag_y}")
                
                zigzag = ET.SubElement(parent, "polyline")
                zigzag.set("points", " ".join(zigzag_points))
                zigzag.set("fill", "none")
                zigzag.set("stroke", self._adjust_brightness(base_color, 0.3))
                zigzag.set("stroke-width", "2")
                
        # Add stylized entrance
        door_width = width * 0.2
        door_height = height * 0.15
        door_x = x + (width - door_width) / 2
        door_y = y + height - door_height
        
        # Door frame with geometric pattern
        door_frame = ET.SubElement(parent, "rect")
        door_frame.set("x", str(door_x - door_width * 0.1))
        door_frame.set("y", str(door_y - door_height * 0.1))
        door_frame.set("width", str(door_width * 1.2))
        door_frame.set("height", str(door_height * 1.2))
        door_frame.set("fill", accent_color)
        
        # Door itself
        door = ET.SubElement(parent, "rect")
        door.set("x", str(door_x))
        door.set("y", str(door_y))
        door.set("width", str(door_width))
        door.set("height", str(door_height))
        door.set("fill", self._adjust_brightness(base_color, 0.1))
        door.set("stroke", accent_color)
        door.set("stroke-width", "1")
        
    def _add_victorian_elements(self, parent, x, y, width, height, detail_level, time_of_day):
        """Add Victorian architectural elements."""
        # Victorian architecture is characterized by ornate details, bay windows,
        # decorative trim, towers/turrets, and steep roofs
        
        # Main building facade
        facade = ET.SubElement(parent, "rect")
        facade.set("x", str(x))
        facade.set("y", str(y))
        facade.set("width", str(width))
        facade.set("height", str(height))
        
        # Victorian color palette - often featured rich, bold colors
        if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
            base_color = "#C62828"  # Rich warm red
            trim_color = "#FFD54F"  # Warm gold
        elif time_of_day == "night":
            base_color = "#4A148C"  # Deep purple
            trim_color = "#5D4037"  # Dark brown
        else:  # Day
            base_color = "#7B1FA2"  # Purple
            trim_color = "#3E2723"  # Dark brown
        
        facade.set("fill", base_color)
        facade.set("stroke", trim_color)
        facade.set("stroke-width", "2")
        
        # Add bay windows - a Victorian signature
        if detail_level > 0.5:
            # Number of bay windows depends on building width
            bay_count = max(1, int(width / 80))
            bay_width = width / (bay_count * 3)
            bay_height = height * 0.4
            
            for i in range(bay_count):
                # Position bay windows with even spacing
                bay_x = x + width * (i + 0.5) / (bay_count + 1) - bay_width / 2
                bay_y = y + height * 0.3
                
                # Protruding bay window
                bay = ET.SubElement(parent, "rect")
                bay.set("x", str(bay_x))
                bay.set("y", str(bay_y))
                bay.set("width", str(bay_width))
                bay.set("height", str(bay_height))
                bay.set("fill", self._adjust_brightness(base_color, 0.1))
                bay.set("stroke", trim_color)
                bay.set("stroke-width", "1")
                
                # Add ornate top to bay window
                ornate_height = bay_height * 0.1
                ornate = ET.SubElement(parent, "rect")
                ornate.set("x", str(bay_x - bay_width * 0.1))
                ornate.set("y", str(bay_y - ornate_height))
                ornate.set("width", str(bay_width * 1.2))
                ornate.set("height", str(ornate_height))
                ornate.set("fill", trim_color)
                
                # Add windows to bay
                window_width = bay_width * 0.7
                window_height = bay_height * 0.3
                window_x = bay_x + (bay_width - window_width) / 2
                
                # Add 2-3 windows per bay depending on detail level
                window_count = 3 if detail_level > 0.7 else 2
                window_spacing = bay_height * 0.6 / window_count
                
                for j in range(window_count):
                    window_y = bay_y + bay_height * 0.1 + j * window_spacing
                    
                    window = ET.SubElement(parent, "rect")
                    window.set("x", str(window_x))
                    window.set("y", str(window_y))
                    window.set("width", str(window_width))
                    window.set("height", str(window_height))
                    window.set("fill", "#FFFDE7")
                    window.set("stroke", trim_color)
                    window.set("stroke-width", "1")
        
        # Add decorative crown/cornice at top
        if detail_level > 0.4:
            cornice_height = height * 0.05
            cornice = ET.SubElement(parent, "rect")
            cornice.set("x", str(x - width * 0.02))
            cornice.set("y", str(y))
            cornice.set("width", str(width * 1.04))  # Slightly wider than building
            cornice.set("height", str(cornice_height))
            cornice.set("fill", trim_color)
            
            # Add ornate brackets under cornice
            bracket_count = max(4, int(width / 30))
            bracket_width = width * 0.02
            bracket_height = cornice_height * 2
            bracket_spacing = width / bracket_count
            
            for i in range(bracket_count):
                bracket_x = x + i * bracket_spacing
                bracket_y = y + cornice_height
                
                bracket = ET.SubElement(parent, "rect")
                bracket.set("x", str(bracket_x))
                bracket.set("y", str(bracket_y))
                bracket.set("width", str(bracket_width))
                bracket.set("height", str(bracket_height))
                bracket.set("fill", trim_color)
        
        # Add ornate Victorian door
        door_width = width * 0.15
        door_height = height * 0.25
        door_x = x + (width - door_width) / 2
        door_y = y + height - door_height
        
        # Decorative door frame
        door_frame = ET.SubElement(parent, "rect")
        door_frame.set("x", str(door_x - door_width * 0.2))
        door_frame.set("y", str(door_y - door_height * 0.1))
        door_frame.set("width", str(door_width * 1.4))
        door_frame.set("height", str(door_height * 1.1))
        door_frame.set("fill", trim_color)
        
        # Door itself
        door = ET.SubElement(parent, "rect")
        door.set("x", str(door_x))
        door.set("y", str(door_y))
        door.set("width", str(door_width))
        door.set("height", str(door_height))
        door.set("fill", "#5D4037")  # Dark wood
        door.set("stroke", trim_color)
        door.set("stroke-width", "1")
        
        # Add detailed trim at roof line if detail level is high
        if detail_level > 0.6:
            # Scalloped/gingerbread trim
            scallop_height = height * 0.03
            scallop_count = max(8, int(width / 15))
            scallop_width = width / scallop_count
            
            for i in range(scallop_count):
                scallop_x = x + i * scallop_width
                scallop_y = y + cornice_height  # Position right below cornice
                
                # Simplified scallop as a small rectangle
                scallop = ET.SubElement(parent, "rect")
                scallop.set("x", str(scallop_x))
                scallop.set("y", str(scallop_y))
                scallop.set("width", str(scallop_width * 0.7))
                scallop.set("height", str(scallop_height))
                scallop.set("fill", trim_color)
                
    def _add_futuristic_elements(self, parent, x, y, width, height, detail_level, time_of_day):
        """Add futuristic architectural elements."""
        # Futuristic architecture is characterized by sleek shapes, glass/metal materials,
        # unusual geometric forms, and high-tech elements
        
        # Main building structure with asymmetrical design
        # Determine if we'll have an angled top or not
        has_angled_top = random.random() > 0.5
        
        # Vary the building shape a bit
        if has_angled_top:
            # Create a polygon for the main structure
            angle_height = height * random.uniform(0.2, 0.4)  # Height differential for angle
            building_points = [ \
                f"{x},{y + height}",  # Bottom left
                f"{x + width},{y + height}",  # Bottom right
                f"{x + width},{y + angle_height}",  # Top right
                f"{x},{y}"  # Top left
            ]
            
            main_structure = ET.SubElement(parent, "polygon")
            main_structure.set("points", " ".join(building_points))
        else:
            # Use a simple rectangle but with some extension on one side
            extension_width = width * 0.3
            extension_height = height * 0.5
            extension_x = random.choice([x - extension_width * 0.2, x + width - extension_width * 0.8])
            extension_y = y + random.uniform(0.1, 0.3) * height
            
            # Main rectangle
            main_structure = ET.SubElement(parent, "rect")
            main_structure.set("x", str(x))
            main_structure.set("y", str(y))
            main_structure.set("width", str(width))
            main_structure.set("height", str(height))
            
            # Extension piece
            extension = ET.SubElement(parent, "rect")
            extension.set("x", str(extension_x))
            extension.set("y", str(extension_y))
            extension.set("width", str(extension_width))
            extension.set("height", str(extension_height))
        
        # Color palette based on time of day
        if time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
            base_color = "#FF5722"  # Deep orange
            accent_color = "#FFD54F"  # Amber
            glass_color = "rgba(255, 235, 59, 0.6)"  # Yellow with transparency
        elif time_of_day == "night":
            base_color = "#212121"  # Very dark gray
            accent_color = "#00BCD4"  # Cyan
            glass_color = "rgba(0, 188, 212, 0.7)"  # Cyan with transparency
        else:  # Day
            base_color = "#607D8B"  # Blue gray
            accent_color = "#00BFA5"  # Teal
            glass_color = "rgba(178, 235, 242, 0.6)"  # Light cyan with transparency
        
        # Apply colors
        if has_angled_top:
            main_structure.set("fill", base_color)
        else:
            main_structure.set("fill", base_color)
            extension.set("fill", self._adjust_brightness(base_color, 0.1))
        
        # Add glass panels
        panel_count = max(3, int(width / 30))
        panel_width = width * 0.7 / panel_count
        
        for i in range(panel_count):
            # Vary panel heights for visual interest
            panel_height = height * random.uniform(0.5, 0.8)
            panel_x = x + width * 0.15 + i * panel_width
            panel_y = y + height - panel_height
            
            panel = ET.SubElement(parent, "rect")
            panel.set("x", str(panel_x))
            panel.set("y", str(panel_y))
            panel.set("width", str(panel_width * 0.8))
            panel.set("height", str(panel_height))
            panel.set("fill", glass_color)
            panel.set("stroke", accent_color)
            panel.set("stroke-width", "1")
        
        # Add high-tech elements if detail level is high
        if detail_level > 0.6:
            # Add futuristic antenna or spire
            antenna_height = height * 0.2
            antenna_width = width * 0.02
            antenna_x = x + width * 0.7
            
            antenna = ET.SubElement(parent, "line")
            antenna.set("x1", str(antenna_x))
            antenna.set("y1", str(y))
            antenna.set("x2", str(antenna_x))
            antenna.set("y2", str(y - antenna_height))
            antenna.set("stroke", accent_color)
            antenna.set("stroke-width", str(antenna_width))
            
            # Add lights at different positions along the antenna
            light_positions = [0.3, 0.6, 0.9]  # Positions along the antenna
            for pos in light_positions:
                light_y = y - antenna_height * pos
                light = ET.SubElement(parent, "circle")
                light.set("cx", str(antenna_x))
                light.set("cy", str(light_y))
                light.set("r", str(width * 0.01))
                light.set("fill", "#FFFF00")  # Yellow light
                
                # Add animation for blinking effect if at night
                if time_of_day == "night":
                    blink_anim = ET.SubElement(light, "animate")
                    blink_anim.set("attributeName", "opacity")
                    blink_anim.set("values", "1;0.3;1")
                    blink_anim.set("dur", f"{random.uniform(1.5, 3.0)}s")
                    blink_anim.set("repeatCount", "indefinite")
            
            # Add solar panels or other high-tech roof elements
            tech_element_width = width * 0.5
            tech_element_height = height * 0.05
            tech_element_x = x + (width - tech_element_width) / 2
            tech_element_y = y
            
            tech_element = ET.SubElement(parent, "rect")
            tech_element.set("x", str(tech_element_x))
            tech_element.set("y", str(tech_element_y))
            tech_element.set("width", str(tech_element_width))
            tech_element.set("height", str(tech_element_height))
            tech_element.set("fill", accent_color)
            tech_element.set("stroke", self._adjust_brightness(accent_color, -0.2))
            tech_element.set("stroke-width", "1")
            
            # Add geometric patterns on the tech element
            pattern_count = int(tech_element_width / 10)
            pattern_width = tech_element_width / pattern_count
            
            for i in range(pattern_count):
                pattern_x = tech_element_x + i * pattern_width
                
                pattern = ET.SubElement(parent, "line")
                pattern.set("x1", str(pattern_x))
                pattern.set("y1", str(tech_element_y))
                pattern.set("x2", str(pattern_x))
                pattern.set("y2", str(tech_element_y + tech_element_height))
                pattern.set("stroke", self._adjust_brightness(accent_color, 0.2))
                pattern.set("stroke-width", "0.5")
        
        # Add sleek entrance
        door_width = width * 0.15
        door_height = height * 0.1
        door_x = x + (width - door_width) / 2
        door_y = y + height - door_height
        
        door = ET.SubElement(parent, "rect")
        door.set("x", str(door_x))
        door.set("y", str(door_y))
        door.set("width", str(door_width))
        door.set("height", str(door_height))
        door.set("fill", glass_color)
        door.set("stroke", accent_color)
        door.set("stroke-width", "1")
        
        # Fixing indentation for decorative line settings
        deco_line.set("x1", str(spire_x + (spire_width - width_at_height) / 2))
        deco_line.set("y1", str(y_pos))
        deco_line.set("x2", str(spire_x + (spire_width + width_at_height) / 2))
        deco_line.set("y2", str(y_pos))
        deco_line.set("stroke", lighter_color)
        deco_line.set("stroke-width", "1")
        
        # Add pointed arch windows if detail level is high
        if detail_level > 0.6:
            # Placeholder for window pattern - will be handled later in specialized window method
            pass
            
        elif style == "futuristic":
            # Add futuristic elements
            # Angled top or asymmetrical element
            if random.random() > 0.5:  # 50% chance for angled top
                top_angle = ET.SubElement(building_group, "polygon")
                angle_height = height * random.uniform(0.1, 0.3)
                points = [ \
                    f"{x},{y}",  # Bottom left
                    f"{x + width},{y + angle_height}",  # Bottom right
                    f"{x + width},{y + height}",  # Top right
                    f"{x},{y + height}"  # Top left
                ]
                top_angle.set("points", " ".join(points))
                top_angle.set("fill", lighter_color)
                top_angle.set("opacity", "0.5")
            
            # Add antenna or spire for futuristic buildings
            if detail_level > 0.5:
                antenna_height = height * 0.15
                antenna_width = width * 0.02
                antenna_x = x + width * 0.8
                
                antenna = ET.SubElement(building_group, "line")
                antenna.set("x1", str(antenna_x))
                antenna.set("y1", str(y))
                antenna.set("x2", str(antenna_x))
                antenna.set("y2", str(y - antenna_height))
                antenna.set("stroke", lighter_color)
                antenna.set("stroke-width", str(antenna_width))
                
                # Add blinking light on top if it's night
                if time_of_day == "night":
                    blink_light = ET.SubElement(building_group, "circle")
                    blink_light.set("cx", str(antenna_x))
                    blink_light.set("cy", str(y - antenna_height))
                    blink_light.set("r", str(width * 0.01))
                    blink_light.set("fill", "#FF0000")  # Red light
                    
                    # Add animation for blinking effect
                    blink_anim = ET.SubElement(blink_light, "animate")
                    blink_anim.set("attributeName", "opacity")
                    blink_anim.set("values", "1;0;1")
                    blink_anim.set("dur", "2s")
                    blink_anim.set("repeatCount", "indefinite")
        
        # Generate architectural details based on style and features
        self._generate_architectural_details( \
            building_group, x, y, width, height, style,
            architectural_features, materials, time_of_day, detail_level
        )
        
        # Add environmental details based on time of day
        if time_of_day == "night":
            # Add random lit windows
            self._add_lit_windows(building_group, x, y, width, height, style, floors, detail_level)
        elif time_of_day in ["dawn", "sunrise", "golden hour", "sunset"]:
            # Add sun reflection
            if style in ["modern", "futuristic"]:
                self._add_sun_reflection(building_group, x, y, width, height, time_of_day)
        
        return building_group
    
    def _adjust_brightness(self, hex_color: str, amount: float, material: str = None, light_angle: float = 45.0, time_of_day: str = 'midday') -> str:
        """Advanced color adjustment with photometric inference and material-specific properties.
        
        This implements a sophisticated color transformation that simulates realistic lighting
        effects based on material properties, light angle, and time of day. It uses perceptual
        color adjustments to maintain visual consistency across different lighting conditions.
        
        Args:
            hex_color: Hex color code (e.g., '#RRGGBB')
            amount: Base adjustment amount (-1 to 1, negative darkens, positive lightens)
            material: Material type affecting light behavior (e.g., 'glass', 'metal', 'concrete')
            light_angle: Angle of incident light in degrees (0-360)
            time_of_day: Time period affecting lighting quality ('dawn', 'midday', 'sunset', 'night', etc.)
            
        Returns:
            Perceptually adjusted hex color with material-specific characteristics
        """
        # Convert hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        
        # Convert RGB to HSL for more perceptually accurate adjustments
        r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
        c_max = max(r_norm, g_norm, b_norm)
        c_min = min(r_norm, g_norm, b_norm)
        delta = c_max - c_min
        
        # Calculate lightness
        lightness = (c_max + c_min) / 2.0
        
        # Calculate saturation
        saturation = 0
        if delta != 0:
            saturation = delta / (1 - abs(2 * lightness - 1)) if lightness != 0 and lightness != 1 else 0
        
        # Calculate hue
        hue = 0
        if delta != 0:
            if c_max == r_norm:
                hue = 60 * (((g_norm - b_norm) / delta) % 6)
            elif c_max == g_norm:
                hue = 60 * (((b_norm - r_norm) / delta) + 2)
            else:  # c_max == b_norm
                hue = 60 * (((r_norm - g_norm) / delta) + 4)
        
        # Apply material-specific modifiers
        material_modifiers = { \
            'glass': {'lightness': 0.15, 'saturation': -0.1, 'hue_shift': 5},
            'metal': {'lightness': 0.05, 'saturation': -0.3, 'hue_shift': -5}, \
            'concrete': {'lightness': -0.1, 'saturation': -0.2, 'hue_shift': 0}, \
            'brick': {'lightness': -0.05, 'saturation': 0.1, 'hue_shift': 10}, \
            'wood': {'lightness': 0.0, 'saturation': 0.15, 'hue_shift': 15}, \
            'stone': {'lightness': -0.1, 'saturation': -0.05, 'hue_shift': 0}, \
            'marble': {'lightness': 0.2, 'saturation': -0.2, 'hue_shift': 0}, \
            'steel': {'lightness': 0.1, 'saturation': -0.4, 'hue_shift': -10}, \
            'plastic': {'lightness': 0.05, 'saturation': 0.2, 'hue_shift': 5}
        }
        
        # Apply time of day color modifications
        time_of_day_modifiers = { \
            'dawn': {'lightness': 0.05, 'saturation': -0.1, 'hue_shift': 15, 'red_bias': 0.15, 'blue_bias': -0.05},
            'sunrise': {'lightness': 0.1, 'saturation': 0.15, 'hue_shift': 10, 'red_bias': 0.2, 'blue_bias': -0.1}, \
            'morning': {'lightness': 0.05, 'saturation': 0.05, 'hue_shift': 0, 'red_bias': 0, 'blue_bias': 0.05}, \
            'midday': {'lightness': 0, 'saturation': 0, 'hue_shift': 0, 'red_bias': 0, 'blue_bias': 0}, \
            'afternoon': {'lightness': -0.02, 'saturation': 0.05, 'hue_shift': -5, 'red_bias': 0.05, 'blue_bias': -0.02}, \
            'golden hour': {'lightness': 0, 'saturation': 0.2, 'hue_shift': -15, 'red_bias': 0.2, 'blue_bias': -0.15}, \
            'sunset': {'lightness': -0.05, 'saturation': 0.25, 'hue_shift': -20, 'red_bias': 0.25, 'blue_bias': -0.2}, \
            'dusk': {'lightness': -0.1, 'saturation': -0.1, 'hue_shift': -25, 'red_bias': 0.1, 'blue_bias': 0.05}, \
            'night': {'lightness': -0.25, 'saturation': -0.2, 'hue_shift': -30, 'red_bias': -0.2, 'blue_bias': 0.2}, \
            'blue hour': {'lightness': -0.15, 'saturation': 0.05, 'hue_shift': 30, 'red_bias': -0.15, 'blue_bias': 0.25}
        }
        
        # Apply photometric inference based on light angle
        # Lower angles create more dramatic lighting effects
        angle_factor = abs(math.sin(math.radians(light_angle)))
        angle_intensity = 1.0 - (0.5 * angle_factor)
        
        # Initialize adjustment factors
        lightness_adjust = amount  # Base adjustment from the input amount
        saturation_adjust = 0
        hue_shift = 0
        red_bias = 0
        blue_bias = 0
        
        # Apply material modifiers if a material is specified
        if material and material in material_modifiers:
            lightness_adjust += material_modifiers[material]['lightness']
            saturation_adjust += material_modifiers[material]['saturation']
            hue_shift += material_modifiers[material]['hue_shift']
        
        # Apply time of day modifiers
        if time_of_day in time_of_day_modifiers:
            tod_mod = time_of_day_modifiers[time_of_day]
            lightness_adjust += tod_mod['lightness']
            saturation_adjust += tod_mod['saturation']
            hue_shift += tod_mod['hue_shift']
            red_bias += tod_mod['red_bias']
            blue_bias += tod_mod['blue_bias']
        
        # Apply the angle intensity as a multiplier for all adjustments
        lightness_adjust *= angle_intensity
        saturation_adjust *= angle_intensity
        
        # Apply adjustments to HSL values
        new_lightness = max(0, min(1, lightness + lightness_adjust))
        new_saturation = max(0, min(1, saturation + saturation_adjust))
        new_hue = (hue + hue_shift) % 360
        
        # Convert back to RGB using improved HSL to RGB conversion
        def hue_to_rgb(p, q, t):
            if t < 0:
                t += 1
            if t > 1:
                t -= 1
            if t < 1/6:
                return p + (q - p) * 6 * t
            if t < 1/2:
                return q
            if t < 2/3:
                return p + (q - p) * (2/3 - t) * 6
            return p
        
        q = new_lightness * (1 + new_saturation) if new_lightness < 0.5 else new_lightness + new_saturation - new_lightness * new_saturation
        p = 2 * new_lightness - q
        
        r_new = hue_to_rgb(p, q, (new_hue / 360 + 1/3))
        g_new = hue_to_rgb(p, q, new_hue / 360)
        b_new = hue_to_rgb(p, q, (new_hue / 360 - 1/3))
        
        # Apply the red and blue bias for time of day effects
        r_new = max(0, min(1, r_new + red_bias))
        g_new = max(0, min(1, g_new))
        b_new = max(0, min(1, b_new + blue_bias))
        
        # Convert back to 8-bit RGB values
        r_out = int(r_new * 255)
        g_out = int(g_new * 255)
        b_out = int(b_new * 255)
        
        # Return as hex
        return f'#{r_out:02x}{g_out:02x}{b_out:02x}'
        
    def _add_material_texture(self, element: ET.Element, material: str, x: float, y: float, width: float, height: float, detail_level: float):
        """Add procedural texture patterns to simulate material properties.
        
        This method generates vector-based micro-patterns to simulate material textures, \
        maintaining SVG scalability while enhancing visual richness without increasing file size.
        
        Args:
            element: SVG element to enhance with texture
            material: Material type to simulate ('brick', 'concrete', 'wood', etc.)
            x, y: Position coordinates
            width, height: Size of the element
            detail_level: Level of detail (0-1) to control texture complexity
        """
        # Skip texture for very low detail levels or small elements
        if detail_level < 0.4 or width < 10 or height < 10:
            return
            
        # Adjust pattern complexity based on detail level
        pattern_density = detail_level * 0.8  # scale to keep patterns reasonable
        
        # Create a pattern definition if not present
        pattern_id = f"pattern_{material}_{hash(str(x) + str(y) + str(width) + str(height))}"
        pattern_exists = False
        
        # Check if the pattern already exists in document definitions
        if hasattr(self, 'document') and hasattr(self.document, 'defs'):
            for child in self.document.defs:
                if child.get('id') == pattern_id:
                    pattern_exists = True
                    break
        
        if not pattern_exists:
            # Create pattern element
            pattern = ET.SubElement(element, "pattern")
            pattern.set("id", pattern_id)
            pattern.set("patternUnits", "userSpaceOnUse")
            
            # Material-specific texture patterns
            if material == "brick":
                self._create_brick_pattern(pattern, width, height, pattern_density)
            elif material == "concrete":
                self._create_concrete_pattern(pattern, width, height, pattern_density)
            elif material == "metal":
                self._create_metal_pattern(pattern, width, height, pattern_density)
            elif material == "glass":
                self._create_glass_pattern(pattern, width, height, pattern_density)
            elif material == "wood":
                self._create_wood_pattern(pattern, width, height, pattern_density)
            elif material == "stone":
                self._create_stone_pattern(pattern, width, height, pattern_density)
            elif material == "marble":
                self._create_marble_pattern(pattern, width, height, pattern_density)
            
        # Apply pattern to element
        if not pattern_exists:
            element.set("fill", f"url(#{pattern_id})")
    
    def _create_brick_pattern(self, parent: ET.Element, width: float, height: float, density: float):
        """Create a brick texture pattern."""
        # Set pattern properties
        brick_width = min(20, max(10, width * 0.1))
        brick_height = brick_width * 0.4
        pattern_width = brick_width * 2
        pattern_height = brick_height * 2
        
        parent.set("width", str(pattern_width))
        parent.set("height", str(pattern_height))
        
        # Background fill
        bg = ET.SubElement(parent, "rect")
        bg.set("width", str(pattern_width))
        bg.set("height", str(pattern_height))
        bg.set("fill", "#d6c0b4")
        
        # Create brick rows with staggered pattern
        for row in range(3):
            y_pos = row * brick_height
            offset = 0 if row % 2 == 0 else brick_width * 0.5
            
            for col in range(3):
                x_pos = col * brick_width + offset
                if x_pos < pattern_width and y_pos < pattern_height:
                    # Individual brick
                    brick = ET.SubElement(parent, "rect")
                    brick.set("x", str(x_pos))
                    brick.set("y", str(y_pos))
                    brick.set("width", str(brick_width * 0.95))
                    brick.set("height", str(brick_height * 0.9))
                    brick.set("rx", "1")
                    brick.set("ry", "1")
                    brick.set("fill", "#c9b1a3")
                    brick.set("stroke", "#b9a193")
                    brick.set("stroke-width", "0.5")
    
    def _create_concrete_pattern(self, parent: ET.Element, width: float, height: float, density: float):
        """Create a concrete texture pattern."""
        pattern_size = min(30, max(15, width * 0.05))
        parent.set("width", str(pattern_size))
        parent.set("height", str(pattern_size))
        
        # Background fill
        bg = ET.SubElement(parent, "rect")
        bg.set("width", str(pattern_size))
        bg.set("height", str(pattern_size))
        bg.set("fill", "#d8d8d8")
        
        # Add noise dots for concrete texture
        num_dots = int(pattern_size * pattern_size * density * 0.1)
        for _ in range(num_dots):
            x_pos = random.uniform(0, pattern_size)
            y_pos = random.uniform(0, pattern_size)
            size = random.uniform(0.5, 1.5)
            
            dot = ET.SubElement(parent, "circle")
            dot.set("cx", str(x_pos))
            dot.set("cy", str(y_pos))
            dot.set("r", str(size))
            
            # Randomly make dots darker or lighter than background
            if random.random() > 0.5:
                dot.set("fill", "#c0c0c0")
            else:
                dot.set("fill", "#e5e5e5")
            
            dot.set("opacity", str(random.uniform(0.1, 0.3)))
    
    def _create_metal_pattern(self, parent: ET.Element, width: float, height: float, density: float):
        """Create a metal texture pattern."""
        pattern_width = min(40, max(20, width * 0.05))
        pattern_height = pattern_width
        
        parent.set("width", str(pattern_width))
        parent.set("height", str(pattern_height))
        
        # Background fill - metallic base
        bg = ET.SubElement(parent, "rect")
        bg.set("width", str(pattern_width))
        bg.set("height", str(pattern_height))
        bg.set("fill", "#c0c0c0")
        
        # Add linear gradient to simulate light reflection
        gradient = ET.SubElement(parent, "linearGradient")
        gradient.set("id", f"metalGradient_{id(parent)}")
        gradient.set("x1", "0%")
        gradient.set("y1", "0%")
        gradient.set("x2", "100%")
        gradient.set("y2", "100%")
        
        # Define gradient stops
        stop1 = ET.SubElement(gradient, "stop")
        stop1.set("offset", "0%")
        stop1.set("stop-color", "#ffffff")
        stop1.set("stop-opacity", "0.7")
        
        stop2 = ET.SubElement(gradient, "stop")
        stop2.set("offset", "50%")
        stop2.set("stop-color", "#c0c0c0")
        stop2.set("stop-opacity", "0.2")
        
        stop3 = ET.SubElement(gradient, "stop")
        stop3.set("offset", "100%")
        stop3.set("stop-color", "#808080")
        stop3.set("stop-opacity", "0.5")
        
        # Apply gradient
        overlay = ET.SubElement(parent, "rect")
        overlay.set("width", str(pattern_width))
        overlay.set("height", str(pattern_height))
        overlay.set("fill", f"url(#metalGradient_{id(parent)})")
        
        # Add subtle scratches based on density
        num_scratches = int(pattern_width * density * 2)
        for _ in range(num_scratches):
            x1 = random.uniform(0, pattern_width)
            y1 = random.uniform(0, pattern_height)
            length = random.uniform(2, 8)
            angle = random.uniform(0, 360)
            
            # Calculate end point based on angle and length
            x2 = x1 + length * math.cos(math.radians(angle))
            y2 = y1 + length * math.sin(math.radians(angle))
            
            scratch = ET.SubElement(parent, "line")
            scratch.set("x1", str(x1))
            scratch.set("y1", str(y1))
            scratch.set("x2", str(x2))
            scratch.set("y2", str(y2))
            scratch.set("stroke", "#ffffff")
            scratch.set("stroke-width", "0.3")
            scratch.set("opacity", str(random.uniform(0.1, 0.3)))
    
    def _create_glass_pattern(self, parent: ET.Element, width: float, height: float, density: float):
        """Create a glass texture pattern."""
        pattern_size = min(50, max(30, width * 0.08))
        parent.set("width", str(pattern_size))
        parent.set("height", str(pattern_size))
        
        # Background - transparent with slight color
        bg = ET.SubElement(parent, "rect")
        bg.set("width", str(pattern_size))
        bg.set("height", str(pattern_size))
        bg.set("fill", "#e0f0ff")
        bg.set("opacity", "0.2")
        
        # Create radial gradient for glass reflection
        gradient = ET.SubElement(parent, "radialGradient")
        gradient.set("id", f"glassGradient_{id(parent)}")
        gradient.set("cx", "30%")
        gradient.set("cy", "30%")
        gradient.set("r", "70%")
        
        # Define gradient stops
        stop1 = ET.SubElement(gradient, "stop")
        stop1.set("offset", "0%")
        stop1.set("stop-color", "#ffffff")
        stop1.set("stop-opacity", "0.9")
        
        stop2 = ET.SubElement(gradient, "stop")
        stop2.set("offset", "40%")
        stop2.set("stop-color", "#ffffff")
        stop2.set("stop-opacity", "0.1")
        
        stop3 = ET.SubElement(gradient, "stop")
        stop3.set("offset", "100%")
        stop3.set("stop-color", "#ffffff")
        stop3.set("stop-opacity", "0")
        
        # Apply reflection gradient
        reflection = ET.SubElement(parent, "rect")
        reflection.set("width", str(pattern_size))
        reflection.set("height", str(pattern_size))
        reflection.set("fill", f"url(#glassGradient_{id(parent)})")
        reflection.set("opacity", "0.3")
    
    def _create_wood_pattern(self, parent: ET.Element, width: float, height: float, density: float):
        """Create a wood grain texture pattern."""
        pattern_width = min(60, max(30, width * 0.1))
        pattern_height = pattern_width * 2
        
        parent.set("width", str(pattern_width))
        parent.set("height", str(pattern_height))
        
        # Background color
        bg = ET.SubElement(parent, "rect")
        bg.set("width", str(pattern_width))
        bg.set("height", str(pattern_height))
        bg.set("fill", "#d2aa6d")
        
        # Create wood grain using curved paths
        num_grain_lines = int(pattern_height * density * 0.3)
        y_interval = pattern_height / (num_grain_lines + 1)
        
        for i in range(num_grain_lines):
            y_pos = (i + 1) * y_interval
            grain_line = ET.SubElement(parent, "path")
            
            # Create wavy path for wood grain
            path_data = f"M 0,{y_pos}"
            
            # Generate control points for curvy grain
            num_segments = 5
            segment_width = pattern_width / num_segments
            
            for j in range(1, num_segments + 1):
                x = j * segment_width
                # Vary the y position slightly to create a natural grain effect
                y_variation = random.uniform(-3, 3) * density
                new_y = y_pos + y_variation
                
                cp1_x = x - (segment_width * 0.7)
                cp1_y = y_pos + random.uniform(-2, 2) * density
                cp2_x = x - (segment_width * 0.3)
                cp2_y = new_y + random.uniform(-2, 2) * density
                
                path_data += f" C {cp1_x},{cp1_y} {cp2_x},{cp2_y} {x},{new_y}"
            
            grain_line.set("d", path_data)
            grain_line.set("stroke", "#b38d5d")
            grain_line.set("stroke-width", str(random.uniform(0.3, 1.0)))
            grain_line.set("fill", "none")
            grain_line.set("opacity", str(random.uniform(0.2, 0.6)))
            
        # Add knots randomly
        num_knots = int(random.uniform(0, 2) * density)
        for _ in range(num_knots):
            cx = random.uniform(pattern_width * 0.2, pattern_width * 0.8)
            cy = random.uniform(pattern_height * 0.2, pattern_height * 0.8)
            radius = random.uniform(3, 8) * density
            
            # Create the knot with concentric circles
            for i in range(3):
                knot = ET.SubElement(parent, "circle")
                knot.set("cx", str(cx))
                knot.set("cy", str(cy))
                knot.set("r", str(radius * (1 - i * 0.25)))
                
                if i == 0:
                    knot.set("fill", "#9b7b56")  # Outer circle darker
                elif i == 1:
                    knot.set("fill", "#8d6e4b")  # Middle circle even darker
                else:
                    knot.set("fill", "#7d614b")  # Center darkest
    
    def _create_stone_pattern(self, parent: ET.Element, width: float, height: float, density: float):
        """Create a stone texture pattern."""
        pattern_size = min(60, max(30, width * 0.1))
        parent.set("width", str(pattern_size))
        parent.set("height", str(pattern_size))
        
        # Background fill
        bg = ET.SubElement(parent, "rect")
        bg.set("width", str(pattern_size))
        bg.set("height", str(pattern_size))
        bg.set("fill", "#c2c2c2")
        
        # Add stone cracks and surface variations based on density
        num_cracks = int(pattern_size * density * 0.5)
        for _ in range(num_cracks):
            # Create random crack line
            x1 = random.uniform(0, pattern_size)
            y1 = random.uniform(0, pattern_size)
            
            # Random angles and lengths for crack segments
            num_segments = random.randint(2, 5)
            crack_path = f"M {x1},{y1}"
            
            x, y = x1, y1
            for _ in range(num_segments):
                length = random.uniform(5, 15) * density
                angle = random.uniform(0, 360)  # Random direction
                
                # Calculate end point
                dx = length * math.cos(math.radians(angle))
                dy = length * math.sin(math.radians(angle))
                x += dx
                y += dy
                
                crack_path += f" L {x},{y}"
            
            crack = ET.SubElement(parent, "path")
            crack.set("d", crack_path)
            crack.set("stroke", "#a0a0a0")
            crack.set("stroke-width", "0.5")
            crack.set("fill", "none")
            
        # Add texture spots for stone surface variations
        num_spots = int(pattern_size * pattern_size * density * 0.05)
        for _ in range(num_spots):
            x_pos = random.uniform(0, pattern_size)
            y_pos = random.uniform(0, pattern_size)
            radius = random.uniform(1, 3)
            
            spot = ET.SubElement(parent, "circle")
            spot.set("cx", str(x_pos))
            spot.set("cy", str(y_pos))
            spot.set("r", str(radius))
            
            # Random variations in color
            color_variation = random.randint(-20, 20)
            spot_color = 194 + color_variation  # base on #c2c2c2
            spot_color = max(0, min(255, spot_color))
            
            spot.set("fill", f"#{spot_color:02x}{spot_color:02x}{spot_color:02x}")
            spot.set("opacity", str(random.uniform(0.3, 0.7)))
    
    def _create_marble_pattern(self, parent: ET.Element, width: float, height: float, density: float, base_color: str = '#f5f5f5', seed: Optional[int] = None) -> ET.Element:
        """Create a procedurally generated marble texture pattern using deterministic algorithms.
        
        Args:
            parent: Parent element to attach the pattern to
            width: Width of the target area
            height: Height of the target area
            density: Density of marble veining (0.0-1.0)
            base_color: Base color of the marble
            seed: Optional seed for deterministic generation
            
        Returns:
            The parent element with marble pattern attached
        """
        # Use deterministic RNG with explicit seeding for reproducibility
        local_rng = random.Random(seed if seed is not None else hash(f"{width}:{height}:{density}:{base_color}"))
        
        # Quantize dimensions to reduce size and improve caching
        pattern_width = min(80, max(40, int(width * 0.15)))
        pattern_height = pattern_width
        
        # Set pattern container attributes with minimal decimal precision
        parent.set("width", f"{pattern_width:.0f}")
        parent.set("height", f"{pattern_height:.0f}")
        
        # Background fill - marble base color
        bg = ET.SubElement(parent, "rect")
        bg.set("width", f"{pattern_width:.0f}")
        bg.set("height", f"{pattern_height:.0f}")
        bg.set("fill", base_color)
        
        # Add subtle base texture gradient to simulate marble depth
        base_grad = ET.SubElement(parent, "linearGradient", { \
            "id": f"marble_base_{hash(base_color) % 10000}",
            "x1": "0%", \
            "y1": "0%", \
            "x2": "100%", \
            "y2": "100%"
        })
        
        # Create quantized color variations for the gradient
        color_components = self._parse_hex_color(base_color)
        if color_components:
            r, g, b = color_components
            
            # Add slight variations for realistic marble look
            ET.SubElement(base_grad, "stop", { \
                "offset": "0%",
                "stop-color": self._adjust_color_brightness(r, g, b, 1.05)
            })
            ET.SubElement(base_grad, "stop", { \
                "offset": "50%",
                "stop-color": base_color
            })
            ET.SubElement(base_grad, "stop", { \
                "offset": "100%",
                "stop-color": self._adjust_color_brightness(r, g, b, 0.95)
            })
            
            # Apply gradient fill
            bg.set("fill", f"url(#marble_base_{hash(base_color) % 10000})")
        
        # Create primary vein structure using a consistent pattern
        # Number of veins is deterministic based on density and dimensions
        vein_density_matrix = [ \
            [0.2, 0.3, 0.4],  # Low density ranges
            [0.4, 0.6, 0.7],  # Medium density ranges
            [0.7, 0.9, 1.1]   # High density ranges
        ]
        
        # Quantized density for consistent results
        density_index = min(2, int(density * 3))
        size_index = min(2, int((pattern_width / 80.0) * 3))
        vein_factor = vein_density_matrix[density_index][size_index]
        
        num_veins = int(pattern_width * vein_factor)
        
        # Pre-compute vein colors for efficiency and consistency
        vein_base = self._parse_hex_color(base_color)
        if vein_base:
            r, g, b = vein_base
            vein_colors = [ \
                self._adjust_color_brightness(r, g, b, 0.85),  # Darker
                self._adjust_color_brightness(r, g, b, 0.90),  # Dark
                self._adjust_color_brightness(r, g, b, 0.95),  # Slightly dark
                self._adjust_color_brightness(r, g, b, 1.05),  # Slightly light
                self._adjust_color_brightness(r, g, b, 1.10)   # Lighter
            ]
        else:
            # Fallback grayscale colors
            vein_colors = ["#e0e0e0", "#d0d0d0", "#c0c0c0", "#b0b0b0", "#a0a0a0"]
        
        # Generate veins with controlled randomness
        for vein_idx in range(num_veins):
            # Derive vein parameters from index for deterministic generation
            vein_seed = vein_idx + (hash(str(seed)) % 1000 if seed else 0)
            
            # Calculate starting positions using constrained distribution
            # This ensures more realistic marble patterns with clustering
            quad_x = vein_idx % 2
            quad_y = (vein_idx // 2) % 2
            x_range = (quad_x * pattern_width/2, (quad_x+1) * pattern_width/2)
            y_range = (quad_y * pattern_height/2, (quad_y+1) * pattern_height/2)
            
            x_start = x_range[0] + local_rng.uniform(0, 1) * (x_range[1] - x_range[0])
            y_start = y_range[0] + local_rng.uniform(0, 1) * (y_range[1] - y_range[0])
            
            # Quantize starting positions to reduce path complexity
            x_start = int(x_start * 10) / 10
            y_start = int(y_start * 10) / 10
            
            # Control the general direction of the vein - each quadrant gets different angles
            # for more realistic distribution of veins
            base_angle = 45 + 90 * quad_x + 45 * quad_y
            angle_variation = 40  # Degrees of variation
            main_angle = base_angle + local_rng.uniform(-angle_variation, angle_variation)
            
            # Start the path with minimal precision
            vein_path = f"M{x_start:.1f},{y_start:.1f}"
            
            # Fixed number of segments based on density for efficiency
            segment_count = 3 + int(density * 5)
            segment_length = pattern_width / (6.0 + local_rng.uniform(0, 2))
            
            # Create bezier segments with reduced precision points
            x, y = x_start, y_start
            for i in range(segment_count):
                # Slightly vary the angle as we progress using a consistent algorithm
                segment_angle = main_angle + (local_rng.uniform(-30, 30) * (i % 3 - 1) * 0.5)
                
                # Scale segment length to taper at the end for natural look
                taper_factor = 1.0 - (i / segment_count) * 0.3
                actual_segment_length = segment_length * taper_factor
                
                # Calculate end point with reduced precision
                rad_angle = math.radians(segment_angle)
                end_x = x + actual_segment_length * math.cos(rad_angle)
                end_y = y + actual_segment_length * math.sin(rad_angle)
                
                # Create control points with deterministic offsets
                ctrl_offset_1 = 25 + (i * 5) % 20
                ctrl_offset_2 = 35 - (i * 7) % 15
                
                cp1_rad = math.radians(segment_angle + ctrl_offset_1)
                cp2_rad = math.radians(segment_angle - ctrl_offset_2)
                
                cp1_x = x + actual_segment_length * 0.5 * math.cos(cp1_rad)
                cp1_y = y + actual_segment_length * 0.5 * math.sin(cp1_rad)
                
                cp2_x = end_x - actual_segment_length * 0.4 * math.cos(cp2_rad)
                cp2_y = end_y - actual_segment_length * 0.4 * math.sin(cp2_rad)
                
                # Reduce precision for smaller output size
                vein_path += f" C{cp1_x:.1f},{cp1_y:.1f} {cp2_x:.1f},{cp2_y:.1f} {end_x:.1f},{end_y:.1f}"
                
                x, y = end_x, end_y
                
                # Break if we went outside the pattern
                if not (0 <= x <= pattern_width and 0 <= y <= pattern_height):
                    break
            
            # Create the vein path
            vein = ET.SubElement(parent, "path")
            vein.set("d", vein_path)
            
            # Choose color deterministically based on vein index
            vein_color = vein_colors[vein_idx % len(vein_colors)]
            vein.set("stroke", vein_color)
            
            # Vary stroke width by position and index
            width_factor = 0.5 + (vein_idx % 4) * 0.3
            vein.set("stroke-width", f"{width_factor:.1f}")
            vein.set("fill", "none")
            
            # Vary opacity by position for depth effect
            opacity = 0.2 + (quad_y * 0.1) + (vein_idx % 3) * 0.1
            vein.set("opacity", f"{opacity:.1f}")
        
        # Add crystalline structure using a grid-based approach for efficiency
        if density > 0.3:  # Only add specks for higher detail levels
            # Use a grid-based distribution to ensure even coverage with minimal elements
            grid_cells = 8 + int(density * 8)
            cell_w = pattern_width / grid_cells
            cell_h = pattern_height / grid_cells
            
            # Pre-compute number of specks per cell based on density for efficiency
            specks_per_cell = max(1, int(density * 3))
            
            # Create a sparse grid of crystalline highlights
            for gx in range(grid_cells):
                for gy in range(grid_cells):
                    # Skip cells deterministically to create natural variation
                    if (gx + gy) % 3 == 0 and local_rng.random() > 0.3:
                        continue
                        
                    for s in range(specks_per_cell):
                        # Position within cell with slight jitter
                        jitter_x = local_rng.uniform(0.2, 0.8)
                        jitter_y = local_rng.uniform(0.2, 0.8)
                        
                        x_pos = (gx + jitter_x) * cell_w
                        y_pos = (gy + jitter_y) * cell_h
                        
                        # Size varies by position and density
                        radius = 0.2 + (density * 0.6) * ((gx + gy) % 3) / 3.0
                        radius = round(radius * 10) / 10  # Quantize to 0.1 precision
                        
                        # Create highlight speck
                        speck = ET.SubElement(parent, "circle")
                        speck.set("cx", f"{x_pos:.1f}")
                        speck.set("cy", f"{y_pos:.1f}")
                        speck.set("r", f"{radius:.1f}")
                        
                        # Use brightest vein color
                        speck.set("fill", vein_colors[-1])
                        
                        # Lower opacity for subtle effect
                        speck_opacity = 0.1 + (s * 0.05)
                        speck.set("opacity", f"{speck_opacity:.1f}")
        
        return parent
    
    def _parse_hex_color(self, hex_color: str) -> Optional[Tuple[int, int, int]]:
        """Parse a hex color string into RGB components.
        
        Args:
            hex_color: Hex color string (e.g., '#f5f5f5')
            
        Returns:
            Tuple of (r, g, b) values or None if invalid format
        """
        if not hex_color or not hex_color.startswith('#') or len(hex_color) != 7:
            return None
            
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            return (r, g, b)
        except ValueError:
            return None
    
    def _adjust_color_brightness(self, r: int, g: int, b: int, factor: float) -> str:
        """Adjust the brightness of a color by the given factor.
        
        Args:
            r: Red color component (0-255)
            g: Green color component (0-255)
            b: Blue color component (0-255)
            factor: Brightness factor (>1 for lighter, <1 for darker)
            
        Returns:
            Hex color string with adjusted brightness
        """
        # Apply factor with bounds checking
        new_r: int = max(0, min(255, int(r * factor)))
        new_g: int = max(0, min(255, int(g * factor)))
        new_b: int = max(0, min(255, int(b * factor)))
        
        # Return hex color
        return f"#{new_r:02x}{new_g:02x}{new_b:02x}"
    
    def _add_material_texture(self, element: ET.Element, material: str, x: float, y: float, width: float, height: float, detail_level: float):
        """Add procedural texture patterns to simulate material properties.
        
        This method generates vector-based micro-patterns to simulate material textures, \
        maintaining SVG scalability while enhancing visual richness without increasing file size.
        
        Args:
            element: SVG element to enhance with texture
            material: Material type to simulate ('brick', 'concrete', 'wood', etc.)
            x, y: Position coordinates
            width, height: Size of the element
            detail_level: Level of detail (0-1) to control texture complexity
        """
        # Skip texture for very low detail levels or small elements
        if detail_level < 0.4 or width < 10 or height < 10:
            return
            
        # Adjust pattern complexity based on detail level
        pattern_density = detail_level * 0.8  # scale to keep patterns reasonable
        
        # Get base color for patterns
        base_color = element.get("fill", "#888888")
        
        # Add base color for the element
        if not element.get("fill"):
            element.set("fill", base_color)
            
        # Apply appropriate texture based on material
        if material == "brick":
            # Create brick pattern code would go here
            pass
        elif material == "concrete":
            # Create concrete pattern with subtle texture
            concrete_group = ET.SubElement(element.getparent(), "g")
            concrete_group.set("class", "concrete-texture")
            
            # Add subtle lines to simulate concrete formwork
            lines_count = max(3, int(10 * pattern_density))
            for i in range(lines_count):
                y_pos = y + i * (height / lines_count)
                concrete_line = ET.SubElement(concrete_group, "line")
                concrete_line.set("x1", str(x))
                concrete_line.set("y1", str(y_pos))
                concrete_line.set("x2", str(x + width))
                concrete_line.set("y2", str(y_pos))
                concrete_line.set("stroke", self._adjust_brightness(base_color, -0.1, "concrete"))
                concrete_line.set("stroke-width", "0.3")
                concrete_line.set("opacity", "0.3")
            
            # Add subtle vertical lines
            for i in range(int(5 * pattern_density)):
                x_pos = x + random.uniform(0, width)
                concrete_line = ET.SubElement(concrete_group, "line")
                concrete_line.set("x1", str(x_pos))
                concrete_line.set("y1", str(y))
                concrete_line.set("x2", str(x_pos))
                concrete_line.set("y2", str(y + height))
                concrete_line.set("stroke", self._adjust_brightness(base_color, -0.05, "concrete"))
                concrete_line.set("stroke-width", "0.2")
                concrete_line.set("opacity", "0.2")
        
        elif material == "metal" or material == "steel":
            # Create metal texture with subtle highlights
            metal_group = ET.SubElement(element.getparent(), "g")
            metal_group.set("class", "metal-texture")
            
            # Add horizontal highlight lines
            highlight_count = max(2, int(5 * pattern_density))
            for i in range(highlight_count):
                y_pos = y + random.uniform(0, height)
                highlight = ET.SubElement(metal_group, "line")
                highlight.set("x1", str(x))
                highlight.set("y1", str(y_pos))
                highlight.set("x2", str(x + width))
                highlight.set("y2", str(y_pos))
                highlight.set("stroke", self._adjust_brightness(base_color, 0.3, "metal"))
                highlight.set("stroke-width", "0.5")
                highlight.set("opacity", "0.4")
            
            # Add reflection gradient
            if detail_level > 0.7:
                reflection = ET.SubElement(metal_group, "rect")
                reflection.set("x", str(x))
                reflection.set("y", str(y))
                reflection.set("width", str(width))
                reflection.set("height", str(height/5))
                reflection.set("fill", self._adjust_brightness(base_color, 0.4, "metal"))
                reflection.set("opacity", "0.2")
                
        elif material == "glass":
            # Create glass texture with reflections
            glass_group = ET.SubElement(element.getparent(), "g")
            glass_group.set("class", "glass-texture")
            
            # Add horizontal reflection
            reflection = ET.SubElement(glass_group, "rect")
            reflection.set("x", str(x))
            reflection.set("y", str(y + height * 0.2))
            reflection.set("width", str(width))
            reflection.set("height", str(height * 0.1))
            reflection.set("fill", "#ffffff")
            reflection.set("opacity", "0.15")
            
            # Add diagonal highlight for curved glass effect
            if detail_level > 0.6:
                highlight = ET.SubElement(glass_group, "line")
                highlight.set("x1", str(x))
                highlight.set("y1", str(y))
                highlight.set("x2", str(x + width))
                highlight.set("y2", str(y + height))
                highlight.set("stroke", "#ffffff")
                highlight.set("stroke-width", str(2 * pattern_density))
                highlight.set("opacity", "0.1")
        
        elif material == "wood":
            # Create wood grain texture
            wood_group = ET.SubElement(element.getparent(), "g")
            wood_group.set("class", "wood-texture")
            
            # Add wood grain lines
            grain_count = max(5, int(15 * pattern_density))
            grain_color = self._adjust_brightness(base_color, -0.1, "wood")
            
            for i in range(grain_count):
                # Create flowing curves to simulate wood grain
                path = ET.SubElement(wood_group, "path")
                
                # Generate a curved path using quadratic bezier curves
                curve_y = y + (i + 0.5) * (height / grain_count)
                control_x1 = x + random.uniform(0, width * 0.3)
                control_x2 = x + width - random.uniform(0, width * 0.3)
                
                path_data = f"M {x},{curve_y} "
                path_data += f"Q {control_x1},{curve_y + random.uniform(-5, 5)} {x + width/3},{curve_y + random.uniform(-3, 3)} "
                path_data += f"Q {control_x2},{curve_y + random.uniform(-5, 5)} {x + width},{curve_y + random.uniform(-2, 2)}"
                
                path.set("d", path_data)
                path.set("fill", "none")
                path.set("stroke", grain_color)
                path.set("stroke-width", str(0.5 + random.uniform(0, 1) * pattern_density))
                path.set("opacity", str(0.1 + random.uniform(0, 0.2)))
            
            # Add some knots if detail level is high enough
            if detail_level > 0.7:
                knot_count = max(1, int(3 * pattern_density))
                for _ in range(knot_count):
                    knot_x = x + random.uniform(width * 0.2, width * 0.8)
                    knot_y = y + random.uniform(height * 0.2, height * 0.8)
                    knot_radius = random.uniform(3, 8) * pattern_density
                    
                    knot = ET.SubElement(wood_group, "circle")
                    knot.set("cx", str(knot_x))
                    knot.set("cy", str(knot_y))
                    knot.set("r", str(knot_radius))
                    knot.set("fill", self._adjust_brightness(base_color, random.uniform(-0.2, 0.1), "wood"))
                    knot.set("opacity", str(random.uniform(0.1, 0.3)))
        
        elif material == "stone":
            # Create stone texture with cracks and surface variations
            stone_group = ET.SubElement(element.getparent(), "g")
            stone_group.set("class", "stone-texture")
            
            # Add subtle cracks
            cracks_count = max(2, int(8 * pattern_density))
            for i in range(cracks_count):
                # Create random crack path
                path = ET.SubElement(stone_group, "path")
                start_x = x + random.uniform(0, width)
                start_y = y + random.uniform(0, height)
                
                crack_points = []
                segments = max(2, int(4 * pattern_density))
                current_x, current_y = start_x, start_y
                
                for j in range(segments):
                    angle = random.uniform(0, 2 * math.pi)
                    length = random.uniform(width/20, width/10) * pattern_density
                    current_x += math.cos(angle) * length
                    current_y += math.sin(angle) * length
                    crack_points.append((current_x, current_y))
                
                # Build the path data
                path_data = f"M {start_x},{start_y} "
                for cp_x, cp_y in crack_points:
                    path_data += f"L {cp_x},{cp_y} "
                
                path.set("d", path_data)
                path.set("stroke", self._adjust_brightness(base_color, -0.2, "stone"))
                path.set("stroke-width", "0.3")
                path.set("fill", "none")
                path.set("opacity", "0.5")
                
            # Add surface variation
            for i in range(int(10 * pattern_density)):
                spot = ET.SubElement(stone_group, "circle")
                spot_x = x + random.uniform(0, width)
                spot_y = y + random.uniform(0, height)
                spot_radius = random.uniform(1, 3) * pattern_density
                
                spot.set("cx", str(spot_x))
                spot.set("cy", str(spot_y))
                spot.set("r", str(spot_radius))
                spot.set("fill", self._adjust_brightness(base_color, random.uniform(-0.2, 0.1), "stone"))
                spot.set("opacity", str(random.uniform(0.1, 0.3)))
        
        elif material == "marble":
            # Create marble texture with veins
            marble_group = ET.SubElement(element.getparent(), "g")
            marble_group.set("class", "marble-texture")
            
            # Add marble veins
            veins_count = max(3, int(7 * pattern_density))
            
            for i in range(veins_count):
                path = ET.SubElement(marble_group, "path")
                
                # Start point for the vein
                start_x = x + random.uniform(0, width)
                start_y = y + random.uniform(0, height)
                
                control_points = []
                segments = max(3, int(5 * pattern_density))
                wave_magnitude = height / 10
                
                current_x, current_y = start_x, start_y
                
                for j in range(segments):
                    # Veins generally flow horizontally with sinusoidal variation
                    current_x += random.uniform(width/10, width/5)
                    current_y += random.uniform(-wave_magnitude, wave_magnitude)
                    
                    # Keep within bounds
                    current_x = min(max(current_x, x), x + width)
                    current_y = min(max(current_y, y), y + height)
                    
                    control_points.append((current_x, current_y))
                
                # Build the path data with bezier curves for smoother veins
                path_data = f"M {start_x},{start_y} "
                for j in range(len(control_points)):
                    if j < len(control_points) - 1:
                        cp1_x, cp1_y = control_points[j]
                        cp2_x, cp2_y = control_points[j+1]
                        
                        # Calculate a control point between the two points for a bezier curve
                        ctrl_x = (cp1_x + cp2_x) / 2
                        ctrl_y = cp1_y + random.uniform(-height/15, height/15)
                        
                        path_data += f"Q {ctrl_x},{ctrl_y} {cp2_x},{cp2_y} "
                    else:
                        cp_x, cp_y = control_points[j]
                        path_data += f"L {cp_x},{cp_y}"
                
                path.set("d", path_data)
                vein_color = self._adjust_brightness(base_color, random.uniform(-0.1, 0.2), "marble")
                path.set("stroke", vein_color)
                path.set("stroke-width", str(random.uniform(0.5, 1.5)))
                path.set("fill", "none")
                path.set("opacity", str(random.uniform(0.2, 0.4)))
                
    def _add_windows(self, parent: ET.Element, x: float, y: float, width: float, height: float, style: str):
        """
        Add windows to a building element.
        
        Args:
            parent: Parent SVG element
            x, y: Position of building
            width, height: Size of building
            style: Building style
        """
        # Window properties
        window_count = int((width * height) / 1000)  # Scale with building size
        window_min_size = min(width, height) * 0.05
        window_max_size = min(width, height) * 0.15
        
        # Window color based on style
        if style == "modern":
            window_color = "rgba(135, 206, 250, 0.7)"  # Light sky blue with transparency
        elif style == "futuristic":
            window_color = "rgba(32, 178, 170, 0.7)"  # Light sea green with transparency
        else:
            window_color = "rgba(255, 255, 255, 0.5)"  # White with transparency
        
        # Create random windows
        for i in range(window_count):
            window_width = window_min_size + random.random() * (window_max_size - window_min_size)
            window_height = window_min_size + random.random() * (window_max_size - window_min_size)
            
            # Position windows with margins
            margin = min(width, height) * 0.05
            window_x = x + margin + random.random() * (width - 2 * margin - window_width)
            window_y = y + margin + random.random() * (height - 2 * margin - window_height)
            
            window = ET.SubElement(parent, "{%s}rect" % SVGNS)
            window.set("x", str(window_x))
            window.set("y", str(window_y))
            window.set("width", str(window_width))
            window.set("height", str(window_height))
            window.set("fill", window_color)
            
    def _add_window_stripes(self, parent: ET.Element, x: float, y: float, width: float, height: float, style: str):
        """Add windows in vertical stripes.
        
        Args:
            parent: Parent SVG element
            x, y: Position of building
            width, height: Size of building
            style: Building style
        """
        # Calculate window columns
        column_count = max(2, int(width / 30))  # At least 2 columns, or more for wider buildings
        column_width = width * 0.7 / column_count
        column_spacing = width * 0.3 / (column_count + 1)
        
        # Window column properties based on style
        if style == "modern":
            column_color = "rgba(135, 206, 250, 0.7)"  # Light sky blue with transparency
        elif style == "futuristic":
            column_color = "rgba(32, 178, 170, 0.7)"  # Light sea green with transparency
        else:
            column_color = "rgba(255, 255, 255, 0.5)"  # White with transparency
        
        # Create vertical window columns
        for i in range(column_count):
            column_x = x + column_spacing + i * (column_width + column_spacing)
            
            column = ET.SubElement(parent, "{%s}rect" % SVGNS)
            column.set("x", str(column_x))
            column.set("y", str(y + height * 0.1))  # Inset from building edge
            column.set("width", str(column_width))
            column.set("height", str(height * 0.8))  # Inset from building edge
            column.set("fill", column_color)
            column.set("stroke", "#FFFFFF")
            column.set("stroke-width", "1")


class SVGVisualTranslator:
    """Main class for translating scene graph to SVG elements."""
    
    def __init__(self, document: 'SVGDocument'):
        self.document = document
        self.generators = {}
        self._init_generators()
    
    def _init_generators(self) -> None:
        """Initialize element generators for each node type."""
        self.generators["sky"] = SkyGenerator(self.document)
        self.generators["mountain"] = MountainGenerator(self.document)
        self.generators["sea"] = SeaGenerator(self.document)
        self.generators["tree"] = TreeGenerator(self.document)
        self.generators["building"] = BuildingGenerator(self.document)
        # Additional generators would be added here
    
    def translate(self, scene_graph: SceneGraph) -> None:
        """Translate the scene graph into SVG elements and add them to the document.
        
        Args:
            scene_graph: The scene graph to translate
        """
        # Set document dimensions
        self.document.set_size(scene_graph.width, scene_graph.height)
        
        # Process global scene properties (contrast, brightness, etc.)
        scene_properties = scene_graph.root.properties
        if scene_properties:
            self._apply_global_effects(scene_properties)
        
        # Create main group for all elements
        main_group = ET.Element("{%s}g" % SVGNS)
        self.document.root.append(main_group)
        
        # Get all nodes sorted by z-index for proper layering
        all_nodes = self._collect_nodes_by_z_index(scene_graph.root)
        
        # Process each node to generate SVG elements
        for node in all_nodes:
            if node.node_type in self.generators:
                # Generate SVG element using the appropriate generator
                element = self.generators[node.node_type].generate(node)
                if element is not None:
                    # Add to document
                    main_group.append(element)
            else:
                # For unknown node types, use a fallback visualization
                self._generate_fallback(main_group, node)
    
    def _collect_nodes_by_z_index(self, root: SceneNode) -> List[SceneNode]:
        """Collect all nodes from the scene graph and sort by z-index.
        
        Args:
            root: Root node of the scene graph
            
        Returns:
            List of nodes sorted by z-index (ascending)
        """
        nodes = []
        
        def collect_recursive(node: SceneNode) -> None:
            # Skip the root node itself
            if node.node_id != "root":
                nodes.append(node)
                
            for child in node.children:
                collect_recursive(child)
        
        collect_recursive(root)
        
        # Sort by z-index property if present, otherwise use node position z
        def get_z_index(node: SceneNode) -> float:
            if "z_index" in node.properties:
                return node.properties["z_index"]
            return node.position[2]  # Z coordinate
        
        return sorted(nodes, key=get_z_index)
    
    def _apply_global_effects(self, properties: Dict[str, Any]) -> None:
        """Apply global visual effects to the SVG document.
        
        Args:
            properties: Global scene properties
        """
        # Apply contrast filter if specified
        if "contrast" in properties or "brightness" in properties or "saturation" in properties:
            contrast = properties.get("contrast", 1.0)
            brightness = properties.get("brightness", 1.0)
            saturation = properties.get("saturation", 1.0)
            
            # Create filter for adjusting contrast/brightness
            filter_id = "mood_filter"
            filter_element = ET.Element("{%s}filter" % SVGNS)
            filter_element.set("id", filter_id)
            
            # Add feComponentTransfer for contrast adjustment
            if contrast != 1.0:
                component_transfer = ET.SubElement(filter_element, "{%s}feComponentTransfer" % SVGNS)
                for channel in ["R", "G", "B"]:
                    func = ET.SubElement(component_transfer, "{%s}feFuncA" % SVGNS)
                    func.set("type", "linear")
                    func.set("slope", str(contrast))
                    func.set("intercept", str((1 - contrast) / 2))
            
            # Add feColorMatrix for saturation adjustment
            if saturation != 1.0:
                color_matrix = ET.SubElement(filter_element, "{%s}feColorMatrix" % SVGNS)
                color_matrix.set("type", "saturate")
                color_matrix.set("values", str(saturation))
            
            # Add feComponentTransfer for brightness adjustment
            if brightness != 1.0:
                component_transfer = ET.SubElement(filter_element, "{%s}feComponentTransfer" % SVGNS)
                for channel in ["R", "G", "B"]:
                    func = ET.SubElement(component_transfer, "{%s}feFuncA" % SVGNS)
                    func.set("type", "linear")
                    func.set("slope", str(brightness))
            
            # Add the filter to the defs section
            self.document.add_definition(filter_element)
            
            # Apply the filter to the root group
            self.document.root.set("filter", f"url(#{filter_id})")
    
    def _generate_fallback(self, parent: ET.Element, node: SceneNode) -> None:
        """Generate a fallback visualization for unknown node types.
        
        Args:
            parent: Parent SVG element
            node: Scene node to visualize
        """
        # Create a simple rectangle with a question mark
        x, y = node.position[0], node.position[1]
        width, height = node.size[0], node.size[1]
        
        # Create a group for the fallback
        g = ET.SubElement(parent, "{%s}g" % SVGNS)
        
        # Create rectangle
        rect = ET.SubElement(g, "{%s}rect" % SVGNS)
        rect.set("x", str(x))
        rect.set("y", str(y))
        rect.set("width", str(width))
        rect.set("height", str(height))
        rect.set("fill", "#CCCCCC")
        rect.set("stroke", "#000000")
        rect.set("stroke-width", "1")
        
        # Add a question mark for unknown elements
        text = ET.SubElement(g, "{%s}text" % SVGNS)
        text.set("x", str(x + width/2))
        text.set("y", str(y + height/2))
        text.set("font-family", "Arial")
        text.set("font-size", "24")
        text.set("fill", "#000000")
        text.set("text-anchor", "middle")
        text.set("dominant-baseline", "middle")
        text.text = "?"

def translate_scene_to_svg(scene_graph: SceneGraph, document: 'SVGDocument') -> None:
    """Translate a scene graph into SVG elements in the given document.
    
    Args:
        scene_graph: Scene graph representing the scene
        document: SVG document to populate
    """
    translator = SVGVisualTranslator(document)
    translator.translate(scene_graph)

#                4. COMPOSITIONAL RENDERING                                     #
# ============================================================================ #

# ============================================================================ #
#                5. ORCHESTRATION AND RUNTIME PIPELINE                         #
# ============================================================================ #
# This stage ties everything together:

def generate_svg(prompt: str, width: int = 800, height: int = 600, detail_level: float = 0.7) -> Dict[str, Any]:
    """
    Main orchestration function that transforms a text prompt into an SVG.
    
    This function coordinates all five stages of the pipeline:
    1. Lexical-semantic interpretation (HybridSemanticParser)
    2. Conceptual mapping to scene graph (SceneGraphBuilder)
    3. Procedural SVG generation (SVGGenerator)
    4. Validation and sanitization (SVGValidator)
    5. Correction and semantic feedback (SVGCorrectionEngine)
    
    Args:
        prompt: Text description to transform into SVG
        width: Width of the output SVG in pixels
        height: Height of the output SVG in pixels
        detail_level: Amount of detail to include (0.0-1.0)
        
    Returns:
        Dictionary containing:
        - svg: The final SVG string
        - status: Success or error status
        - feedback: Semantic feedback about the generation process
        - corrections: Any corrections that were applied
    """
    result = { \
        "svg": "",
        "status": "error", \
        "feedback": [], \
        "corrections": []
    }
    
    try:
        # Stage 1: Parse the text prompt into semantic vectors
        parser = HybridSemanticParser()
        scene_vector = parser.parse(prompt)
        
        # Add confidence information to result
        result["feedback"].append(f"Parsed prompt with {parser.get_confidence():.1%} confidence")
        
        if not scene_vector or "objects" not in scene_vector or not scene_vector["objects"]:
            result["feedback"].append("Failed to extract any objects from the prompt")
            return result
        
        # Stage 2: Build a scene graph from the semantic vectors
        graph_builder = SceneGraphBuilder(width=width, height=height)
        scene_graph = graph_builder.build_scene_graph(scene_vector)
        
        result["feedback"].append(f"Built scene graph with {len(scene_graph.nodes)} nodes")
        
        # Stage 3: Generate SVG from the scene graph
        svg_generator = SVGGenerator(detail_level=detail_level)
        svg_string = svg_generator.generate(scene_graph)
        
        result["feedback"].append(f"Generated SVG with size {len(svg_string)} bytes")
        
        # Stage 4: Validate the SVG for safety and standards compliance
        validator = SVGValidator()
        validation_result = validator.validate(svg_string)
        
        result["feedback"].append(f"Validation: {validation_result['status']}")
        
        if validation_result["status"] == "error" or validation_result["warnings"]:
            # Stage 5: Apply corrections if needed
            correction_engine = SVGCorrectionEngine()
            correction_result = correction_engine.correct_svg(svg_string)
            
            svg_string = correction_result["svg"]
            result["corrections"] = correction_result["corrections"]
            result["feedback"].append(f"Applied {len(correction_result['corrections'])} corrections")
            
            # Re-validate to ensure corrections fixed the issues
            final_validation = validator.validate(svg_string)
            if final_validation["status"] == "error":
                result["feedback"].append("Critical issues remain after correction")
                result["status"] = "partial_success"
            else:
                result["status"] = "success"
        else:
            result["status"] = "success"
        
        # Set the final SVG
        result["svg"] = svg_string
        
    except Exception as e:
        result["feedback"].append(f"Error: {str(e)}")
    
    return result


def generate_svg_file(prompt: str, output_path: str, width: int = 800, height: int = 600, detail_level: float = 0.7) -> Dict[str, Any]:
    """
    Generate an SVG from a prompt and save it to a file.
    
    Args:
        prompt: Text description to transform into SVG
        output_path: Path to save the SVG file
        width: Width of the output SVG in pixels
        height: Height of the output SVG in pixels
        detail_level: Amount of detail to include (0.0-1.0)
        
    Returns:
        Dictionary with the result of the generation process
    """
    result = generate_svg(prompt, width, height, detail_level)
    
    if result["status"] in ["success", "partial_success"]:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result["svg"])
            result["feedback"].append(f"SVG saved to {output_path}")
        except Exception as e:
            result["feedback"].append(f"Error saving file: {str(e)}")
            result["status"] = "error_saving"
    
    return result


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 3:
        print("Usage: python svg_hyper_realistic_unified.py <prompt> <output_path> [width] [height] [detail_level]")
        sys.exit(1)
    
    prompt = sys.argv[1]
    output_path = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 800
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    detail_level = float(sys.argv[5]) if len(sys.argv) > 5 else 0.7
    
    result = generate_svg_file(prompt, output_path, width, height, detail_level)
    
    # Print feedback
    print(f"Status: {result['status']}")
    for feedback in result["feedback"]:
        print(f"- {feedback}")
    
    if result["corrections"]:
        print("\nCorrections applied:")
        for correction in result["corrections"]:
            print(f"- {correction}")
    
    if result["status"] in ["success", "partial_success"]:
        print(f"\nSVG generated and saved to: {os.path.abspath(output_path)}")
        print(f"SVG size: {len(result['svg'])} bytes")
    else:
        print("\nFailed to generate SVG")
        sys.exit(1)

class SVGConstraints:
    """Enforces constraints and security rules on SVG outputs."""
    
    # SVG element whitelist (allowed elements)
    ALLOWED_ELEMENTS = { \
        "svg", "g", "rect", "circle", "ellipse", "line", "polyline",
        "polygon", "path", "text", "linearGradient", "radialGradient", \
        "stop", "defs", "filter", "feGaussianBlur", "feOffset", "feBlend", \
        "feColorMatrix", "feTurbulence", "feDisplacementMap", "title", "desc"
    }
    
    # SVG attribute whitelist
    ALLOWED_ATTRIBUTES = { \
        "id", "class", "x", "y", "cx", "cy", "r", "rx", "ry", "x1", "y1", "x2", "y2",
        "width", "height", "fill", "stroke", "stroke-width", "stroke-linecap", \
        "stroke-linejoin", "stroke-dasharray", "stroke-opacity", "fill-opacity", \
        "opacity", "transform", "d", "points", "viewBox", "preserveAspectRatio", \
        "xmlns", "xmlns:xlink", "version", "style", "font-family", "font-size", \
        "text-anchor", "dominant-baseline", "offset", "stop-color", "stop-opacity", \
        "gradientUnits", "spreadMethod", "gradientTransform", "text-decoration", \
        "filter", "filterUnits", "stdDeviation", "result", "in", "in2", "dx", "dy", \
        "mode", "type", "values", "baseFrequency", "numOctaves", "seed", "scale", \
        "xChannelSelector", "yChannelSelector"
    }
    
    # Forbidden attributes that may pose security risks
    FORBIDDEN_ATTRIBUTES = { \
        "onclick", "onload", "onmouseover", "onmouseout", "onerror",
        "script", "href", "xlink:href", "ev:event", "externalResourcesRequired"
    }
    
    # Maximum size allowed for SVG output (10KB)
    MAX_SIZE_BYTES = 10 * 1024
    
    @classmethod
    def validate_svg(cls, svg_string: str) -> Tuple[bool, List[str]]:
        """Validate SVG for security, size and structure compliance.
        
        Args:
            svg_string: SVG markup string to validate
            
        Returns:
            Tuple of (valid, list of error messages)
        """
        errors = []
        
        # 1. Size validation
        size_bytes = len(svg_string.encode('utf-8'))
        if size_bytes > cls.MAX_SIZE_BYTES:
            errors.append(f"SVG exceeds maximum size: {size_bytes} bytes (limit: {cls.MAX_SIZE_BYTES})")
        
        # 2. Content security screening (no scripts, external resources, etc.)
        if "<script" in svg_string or "javascript:" in svg_string:
            errors.append("SVG contains script elements or JavaScript URIs")
            
        if "data:" in svg_string:
            errors.append("SVG contains data URIs which are not allowed")
        
        try:
            # 3. Parse and check for disallowed elements/attributes
            # Using ElementTree for simplicity - in production would use defusedxml
            root = ET.fromstring(svg_string)
            cls._validate_element(root, errors)
        except ET.ParseError as e:
            errors.append(f"Invalid XML structure: {str(e)}")
        
        return len(errors) == 0, errors
    
    @classmethod
    def _validate_element(cls, element: ET.Element, errors: List[str]) -> None:
        """Recursively validate element and its children for compliance.
        
        Args:
            element: Element to validate
            errors: List to append errors to
        """
        # Strip namespace for tag comparison
        tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
        
        # Check element against whitelist
        if tag not in cls.ALLOWED_ELEMENTS:
            errors.append(f"Disallowed element: {tag}")
        
        # Check attributes against whitelist/blacklist
        for attr in element.attrib:
            # Strip namespace from attribute if present
            attr_name = attr.split('}')[-1] if '}' in attr else attr
            
            if attr_name in cls.FORBIDDEN_ATTRIBUTES:
                errors.append(f"Forbidden attribute: {attr_name}")
            elif attr_name not in cls.ALLOWED_ATTRIBUTES:
                errors.append(f"Disallowed attribute: {attr_name}")
            
            # Check for potentially malicious values
            value = element.attrib[attr]
            if "javascript:" in value or "data:" in value:
                errors.append(f"Potentially unsafe attribute value in {attr_name}")
        
        # Recursively check all children
        for child in element:
            cls._validate_element(child, errors)

# ============================================================================ #
#                5. CORRECTION AND SEMANTIC FEEDBACK                           #
# ============================================================================ #
# This stage automatically corrects issues detected during validation:
# - Regenerates missing elements inferred from context
# - Adjusts positions and attributes to resolve constraint violations
# - Ensures semantic completeness of the visual representation

class SVGFeedbackCorrector:
    """Provides correction mechanisms for SVG based on semantic feedback."""
    
    def __init__(self, scene_graph: SceneGraph, generators: Dict[str, SVGElementGenerator]):
        """Initialize the corrector with a scene graph and generators.
        
        Args:
            scene_graph: The source scene graph
            generators: Dictionary of element generators
        """
        self.scene_graph = scene_graph
        self.generators = generators
        self.max_correction_passes = 3
    
    def validate_and_correct(self, svg_document: 'SVGDocument') -> Tuple[bool, str]:
        """Validate SVG against scene graph semantics and correct if needed.
        
        Args:
            svg_document: SVG document to validate and correct
            
        Returns:
            Tuple of (success, message)
        """
        # Convert to string for validation
        svg_string = svg_document.to_string()
        
        # First check size and security constraints
        valid, errors = SVGConstraints.validate_svg(svg_string)
        
        if not valid:
            # Try to fix size-related issues
            if any("exceeds maximum size" in err for err in errors):
                self._reduce_svg_size(svg_document)
                return self.validate_and_correct(svg_document)  # Re-validate after correction
            
            return False, f"Validation errors: {', '.join(errors)}"
        
        # Check semantic completeness
        missing_elements = self._check_semantic_completeness(svg_document)
        
        if missing_elements:
            if self._attempt_correction(svg_document, missing_elements):
                return True, "Corrected missing semantic elements"
            else:
                return False, f"Failed to correct all missing elements: {', '.join(missing_elements)}"
        
        return True, "SVG is valid and semantically complete"
    
    def _check_semantic_completeness(self, svg_document: 'SVGDocument') -> List[str]:
        """Check if all high-priority semantic elements are present.
        
        Args:
            svg_document: SVG document to check
            
        Returns:
            List of missing element types
        """
        # Get all focal elements from scene graph
        focal_elements = []
        for node in self.scene_graph.find_nodes_by_type("focal_point"):
            focal_elements.append(node.properties.get("element_type", ""))
        
        # Check if they exist in the SVG
        missing = []
        for element_type in focal_elements:
            if not svg_document.find_elements_by_class(element_type):
                missing.append(element_type)
        
        return missing
    
    def _attempt_correction(self, svg_document: 'SVGDocument', missing_elements: List[str]) -> bool:
        """Attempt to add missing elements to the SVG.
        
        Args:
            svg_document: SVG document to modify
            missing_elements: List of missing element types
            
        Returns:
            True if all corrections were successful
        """
        success = True
        
        for element_type in missing_elements:
            # Find nodes of this type in the scene graph
            nodes = self.scene_graph.find_nodes_by_type(element_type)
            if not nodes:
                success = False
                continue
                
            # Try to regenerate each node
            for node in nodes:
                if element_type in self.generators:
                    element = self.generators[element_type].generate(node)
                    if element is not None:
                        svg_document.add_element(element)
                    else:
                        success = False
                else:
                    success = False
        
        return success
    
    def _reduce_svg_size(self, svg_document: 'SVGDocument') -> None:
        """Reduce the size of the SVG to meet constraints.
        
        Args:
            svg_document: SVG document to modify
        """
        # 1. Reduce precision of floating point values
        self._reduce_precision(svg_document)
        
        # 2. Remove non-essential elements if still too large
        if len(svg_document.to_string().encode('utf-8')) > SVGConstraints.MAX_SIZE_BYTES:
            self._remove_non_essential_elements(svg_document)
    
    def _reduce_precision(self, svg_document: 'SVGDocument') -> None:
        """Reduce decimal precision in the SVG.
        
        Args:
            svg_document: SVG document to modify
        """
        # Implementation would truncate decimal places in coordinates and measurements
        pass
    
    def _remove_non_essential_elements(self, svg_document: 'SVGDocument') -> None:
        """Remove non-essential decorative elements to reduce size.
        
        Args:
            svg_document: SVG document to modify
        """
        # Implementation would remove low-priority elements based on their semantic importance
        pass

class AdvancedSVGGenerator:
    """Main class for the advanced SVG generator, integrating all components."""
    
    def __init__(self, width: int = 800, height: int = 600):
        """Initialize the generator with default dimensions.
        
        Args:
            width: Width of the SVG canvas in pixels
            height: Height of the SVG canvas in pixels
        """
        self.width = width
        self.height = height
        self.optimizer = SVGOptimizer()
    
    def generate_from_prompt(self, prompt: str, output_path: str = None) -> str:
        """Generate an SVG illustration from a text prompt.
        
        This is the main entry point for the SVG generation process, integrating all pipeline stages:
        1. Semantic analysis - extract scene features from prompt
        2. Scene representation - create scene graph
        3. Visual translation - convert graph to SVG elements
        4. Optimization - improve SVG quality and size
        
        Args:
            prompt: Text prompt describing the desired illustration
            output_path: Path to save the generated SVG file (optional)
            
        Returns:
            SVG document as a string
        """
        # 1. Semantic Analysis
        print(f"Analyzing prompt: {prompt}")
        scene_features = extract_scene_features(prompt)
        
        # 2. Scene Representation
        print("Creating scene representation...")
        scene_graph = generate_intermediate_representation(scene_features, self.width, self.height)
        
        # 3. Visual Translation
        print("Translating to SVG elements...")
        document = SVGDocument(width=self.width, height=self.height)
        translate_scene_to_svg(scene_graph, document)
        
        # 4. Optimization
        print("Optimizing SVG output...")
        self.optimizer.optimize_svg_paths(document)
        self.optimizer.optimize_colors(document)
        
        # Save if path provided
        svg_output = document.to_string()
        if output_path:
            with open(output_path, 'w') as f:
                f.write(svg_output)
                print(f"SVG saved to {output_path}")
        
        return svg_output
    
    def generate_with_constraints(self, prompt: str, constraints: Dict[str, Any], output_path: str = None) -> str:
        """Generate an SVG with additional constraints and specifications.
        
        Similar to generate_from_prompt, but allows customization of:
        - Canvas dimensions
        - Scene elements properties
        - Style specifications
        - Optimization parameters
        
        Args:
            prompt: Text prompt describing the desired illustration
            constraints: Dictionary of constraints and specifications
            output_path: Path to save the generated SVG file (optional)
            
        Returns:
            SVG document as a string
        """
        # Extract constraints
        width = constraints.get("width", self.width)
        height = constraints.get("height", self.height)
        style_override = constraints.get("style", {})
        elements_override = constraints.get("elements", {})
        optimization_params = constraints.get("optimization", {})
        
        # 1. Semantic Analysis (with possible overrides)
        print(f"Analyzing prompt with constraints: {prompt}")
        scene_features = extract_scene_features(prompt)
        
        # Apply style overrides to features
        if style_override:
            for key, value in style_override.items():
                if key in scene_features:
                    scene_features[key]["value"] = value
                    scene_features[key]["confidence"] = 1.0  # User override has max confidence
                else:
                    scene_features[key] = { \
                        "value": value,
                        "confidence": 1.0, \
                        "alternatives": []
                    }
        
        # 2. Scene Representation
        print("Creating constrained scene representation...")
        scene_graph = generate_intermediate_representation(scene_features, width, height)
        
        # Apply element-specific overrides to scene graph
        if elements_override:
            self._apply_element_overrides(scene_graph, elements_override)
        
        # 3. Visual Translation
        print("Translating to SVG elements...")
        document = SVGDocument(width=width, height=height)
        translate_scene_to_svg(scene_graph, document)
        
        # 4. Optimization with custom parameters
        print("Optimizing SVG output with custom parameters...")
        self.optimizer.optimize_svg_paths(document)
        self.optimizer.optimize_colors(document)
        
        if "precision" in optimization_params:
            # Adjust numeric precision in SVG
            precision = optimization_params["precision"]
            self._adjust_precision(document, precision)
        
        # Additional optimization steps based on parameters
        if optimization_params.get("minimize_size", False):
            self.optimizer.optimize_svg_for_size(document)
        
        # Save if path provided
        svg_output = document.to_string()
        if output_path:
            with open(output_path, 'w') as f:
                f.write(svg_output)
                print(f"Constrained SVG saved to {output_path}")
        
        return svg_output
    
    def regenerate_element(self, svg_data: str, element_id: str, new_properties: Dict[str, Any]) -> str:
        """Regenerate a specific element in an existing SVG.
        
        This allows for targeted updates to elements without regenerating the entire SVG.
        
        Args:
            svg_data: Existing SVG data (string or path)
            element_id: ID of the element to regenerate
            new_properties: New properties for the element
            
        Returns:
            Updated SVG document as a string
        """
        # Parse the existing SVG
        document = SVGDocument()
        
        if svg_data.endswith(".svg") and os.path.exists(svg_data):
            # If svg_data is a path to an SVG file
            with open(svg_data, 'r') as f:
                svg_content = f.read()
                document.from_string(svg_content)
        else:
            # If svg_data is SVG content
            document.from_string(svg_data)
        
        # Find the element by ID
        element = document.find_element_by_id(element_id)
        if element is None:
            raise ValueError(f"Element with ID '{element_id}' not found in SVG")
        
        # Update the element properties
        for attr, value in new_properties.items():
            if attr == "style":
                # Parse style string into individual attributes
                style_dict = {}
                if element.get("style"):
                    style_parts = element.get("style").split(';')
                    for part in style_parts:
                        if ':' in part:
                            k, v = part.strip().split(':', 1)
                            style_dict[k.strip()] = v.strip()
                
                # Update with new style properties
                for style_key, style_value in value.items():
                    style_dict[style_key] = style_value
                
                # Convert back to style string
                style_str = ';'.join([f"{k}:{v}" for k, v in style_dict.items()])
                element.set("style", style_str)
            elif attr == "d" and element.tag.endswith("path"):
                # Special handling for path data
                element.set("d", value)
            elif attr == "points" and element.tag.endswith("polygon"):
                # Special handling for polygon points
                element.set("points", value)
            else:
                # Regular attributes
                element.set(attr, str(value))
        
        # Re-optimize the document
        self.optimizer.optimize_svg_paths(document)
        
        return document.to_string()
    
    def _apply_element_overrides(self, scene_graph: SceneGraph, overrides: Dict[str, Any]) -> None:
        """Apply custom overrides to scene graph elements.
        
        Args:
            scene_graph: Scene graph to modify
            overrides: Dictionary of element overrides
        """
        # Process overrides by element type
        for element_type, properties in overrides.items():
            # Find all nodes of this type
            nodes = scene_graph.find_nodes_by_type(element_type)
            
            # Apply properties to all matching nodes
            for node in nodes:
                for prop_key, prop_value in properties.items():
                    if prop_key == "size" and isinstance(prop_value, (tuple, list)) and len(prop_value) >= 2:
                        # Update size
                        node.size = (prop_value[0], prop_value[1], node.size[2] if len(node.size) > 2 else 0)
                    elif prop_key == "position" and isinstance(prop_value, (tuple, list)) and len(prop_value) >= 2:
                        # Update position
                        node.position = (prop_value[0], prop_value[1], node.position[2] if len(node.position) > 2 else 0)
                    else:
                        # Update other properties
                        node.properties[prop_key] = prop_value
    
    def _adjust_precision(self, document: SVGDocument, precision: int) -> None:
        """Adjust the numeric precision of all measurements in the SVG.
        
        Args:
            document: SVG document to modify
            precision: Number of decimal places to keep
        """
        def process_element(element: ET.Element) -> None:
            # List of attributes that may contain numeric values
            numeric_attrs = ["x", "y", "width", "height", "cx", "cy", "r", "rx", "ry", \
                            "x1", "y1", "x2", "y2", "offset"]
            
            # Special handling for path data
            if element.tag.endswith("path") and "d" in element.attrib:
                path_data = element.get("d")
                # Find all numeric values in path data and round them
                def round_path_numbers(match):
                    num = float(match.group(0))
                    return str(round(num, precision))
                
                path_data = re.sub(r'[-+]?\d*\.\d+|[-+]?\d+', round_path_numbers, path_data)
                element.set("d", path_data)
            
            # Process regular numeric attributes
            for attr in numeric_attrs:
                if attr in element.attrib:
                    try:
                        value = float(element.get(attr))
                        element.set(attr, str(round(value, precision)))
                    except (ValueError, TypeError):
                        # Skip if not a valid number
                        pass
            
            # Process transform attribute if present
            if "transform" in element.attrib:
                transform = element.get("transform")
                # Find numeric values in transform functions like translate(x,y) or scale(x,y)
                def round_transform_numbers(match):
                    functions = match.group(1)
                    numbers = match.group(2)
                    
                    # Round each number in the function arguments
                    rounded_numbers = re.sub( \
                        r'[-+]?\d*\.\d+|[-+]?\d+',
                        lambda m: str(round(float(m.group(0)), precision)), \
                        numbers
                    )
                    
                    return f"{functions}({rounded_numbers})"
                
                transform = re.sub(r'(\w+)\(([^)]+)\)', round_transform_numbers, transform)
                element.set("transform", transform)
            
            # Process all children
            for child in element:
                process_element(child)
        
        # Start processing from the root element
        process_element(document.root)

def generate_svg_illustration(prompt: str, width: int = 800, height: int = 600,  \
                            output_path: str = None, constraints: Dict[str, Any] = None) -> str:
    """Generate an SVG illustration from a text prompt.
    
    This function provides a simple interface to the SVG generation pipeline.
    
    Args:
        prompt: Text prompt describing the desired scene
        width: Width of the output SVG
        height: Height of the output SVG
        output_path: Path to save the SVG file (optional)
        constraints: Additional constraints for generation (optional)
        
    Returns:
        SVG document as a string
    """
    generator = AdvancedSVGGenerator(width, height)
    
    if constraints:
        return generator.generate_with_constraints(prompt, constraints, output_path)
    else:
        return generator.generate_from_prompt(prompt, output_path)

class SVGOptimizer:
    """
    Advanced implementation of SVG optimization techniques using:
    1. Informational path compression (path compounding)
    2. Edge-merging for adjacent shapes
    3. Bezier curve hierarchy (Quadratic > Cubic and Arc usage)
    4. Vector volumetry with light/shadow techniques
    5. Chromatic compression through quantization
    6. Optimized vector pseudo-noise for textures
    7. Advanced usage of clipping paths
    8. Statistical analysis for final optimization
    9. Visual hierarchy for balanced rendering
    10. Adaptive illumination based on geometric orientation
    11. Spatial prediction for pattern recognition
    12. Semantic DOM reduction
    """
    
    def __init__(self, light_angle: float = 45.0):
        """
        Initialize the SVG optimizer with default settings.
        
        Args:
            light_angle: Angle in degrees of the virtual light source (default: 45 degrees)
        """
        # Compression settings
        self.precision = 2  # Default decimal precision
        self.merge_tolerance = 0.01  # Tolerance for edge merging
        self.color_palette = None  # Limited color palette
        
        # Visual hierarchy settings - controls rendering order for binary tree optimization
        self.visual_hierarchy = { \
            'primary': {'z_index': 0, 'opacity': 1.0},    # Backgrounds, silhouettes
            'secondary': {'z_index': 1, 'opacity': 1.0},  # Main structures
            'tertiary': {'z_index': 2, 'opacity': 0.95},  # Details
            'quaternary': {'z_index': 3, 'opacity': 0.9}  # Fine details, highlights
        }
        
        # Light source configuration for adaptive illumination
        self.light_angle = light_angle  # degrees (0 = top, 90 = right)
        self.light_vector = self._calculate_light_vector(light_angle)
        
        # Spatial prediction parameters for pattern recognition
        self.detected_patterns = {}
        self.pattern_thresholds = { \
            'linear': 0.95,    # Threshold for detecting linear patterns
            'grid': 0.90,      # Threshold for detecting grid patterns
            'radial': 0.85     # Threshold for detecting radial patterns
        }
        
        # ViewBox optimization settings
        self.auto_crop_viewbox = True  # Automatically crop viewBox to content
        self.viewbox_padding = 1       # Padding in pixels around content
        
    def _calculate_light_vector(self, angle: float) -> Tuple[float, float]:
        """
        Calculate the normalized light vector from the angle
        
        Args:
            angle: Light angle in degrees
            
        Returns:
            Tuple of (x, y) components of the light vector
        """
        radians = math.radians(angle)
        return (math.cos(radians), math.sin(radians))
        
    def create_limited_palette(self, num_colors: int = 6) -> None:
        """
        Create a limited color palette for chromatic compression.
        
        Args:
            num_colors: Number of basic colors in the palette
        """
        # High compressibility base colors
        base_colors = [ \
            "#000000",  # Black
            "#FFFFFF",  # White
            "#0077BE",  # Blue
            "#44AA44",  # Green
            "#DD2222",  # Red
            "#FFDD44",  # Yellow
            "#884422",  # Brown
            "#666666",  # Gray
            "#FF6600",  # Orange
            "#7744FF"   # Purple
        ]
        
        self.color_palette = base_colors[:min(num_colors, len(base_colors))]
        
    def optimize_svg_paths(self, doc: 'SVGDocument') -> None:
        """
        Optimizes all path data in an SVG document by finding and processing all path elements.
        
        Args:
            doc: SVG document to optimize
        """
        # Process all elements in the document recursively
        self._optimize_elements_recursive(doc.root)
        
        # Process definitions as well
        if hasattr(doc, 'defs') and doc.defs:
            for def_element in doc.defs:
                self._optimize_elements_recursive(def_element)
    
    def optimize_colors(self, doc: 'SVGDocument') -> None:
        """
        Optimizes color usage in an SVG document by standardizing color formats
        and replacing similar colors with color variables.
        
        Args:
            doc: SVG document to optimize
        """
        # Get all elements that have color attributes
        self._optimize_colors_recursive(doc.elements)
        
        # Process definitions as well
        if hasattr(doc, 'defs') and doc.defs:
            self._optimize_colors_recursive(doc.defs)
    
    def _optimize_colors_recursive(self, elements: List[ET.Element]) -> None:
        """
        Recursively optimizes color attributes in a list of SVG elements and their children.
        
        Args:
            elements: List of SVG elements to process
        """
        # Color attributes to check and optimize
        color_attrs = ['fill', 'stroke', 'stop-color', 'color']
        
        # Validate input
        if not elements or not isinstance(elements, list):
            return
        
        for elem in elements:
            # Skip non-Element objects
            if not isinstance(elem, ET.Element):
                continue
            
            # Process color attributes if present
            if hasattr(elem, 'attrib'):
                for attr in color_attrs:
                    if attr in elem.attrib:
                        try:
                            # Optimize the color value (e.g., convert #ff0000 to #f00)
                            elem.attrib[attr] = self._optimize_color(elem.attrib[attr])
                        except Exception:
                            # Silently continue on error
                            pass
            
            # Process children recursively
            try:
                children = list(elem)
                if children:
                    self._optimize_colors_recursive(children)
            except Exception:
                # Skip any elements that can't be iterated
                continue
    
    def _optimize_color(self, color: str) -> str:
        """
        Optimizes a color value by standardizing the format and minimizing the representation.
        
        Args:
            color: Color value as string
            
        Returns:
            Optimized color string
        """
        # Skip optimization for non-hex colors, URLs, or "none"
        if not color or not isinstance(color, str) or color.startswith('url(') or color.lower() == 'none': \
            return color
        
        # Check if it's a hex color
        if color.startswith('#'):
            # Remove the # temporarily
            hex_value = color[1:]
            
            # Convert 6 digit hex to 3 digit hex if possible
            if len(hex_value) == 6:
                if hex_value[0] == hex_value[1] and hex_value[2] == hex_value[3] and hex_value[4] == hex_value[5]:
                    return f'#{hex_value[0]}{hex_value[2]}{hex_value[4]}'
        
        return color
    
    def _optimize_elements_recursive(self, elements: List[ET.Element]) -> None:
        """
        Recursively optimizes path data in a list of SVG elements and their children.
        
        Args:
            elements: List of SVG elements to process
        """
        # Validate input
        if not elements or not isinstance(elements, list):
            return
            
        for elem in elements:
            # Skip non-Element objects
            if not isinstance(elem, ET.Element):
                continue
                
            # Check if this is a path element with 'd' attribute
            if hasattr(elem, 'tag') and str(elem.tag).endswith('path') and hasattr(elem, 'attrib') and 'd' in elem.attrib:
                try:
                    # Optimize the path data with safety checks
                    elem.attrib['d'] = self.optimize_path_data(elem.attrib['d'])
                except Exception as e:
                    # Silently continue on error to maintain robustness
                    pass
            
            # Check if this element has children and process them
            try:
                children = list(elem)
                if children:
                    self._optimize_elements_recursive(children)
            except Exception:
                # Skip any elements that can't be iterated
                continue
    
    def optimize_path_data(self, path_data: str) -> str:
        """
        Optimizes SVG path data string by applying compression techniques.
        
        Args:
            path_data: SVG path data string (d attribute)
            
        Returns:
            Optimized path data string
        """
        # Handle edge cases
        if not path_data or not isinstance(path_data, str):
            return path_data if isinstance(path_data, str) else ""
            
        # 1. Normalize spaces and separate commands
        path_data = re.sub(r'\s+', ' ', path_data.strip())
        path_data = re.sub(r'([MLHVCSQTAZmlhvcsqtaz])\s*', r'\1 ', path_data)
        
        # 2. Split into tokens (command + parameters)
        tokens = []
        current_cmd = None
        parts = path_data.split()
        
        i = 0
        while i < len(parts):
            if parts[i][0].isalpha():
                current_cmd = parts[i][0]
                tokens.append(parts[i])
            else:
                # If not a command, add with implicit command
                # But preserve the compressed form without repeating the command
                tokens.append(parts[i])
            i += 1
            
        # 3. Main optimization: remove redundant commands
        optimized = []
        last_cmd = None
        
        for token in tokens:
            if token[0].isalpha():
                cmd = token[0].upper()
                params = token[1:]
                
                # If command changes, we need to make it explicit
                if cmd != last_cmd:
                    optimized.append(token)
                    last_cmd = cmd
                else:
                    # Same command letter, we can omit it
                    optimized.append(params)
            else:
                optimized.append(token)
                
        # 4. Remove unnecessary zeros and optimize numbers
        for i in range(len(optimized)):
            # Skip empty strings or check if first character is not alphabetic
            if optimized[i] and not optimized[i][0].isalpha():
                # Optimize numeric values
                optimized[i] = self._optimize_numeric(optimized[i])
                
        # 5. Join tokens back into a single string
        return ' '.join(optimized)
    
    def _optimize_numeric(self, value_str: str) -> str:
        """
        Optimizes the textual representation of a numeric value.
        
        Args:
            value_str: String representing a numeric value
            
        Returns:
            Optimized string
        """
        # Split values if there are commas
        if ',' in value_str:
            parts = value_str.split(',')
            return ','.join(self._optimize_numeric(p) for p in parts)
            
        try:
            value = float(value_str)
            
            # Integer
            if value == int(value):
                return str(int(value))
                
            # Decimal value - limit decimals by precision
            formatted = f"{value:.{self.precision}f}"
            
            # Remove trailing zeros and decimal point if x.0
            formatted = formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted
            
            # Additional optimization: remove leading zero for values between -1 and 1
            if formatted.startswith('0.') and len(formatted) > 2:
                return formatted[1:]
            elif formatted.startswith('-0.') and len(formatted) > 3:
                return '-' + formatted[2:]
                
            return formatted
        except ValueError:
            return value_str
            
    def convert_cubic_to_quadratic(self, path_data: str, tolerance: float = 0.1) -> str:
        """
        Converts cubic Bezier curves (C) to quadratic (Q) when possible, \
        saving bytes without perceptible quality loss.
        
        Args:
            path_data: SVG path data string
            tolerance: Conversion tolerance (lower = more precise, higher = more compact)
            
        Returns:
            Optimized path data string
        """
        # This is a simplification that would require more complex algebra in a real implementation
        # Here we identify patterns of cubic curves that are almost quadratic
        # Complete implementation would require calculation of optimized control points
        
        # Basic example: look for cubic curves where control points are almost symmetric
        pattern = r'C\s+([\d.-]+)[,\s]+([\d.-]+)[,\s]+([\d.-]+)[,\s]+([\d.-]+)[,\s]+([\d.-]+)[,\s]+([\d.-]+)'
        
        def convert_match(match):
            x1, y1 = float(match.group(1)), float(match.group(2))  # First control point
            x2, y2 = float(match.group(3)), float(match.group(4))  # Second control point
            x, y = float(match.group(5)), float(match.group(6))    # End point
            
            # Check if control points are approximately symmetric
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            
            # Distance between control points
            dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            # If distance is small or points are almost collinear, convert to quadratic
            if dist < tolerance * 100:  # Scale based on tolerance
                return f"Q {self._optimize_numeric(str(mid_x))},{self._optimize_numeric(str(mid_y))} {self._optimize_numeric(str(x))},{self._optimize_numeric(str(y))}"
            
            # Keep the original cubic curve
            return match.group(0)
            
        # Apply the conversion where possible
        return re.sub(pattern, convert_match, path_data)
        
    def use_arcs_when_possible(self, path_data: str) -> str:
        """
        Replaces sequences of curves that form circles/arcs with
        more efficient 'A' (elliptical arc) commands.
        
        Args:
            path_data: SVG path data string
            
        Returns:
            Optimized path data string
        """
        # Simplified implementation - a complete version would require
        # algorithms to detect arcs in sequences of curves
        
        # For now, we detect specific patterns that look like arcs
        # In a real implementation, we would need to adjust the parameters
        # rx, ry, x-axis-rotation, large-arc-flag and sweep-flag appropriately
        
        # Basic example of a pattern to be replaced
        # This example is highly simplified and wouldn't work in all cases
        return path_data
        
    def optimize_svg_element(self, element: ET.Element) -> ET.Element:
        """
        Applies advanced optimizations to an SVG element.
        
        Args:
            element: SVG Element (ET.Element)
            
        Returns:
            Optimized SVG element
        """
        # Optimize the 'd' attribute of path elements
        if element.tag.endswith('path'):
            path_data = element.get('d', '')
            if path_data:
                # Apply all path optimizations in sequence
                optimized_data = self.optimize_path_data(path_data)
                optimized_data = self.convert_cubic_to_quadratic(optimized_data)
                optimized_data = self.use_arcs_when_possible(optimized_data)
                
                element.set('d', optimized_data)
                
        # Optimize colors, removing redundant values
        for attr in ['fill', 'stroke']:
            color = element.get(attr, '')
            if color and color.startswith('#'):
                # Convert to compact form if possible (#RRGGBB -> #RGB)
                if len(color) == 7 and color[1:3] == color[3:5] == color[5:7]:
                    element.set(attr, f"#{color[1]}{color[3]}{color[5]}")
                    
        # Optimize opacity - remove if it's 1
        for attr in ['fill-opacity', 'stroke-opacity', 'opacity']:
            opacity = element.get(attr, '')
            if opacity == '1' or opacity == '1.0':
                del element.attrib[attr]
                
        # Process children recursively
        for child in element:
            self.optimize_svg_element(child)
            
        return element
        
# Register SVG namespaces
ET.register_namespace("", SVGNS)
ET.register_namespace("xlink", XLINKNS)


class Point:
    """Represents a 2D point with optional precision control for optimization."""
    
    def __init__(self, x: float, y: float, precision: int = 1):
        """
        Initialize a point with given coordinates.
        
        Args:
            x: X coordinate
            y: Y coordinate
            precision: Decimal precision for coordinate values (default=1)
        """
        self.x = self._quantize(x, precision)
        self.y = self._quantize(y, precision)
        self.precision = precision
        
    def _quantize(self, value: float, precision: int) -> float:
        """Quantize a value to the specified decimal precision."""
        factor = 10 ** precision
        return round(value * factor) / factor
        
    def __str__(self) -> str:
        """Return string representation with optimized precision."""
        if self.precision == 0:
            return f"{int(self.x)},{int(self.y)}"
        else:
            x_str = f"{self.x:.{self.precision}f}".rstrip('0').rstrip('.')
            y_str = f"{self.y:.{self.precision}f}".rstrip('0').rstrip('.')
            return f"{x_str},{y_str}"
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def is_redundant(self, other: 'Point', threshold: float = 0.2) -> bool:
        """Check if this point is redundant compared to another within threshold."""
        return self.distance_to(other) < threshold


class Path:
    """Represents an SVG path with optimization capabilities."""
    
    def __init__(self, precision: int = 1):
        """
        Initialize an empty path with given precision for coordinates.
        
        Args:
            precision: Decimal precision for coordinate values (default=1)
        """
        self.commands = []
        self.precision = precision
        self.current_point = Point(0, 0, precision)
        self.path_start = None
        
    def move_to(self, x: float, y: float, relative: bool = False) -> 'Path':
        """
        Add move command (M/m) to the path.
        
        Args:
            x: X coordinate
            y: Y coordinate
            relative: Whether to use relative coordinates (m) instead of absolute (M)
        
        Returns:
            Self for method chaining
        """
        point = Point(x, y, self.precision)
        cmd = "m" if relative else "M"
        self.commands.append(f"{cmd}{point}")
        
        if not relative:
            self.current_point = point
            self.path_start = point
        else:
            self.current_point = Point(self.current_point.x + x, self.current_point.y + y, self.precision)
            self.path_start = self.current_point
            
        return self
        
    def M(self, x: float, y: float) -> 'Path':
        """
        Shorthand for absolute move_to command (M).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.move_to(x, y, relative=False)
        
    def m(self, x: float, y: float) -> 'Path':
        """
        Shorthand for relative move_to command (m).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.move_to(x, y, relative=True)
    
    def line_to(self, x: float, y: float, relative: bool = False) -> 'Path':
        """
        Add line command (L/l) to the path.
        
        Args:
            x: X coordinate
            y: Y coordinate
            relative: Whether to use relative coordinates (l) instead of absolute (L)
        
        Returns:
            Self for method chaining
        """
        point = Point(x, y, self.precision)
        cmd = "l" if relative else "L"
        self.commands.append(f"{cmd}{point}")
        
        if not relative:
            self.current_point = point
        else:
            self.current_point = Point(self.current_point.x + x, self.current_point.y + y, self.precision)
        
        return self
        
    def L(self, x: float, y: float) -> 'Path':
        """
        Shorthand for absolute line_to command (L).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.line_to(x, y, relative=False)
        
    def l(self, x: float, y: float) -> 'Path':
        """
        Shorthand for relative line_to command (l).
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.line_to(x, y, relative=True)
    
    def curve_to(self, x: float, y: float, cx1: float, cy1: float, cx2: float, cy2: float, \
                    relative: bool = True) -> 'Path':
        """
        Add cubic Bézier curve command (C/c) to the path.
        
        Args:
            x: End X coordinate
            y: End Y coordinate
            cx1: First control point X coordinate
            cy1: First control point Y coordinate
            cx2: Second control point X coordinate
            cy2: Second control point Y coordinate
            relative: Whether to use relative coordinates (c) instead of absolute (C)
        
        Returns:
            Self for method chaining
        """
        end_point = Point(x, y, self.precision)
        control1 = Point(cx1, cy1, self.precision)
        control2 = Point(cx2, cy2, self.precision)
        cmd = "c" if relative else "C"
        
        self.commands.append(f"{cmd}{control1} {control2} {end_point}")
        
        if not relative:
            self.current_point = end_point
        else:
            self.current_point = Point(self.current_point.x + x, self.current_point.y + y, self.precision)
        
        return self
        
    def quad_curve_to(self, x: float, y: float, cx: float, cy: float, relative: bool = True) -> 'Path':
        """
        Add quadratic Bézier curve command (Q/q) to the path.
        
        Args:
            x: End X coordinate
            y: End Y coordinate
            cx: Control point X coordinate
            cy: Control point Y coordinate
            relative: Whether to use relative coordinates (q) instead of absolute (Q)
        
        Returns:
            Self for method chaining
        """
        end_point = Point(x, y, self.precision)
        control = Point(cx, cy, self.precision)
        cmd = "q" if relative else "Q"
        
        self.commands.append(f"{cmd}{control} {end_point}")
        
        if not relative:
            self.current_point = end_point
        else:
            self.current_point = Point(self.current_point.x + x, self.current_point.y + y, self.precision)
        
        return self
    
    def arc_to(self, x: float, y: float, rx: float, ry: float, x_rotation: float = 0, \
                large_arc: bool = False, sweep: bool = True, relative: bool = True) -> 'Path':
        """
        Add elliptical arc command (A/a) to the path.
        
        Args:
            x: End X coordinate
            y: End Y coordinate
            rx: X radius of the arc
            ry: Y radius of the arc
            x_rotation: Rotation of the arc in degrees
            large_arc: Use large arc (1) or small arc (0)
            sweep: Sweep flag (1) or not (0)
            relative: Whether to use relative coordinates (a) instead of absolute (A)
        
        Returns:
            Self for method chaining
        """
        end_point = Point(x, y, self.precision)
        rx = self._quantize_value(rx)
        ry = self._quantize_value(ry)
        x_rotation = self._quantize_value(x_rotation)
        large_arc_flag = 1 if large_arc else 0
        sweep_flag = 1 if sweep else 0
        
        cmd = "a" if relative else "A"
        
        self.commands.append(f"{cmd}{rx},{ry} {x_rotation} {large_arc_flag},{sweep_flag} {end_point}")
        
        if not relative:
            self.current_point = end_point
        else:
            self.current_point = Point(self.current_point.x + x, self.current_point.y + y, self.precision)
        
        return self
    
    def close_path(self) -> 'Path':
        """
        Add close path command (Z/z) to the path.
        
        Returns:
            Self for method chaining
        """
        self.commands.append("z")
        if self.path_start:
            self.current_point = self.path_start
        
        return self
        
    def Z(self) -> 'Path':
        """
        Shorthand for close path command (Z).
        
        Returns:
            Self for method chaining
        """
        return self.close_path()
        
    def z(self) -> 'Path':
        """
        Shorthand for close path command (z).
        
        Returns:
            Self for method chaining
        """
        return self.close_path()
        
    def Q(self, x: float, y: float, cx: float, cy: float) -> 'Path':
        """
        Shorthand for absolute quadratic Bézier curve command (Q).
        
        Args:
            x: End X coordinate
            y: End Y coordinate
            cx: Control point X coordinate
            cy: Control point Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.quad_curve_to(x, y, cx, cy, relative=False)
        
    def q(self, x: float, y: float, cx: float, cy: float) -> 'Path':
        """
        Shorthand for relative quadratic Bézier curve command (q).
        
        Args:
            x: End X coordinate
            y: End Y coordinate
            cx: Control point X coordinate
            cy: Control point Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.quad_curve_to(x, y, cx, cy, relative=True)
        
    def C(self, x: float, y: float, cx1: float, cy1: float, cx2: float, cy2: float) -> 'Path':
        """
        Shorthand for absolute cubic Bézier curve command (C).
        
        Args:
            x: End X coordinate
            y: End Y coordinate
            cx1: First control point X coordinate
            cy1: First control point Y coordinate
            cx2: Second control point X coordinate
            cy2: Second control point Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.curve_to(x, y, cx1, cy1, cx2, cy2, relative=False)
        
    def c(self, x: float, y: float, cx1: float, cy1: float, cx2: float, cy2: float) -> 'Path':
        """
        Shorthand for relative cubic Bézier curve command (c).
        
        Args:
            x: End X coordinate
            y: End Y coordinate
            cx1: First control point X coordinate
            cy1: First control point Y coordinate
            cx2: Second control point X coordinate
            cy2: Second control point Y coordinate
            
        Returns:
            Self for method chaining
        """
        return self.curve_to(x, y, cx1, cy1, cx2, cy2, relative=True)
        
    def optimize_path(self) -> 'Path':
        """
        Optimizes the path applying advanced vector compression techniques:
        1. Eliminates redundant commands (consecutive L become coordinates only)
        2. Minimizes coordinates while maintaining visual precision
        3. Applies perception-based simplification
        
        Returns:
            Self for method chaining
        """
        if not self.commands:
            return self
            
        # Current command for comparison
        current_cmd = None
        optimized_commands = []
        
        for i, cmd in enumerate(self.commands):
            # If it's a coordinate without command letter (implicit continuation)
            if all(c.isdigit() or c in '.-,' for c in cmd):
                optimized_commands.append(cmd)
                continue
                
            # Extract the command letter (first character)
            if cmd and cmd[0].isalpha():
                cmd_letter = cmd[0].upper()
                cmd_value = cmd[1:].strip() if len(cmd) > 1 else ""
                
                # If it's the same command as the previous one, we can omit the letter
                if cmd_letter == current_cmd:
                    optimized_commands.append(cmd_value)
                else:
                    optimized_commands.append(cmd)
                    current_cmd = cmd_letter
            else:
                # Keep closing commands or other special ones like 'z'
                optimized_commands.append(cmd)
                current_cmd = None
        
        # Substitui os comandos originais pelos otimizados
        self.commands = optimized_commands
        return self
        
    def merge_with_adjacent(self, other_path: 'Path') -> bool:
        """
        Attempts to merge this path with another adjacent path, \
        if they share a collinear edge.
        
        Args:
            other_path: Another Path to try to merge with
            
        Returns:
            bool: True if the paths were merged, False otherwise
        """
        # Extract the points from this path
        points1 = self._extract_path_points()
        # Extract points from the other path
        points2 = other_path._extract_path_points()
        
        if not points1 or not points2:
            return False
            
        # Check if there's any shared edge (same points in reverse order)
        shared_edges = []
        
        for i in range(len(points1)):
            pt1 = points1[i]
            pt2 = points1[(i+1) % len(points1)]
            
            for j in range(len(points2)):
                pt3 = points2[j]
                pt4 = points2[(j+1) % len(points2)]
                
                # If the edge is the same but in opposite directions
                if (self._points_equal(pt1, pt4) and self._points_equal(pt2, pt3)):
                    shared_edges.append((i, j))
                    
        if shared_edges:
            # Implement the fusion of the first shared edge found
            edge = shared_edges[0]
            i, j = edge
            
            # Create a new merged path
            merged_path = Path(self.precision)
            
            # Add points from the first path up to the shared edge
            merged_path.move_to(points1[0][0], points1[0][1])  # First point
            for k in range(1, i+1):
                merged_path.line_to(points1[k][0], points1[k][1])
                
            # Add points from the second path after the shared edge
            for k in range((j+2) % len(points2), len(points2)):
                merged_path.line_to(points2[k][0], points2[k][1])
            # Complete with the beginning of the second path
            for k in range(0, j+1):
                merged_path.line_to(points2[k][0], points2[k][1])
                
            # Continue with the rest of the first path
            for k in range(i+2, len(points1)):
                merged_path.line_to(points1[k][0], points1[k][1])
                
            merged_path.close_path()
            
            # Replace the commands of this path with those of the merged one
            self.commands = merged_path.commands
            self.current_point = merged_path.current_point
            self.path_start = merged_path.path_start
            
            return True
            
        return False
        
    def _extract_path_points(self) -> List[Tuple[float, float]]:
        """
        Extracts a list of points (x,y) from the current path.
        
        Returns:
            List of tuples (x,y) representing the path points
        """
        points = []
        current_x, current_y = 0, 0
        first_x, first_y = 0, 0
        first_point_set = False
        
        for cmd in self.commands:
            if not cmd:
                continue
                
            if cmd == 'z' or cmd == 'Z':
                # Close the path by returning to the first point
                if first_point_set:
                    points.append((first_x, first_y))
                continue
                
            cmd_letter = cmd[0] if cmd[0].isalpha() else None
            params = cmd[1:] if cmd_letter else cmd
            
            if cmd_letter == 'M':
                # absolute moveto
                parts = params.strip().split()
                if len(parts) >= 2:
                    current_x = float(parts[0])
                    current_y = float(parts[1])
                    if not first_point_set:
                        first_x, first_y = current_x, current_y
                        first_point_set = True
                    points.append((current_x, current_y))
            elif cmd_letter == 'L':
                # absolute lineto
                parts = params.strip().split()
                if len(parts) >= 2:
                    current_x = float(parts[0])
                    current_y = float(parts[1])
                    points.append((current_x, current_y))
            # Process other types of commands as needed...
        
        return points
        
    def _points_equal(self, p1: Tuple[float, float], p2: Tuple[float, float], tolerance: float = 0.01) -> bool:
        """
        Checks if two points are considered equal within a tolerance.
        
        Args:
            p1: First point (x1, y1)
            p2: Second point (x2, y2)
            tolerance: Tolerance for equality
            
        Returns:
            bool: True if the points are considered equal
        """
        return (abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance)
    
    def _quantize_value(self, value: float) -> str:
        """
        Quantize a value to the specified precision and convert to string, \
        stripping trailing zeros.
        """
        if self.precision == 0:
            return str(int(round(value)))
        else:
            return f"{value:.{self.precision}f}".rstrip('0').rstrip('.')
    
    def simplify(self, tolerance: float = 0.2) -> 'Path':
        """
        Simplify the path by removing redundant points.This is a basic implementation of path simplification.
        
        Args:
            tolerance: Distance threshold for point removal
            
        Returns:
            Self with simplified path
        """
        # TODO: Implement Douglas-Peucker algorithm for path simplification
        # This is a placeholder for the actual implementation
        return self
    
    def to_svg_path_data(self) -> str:
        """
        Convert the path commands to SVG path data string.
        
        Returns:
            SVG path data string (d attribute)
        """
        return " ".join(self.commands)
    
    def to_svg_element(self, attributes: Dict[str, str] = None) -> ET.Element:
        """
        Convert the path to an SVG path element.
        
        Args:
            attributes: Additional attributes for the path element
            
        Returns:
            SVG path element
        """
        attribs = {"d": self.to_svg_path_data()}
        if attributes:
            attribs.update(attributes)
            
        path_elem = ET.Element("path", attribs)
        return path_elem


class SVGShape:
    """Base class for SVG shape elements."""
    
    def __init__(self, precision: int = 1):
        """Initialize a shape with given precision for coordinates."""
        self.precision = precision
        
    def _quantize_value(self, value: float) -> str:
        """Quantize a value to the specified precision and convert to string."""
        if self.precision == 0:
            return str(int(round(value)))
        else:
            return f"{value:.{self.precision}f}".rstrip('0').rstrip('.')
    
    def to_svg_element(self, attributes: Dict[str, str] = None) -> ET.Element:
        """Convert the shape to an SVG element."""
        raise NotImplementedError("Subclasses must implement this method")


class Rectangle(SVGShape):
    """Represents an SVG rectangle element."""
    
    def __init__(self, x: float, y: float, width: float, height: float, precision: int = 1):
        """
        Initialize a rectangle with given position and dimensions.
        
        Args:
            x: X coordinate of the top-left corner
            y: Y coordinate of the top-left corner
            width: Width of the rectangle
            height: Height of the rectangle
            precision: Decimal precision for coordinate values (default=1)
        """
        super().__init__(precision)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def to_svg_element(self, attributes: Dict[str, str] = None) -> ET.Element:
        """
        Convert the rectangle to an SVG rect element.
        
        Args:
            attributes: Additional attributes for the element
            
        Returns:
            SVG rect element
        """
        x = self._quantize_value(self.x)
        y = self._quantize_value(self.y)
        width = self._quantize_value(self.width)
        height = self._quantize_value(self.height)
        
        attribs = {"x": x, "y": y, "width": width, "height": height}
        if attributes:
            attribs.update(attributes)
            
        rect_elem = ET.Element("rect", attribs)
        return rect_elem
    
    def to_path(self) -> Path:
        """
        Convert the rectangle to a path.
        
        Returns:
            Path object representing the rectangle
        """
        path = Path(self.precision)
        path.move_to(self.x, self.y, False)
        path.line_to(self.width, 0)
        path.line_to(0, self.height)
        path.line_to(-self.width, 0)
        path.close_path()
        return path


class Circle(SVGShape):
    """Represents an SVG circle element."""
    
    def __init__(self, cx: float, cy: float, r: float, precision: int = 1):
        """
        Initialize a circle with given center and radius.
        
        Args:
            cx: X coordinate of the center
            cy: Y coordinate of the center
            r: Radius of the circle
            precision: Decimal precision for coordinate values (default=1)
        """
        super().__init__(precision)
        self.cx = cx
        self.cy = cy
        self.r = r
    
    def to_svg_element(self, attributes: Dict[str, str] = None) -> ET.Element:
        """
        Convert the circle to an SVG circle element.
        
        Args:
            attributes: Additional attributes for the element
            
        Returns:
            SVG circle element
        """
        cx = self._quantize_value(self.cx)
        cy = self._quantize_value(self.cy)
        r = self._quantize_value(self.r)
        
        attribs = {"cx": cx, "cy": cy, "r": r}
        if attributes:
            attribs.update(attributes)
            
        circle_elem = ET.Element("circle", attribs)
        return circle_elem
    
    def to_path(self) -> Path:
        """
        Convert the circle to a path using arcs.
        
        Returns:
            Path object representing the circle
        """
        path = Path(self.precision)
        # Move to leftmost point of the circle
        path.move_to(self.cx - self.r, self.cy, False)
        # Draw the top semicircle
        path.arc_to(self.cx + self.r, self.cy, self.r, self.r, 0, False, True, False)
        # Draw the bottom semicircle
        path.arc_to(self.cx - self.r, self.cy, self.r, self.r, 0, False, True, False)
        path.close_path()
        return path


class Ellipse(SVGShape):
    """Represents an SVG ellipse element."""
    
    def __init__(self, cx: float, cy: float, rx: float, ry: float, precision: int = 1):
        """
        Initialize an ellipse with given center and radii.
        
        Args:
            cx: X coordinate of the center
            cy: Y coordinate of the center
            rx: X radius of the ellipse
            ry: Y radius of the ellipse
            precision: Decimal precision for coordinate values (default=1)
        """
        super().__init__(precision)
        self.cx = cx
        self.cy = cy
        self.rx = rx
        self.ry = ry
    
    def to_svg_element(self, attributes: Dict[str, str] = None) -> ET.Element:
        """
        Convert the ellipse to an SVG ellipse element.
        
        Args:
            attributes: Additional attributes for the element
            
        Returns:
            SVG ellipse element
        """
        cx = self._quantize_value(self.cx)
        cy = self._quantize_value(self.cy)
        rx = self._quantize_value(self.rx)
        ry = self._quantize_value(self.ry)
        
        attribs = {"cx": cx, "cy": cy, "rx": rx, "ry": ry}
        if attributes:
            attribs.update(attributes)
            
        ellipse_elem = ET.Element("ellipse", attribs)
        return ellipse_elem
    
    def to_path(self) -> Path:
        """
        Convert the ellipse to a path using arcs.
        
        Returns:
            Path object representing the ellipse
        """
        path = Path(self.precision)
        # Move to leftmost point of the ellipse
        path.move_to(self.cx - self.rx, self.cy, False)
        # Draw the top semicircle
        path.arc_to(self.cx + self.rx, self.cy, self.rx, self.ry, 0, False, True, False)
        # Draw the bottom semicircle
        path.arc_to(self.cx - self.rx, self.cy, self.rx, self.ry, 0, False, True, False)
        path.close_path()
        return path


class LinearGradient:
    """Represents an SVG linear gradient for efficient color transitions."""
    
    def __init__(self, id: str, x1: float = 0, y1: float = 0, x2: float = 0, y2: float = 1, precision: int = 1):
        """
        Initialize a linear gradient with given coordinates and ID.
        
        Args:
            id: Unique identifier for the gradient
            x1: X coordinate of the start point
            y1: Y coordinate of the start point
            x2: X coordinate of the end point
            y2: Y coordinate of the end point
            precision: Decimal precision for coordinate values (default=1)
        """
        self.id = id
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.precision = precision
        self.stops = []
    
    def _quantize_value(self, value: float) -> str:
        """Quantize a value to the specified precision and convert to string."""
        if self.precision == 0:
            return str(int(round(value)))
        else:
            return f"{value:.{self.precision}f}".rstrip('0').rstrip('.')
    
    def add_stop(self, offset: float, color: str, opacity: float = None) -> 'LinearGradient':
        """
        Add a color stop to the gradient.
        
        Args:
            offset: Position of the stop (0 to 1)
            color: Color value of the stop (hex, rgb, etc.)
            opacity: Optional opacity value (0 to 1)
            
        Returns:
            Self for method chaining
        """
        self.stops.append((offset, color, opacity))
        return self
    
    def to_svg_element(self) -> ET.Element:
        """
        Convert the linear gradient to an SVG linearGradient element.
        
        Returns:
            SVG linearGradient element
        """
        x1 = self._quantize_value(self.x1)
        y1 = self._quantize_value(self.y1)
        x2 = self._quantize_value(self.x2)
        y2 = self._quantize_value(self.y2)
        
        attribs = {"id": self.id, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        
        grad_elem = ET.Element("linearGradient", attribs)
        
        for offset, color, opacity in self.stops:
            stop_attribs = { \
                "offset": f"{offset:.2f}".rstrip('0').rstrip('.'),
                "stop-color": color
            }
            if opacity is not None:
                stop_attribs["stop-opacity"] = f"{opacity:.2f}".rstrip('0').rstrip('.')
                
            stop_elem = ET.SubElement(grad_elem, "{%s}stop" % SVGNS, stop_attribs)
            
        return grad_elem


class SVGDocument:
    """Represents an SVG document with size optimization capabilities."""
    
    def __init__(self, width: float = 800, height: float = 600, precision: int = 1):
        """
        Initialize an SVG document with given dimensions.
        
        Args:
            width: Width of the SVG viewport
            height: Height of the SVG viewport
            precision: Decimal precision for coordinate values (default=1)
        """
        self.width = width
        self.height = height
        self.precision = precision
        self.elements = []
        self.defs = []
        
        # Create root element
        self.root = ET.Element("svg")
        self.root.set("xmlns", SVGNS)
        self.root.set("xmlns:xlink", XLINKNS)
        self.root.set("width", str(width))
        self.root.set("height", str(height))
        self.root.set("viewBox", f"0 0 {width} {height}")
        
        # Create defs element
        self.defs_element = ET.SubElement(self.root, "defs")
        
    def _quantize_value(self, value: float) -> str:
        """Quantize a value to the specified precision and convert to string."""
        if self.precision == 0:
            return str(int(round(value)))
        else:
            return f"{value:.{self.precision}f}".rstrip('0').rstrip('.')
        
    def add_element(self, element: ET.Element) -> 'SVGDocument':
        """
        Add an SVG element to the document.
        
        Args:
            element: SVG element to add
            
        Returns:
            Self for method chaining
        """
        self.elements.append(element)
        self.root.append(element)
        return self
    
    def add_definition(self, definition: ET.Element) -> 'SVGDocument':
        """
        Add a definition element (gradient, pattern, etc.) to the document.
        
        Args:
            definition: Definition element to add
            
        Returns:
            Self for method chaining
        """
        self.defs.append(definition)
        self.defs_element.append(definition)
        return self
    
    def set_size(self, width: float, height: float) -> 'SVGDocument':
        """
        Set or update the document dimensions.
        
        Args:
            width: New width of the SVG viewport
            height: New height of the SVG viewport
            
        Returns:
            Self for method chaining
        """
        self.width = width
        self.height = height
        return self
        
    def create_clip_path(self, clip_id: str) -> ET.Element:
        """
        Create a clipping path element for masking content.
        
        Args:
            clip_id: Unique identifier for the clip path
            
        Returns:
            ClipPath element that can be populated with path elements
        """
        # Create the clipPath element with the given ID
        clip_path = ET.Element("clipPath", attrib={ \
            "id": clip_id
        })
        
        return clip_path
    
    def create_group_element(self, group_id: str, attributes: Dict[str, str] = None) -> ET.Element:
        """
        Create an SVG group element (g) with the given ID and attributes.
        
        Args:
            group_id: Unique identifier for the group
            attributes: Dictionary of additional attributes for the group
            
        Returns:
            Group element that can contain other SVG elements
        """
        # Start with the base attributes including the ID
        attribs = {"id": group_id}
        
        # Add any additional attributes if provided
        if attributes:
            attribs.update(attributes)
        
        # Create the group element with the given attributes
        group = ET.Element("g", attrib=attribs)
        
        # Add the group to the document elements
        self.add_element(group)
        
        return group
    
    def create_element(self, element_type: str, attributes: Dict[str, str] = None) -> ET.Element:
        """
        Create an SVG element of the specified type.
        
        Args:
            element_type: Type of SVG element to create (path, rect, etc.)
            attributes: Dictionary of attributes for the element
            
        Returns:
            Created SVG element
        """
        # Create the basic element with the specified type
        element = ET.Element(element_type)
        
        # Add attributes if provided
        if attributes:
            for key, value in attributes.items():
                element.set(key, value)
        
        # Add the element to the document
        self.add_element(element)
        
        return element
    
    def from_string(self, svg_string: str) -> 'SVGDocument':
        """
        Initialize the document from an SVG string.
        
        Args:
            svg_string: SVG document as string
            
        Returns:
            Self for method chaining
        """
        # Remove XML declaration if present
        if svg_string.startswith('<?xml'):
            svg_string = svg_string[svg_string.find('?>')+2:].strip()
        
        # Parse the SVG content
        try:
            self.root = ET.fromstring(svg_string)
            
            # Extract width and height from root element
            if 'width' in self.root.attrib:
                width_str = self.root.get('width')
                self.width = float(width_str.rstrip('px')) if 'px' in width_str else float(width_str)
            
            if 'height' in self.root.attrib:
                height_str = self.root.get('height')
                self.height = float(height_str.rstrip('px')) if 'px' in height_str else float(height_str)
            
            # Find and store defs_element
            self.defs_element = self.root.find(f".//{{{SVGNS}}}defs")
            if self.defs_element is None:
                self.defs_element = ET.SubElement(self.root, "{%s}defs" % SVGNS)
                
            # Reset elements and defs lists
            self.elements = []
            self.defs = []
            
            # Populate elements list (excluding defs)
            for child in self.root:
                if child != self.defs_element:
                    self.elements.append(child)
            
            # Populate defs list
            if self.defs_element is not None:
                for child in self.defs_element:
                    self.defs.append(child)
                    
        except Exception as e:
            print(f"Error parsing SVG string: {e}")
        
        return self
    
    def to_string(self, pretty_print: bool = False) -> str:
        """
        Convert the document to an SVG string.
        
        Args:
            pretty_print: Whether to format the XML with indentation (increases file size)
            
        Returns:
            SVG document as string
        """
        # Update root attributes with current dimensions
        self.root.set("width", str(self.width))
        self.root.set("height", str(self.height))
        self.root.set("viewBox", f"0 0 {self.width} {self.height}")
        
        # Convert to string
        xml_str = ET.tostring(self.root, encoding="unicode")
        
        if not pretty_print:
            # Remove whitespace between tags to reduce file size
            xml_str = re.sub(r'>\s+<', '><', xml_str)
        
        # Add XML declaration
        return f'<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n{xml_str}'
    
    def find_element_by_id(self, element_id: str) -> Optional[ET.Element]:
        """
        Find an element in the document by its ID attribute.
        
        Args:
            element_id: ID attribute value to search for
            
        Returns:
            Element with the specified ID, or None if not found
        """
        # Helper function to search recursively
        def find_by_id_recursive(elements):
            for elem in elements:
                # Check if current element has the ID
                if 'id' in elem.attrib and elem.attrib['id'] == element_id:
                    return elem
                
                # Recursively search child elements
                children = list(elem)
                if children:
                    result = find_by_id_recursive(children)
                    if result is not None:
                        return result
            return None
        
        # Search in regular elements
        result = find_by_id_recursive(self.elements)
        if result is not None:
            return result
        
        # Also search in definitions
        if self.defs:
            return find_by_id_recursive(self.defs)
        
        return None
        
    def remove_element(self, element: ET.Element) -> bool:
        """
        Remove an element from the document.
        
        Args:
            element: The element to remove
            
        Returns:
            True if element was found and removed, False otherwise
        """
        # Direct removal if element is in the top-level elements list
        if element in self.elements:
            self.elements.remove(element)
            return True
            
        # Direct removal if element is in the top-level definitions list
        if self.defs and element in self.defs:
            self.defs.remove(element)
            return True
            
        # Helper function to find parent element
        def find_and_remove_from_parent(container_elements):
            for container in container_elements:
                children = list(container)
                if element in children:
                    container.remove(element)
                    return True
                    
                # Recursively search deeper
                if children and find_and_remove_from_parent(children):
                    return True
                    
            return False
        
        # Try to find and remove from document structure
        if find_and_remove_from_parent(self.elements):
            return True
            
        # Also check in definitions
        if self.defs and find_and_remove_from_parent(self.defs):
            return True
            
        return False
    
    def save(self, filename: str, pretty_print: bool = False) -> int:
        """
        Save the document to an SVG file.
        
        Args:
            filename: Path to save the file
            pretty_print: Whether to format the XML with indentation (increases file size)
            
        Returns:
            Size of the saved file in bytes
        """
        xml_str = self.to_string(pretty_print)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(xml_str)
            
        return len(xml_str.encode('utf-8'))
    
    def optimize(self) -> 'SVGDocument':
        """
        Optimize the SVG document for size reduction.This method applies various optimization techniques to reduce file size.
        
        Returns:
            Self with optimized content
        """
        # Basic optimizations
        # 1. Convert absolute coordinates to relative in paths
        # 2. Remove unnecessary precision in numeric values
        # 3. Minimize attribute duplication
        
        for element in list(self.elements):
            self._optimize_element(element)
            
        return self
    
    def extreme_optimize(self) -> 'SVGDocument':
        """
        Apply extreme optimization techniques for maximum compression while
        maintaining visual fidelity. This implements advanced mathematical and
        information-theoretic approaches to SVG optimization.
        
        Returns:
            Self with extremely optimized content
        """
        # Create an optimizer with appropriate settings
        optimizer = SVGOptimizer()
        optimizer.create_limited_palette(6)  # Limit to optimal palette size
        
        # 1. Path optimization: Apply path compounding and mathematical simplification
        self._apply_path_optimization(optimizer)
        
        # 2. Edge merging: Merge adjacent paths with shared edges
        self._apply_edge_merging()
        
        # 3. Bezier hierarchy: Convert cubic curves to quadratic when possible
        self._apply_bezier_optimization(optimizer)
        
        # 4. Color optimization: Apply semantic compression to colors
        self._apply_color_optimization(optimizer)
        
        # 5. Attributes optimization: Remove redundant attributes
        self._optimize_attributes()
        
        # 6. Structure optimization: Group elements with shared attributes
        self._apply_structural_optimization()
        
        return self
    
    def _optimize_element(self, element: ET.Element) -> None:
        """
        Apply basic optimizations to an SVG element.
        
        Args:
            element: SVG element to optimize
        """
        # Optimize path data if it's a path element
        if element.tag.endswith('path') and 'd' in element.attrib:
            path_data = element.get('d', '')
            # Convert to relative coordinates when beneficial
            path_data = self._optimize_path_data(path_data)
            element.set('d', path_data)
            
        # Optimize numeric attributes
        for attr in ['x', 'y', 'width', 'height', 'cx', 'cy', 'r', 'rx', 'ry']:
            if attr in element.attrib:
                try:
                    value = float(element.attrib[attr])
                    element.attrib[attr] = self._quantize_value(value)
                except ValueError:
                    pass
        
        # Process child elements recursively
        for child in element:
            self._optimize_element(child)
    
    def _optimize_path_data(self, path_data: str) -> str:
        """
        Optimize path data by converting absolute to relative coordinates
        and removing unnecessary precision.
        
        Args:
            path_data: SVG path data string
            
        Returns:
            Optimized path data string
        """
        # This is a simplified implementation
        # A full implementation would parse the path and convert commands intelligently
        return path_data
    
    def _apply_path_optimization(self, optimizer: SVGOptimizer) -> None:
        """
        Apply advanced path optimization techniques.
        
        Args:
            optimizer: SVGOptimizer instance
        """
        for i, element in enumerate(self.elements):
            if element.tag.endswith('path') and 'd' in element.attrib:
                path_data = element.get('d', '')
                # Apply extreme path optimization
                optimized_data = optimizer.optimize_path_data(path_data)
                element.set('d', optimized_data)
                
    def _apply_edge_merging(self) -> None:
        """
        Merge adjacent paths with shared edges to reduce total markup size.
        """
        # This would implement edge detection and merging for paths with the same style
        # Placeholder for advanced implementation
        pass
    
    def _apply_bezier_optimization(self, optimizer: SVGOptimizer) -> None:
        """
        Apply Bezier curve hierarchy optimization.
        
        Args:
            optimizer: SVGOptimizer instance
        """
        for element in self.elements:
            if element.tag.endswith('path') and 'd' in element.attrib:
                path_data = element.get('d', '')
                # Convert cubic curves to quadratic when possible
                optimized_data = optimizer.convert_cubic_to_quadratic(path_data)
                # Use arcs for circular segments
                optimized_data = optimizer.use_arcs_when_possible(optimized_data)
                element.set('d', optimized_data)
    
    def _apply_color_optimization(self, optimizer: SVGOptimizer) -> None:
        """
        Apply color optimization techniques.
        
        Args:
            optimizer: SVGOptimizer instance
        """
        # Collect all colors from elements
        colors = []
        for element in self.elements:
            for attr in ['fill', 'stroke']:
                color = element.get(attr, '')
                if color and color.startswith('#') and color not in colors and color != 'none':
                    colors.append(color)
        
        # Build optimized palette
        optimizer.build_palette([c for c in colors if c != 'none'])
        
        # Apply optimized colors to elements
        for element in self.elements:
            for attr in ['fill', 'stroke']:
                color = element.get(attr, '')
                if color and color.startswith('#') and color != 'none':
                    element.set(attr, optimizer.optimize_color(color))
    
    def _optimize_attributes(self) -> None:
        """
        Remove redundant attributes and optimize attribute values.
        """
        for element in self.elements:
            # Remove default attributes
            for attr, default in [('fill-opacity', '1'), ('stroke-opacity', '1'), ('opacity', '1')]:
                if attr in element.attrib and element.attrib[attr] in ['1', '1.0']:
                    del element.attrib[attr]
            
            # Optimize attribute order for better gzip compression
            # This would reorder attributes based on frequency analysis
            pass
    
    def _apply_structural_optimization(self) -> None:
        """
        Apply structural optimizations like grouping elements with shared attributes.
        """
        # Group elements with shared attributes
        # This would analyze elements to find shared attributes and move them to parent groups
        pass
        
    def _quantize_value(self, value: float, precision: int = 2) -> str:
        """
        Optimize a numeric value by reducing precision without sacrificing visual quality.
        
        Args:
            value: The numeric value to optimize
            precision: Decimal places to keep
            
        Returns:
            Optimized string representation of the value
        """
        # Round to specified precision
        rounded = round(value, precision)
        
        # Convert to string with minimal representation
        if rounded == int(rounded):
            # Integer value, no decimal point needed
            return str(int(rounded))
        else:
            # Remove trailing zeros
            s = str(rounded)
            if '.' in s:
                s = s.rstrip('0').rstrip('.') if s.endswith('0') or s.endswith('.') else s
            return s


def create_santorini_illustration(width: float = 1000, height: float = 600, precision: int = 1) -> SVGDocument:
    """
    Create a hyper-realistic SVG illustration of Santorini under 10KB.
    
    This demonstrates the techniques described for creating compact, \
    realistic SVG illustrations through mathematical and graphical optimization.
    
    Args:
        width: Width of the SVG viewport
        height: Height of the SVG viewport
        precision: Decimal precision for coordinate values
        
    Returns:
        SVG document with the Santorini illustration
    """
    # Create SVG document
    doc = SVGDocument(width, height, precision)
    
    # Create sky gradient
    sky_gradient = LinearGradient("sky", 0, 0, 0, height * 0.6, precision)
    sky_gradient.add_stop(0, "#4a90e2")
    sky_gradient.add_stop(0.7, "#c8e6ff")
    sky_gradient.add_stop(1, "#fff")
    doc.add_definition(sky_gradient.to_svg_element())
    
    # Create sea gradient
    sea_gradient = LinearGradient("sea", 0, height * 0.6, 0, height, precision)
    sea_gradient.add_stop(0, "#2a7de1")
    sea_gradient.add_stop(1, "#1c5aaa")
    doc.add_definition(sea_gradient.to_svg_element())
    
    # Create sky
    sky = Rectangle(0, 0, width, height * 0.6, precision)
    doc.add_element(sky.to_svg_element({"fill": "url(#sky)", "stroke": "none"}))
    
    # Create sea
    sea = Rectangle(0, height * 0.6, width, height * 0.4, precision)
    doc.add_element(sea.to_svg_element({"fill": "url(#sea)", "stroke": "none"}))
    
    # Create cliff silhouette (Caldera)
    cliff_path = Path(precision)
    cliff_path.move_to(0, height * 0.6, False)  # Start at water line, left side
    cliff_path.line_to(width * 0.2, height * 0.6)  # Water line to cliff start
    cliff_path.line_to(width * 0.25, height * 0.5)  # Cliff rise
    cliff_path.line_to(width * 0.35, height * 0.45)  # Cliff plateau
    
    # Create buildings on cliff
    cliff_path.line_to(width * 0.38, height * 0.43)  # First building
    cliff_path.line_to(width * 0.38, height * 0.4)
    cliff_path.line_to(width * 0.4, height * 0.4)
    cliff_path.line_to(width * 0.4, height * 0.39)  # Second level
    cliff_path.line_to(width * 0.43, height * 0.39)
    cliff_path.line_to(width * 0.43, height * 0.37)
    
    # Create domed church
    cliff_path.line_to(width * 0.46, height * 0.37)  # Approach to dome
    cliff_path.line_to(width * 0.46, height * 0.35)  # Wall
    cliff_path.arc_to(width * 0.51, height * 0.35, width * 0.025, width * 0.025, 0, False, True, False)  # Dome
    cliff_path.line_to(width * 0.51, height * 0.37)  # Wall
    
    # Continue cliff with more buildings
    cliff_path.line_to(width * 0.55, height * 0.37)  # Next building
    cliff_path.line_to(width * 0.55, height * 0.35)
    cliff_path.line_to(width * 0.58, height * 0.35)
    cliff_path.line_to(width * 0.58, height * 0.39)
    cliff_path.line_to(width * 0.62, height * 0.39)
    cliff_path.line_to(width * 0.62, height * 0.36)
    
    # Another domed building
    cliff_path.line_to(width * 0.65, height * 0.36)  # Approach to dome
    cliff_path.line_to(width * 0.65, height * 0.34)  # Wall
    cliff_path.arc_to(width * 0.7, height * 0.34, width * 0.025, width * 0.025, 0, False, True, False)  # Dome
    cliff_path.line_to(width * 0.7, height * 0.37)  # Wall
    
    # Continue cliff edge
    cliff_path.line_to(width * 0.75, height * 0.42)  # Cliff descent
    cliff_path.line_to(width * 0.8, height * 0.5)  # Cliff descent continues
    cliff_path.line_to(width * 0.85, height * 0.55)  # Approaching water
    cliff_path.line_to(width * 0.9, height * 0.6)  # Back to water line
    cliff_path.line_to(width, height * 0.6)  # Rest of water line to edge
    cliff_path.line_to(width, height)  # Down to bottom-right corner
    cliff_path.line_to(0, height)  # Across to bottom-left corner
    cliff_path.close_path()  # Back to start
    
    # Add cliff to document
    doc.add_element(cliff_path.to_svg_element({"fill": "#584d41", "stroke": "none"}))
    
    # Add white buildings
    buildings_path = Path(precision)
    
    # First building cluster
    buildings_path.move_to(width * 0.39, height * 0.42, False)
    buildings_path.line_to(width * 0.42, height * 0.42)
    buildings_path.line_to(width * 0.42, height * 0.38)
    buildings_path.line_to(width * 0.41, height * 0.38)
    buildings_path.line_to(width * 0.41, height * 0.36)
    buildings_path.line_to(width * 0.39, height * 0.36)
    buildings_path.close_path()
    
    # Church dome
    buildings_path.move_to(width * 0.46, height * 0.35, False)
    buildings_path.arc_to(width * 0.51, height * 0.35, width * 0.025, width * 0.025, 0, True, True, False)
    buildings_path.close_path()
    
    # Second building cluster
    buildings_path.move_to(width * 0.56, height * 0.36, False)
    buildings_path.line_to(width * 0.59, height * 0.36)
    buildings_path.line_to(width * 0.59, height * 0.33)
    buildings_path.line_to(width * 0.56, height * 0.33)
    buildings_path.close_path()
    
    # Second dome
    buildings_path.move_to(width * 0.65, height * 0.34, False)
    buildings_path.arc_to(width * 0.7, height * 0.34, width * 0.025, width * 0.025, 0, True, True, False)
    buildings_path.close_path()
    
    # Add buildings to document
    doc.add_element(buildings_path.to_svg_element({"fill": "#fff", "stroke": "none"}))
    
    # Add blue domes
    domes_path = Path(precision)
    
    # First dome
    domes_path.move_to(width * 0.485, height * 0.35, False)
    domes_path.arc_to(width * 0.485 - width * 0.02, height * 0.35, width * 0.02, width * 0.015, 0, True, False, False)
    domes_path.close_path()
    
    # Second dome
    domes_path.move_to(width * 0.675, height * 0.34, False)
    domes_path.arc_to(width * 0.675 - width * 0.02, height * 0.34, width * 0.02, width * 0.015, 0, True, False, False)
    domes_path.close_path()
    
    # Add domes to document
    doc.add_element(domes_path.to_svg_element({"fill": "#1a4d8c", "stroke": "none"}))
    
    # Add shadows
    shadows_path = Path(precision)
    
    # Cliff shadow
    shadows_path.move_to(width * 0.25, height * 0.55, False)
    shadows_path.line_to(width * 0.35, height * 0.49)
    shadows_path.line_to(width * 0.4, height * 0.47)
    shadows_path.line_to(width * 0.45, height * 0.46)
    shadows_path.line_to(width * 0.5, height * 0.45)
    shadows_path.line_to(width * 0.55, height * 0.46)
    shadows_path.line_to(width * 0.6, height * 0.48)
    shadows_path.line_to(width * 0.65, height * 0.5)
    shadows_path.line_to(width * 0.7, height * 0.53)
    shadows_path.line_to(width * 0.75, height * 0.56)
    shadows_path.line_to(width * 0.25, height * 0.56)
    shadows_path.close_path()
    
    # Add shadows to document
    doc.add_element(shadows_path.to_svg_element({ \
        "fill": "#000",
        "stroke": "none", \
        "fill-opacity": "0.1"
    }))
    
    # Optimize the document
    doc.optimize()
    
    return doc


# The main function has been moved to the end of the file
# Landscape example integrated into the unified SVG optimizer
import math
import random
import sys
from typing import List, Tuple, Dict

def generate_mountain_profile(width: float, height: float, complexity: int = 10, roughness: float = 0.5) -> List[Tuple[float, float]]:
    """
    Generate a realistic mountain silhouette profile using the diamond-square algorithm.
    
    Args:
        width: Width of the mountain range
        height: Maximum height of the mountains
        complexity: Number of segments (higher means more detailed)
        roughness: Roughness factor (0-1)
        
    Returns:
        List of points defining the mountain profile
    """
    # Start with a baseline and the highest point in the middle
    points = [(0, 0), (width/2, -height), (width, 0)]
    
    # Diamond-square algorithm simplified for 1D
    iterations = int(math.log2(complexity)) + 1
    current_roughness = roughness * height
    
    for i in range(iterations):
        new_points = []
        
        # Add points in between each pair of existing points
        for j in range(len(points) - 1):
            p1 = points[j]
            p2 = points[j + 1]
            
            # Add the first point to the new list
            new_points.append(p1)
            
            # Add a new point in the middle with some random variation
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            rand_offset = (random.random() * 2 - 1) * current_roughness
            new_points.append((mid_x, mid_y + rand_offset))
        
        # Add the last point
        new_points.append(points[-1])
        
        # Update points and reduce roughness for the next iteration
        points = new_points
        current_roughness *= 0.5
    
    # Ensure y-coordinates are negative (above the baseline)
    points = [(p[0], min(0, p[1])) for p in points]
    
    return points

def create_landscape(width: float = 1000, height: float = 600, precision: int = 1) -> SVGDocument:
    """
    Create a hyper-realistic mountain landscape SVG.
    
    Args:
        width: Width of the SVG viewport
        height: Height of the SVG viewport
        precision: Decimal precision for coordinate values
        
    Returns:
        SVG document with the landscape
    """
    # Create SVG document
    doc = SVGDocument(width, height, precision)
    
    # Create sky gradient
    sky_gradient = LinearGradient("sky", 0, 0, 0, height * 0.6, precision)
    sky_gradient.add_stop(0, "#1a4d8c")
    sky_gradient.add_stop(0.5, "#4a90e2")
    sky_gradient.add_stop(1, "#c8e6ff")
    doc.add_definition(sky_gradient.to_svg_element())
    
    # Create sky background
    sky = Rectangle(0, 0, width, height, precision)
    doc.add_element(sky.to_svg_element({"fill": "url(#sky)", "stroke": "none"}))
    
    # Generate mountain silhouettes (3 layers for depth)
    layers = [ \
        {
            "distance": 0.8,  # Back mountains
            "height": height * 0.35, \
            "complexity": 16, \
            "roughness": 0.7, \
            "fill": "#46618c", \
            "opacity": 0.8
        }, \
        { \
            "distance": 0.5,  # Middle mountains
            "height": height * 0.45, \
            "complexity": 20, \
            "roughness": 0.5, \
            "fill": "#36516c", \
            "opacity": 0.9
        }, \
        { \
            "distance": 0.2,  # Front mountains
            "height": height * 0.5, \
            "complexity": 25, \
            "roughness": 0.4, \
            "fill": "#263c50", \
            "opacity": 1.0
        }
    ]
    
    for layer in layers:
        # Generate mountain profile
        profile = generate_mountain_profile( \
            width,
            layer["height"], \
            layer["complexity"], \
            layer["roughness"]
        )
        
        # Create mountain path
        mountain_path = Path(precision)
        mountain_path.move_to(0, height, False)  # Start at bottom-left
        
        # Add each point in the profile, adjusted to position on screen
        for x, y in profile:
            # Adjust y position - profile y values are negative (above baseline)
            # and we need to position them relative to horizon line
            y_pos = height * (0.6 + layer["distance"] * 0.4) + y
            mountain_path.line_to(x, y_pos, False)
        
        # Close the path
        mountain_path.line_to(width, height, False)  # Right edge
        mountain_path.line_to(0, height, False)     # Bottom edge
        mountain_path.close_path()
        
        # Add mountain to document
        doc.add_element(mountain_path.to_svg_element({ \
            "fill": layer["fill"],
            "stroke": "none", \
            "opacity": str(layer["opacity"])
        }))
    
    # Create ground/foreground
    ground = Rectangle(0, height * 0.8, width, height * 0.2, precision)
    doc.add_element(ground.to_svg_element({ \
        "fill": "#1a4d36",
        "stroke": "none"
    }))
    
    # Add some trees in the foreground
    for i in range(15):
        x_pos = random.random() * width
        tree_height = random.uniform(height * 0.1, height * 0.16)
        trunk_width = tree_height * 0.1
        
        # Only draw trees in the foreground
        y_pos = height - tree_height - random.uniform(0, height * 0.05)
        
        # Create tree trunk
        trunk = Path(precision)
        trunk.move_to(x_pos - trunk_width/2, y_pos + tree_height, False)
        trunk.line_to(x_pos - trunk_width/2, y_pos + tree_height * 0.5)
        trunk.line_to(x_pos + trunk_width/2, y_pos + tree_height * 0.5)
        trunk.line_to(x_pos + trunk_width/2, y_pos + tree_height)
        trunk.close_path()
        
        # Create tree foliage (triangular for conifers)
        foliage = Path(precision)
        foliage.move_to(x_pos - tree_height * 0.3, y_pos + tree_height * 0.5, False)
        foliage.line_to(x_pos, y_pos)
        foliage.line_to(x_pos + tree_height * 0.3, y_pos + tree_height * 0.5)
        foliage.close_path()
        
        # Add shadow (simplified)
        shadow = Path(precision)
        shadow.move_to(x_pos - tree_height * 0.2, y_pos + tree_height, False)
        shadow.line_to(x_pos, y_pos + tree_height * 0.9)
        shadow.line_to(x_pos + tree_height * 0.2, y_pos + tree_height)
        shadow.close_path()
        
        # Add elements to document
        doc.add_element(shadow.to_svg_element({ \
            "fill": "#000",
            "stroke": "none", \
            "fill-opacity": "0.1"
        }))
        doc.add_element(trunk.to_svg_element({ \
            "fill": "#432",
            "stroke": "none"
        }))
        doc.add_element(foliage.to_svg_element({ \
            "fill": "#1a3d36",
            "stroke": "none"
        }))
    
    return doc

class PerceptualSimplifier:
    """
    Advanced path simplification using perceptual thresholds and
    angular deviation analysis.
    """
    
    def __init__(self, angular_threshold: float = 0.15):
        """
        Initialize the perceptual simplifier.
        
        Args:
            angular_threshold: Threshold in radians (default 0.15) for angular deviation.Below this threshold, humans cannot distinguish between a curved line and
                a series of short straight lines.
        """
        self.angular_threshold = angular_threshold
    
    def angle_between_vectors(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """
        Calculate the angle between two vectors.
        
        Args:
            v1: First vector as (x, y)
            v2: Second vector as (x, y)
            
        Returns:
            Angle in radians
        """
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        
        # Prevent division by zero
        if mag_v1 * mag_v2 == 0:
            return 0
            
        cos_angle = dot_product / (mag_v1 * mag_v2)
        # Handle floating point errors
        cos_angle = max(-1, min(1, cos_angle))
        
        return math.acos(cos_angle)
    
    def vector_between_points(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
        """
        Calculate the vector from p1 to p2.
        
        Args:
            p1: First point as (x, y)
            p2: Second point as (x, y)
            
        Returns:
            Vector as (x, y)
        """
        return (p2[0] - p1[0], p2[1] - p1[1])
    
    def simplify_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Simplify a list of points using perceptual angular thresholds.
        
        This algorithm preserves points where the path changes direction beyond
        the perceptual threshold, eliminating points where the change in direction
        is imperceptible.
        
        Args:
            points: List of points as (x, y) tuples
            
        Returns:
            Simplified list of points
        """
        if len(points) < 3:
            return points.copy()
            
        result = [points[0]]  # Always keep the first point
        
        for i in range(1, len(points) - 1):
            prev_vector = self.vector_between_points(points[i-1], points[i])
            next_vector = self.vector_between_points(points[i], points[i+1])
            
            angle = self.angle_between_vectors(prev_vector, next_vector)
            
            # Keep the point if the angle exceeds the threshold
            if abs(angle) > self.angular_threshold:
                result.append(points[i])
        
        result.append(points[-1])  # Always keep the last point
        return result
    
    def douglas_peucker(self, points: List[Tuple[float, float]], epsilon: float) -> List[Tuple[float, float]]:
        """
        Douglas-Peucker algorithm for path simplification.
        
        This algorithm reduces the number of points in a curve by recursively
        eliminating points that don't deviate from the simplified curve by more
        than epsilon.
        
        Args:
            points: List of points as (x, y) tuples
            epsilon: Maximum distance threshold for point elimination
            
        Returns:
            Simplified list of points
        """
        if len(points) <= 2:
            return points
            
        # Find the point with the maximum distance from line between start and end
        dmax = 0
        index = 0
        start, end = points[0], points[-1]
        
        for i in range(1, len(points) - 1):
            d = self._perpendicular_distance(points[i], start, end)
            if d > dmax:
                index = i
                dmax = d
        
        # If max distance is greater than epsilon, recursively simplify
        if dmax > epsilon:
            results1 = self.douglas_peucker(points[:index+1], epsilon)
            results2 = self.douglas_peucker(points[index:], epsilon)
            
            # Combine the results (avoiding duplicating the common point)
            return results1[:-1] + results2
        else:
            # All points in this segment can be approximated by a straight line
            return [start, end]
    
    def _perpendicular_distance(self, point: Tuple[float, float], \
                                line_start: Tuple[float, float],
                                line_end: Tuple[float, float]) -> float:
        """
        Calculate the perpendicular distance from a point to a line.
        
        Args:
            point: The point as (x, y)
            line_start: Start of the line as (x, y)
            line_end: End of the line as (x, y)
            
        Returns:
            Perpendicular distance
        """
        if line_start == line_end:
            # Point to point distance if line has zero length
            return math.sqrt((point[0] - line_start[0])**2 + (point[1] - line_start[1])**2)
            
        # Line length squared
        line_length_sq = (line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2
        
        # Calculate the normalized dot product
        t = max(0, min(1, ( \
            (point[0] - line_start[0]) * (line_end[0] - line_start[0]) +
            (point[1] - line_start[1]) * (line_end[1] - line_start[1])
        ) / line_length_sq))
        
        # Calculate the closest point on the line
        closest_x = line_start[0] + t * (line_end[0] - line_start[0])
        closest_y = line_start[1] + t * (line_end[1] - line_start[1])
        
        # Return the distance to the closest point
        return math.sqrt((point[0] - closest_x)**2 + (point[1] - closest_y)**2)


class AtmosphericPerspective:
    """
    Atmospheric perspective simulator for creating depth perception
    without using filters or raster effects.
    """
    
    def __init__(self, view_distance: float = 1000, atmosphere_density: float = 0.15):
        """
        Initialize the atmospheric perspective simulator.
        
        Args:
            view_distance: Maximum view distance in scene units
            atmosphere_density: Density factor affecting color and opacity falloff
        """
        self.view_distance = view_distance
        self.atmosphere_density = atmosphere_density
        
    def calculate_opacity(self, distance: float) -> float:
        """
        Calculate opacity based on distance.
        
        Args:
            distance: Distance from viewer in scene units
            
        Returns:
            Opacity value between 0 and 1
        """
        if distance <= 0:
            return 1.0
            
        ratio = min(1.0, distance / self.view_distance)
        return max(0.2, 1.0 - (ratio * self.atmosphere_density))
    
    def desaturate_color(self, color: str, distance: float) -> str:
        """
        Desaturate a color based on distance to simulate atmospheric perspective.
        
        Args:
            color: Color as hex string (#rrggbb or #rgb)
            distance: Distance from viewer in scene units
            
        Returns:
            Desaturated color as hex string
        """
        # Expand short hex format
        if len(color) == 4:  # #rgb format
            color = f"#{color[1]}{color[1]}{color[2]}{color[2]}{color[3]}{color[3]}"
            
        # Convert hex to RGB
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        
        # Calculate desaturation factor
        ratio = min(1.0, distance / self.view_distance)
        desaturation = ratio * self.atmosphere_density
        
        # Desaturate by blending with a light blue/white atmospheric color
        atmosphere_r, atmosphere_g, atmosphere_b = 0.85, 0.9, 1.0
        
        # Blend colors
        r = r * (1 - desaturation) + atmosphere_r * desaturation
        g = g * (1 - desaturation) + atmosphere_g * desaturation
        b = b * (1 - desaturation) + atmosphere_b * desaturation
        
        # Convert back to hex
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
    
    def apply_to_element_attribs(self, attribs: Dict[str, str], distance: float) -> Dict[str, str]:
        """
        Apply atmospheric perspective effects to element attributes.
        
        Args:
            attribs: Element attributes
            distance: Distance from viewer in scene units
            
        Returns:
            Modified attributes
        """
        # Make a copy to avoid modifying the original
        result = attribs.copy()
        
        # Modify fill color if present
        if "fill" in result and result["fill"] != "none" and not result["fill"].startswith("url("): \
            result["fill"] = self.desaturate_color(result["fill"], distance)
        
        # Modify stroke color if present
        if "stroke" in result and result["stroke"] != "none" and not result["stroke"].startswith("url("): \
            result["stroke"] = self.desaturate_color(result["stroke"], distance)
            
        # Apply opacity based on distance
        opacity = self.calculate_opacity(distance)
        if "opacity" in result:
            current_opacity = float(result["opacity"])
            result["opacity"] = str(current_opacity * opacity)
        else:
            result["opacity"] = str(opacity)
            
        return result


class ColorOptimizer:
    """
    Color optimization system for minimizing unique colors while maintaining
    perceptual quality.
    """
    
    def __init__(self, max_colors: int = 7):
        """
        Initialize the color optimizer.
        
        Args:
            max_colors: Maximum number of colors to use
        """
        self.max_colors = max_colors
        self.color_map = {}
        self.palette = []
        
    def hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """
        Convert hex color to RGB.
        
        Args:
            hex_color: Color as hex string (#rrggbb or #rgb)
            
        Returns:
            RGB tuple (0-255)
        """
        # Strip # if present
        hex_color = hex_color.lstrip('#')
        
        # Handle short hex format
        if len(hex_color) == 3:
            return tuple(int(c + c, 16) for c in hex_color)
        
        # Handle normal hex format
        return tuple(int(hex_color[i:i+2], 16) for i in range(0, 6, 2))
    
    def rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """
        Convert RGB to hex color.
        
        Args:
            rgb: RGB tuple (0-255)
            
        Returns:
            Color as hex string
        """
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def optimize_hex(self, hex_color: str) -> str:
        """
        Optimize a hex color to use short format if possible.
        
        Args:
            hex_color: Color as hex string (#rrggbb)
            
        Returns:
            Optimized hex color
        """
        # Strip # if present
        hex_color = hex_color.lstrip('#')
        
        # Check if color can be represented in short form
        if len(hex_color) == 6:
            if hex_color[0] == hex_color[1] and hex_color[2] == hex_color[3] and hex_color[4] == hex_color[5]:
                return f"#{hex_color[0]}{hex_color[2]}{hex_color[4]}"
        
        return f"#{hex_color}"
    
    def color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """
        Calculate perceptual distance between two colors using weighted Euclidean distance.
        
        Args:
            color1: First color as RGB tuple
            color2: Second color as RGB tuple
            
        Returns:
            Perceptual distance
        """
        # Human perception is more sensitive to differences in green, then red, then blue
        weights = (0.3, 0.59, 0.11)
        
        return math.sqrt(sum(weights[i] * (color1[i] - color2[i])**2 for i in range(3)))
    
    def find_closest_color(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """
        Find the closest color in the palette.
        
        Args:
            color: Target color as RGB tuple
            
        Returns:
            Closest palette color as RGB tuple
        """
        if not self.palette:
            return color
            
        return min(self.palette, key=lambda c: self.color_distance(color, c))
    
    def build_palette(self, colors: List[str]) -> None:
        """
        Build an optimized color palette from a list of colors.
        
        Args:
            colors: List of hex colors
        """
        # Convert all colors to RGB
        rgb_colors = [self.hex_to_rgb(c) for c in colors]
        
        if len(rgb_colors) <= self.max_colors:
            self.palette = rgb_colors
            return
        
        # Use k-means clustering to reduce the number of colors
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.max_colors, random_state=0).fit(rgb_colors)
            self.palette = [tuple(map(int, center)) for center in kmeans.cluster_centers_]
        except ImportError:
            # Fallback to a simple method if sklearn is not available
            self.palette = rgb_colors[:self.max_colors]
    
    def calculate_adaptive_illumination(self, path_data: str) -> Dict[str, float]:
        """
        Calculate adaptive illumination parameters based on path geometry and light source.Implements Lambertian shading model for SVG vector elements.
        
        Args:
            path_data: SVG path data string
            
        Returns:
            Dictionary of illumination parameters (fill-opacity, relative brightness)
        """
        # Extract path segments and calculate average normal vector
        path_normal = self._estimate_path_normal(path_data)
        if not path_normal:
            return {'fill-opacity': 1.0, 'brightness': 1.0}
        
        # Calculate dot product between path normal and light vector (Lambertian model)
        light_intensity = max(0.3, self._dot_product(path_normal, self.light_vector))
        
        # Calculate illumination parameters
        fill_opacity = min(1.0, max(0.4, light_intensity))
        brightness = light_intensity
        
        return { \
            'fill-opacity': round(fill_opacity, 2),
            'brightness': round(brightness, 2)
        }
    
    def _estimate_path_normal(self, path_data: str) -> Optional[Tuple[float, float]]:
        """
        Estimate the normal vector of a path by analyzing its segments.
        
        Args:
            path_data: SVG path data string
            
        Returns:
            Normalized 2D vector representing path orientation or None if undetermined
        """
        # Extract points from path data using regex
        point_pattern = r'[ML]\s*([\d.-]+)[,\s]+([\d.-]+)'  # Only look for M and L commands for simplicity
        points = re.findall(point_pattern, path_data)
        
        if len(points) < 2:
            return None
            
        # Convert to float coordinates
        points = [(float(x), float(y)) for x, y in points]
        
        # Calculate the average direction vector
        dx_sum, dy_sum = 0, 0
        for i in range(len(points) - 1):
            dx = points[i+1][0] - points[i][0]
            dy = points[i+1][1] - points[i][1]
            mag = math.sqrt(dx*dx + dy*dy)
            if mag > 0:
                dx_sum += dx / mag
                dy_sum += dy / mag
        
        if dx_sum == 0 and dy_sum == 0:
            return None
            
        # Calculate the normal vector (perpendicular to direction)
        # In 2D, normal to (dx, dy) is (-dy, dx) or (dy, -dx)
        normal_x, normal_y = -dy_sum, dx_sum
        normal_mag = math.sqrt(normal_x*normal_x + normal_y*normal_y)
        
        if normal_mag > 0:
            return (normal_x / normal_mag, normal_y / normal_mag)
        return None
        
    def _dot_product(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """
        Calculate the dot product of two 2D vectors
        
        Args:
            v1: First vector as (x, y)
            v2: Second vector as (x, y)
            
        Returns:
            Dot product value
        """
        return v1[0] * v2[0] + v1[1] * v2[1]
    
    def apply_illumination_to_attributes(self, attributes: Dict[str, str], illumination: Dict[str, float]) -> Dict[str, str]:
        """
        Apply adaptive illumination parameters to element attributes.
        
        Args:
            attributes: Original element attributes
            illumination: Illumination parameters
            
        Returns:
            Modified attributes with illumination applied
        """
        result = attributes.copy() if attributes else {}
        
        # Apply fill-opacity for shading effect if not already set
        if 'fill-opacity' not in result and illumination['fill-opacity'] < 1.0:
            result['fill-opacity'] = str(illumination['fill-opacity'])
        
        # Adjust fill color brightness if fill is specified
        if 'fill' in result and result['fill'].startswith('#') and illumination['brightness'] < 1.0:
            result['fill'] = self._adjust_color_brightness(result['fill'], illumination['brightness'])
            
        return result
    
    def _adjust_color_brightness(self, color: str, brightness_factor: float) -> str:
        """
        Adjust the brightness of a color.
        
        Args:
            color: Color to adjust (hex format)
            brightness_factor: Factor to multiply RGB values by (0-1)
            
        Returns:
            Adjusted color in hex format
        """
        color = color.lstrip('#')
        if len(color) == 3:  # Handle shorthand hex color
            color = ''.join([c*2 for c in color])
            
        try:
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        except ValueError:
            return '#' + color  # Return original if invalid
            
        # Adjust brightness, ensuring values stay in 0-255 range
        r = min(255, max(0, int(r * brightness_factor)))
        g = min(255, max(0, int(g * brightness_factor)))
        b = min(255, max(0, int(b * brightness_factor)))
        
        return f'#{r:02x}{g:02x}{b:02x}'
        
    def optimize_color(self, color: str) -> str:
        """
        Optimize a color by finding the closest match in the palette.
        
        Args:
            color: Color to optimize
            
        Returns:
            Optimized color from the palette
        """
        if not self.color_palette:
            return color
            
        # Convert input color to RGB
        color = color.lstrip('#')
        if len(color) == 3:  # Handle shorthand hex color
            color = ''.join([c*2 for c in color])
        try:
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
        except ValueError:
            return color  # Return original if invalid
            
        # Find closest color in palette using Euclidean distance
        closest_color = None
        min_distance = float('inf')
        
        for palette_color in self.color_palette:
            palette_color = palette_color.lstrip('#')
            pr, pg, pb = int(palette_color[0:2], 16), int(palette_color[2:4], 16), int(palette_color[4:6], 16)
            
            # Calculate distance (could be improved with perceptual color models)
            distance = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
            
            if distance < min_distance:
                min_distance = distance
                closest_color = '#' + palette_color
                
        return closest_color
        
    def detect_spatial_patterns(self, elements: List[Dict[str, str]]) -> Dict[str, List]:
        """
        Detect spatial patterns in a list of SVG elements to enable optimization through
        prediction rather than explicit representation.
        
        Args:
            elements: List of element dictionaries with attributes
            
        Returns:
            Dictionary of detected patterns by type
        """
        patterns = { \
            'linear': [],  # Linear patterns like windows, columns, etc.
            'grid': [],    # Grid patterns like tiles, bricks, etc.
            'radial': []   # Radial patterns like spokes, petals, etc.
        }
        
        # Group elements by type and similar attributes
        element_groups = self._group_similar_elements(elements)
        
        # For each group, detect patterns
        for group_id, group in element_groups.items():
            if len(group) < 3:  # Need at least 3 elements to detect a pattern
                continue
                
            # Extract positions
            positions = self._extract_element_positions(group)
            
            # Detect linear patterns
            linear_patterns = self._detect_linear_arrangements(positions)
            if linear_patterns:
                patterns['linear'].append({ \
                    'group_id': group_id,
                    'elements': group, \
                    'pattern': linear_patterns
                })
            
            # Detect grid patterns
            grid_patterns = self._detect_grid_arrangements(positions)
            if grid_patterns:
                patterns['grid'].append({ \
                    'group_id': group_id,
                    'elements': group, \
                    'pattern': grid_patterns
                })
            
            # Detect radial patterns
            radial_patterns = self._detect_radial_arrangements(positions)
            if radial_patterns:
                patterns['radial'].append({ \
                    'group_id': group_id,
                    'elements': group, \
                    'pattern': radial_patterns
                })
        
        return patterns
    
    def _group_similar_elements(self, elements: List[Dict[str, str]]) -> Dict[str, List]:
        """
        Group elements with similar attributes.
        
        Args:
            elements: List of element dictionaries
            
        Returns:
            Dictionary of element groups by similarity ID
        """
        groups = {}
        
        for i, elem in enumerate(elements):
            # Create a signature for each element based on critical attributes
            signature = self._create_element_signature(elem)
            
            if signature in groups:
                groups[signature].append(elem)
            else:
                groups[signature] = [elem]
        
        return groups
    
    def _create_element_signature(self, element: Dict[str, str]) -> str:
        """
        Create a signature for an element based on its type and key attributes.
        
        Args:
            element: Element dictionary
            
        Returns:
            String signature
        """
        # Extract key attributes that define similarity
        elem_type = element.get('tag', '')
        fill = element.get('fill', 'none')
        stroke = element.get('stroke', 'none')
        width = element.get('width', '')
        height = element.get('height', '')
        
        # For paths, use d attribute length as a rough similarity measure
        d_length = str(len(element.get('d', ''))) if 'd' in element else ''
        
        return f"{elem_type}_{fill}_{stroke}_{width}_{height}_{d_length}"
    
    def _extract_element_positions(self, elements: List[Dict[str, str]]) -> List[Tuple[float, float]]:
        """
        Extract center positions of elements.
        
        Args:
            elements: List of element dictionaries
            
        Returns:
            List of (x, y) center positions
        """
        positions = []
        
        for elem in elements:
            # Extract position based on element type
            if 'cx' in elem and 'cy' in elem:  # Circle, ellipse
                try:
                    positions.append((float(elem['cx']), float(elem['cy'])))
                except (ValueError, TypeError):
                    pass
            elif 'x' in elem and 'y' in elem:  # Rect, image, etc.
                try:
                    x = float(elem['x'])
                    y = float(elem['y'])
                    w = float(elem.get('width', '0'))
                    h = float(elem.get('height', '0'))
                    positions.append((x + w/2, y + h/2))  # Use center point
                except (ValueError, TypeError):
                    pass
            elif 'd' in elem:  # Path - estimate center from bounding box
                bbox = self._estimate_path_bbox(elem['d'])
                if bbox:
                    positions.append(((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2))
        
        return positions
    
    def _estimate_path_bbox(self, path_data: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Estimate the bounding box of a path from its data.
        
        Args:
            path_data: SVG path data string
            
        Returns:
            Tuple of (min_x, min_y, max_x, max_y) or None if can't be determined
        """
        # Extract points from path data using regex
        point_pattern = r'[MLHVCSQTAZmlhvcsqtaz]\s*([\d.-]+)(?:[,\s]+([\d.-]+))?'
        matches = re.findall(point_pattern, path_data)
        
        if not matches:
            return None
            
        points = []
        for match in matches:
            try:
                if len(match) == 2 and match[1]:  # Both x and y
                    points.append((float(match[0]), float(match[1])))
                elif len(match) == 1 or not match[1]:  # Just x (for H command)
                    points.append((float(match[0]), 0))
            except (ValueError, TypeError):
                pass
                
        if not points:
            return None
            
        # Calculate bbox
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        
        return (min_x, min_y, max_x, max_y)
    
    def _detect_linear_arrangements(self, positions: List[Tuple[float, float]]) -> Optional[Dict]:
        """
        Detect if positions form a linear pattern (e.g., row, column).
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            Dictionary with pattern info or None if not detected
        """
        if len(positions) < 3:
            return None
            
        # Sort by x coordinate
        sorted_by_x = sorted(positions, key=lambda p: p[0])
        x_diffs = [sorted_by_x[i+1][0] - sorted_by_x[i][0] for i in range(len(sorted_by_x)-1)]
        x_consistency = self._check_consistent_spacing(x_diffs)
        
        # Sort by y coordinate
        sorted_by_y = sorted(positions, key=lambda p: p[1])
        y_diffs = [sorted_by_y[i+1][1] - sorted_by_y[i][1] for i in range(len(sorted_by_y)-1)]
        y_consistency = self._check_consistent_spacing(y_diffs)
        
        # Check for horizontal or vertical alignment
        if x_consistency > self.pattern_thresholds['linear'] and all(abs(p[1] - positions[0][1]) < 5 for p in positions):
            # Horizontal row with consistent spacing
            return { \
                'type': 'horizontal',
                'start': sorted_by_x[0], \
                'spacing': sum(x_diffs) / len(x_diffs), \
                'count': len(positions)
            }
        elif y_consistency > self.pattern_thresholds['linear'] and all(abs(p[0] - positions[0][0]) < 5 for p in positions):
            # Vertical column with consistent spacing
            return { \
                'type': 'vertical',
                'start': sorted_by_y[0], \
                'spacing': sum(y_diffs) / len(y_diffs), \
                'count': len(positions)
            }
        
        # Check for diagonal alignment
        # Compute regression line
        if len(positions) >= 3:
            # Simple linear regression
            x_vals = [p[0] for p in positions]
            y_vals = [p[1] for p in positions]
            
            x_mean = sum(x_vals) / len(x_vals)
            y_mean = sum(y_vals) / len(y_vals)
            
            # Calculate slope and intercept
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
            denominator = sum((x - x_mean) ** 2 for x in x_vals)
            
            if abs(denominator) > 0.0001:  # Avoid division by zero
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
                
                # Calculate distances to regression line
                distances = [abs(y - (slope * x + intercept)) / math.sqrt(1 + slope**2) for x, y in zip(x_vals, y_vals)]
                avg_distance = sum(distances) / len(distances)
                
                if avg_distance < 5:  # Threshold for linearity
                    # Find spacing along the line
                    # Project points onto the line and measure distances
                    projected = []
                    for x, y in zip(x_vals, y_vals):
                        # Project point onto line
                        k = ((x - x_mean) + slope * (y - y_mean)) / (1 + slope**2)
                        proj_x = x_mean + k
                        proj_y = y_mean + slope * k
                        projected.append((proj_x, proj_y))
                    
                    # Sort projected points by their distance along the line
                    projected.sort(key=lambda p: p[0] if abs(slope) < 1 else p[1])
                    
                    # Measure distances between consecutive points
                    proj_diffs = [math.sqrt((projected[i+1][0] - projected[i][0])**2 + \
                                            (projected[i+1][1] - projected[i][1])**2)
                                for i in range(len(projected)-1)]
                    
                    proj_consistency = self._check_consistent_spacing(proj_diffs)
                    
                    if proj_consistency > self.pattern_thresholds['linear']:
                        return { \
                            'type': 'diagonal',
                            'slope': slope, \
                            'intercept': intercept, \
                            'start': projected[0], \
                            'spacing': sum(proj_diffs) / len(proj_diffs), \
                            'count': len(positions)
                        }
        
        return None
    
    def _check_consistent_spacing(self, diffs: List[float]) -> float:
        """
        Check how consistent the spacing is between elements.
        
        Args:
            diffs: List of differences between consecutive positions
            
        Returns:
            Consistency score (0-1), where 1 is perfectly consistent
        """
        if not diffs:
            return 0
            
        mean_diff = sum(diffs) / len(diffs)
        if mean_diff == 0:
            return 1.0  # Perfect consistency (all values are the same)
            
        # Calculate coefficient of variation (lower is more consistent)
        variance = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_diff if mean_diff != 0 else 0
        
        # Convert to consistency score (1 - normalized CV)
        consistency = max(0, min(1, 1 - cv))
        
        return consistency
    
    def _detect_grid_arrangements(self, positions: List[Tuple[float, float]]) -> Optional[Dict]:
        """
        Detect if positions form a grid pattern.
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            Dictionary with pattern info or None if not detected
        """
        if len(positions) < 4:  # Need at least 4 points for a grid (2x2)
            return None
            
        # Get unique x and y coordinates (with tolerance)
        x_coords = self._cluster_coordinates([p[0] for p in positions])
        y_coords = self._cluster_coordinates([p[1] for p in positions])
        
        # Check if we have enough rows and columns
        if len(x_coords) < 2 or len(y_coords) < 2:
            return None
            
        # Check if points exist at grid intersections
        grid_points = [(x, y) for x in x_coords for y in y_coords]
        matched_count = 0
        
        for gx, gy in grid_points:
            # Check if any position is close to this grid point
            if any(abs(p[0] - gx) < 5 and abs(p[1] - gy) < 5 for p in positions):
                matched_count += 1
                
        grid_coverage = matched_count / len(grid_points)
        
        if grid_coverage > self.pattern_thresholds['grid']:
            # Calculate x and y spacing
            x_diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
            
            x_spacing = sum(x_diffs) / len(x_diffs) if x_diffs else 0
            y_spacing = sum(y_diffs) / len(y_diffs) if y_diffs else 0
            
            return { \
                'type': 'grid',
                'origin': (min(x_coords), min(y_coords)), \
                'x_spacing': x_spacing, \
                'y_spacing': y_spacing, \
                'cols': len(x_coords), \
                'rows': len(y_coords), \
                'coverage': grid_coverage
            }
            
        return None
    
    def _cluster_coordinates(self, coords: List[float], tolerance: float = 5.0) -> List[float]:
        """
        Cluster coordinates that are close to each other.
        
        Args:
            coords: List of coordinate values
            tolerance: Maximum distance to consider coordinates as same cluster
            
        Returns:
            List of representative coordinates for each cluster
        """
        if not coords:
            return []
            
        # Sort coordinates
        sorted_coords = sorted(coords)
        
        # Group into clusters
        clusters = []
        current_cluster = [sorted_coords[0]]
        
        for i in range(1, len(sorted_coords)):
            if sorted_coords[i] - sorted_coords[i-1] <= tolerance:
                current_cluster.append(sorted_coords[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_coords[i]]
                
        if current_cluster:
            clusters.append(current_cluster)
            
        # Return average value for each cluster
        return [sum(cluster) / len(cluster) for cluster in clusters]
    
    def _detect_radial_arrangements(self, positions: List[Tuple[float, float]]) -> Optional[Dict]:
        """
        Detect if positions form a radial pattern around a center.
        
        Args:
            positions: List of (x, y) positions
            
        Returns:
            Dictionary with pattern info or None if not detected
        """
        if len(positions) < 3:  # Need at least 3 points for a radial pattern
            return None
            
        # Try different center points (average of all points is a good starting guess)
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        center = (center_x, center_y)
        
        # Calculate distances from center
        distances = [math.sqrt((p[0] - center[0])**2 + (p[1] - center[1])**2) for p in positions]
        avg_distance = sum(distances) / len(distances)
        
        # Compute angles from center
        angles = [math.atan2(p[1] - center[1], p[0] - center[0]) for p in positions]
        
        # Check for consistent angular spacing
        sorted_angles = sorted(angles)
        angle_diffs = []
        for i in range(len(sorted_angles)):
            next_angle = sorted_angles[(i+1) % len(sorted_angles)]
            diff = next_angle - sorted_angles[i] if next_angle > sorted_angles[i] else next_angle + 2*math.pi - sorted_angles[i]
            angle_diffs.append(diff)
            
        angle_consistency = self._check_consistent_spacing(angle_diffs)
        
        # Check for consistent distances
        distance_consistency = self._check_consistent_spacing(distances)
        
        if angle_consistency > self.pattern_thresholds['radial']:
            # Angular pattern detected
            avg_angle_diff = sum(angle_diffs) / len(angle_diffs)
            return { \
                'type': 'radial',
                'center': center, \
                'radius': avg_distance, \
                'angle_spacing': avg_angle_diff, \
                'count': len(positions), \
                'distance_consistency': distance_consistency, \
                'angle_consistency': angle_consistency
            }
            
        return None
        
    def optimize_visual_hierarchy(self, elements: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Optimize element order based on visual hierarchy principles to create
        a balanced binary tree for render efficiency.
        
        Args:
            elements: List of SVG elements
            
        Returns:
            Reordered list of elements based on visual impact
        """
        # Categorize elements by visual importance
        categorized = { \
            'primary': [],    # Background, large shapes
            'secondary': [],  # Main structures
            'tertiary': [],   # Details
            'quaternary': []  # Fine details, highlights
        }
        
        for elem in elements:
            # Determine category based on size, position, and attributes
            category = self._categorize_element_importance(elem)
            categorized[category].append(elem)
            
        # Reorder elements based on hierarchy
        ordered_elements = []
        for category in ['primary', 'secondary', 'tertiary', 'quaternary']:
            # Sort elements within each category by size (descending)
            sorted_elements = sorted(categorized[category], \
                                    key=lambda e: self._estimate_element_size(e),
                                    reverse=True)
            ordered_elements.extend(sorted_elements)
            
        return ordered_elements
    
    def _categorize_element_importance(self, element: Dict[str, any]) -> str:
        """
        Determine the visual importance category of an element.
        
        Args:
            element: SVG element
            
        Returns:
            Category name (primary, secondary, tertiary, quaternary)
        """
        # Extract element type and attributes
        elem_type = element.get('tag', '').lower()
        elem_id = element.get('id', '')
        elem_class = element.get('class', '')
        
        # Check for explicit hierarchy markers in id/class
        for category in ['primary', 'secondary', 'tertiary', 'quaternary']:
            if category in elem_id.lower() or category in elem_class.lower():
                return category
        
        # Background elements are primary
        if 'background' in elem_id.lower() or 'bg' in elem_id.lower():
            return 'primary'
        
        # Check size of element (larger elements have higher visual importance)
        size = self._estimate_element_size(element)
        
        # Determine based on size and element type
        if size > 10000 or elem_type in ['rect', 'polygon'] and size > 5000:
            return 'primary'
        elif size > 2000 or elem_type in ['path', 'polygon', 'g'] and size > 1000:
            return 'secondary'
        elif size > 500 or elem_type in ['path', 'circle', 'line']:
            return 'tertiary'
        else:
            return 'quaternary'
    
    def _estimate_element_size(self, element: Dict[str, any]) -> float:
        """
        Estimate the visual importance size of an element.
        
        Args:
            element: SVG element
            
        Returns:
            Size estimate (higher = more visually important)
        """
        # Different element types have different size calculations
        elem_type = element.get('tag', '').lower()
        
        if elem_type == 'rect':
            try:
                width = float(element.get('width', 0))
                height = float(element.get('height', 0))
                return width * height
            except (ValueError, TypeError):
                return 0
        elif elem_type == 'circle':
            try:
                r = float(element.get('r', 0))
                return math.pi * r * r
            except (ValueError, TypeError):
                return 0
        elif elem_type == 'ellipse':
            try:
                rx = float(element.get('rx', 0))
                ry = float(element.get('ry', 0))
                return math.pi * rx * ry
            except (ValueError, TypeError):
                return 0
        elif elem_type == 'path':
            # Estimate path size from bounding box
            bbox = self._estimate_path_bbox(element.get('d', ''))
            if bbox:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                return width * height
            return 500  # Default size for paths
        elif elem_type == 'g':
            # Groups are as important as their combined children
            children = element.get('children', [])
            return sum(self._estimate_element_size(child) for child in children) \
                    if children else 1000
        else:
            # Default size for other elements
            return 100
    
    def optimize_dom_structure(self, elements: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Apply semantic DOM reduction by eliminating redundant elements and
        grouping elements with shared attributes.
        
        Args:
            elements: List of SVG elements
            
        Returns:
            Optimized list of elements
        """
        # 1. Remove redundant elements (those implicitly defined by others)
        non_redundant = self._filter_redundant_elements(elements)
        
        # 2. Group elements with shared attributes to reduce markup size
        grouped = self._group_elements_with_shared_attributes(non_redundant)
        
        return grouped
    
    def _filter_redundant_elements(self, elements: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Remove elements that are redundant or can be inferred from other elements.
        
        Args:
            elements: List of SVG elements
            
        Returns:
            Filtered list without redundant elements
        """
        # Identify elements covering the same area
        essential_elements = []
        covered_areas = []
        
        # Sort elements by z-index (visual stacking)
        sorted_elements = sorted(elements, key=lambda e: int(e.get('z-index', 0)))
        
        for elem in sorted_elements:
            elem_bbox = self._get_element_bbox(elem)
            if not elem_bbox:
                essential_elements.append(elem)
                continue
                
            # Check if this element is completely covered by another
            if any(self._is_bbox_covered(elem_bbox, covered) for covered in covered_areas):
                # Element is redundant if it's fully covered and has no transparency
                if not self._has_transparency(elem):
                    continue  # Skip this element
            
            essential_elements.append(elem)
            covered_areas.append(elem_bbox)
            
        return essential_elements
    
    def _is_bbox_covered(self, bbox1: Tuple[float, float, float, float], \
                            bbox2: Tuple[float, float, float, float]) -> bool:
        """
        Check if bbox1 is completely covered by bbox2.
        
        Args:
            bbox1: First bounding box (min_x, min_y, max_x, max_y)
            bbox2: Second bounding box (min_x, min_y, max_x, max_y)
            
        Returns:
            True if bbox1 is completely inside bbox2
        """
        return (bbox1[0] >= bbox2[0] and bbox1[1] >= bbox2[1] and \
                bbox1[2] <= bbox2[2] and bbox1[3] <= bbox2[3])
    
    def _has_transparency(self, element: Dict[str, any]) -> bool:
        """
        Check if an element has transparency.
        
        Args:
            element: SVG element
            
        Returns:
            True if element has transparency
        """
        # Check for opacity attributes
        opacity = float(element.get('opacity', 1.0))
        fill_opacity = float(element.get('fill-opacity', 1.0))
        stroke_opacity = float(element.get('stroke-opacity', 1.0))
        
        # Check for rgba/hsla colors with alpha channel
        fill = element.get('fill', '')
        stroke = element.get('stroke', '')
        
        has_transparent_fill = False
        has_transparent_stroke = False
        
        if 'rgba' in fill or 'hsla' in fill:
            has_transparent_fill = True
        if 'rgba' in stroke or 'hsla' in stroke:
            has_transparent_stroke = True
            
        return (opacity < 1.0 or fill_opacity < 1.0 or stroke_opacity < 1.0 or \
                has_transparent_fill or has_transparent_stroke)
    
    def _get_element_bbox(self, element: Dict[str, any]) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the bounding box of an element.
        
        Args:
            element: SVG element
            
        Returns:
            Bounding box as (min_x, min_y, max_x, max_y) or None
        """
        elem_type = element.get('tag', '').lower()
        
        if elem_type == 'rect':
            try:
                x = float(element.get('x', 0))
                y = float(element.get('y', 0))
                width = float(element.get('width', 0))
                height = float(element.get('height', 0))
                return (x, y, x + width, y + height)
            except (ValueError, TypeError):
                return None
        elif elem_type == 'circle':
            try:
                cx = float(element.get('cx', 0))
                cy = float(element.get('cy', 0))
                r = float(element.get('r', 0))
                return (cx - r, cy - r, cx + r, cy + r)
            except (ValueError, TypeError):
                return None
        elif elem_type == 'path':
            return self._estimate_path_bbox(element.get('d', ''))
        # Add more element types as needed
        
        return None
    
    def _group_elements_with_shared_attributes(self, elements: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Group elements that share common attributes to reduce redundancy.
        
        Args:
            elements: List of SVG elements
            
        Returns:
            Optimized list with grouped elements
        """
        # Find elements with same fill/stroke/style attributes
        grouped_elements = []
        attribute_groups = {}
        
        for elem in elements:
            # Create a key based on shared attributes
            shared_attrs = self._extract_shared_attributes(elem)
            key = json.dumps(shared_attrs, sort_keys=True)
            
            if key in attribute_groups:
                attribute_groups[key].append(elem)
            else:
                attribute_groups[key] = [elem]
        
        # Create groups for elements with shared attributes (if more than 2 elements share)
        for attrs_key, group_elems in attribute_groups.items():
            if len(group_elems) >= 2:
                # Create a group element with shared attributes
                shared_attrs = json.loads(attrs_key)
                group = { \
                    'tag': 'g',
                    'children': []
                }
                
                # Add shared attributes to group
                for attr_name, attr_value in shared_attrs.items():
                    group[attr_name] = attr_value
                
                # Add elements to group, removing redundant attributes
                for elem in group_elems:
                    # Remove attributes now defined on the group
                    for attr_name in shared_attrs.keys():
                        if attr_name in elem:
                            del elem[attr_name]
                    group['children'].append(elem)
                
                grouped_elements.append(group)
            else:
                # No grouping needed, add as is
                grouped_elements.extend(group_elems)
                
        return grouped_elements
    
    def _extract_shared_attributes(self, element: Dict[str, any]) -> Dict[str, str]:
        """
        Extract attributes that can be shared among elements.
        
        Args:
            element: SVG element
            
        Returns:
            Dictionary of shareable attributes
        """
        shareable = {}
        
        # List of attributes that can be shared in a group
        sharable_attributes = [ \
            'fill', 'stroke', 'stroke-width', 'opacity', 'fill-opacity',
            'stroke-opacity', 'stroke-linecap', 'stroke-linejoin', \
            'fill-rule', 'transform', 'filter', 'clip-path'
        ]
        
        for attr in sharable_attributes:
            if attr in element:
                shareable[attr] = element[attr]
                
        return shareable
    
    def optimize_viewbox(self, elements: List[Dict[str, any]]) -> Tuple[float, float, float, float]:
        """
        Calculate the optimal viewBox dimensions to encapsulate content precisely.
        
        Args:
            elements: List of SVG elements
            
        Returns:
            Tuple of (min_x, min_y, width, height) for viewBox
        """
        if not elements or not self.auto_crop_viewbox:
            return (0, 0, 100, 100)  # Default viewBox if no elements
            
        # Calculate bounding box of all elements
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for elem in elements:
            bbox = self._get_element_bbox(elem)
            if bbox:
                min_x = min(min_x, bbox[0])
                min_y = min(min_y, bbox[1])
                max_x = max(max_x, bbox[2])
                max_y = max(max_y, bbox[3])
        
        # Add padding
        padding = self.viewbox_padding
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        
        # Ensure minimum dimensions
        width = max(max_x - min_x, 1)  # Avoid zero width
        height = max(max_y - min_y, 1)  # Avoid zero height
        
        # Round to integer values for better compression
        min_x = math.floor(min_x)
        min_y = math.floor(min_y)
        max_x = math.ceil(max_x)
        max_y = math.ceil(max_y)
        width = max_x - min_x
        height = max_y - min_y
        
        return (min_x, min_y, width, height)


class PathOptimizer:
    """
    Advanced path optimization for maximum compression with minimal visual impact.
{{ ... }}
    """
    
    def __init__(self, precision: int = 1, angular_threshold: float = 0.15):
        """
        Initialize the path optimizer.
        
        Args:
            precision: Decimal precision for coordinate values
            angular_threshold: Angular threshold in radians for simplified lines
        """
        self.precision = precision
        self.simplifier = PerceptualSimplifier(angular_threshold)
    
    def _extract_points_from_path(self, path_data: str) -> List[List[Tuple[float, float]]]:
        """
        Extract points from SVG path data.
        
        Args:
            path_data: SVG path data string
            
        Returns:
            List of point lists, one for each subpath
        """
        # This is a simplified implementation - a full parser would be more complex
        subpaths = []
        current_subpath = []
        commands = re.findall(r'([MLHVCSQTAZmlhvcsqtaz])([^MLHVCSQTAZmlhvcsqtaz]*)', path_data)
        
        current_x, current_y = 0, 0
        subpath_start_x, subpath_start_y = 0, 0
        
        for cmd, params in commands:
            # Split parameters into floats, handling both space and comma separators
            params = re.findall(r'-?\d*\.?\d+', params)
            params = [float(p) for p in params]
            
            if cmd == 'M':
                if current_subpath:
                    subpaths.append(current_subpath)
                current_subpath = []
                for i in range(0, len(params), 2):
                    x, y = params[i], params[i+1]
                    current_subpath.append((x, y))
                    current_x, current_y = x, y
                subpath_start_x, subpath_start_y = current_x, current_y
            
            elif cmd == 'm':
                if current_subpath:
                    subpaths.append(current_subpath)
                current_subpath = []
                for i in range(0, len(params), 2):
                    x, y = current_x + params[i], current_y + params[i+1]
                    current_subpath.append((x, y))
                    current_x, current_y = x, y
                subpath_start_x, subpath_start_y = current_x, current_y
            
            elif cmd == 'L':
                for i in range(0, len(params), 2):
                    x, y = params[i], params[i+1]
                    current_subpath.append((x, y))
                    current_x, current_y = x, y
            
            elif cmd == 'l':
                for i in range(0, len(params), 2):
                    x, y = current_x + params[i], current_y + params[i+1]
                    current_subpath.append((x, y))
                    current_x, current_y = x, y
                    
            elif cmd == 'H':
                for x in params:
                    current_subpath.append((x, current_y))
                    current_x = x
            
            elif cmd == 'h':
                for dx in params:
                    current_x += dx
                    current_subpath.append((current_x, current_y))
            
            elif cmd == 'V':
                for y in params:
                    current_subpath.append((current_x, y))
                    current_y = y
            
            elif cmd == 'v':
                for dy in params:
                    current_y += dy
                    current_subpath.append((current_x, current_y))
                    
            elif cmd.lower() == 'z':
                # Close path by returning to start point
                if current_subpath and (current_x, current_y) != (subpath_start_x, subpath_start_y):
                    current_subpath.append((subpath_start_x, subpath_start_y))
                current_x, current_y = subpath_start_x, subpath_start_y
                subpaths.append(current_subpath)
                current_subpath = []
        
        # Add any remaining subpath
        if current_subpath:
            subpaths.append(current_subpath)
            
        return subpaths
    
    def _create_path_data_from_points(self, subpaths: List[List[Tuple[float, float]]], \
                                        use_relative: bool = True,
                                        close_paths: bool = True) -> str:
        """
        Create SVG path data from point lists.
        
        Args:
            subpaths: List of point lists, one for each subpath
            use_relative: Whether to use relative coordinates
            close_paths: Whether to close the paths
            
        Returns:
            SVG path data string
        """
        if not subpaths:
            return ""
            
        path_data = []
        
        for subpath in subpaths:
            if not subpath:
                continue
                
            # Start with an absolute move
            x, y = subpath[0]
            path_data.append(f"M{self._format_number(x)},{self._format_number(y)}")
            
            if use_relative:
                # Use relative lines for the rest
                prev_x, prev_y = x, y
                for x, y in subpath[1:]:
                    dx, dy = x - prev_x, y - prev_y
                    path_data.append(f"l{self._format_number(dx)},{self._format_number(dy)}")
                    prev_x, prev_y = x, y
            else:
                # Use absolute lines for the rest
                for x, y in subpath[1:]:
                    path_data.append(f"L{self._format_number(x)},{self._format_number(y)}")
            
            # Close the path if requested
            if close_paths:
                path_data.append("z")
        
        return " ".join(path_data)
    
    def _format_number(self, value: float) -> str:
        """
        Format a number with the specified precision.
        
        Args:
            value: The number to format
            
        Returns:
            Formatted number as string
        """
        if self.precision == 0:
            return str(int(round(value)))
        else:
            # Round to precision and remove trailing zeros and decimal point if not needed
            return f"{value:.{self.precision}f}".rstrip('0').rstrip('.')
    
    def optimize_path_data(self, path_data: str, simplify: bool = True, \
                            use_relative: bool = True, epsilon: float = 0.5) -> str:
        """
        Optimize SVG path data for minimal size with preserved visual quality.
        
        Args:
            path_data: SVG path data string
            simplify: Whether to simplify the path points
            use_relative: Whether to use relative coordinates
            epsilon: Distance threshold for Douglas-Peucker simplification
            
        Returns:
            Optimized SVG path data string
        """
        # Extract points from path
        subpaths = self._extract_points_from_path(path_data)
        
        # Simplify points if requested
        if simplify:
            simplified_subpaths = []
            for subpath in subpaths:
                # Apply angular simplification
                simplified = self.simplifier.simplify_points(subpath)
                # Apply Douglas-Peucker for further size reduction
                simplified = self.simplifier.douglas_peucker(simplified, epsilon)
                simplified_subpaths.append(simplified)
            subpaths = simplified_subpaths
        
        # Create optimized path data
        return self._create_path_data_from_points(subpaths, use_relative)
    
    def optimize_path(self, path: Path) -> Path:
        """
        Optimize a Path object for minimal size with preserved visual quality.
        
        Args:
            path: Path object to optimize
            
        Returns:
            Optimized Path object
        """
        # Convert path to SVG path data
        path_data = path.to_svg_path_data()
        
        # Optimize path data
        optimized_data = self.optimize_path_data(path_data)
        
        # Create a new path with the optimized data
        optimized_path = Path(self.precision)
        # TODO: Parse the optimized data and rebuild the path commands
        
        return optimized_path


class SVGMarkupOptimizer:
    """
    SVG markup optimizer for extreme compression through intelligent
    restructuring and syntax optimization.
    """
    
    def __init__(self, precision: int = 1):
        """
        Initialize the SVG markup optimizer.
        
        Args:
            precision: Decimal precision for coordinate values
        """
        self.precision = precision
        self.color_optimizer = ColorOptimizer()
        self.path_optimizer = PathOptimizer(precision)
    
    def optimize_viewbox(self, width: float, height: float, content_bbox: Tuple[float, float, float, float]) -> str:
        """
        Optimize the viewBox attribute for maximum coordinate efficiency.
        
        Args:
            width: SVG width
            height: SVG height
            content_bbox: Content bounding box as (min_x, min_y, max_x, max_y)
            
        Returns:
            Optimized viewBox attribute value
        """
        # Calculate the optimal viewBox that:
        # 1. Maintains aspect ratio
        # 2. Uses integer coordinates if possible
        # 3. Ensures smallest detail is visible
        min_x, min_y, max_x, max_y = content_bbox
        content_width = max_x - min_x
        content_height = max_y - min_y
        
        # Calculate optimal scale to normalize coordinates
        scale = min( \
            1000 / content_width if content_width > 0 else 1000,
            1000 / content_height if content_height > 0 else 1000
        )
        
        # Round values based on precision
        min_x = self._format_number(min_x)
        min_y = self._format_number(min_y)
        content_width = self._format_number(content_width * scale)
        content_height = self._format_number(content_height * scale)
        
        return f"{min_x} {min_y} {content_width} {content_height}"
    
    def _format_number(self, value: float) -> str:
        """
        Format a number with the specified precision.
        
        Args:
            value: The number to format
            
        Returns:
            Formatted number as string
        """
        if self.precision == 0:
            return str(int(round(value)))
        else:
            # Round to precision and remove trailing zeros and decimal point if not needed
            return f"{value:.{self.precision}f}".rstrip('0').rstrip('.')
    
    def optimize_attributes(self, attribs: Dict[str, str]) -> Dict[str, str]:
        """
        Optimize element attributes for minimal size.
        
        Args:
            attribs: Element attributes
            
        Returns:
            Optimized attributes
        """
        result = {}
        
        # Process colors
        for attr in ['fill', 'stroke']:
            if attr in attribs and attribs[attr] != 'none':
                result[attr] = self.color_optimizer.optimize_color(attribs[attr])
        
        # Optimize numeric attributes
        for attr in ['stroke-width', 'stroke-miterlimit', 'opacity', 'fill-opacity', 'stroke-opacity']:
            if attr in attribs:
                try:
                    value = float(attribs[attr])
                    result[attr] = self._format_number(value)
                except ValueError:
                    result[attr] = attribs[attr]
        
        # Copy other attributes that don't need optimization
        for attr, value in attribs.items():
            if attr not in result:
                result[attr] = value
        
        return result
    
    def optimize_svg_string(self, svg_string: str) -> str:
        """
        Optimize an SVG string for minimal size.
        
        Args:
            svg_string: SVG XML string
            
        Returns:
            Optimized SVG string
        """
        # Remove XML comments
        svg_string = re.sub(r'<!--[\s\S]*?-->', '', svg_string)
        
        # Remove whitespace between tags
        svg_string = re.sub(r'>\s+<', '><', svg_string)
        
        # Optimize path data
        def optimize_path(match):
            d = match.group(1)
            optimized_d = self.path_optimizer.optimize_path_data(d)
            return f'd="{optimized_d}"'
            
        svg_string = re.sub(r'd="([^"]+)"', optimize_path, svg_string)
        
        # Optimize colors
        def optimize_color(match):
            attr = match.group(1)
            color = match.group(2)
            optimized_color = self.color_optimizer.optimize_color(color)
            return f'{attr}="{optimized_color}"'
            
        svg_string = re.sub(r'(fill|stroke)="([^"]+)"', optimize_color, svg_string)
        
        # Combine duplicate attribute values in groups
        # TODO: Implement group optimization for shared attributes
        
        return svg_string


class HyperRealisticSVG:
    """
    Complete system for creating hyper-realistic SVG illustrations under 10KB.
    """
    
    def __init__(self, width: float = 1000, height: float = 600, precision: int = 1):
        """
        Initialize the hyper-realistic SVG system.
        
        Args:
            width: SVG width
            height: SVG height
            precision: Decimal precision for coordinate values
        """
        self.width = width
        self.height = height
        self.precision = precision
        self.document = SVGDocument(width, height, precision)
        self.color_optimizer = ColorOptimizer()
        self.path_optimizer = PathOptimizer(precision)
        self.markup_optimizer = SVGMarkupOptimizer(precision)
        self.atmosphere = AtmosphericPerspective()
    
    def create_illustration(self, scene_func: Callable) -> SVGDocument:
        """
        Create an illustration using the provided scene function.
        
        Args:
            scene_func: Function that builds the scene
            
        Returns:
            Completed SVG document
        """
        # Build the scene
        scene_func(self)
        
        # Collect all colors used in the document
        colors = self._collect_colors()
        
        # Build an optimized palette
        self.color_optimizer.build_palette(colors)
        
        # Optimize the document
        self._optimize_document()
        
        return self.document
    
    def _collect_colors(self) -> List[str]:
        """
        Collect all colors used in the document.
        
        Returns:
            List of hex colors
        """
        # This is a placeholder - a real implementation would traverse the document
        # to extract all colors from fill and stroke attributes
        return ['#ffffff', '#000000', '#4a90e2', '#c8e6ff', '#2a7de1', '#1c5aaa', '#584d41', '#1a4d8c']
    
    def _optimize_document(self) -> None:
        """
        Apply all optimizations to the document.
        """
        # This is a placeholder - a real implementation would traverse the document
        # and apply various optimizations to each element
        pass
    
    def to_string(self) -> str:
        """
        Convert the illustration to an optimized SVG string.
        
        Returns:
            Optimized SVG string
        """
        # Get the SVG string
        svg_string = self.document.to_string()
        
        # Apply markup optimization
        svg_string = self.markup_optimizer.optimize_svg_string(svg_string)
        
        return svg_string
    
    def save(self, filename: str) -> int:
        """
        Save the illustration to an SVG file.
        
        Args:
            filename: Path to save the file
            
        Returns:
            Size of the saved file in bytes
        """
        # Get the optimized SVG string
        svg_string = self.to_string()
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_string)
            
        return len(svg_string.encode('utf-8'))


def create_parametric_curve(t_start: float, t_end: float, segments: int, x_fn: Callable[[float], float], y_fn: Callable[[float], float], curvature_threshold: float = 0.01, precision: int = 1) -> Path:
    """
    Generate a parametric curve with adaptive point density based on local curvature.
    
    Args:
        t_start: Starting parameter value
        t_end: Ending parameter value
        segments: Initial number of segments
        x_fn: Function that returns x coordinate for parameter t
        y_fn: Function that returns y coordinate for parameter t
        curvature_threshold: Threshold for adapting point density based on curvature
        precision: Decimal precision for coordinate values
    
    Returns:
        Path object with the parametric curve
    """
    path = Path(precision)
    
    def calculate_curvature(t):
        # Aproximação numérica da curvatura
        h = 0.0001  # Pequeno delta para derivada numérica
        
        # Primeiras derivadas
        dx_dt = (x_fn(t + h) - x_fn(t - h)) / (2 * h)
        dy_dt = (y_fn(t + h) - y_fn(t - h)) / (2 * h)
        
        # Segundas derivadas
        d2x_dt2 = (x_fn(t + h) - 2 * x_fn(t) + x_fn(t - h)) / (h * h)
        d2y_dt2 = (y_fn(t + h) - 2 * y_fn(t) + y_fn(t - h)) / (h * h)
        
        # Fórmula da curvatura
        numerator = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2)
        denominator = pow(dx_dt * dx_dt + dy_dt * dy_dt, 1.5)
        
        if denominator < 1e-10:  # Avoid division by near-zero
            return 0
            
        return numerator / denominator
    
    # Recursive function to add points with adaptive density
    def add_curve_segment(t1, t2, depth=0):
        # Calculate midpoint
        tmid = (t1 + t2) / 2
        
        # Calculate points
        p1 = (x_fn(t1), y_fn(t1))
        p2 = (x_fn(t2), y_fn(t2))
        pmid = (x_fn(tmid), y_fn(tmid))
        
        # Calculate distance from midpoint to line segment
        line_len = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        if line_len < 1e-10:  # Points too close, don't subdivide
            return [p1]
            
        # Calculate perpendicular distance from pmid to line p1-p2
        if line_len == 0:
            perp_distance = 0
        else:
            t = ((pmid[0] - p1[0]) * (p2[0] - p1[0]) + (pmid[1] - p1[1]) * (p2[1] - p1[1])) / (line_len * line_len)
            t = max(0, min(1, t))  # Clamp to [0,1]
            proj_x = p1[0] + t * (p2[0] - p1[0])
            proj_y = p1[1] + t * (p2[1] - p1[1])
            perp_distance = math.sqrt((pmid[0] - proj_x)**2 + (pmid[1] - proj_y)**2)
        
        # Get curvature at midpoint
        curve = calculate_curvature(tmid)
        
        # Decide whether to subdivide based on curvature and deviation
        # Higher curvature or deviation means we need more points for accuracy
        should_subdivide = (perp_distance > curvature_threshold or \
                            curve > curvature_threshold) and depth < 10
        
        if should_subdivide:
            # Recursively subdivide
            left_points = add_curve_segment(t1, tmid, depth + 1)
            right_points = add_curve_segment(tmid, t2, depth + 1)
            return left_points + right_points[1:]  # Avoid duplicating the middle point
        else:
            # No need to subdivide further
            return [p1]
    
    # Generate initial sampling
    points = []
    t_step = (t_end - t_start) / segments
    for i in range(segments + 1):
        t = t_start + i * t_step
        points.append((x_fn(t), y_fn(t)))
    
    # Optimize with adaptive sampling if there are enough initial points
    if segments >= 4:
        optimized_points = []
        for i in range(segments):
            t1 = t_start + i * t_step
            t2 = t_start + (i + 1) * t_step
            segment_points = add_curve_segment(t1, t2)
            if i > 0:
                # Avoid duplicating points between segments
                optimized_points.extend(segment_points[1:])
            else:
                optimized_points.extend(segment_points)
        
        # Ensure the end point is included
        optimized_points.append((x_fn(t_end), y_fn(t_end)))
    else:
        optimized_points = points
    
    # Create the path from optimized points
    first_point = True
    for x, y in optimized_points:
        if first_point:
            path.move_to(x, y)
            first_point = False
        else:
            path.line_to(x, y)
    
    return path


def create_fourier_shape(center_x: float, center_y: float, radius: float,  \
        harmonic_coeffs: Optional[List[Tuple[float, float, float]]] = None,
        shape_type: str = "organic", \
        n_harmonics: int = 5, \
        precision: int = 1, \
        seed: Optional[int] = None) -> Path:
    """
    Cria uma forma otimizada usando decomposição em série de Fourier avançada.Implementa uma decomposição real em série de Fourier para modelar formas orgânicas com
    miníma quantidade de dados enquanto preserva características visuais complexas.
    
    Args:
        center_x: Coordenada X do centro da forma
        center_y: Coordenada Y do centro da forma
        radius: Raio base da forma
        harmonic_coeffs: Lista de tuplas (amplitude, frequência, fase) para controle preciso dos harmônicos
                        Se None, os coeficientes serão gerados com base no shape_type
        shape_type: Tipo de forma predefinido ("organic", "leaf", "cloud", "rock", "tree", "flower")
        n_harmonics: Número de harmônicos a usar se harmonic_coeffs for None
        precision: Precisão decimal para valores de coordenadas
        seed: Semente para geração aleatória reproduzível (opcional)
        
    Returns:
        Objeto Path representando a forma harmonicamente otimizada
    """
    path = Path(precision)
    
    # Se uma semente for fornecida, use-a para geração reproduzível
    if seed is not None:
        saved_state = random.getstate()
        random.seed(seed)
    
    # Definir coeficientes com base em modelos de formas se não forem fornecidos
    if harmonic_coeffs is None:
        if shape_type == "leaf":
            # Forma de folha com simetria bilateral e variações naturais
            harmonic_coeffs = [
                # Coeficientes de cosseno (cos) - controle do contorno geral
                (0.3, 1, 0, 'cos'),           # Forma elíptica base (círculo achatado)
                (0.2, 2, math.pi/2, 'cos'),   # Simetria bilateral com achatamento em X
                (-0.1, 3, 0, 'cos'),          # Estreitamentos laterais (3 ciclos ao redor)
                
                # Coeficientes de seno (sin) - detalhes e assimetrias
                (0.4, 1, math.pi/4, 'sin'),    # Extensão em uma direção (ponta da folha)
                (0.03, 4, 0, 'sin'),           # Pequenas ondulações na borda
                (0.02, 7, math.pi/6, 'sin'),   # Micro-texturas na borda da folha
                (0.015, 11, 0, 'sin')          # Detalhes finos das bordas
            ]
        elif shape_type == "cloud":
            # Nuvem com formas arredondadas e limites difusos
            harmonic_coeffs = [
                # Coeficientes principais para a forma suave e arredondada
                (0.2, 1, 0, 'cos'),             # Forma base oval
                (0.15, 2, math.pi/3, 'sin'),    # Curvatura geral
                (0.1, 3, math.pi/6, 'cos'),     # Divisões principais (lobos da nuvem)
                
                # Detalhes para bordas irregulares em escalas diferentes
                (0.08, 4, math.pi/2, 'sin'),    # Ondulações médias
                (0.05, 6, math.pi/4, 'sin'),    # Ondulações menores
                (0.03, 8, 0, 'cos'),            # Microdetalhes
                (0.02, 12, math.pi/5, 'sin')    # Textura sutil nas bordas
            ]
        elif shape_type == "rock":
            # Pedra com arestas e irregularidades
            harmonic_coeffs = [
                # Forma base angular
                (0.1, 1, 0, 'cos'),              # Forma base oval/circular com leve distorção
                (0.2, 2, math.pi/5, 'sin'),      # Primeira deformação principal - cria forma básica da pedra
                (0.25, 3, math.pi/3, 'cos'),     # Segunda deformação - adiciona irregularidade
                
                # Detalhes angulares em diferentes frequências
                (0.15, 5, math.pi/2, 'sin'),     # Arestas médias
                (0.12, 7, math.pi/6, 'cos'),     # Arestas menores
                (0.06, 11, math.pi/7, 'sin'),    # Pequenas fissuras
                (0.03, 15, math.pi/9, 'sin')     # Micro-detalhes erosivos
            ]
        elif shape_type == "tree":
            # Silhueta de árvore com tronco e copa
            harmonic_coeffs = [
                # Forma base com tronco e copa
                (0.1, 1, 0, 'cos'),               # Forma base ovalada/alongada
                (-0.3, 2, math.pi/2, 'sin'),      # Estreitamento para o tronco na parte inferior
                (0.2, 3, 0, 'sin'),               # Expansão da copa na parte superior
                
                # Detalhes da copa e ramificações
                (0.15, 5, math.pi/4, 'cos'),     # Variações principais na copa
                (0.1, 7, math.pi/3, 'sin'),       # Ramificações secundárias
                (0.05, 11, math.pi/5, 'cos'),     # Pequenas variações nas bordas
                (0.02, 17, 0, 'sin')              # Micro-textura da folhagem
            ]
        elif shape_type == "flower":
            # Flor com pétalas simetricamente distribuídas
            # Número de pétalas (varia entre 5 e 8 pétalas)
            n_petals = random.choice([5, 6, 8])
            
            harmonic_coeffs = [
                # Forma base circular
                (0.05, 1, 0, 'cos'),                # Leve oval para forma básica
                
                # Pétalas - usando frequência igual ao número de pétalas
                (0.3, n_petals, 0, 'cos'),         # Cria as pétalas principais
                (0.1, n_petals*2, math.pi/(n_petals*2), 'sin'),  # Suaviza pontas das pétalas
                
                # Variações e detalhes
                (0.03, n_petals*3, math.pi/4, 'cos'),  # Textura sutil nas bordas
                (0.02, n_petals+1, 0, 'sin')            # Pequena assimetria natural
            ]
        else:  # "organic" ou padrão
            # Gera coeficientes com distribuição natural de frequências (padrão 1/f)
            harmonic_coeffs = []
            
            # Primeiro harmônico - define forma geral
            harmonic_coeffs.append((0.15, 1, 0, 'cos'))
            
            # Adiciona sin/cos para cada harmônico
            for h in range(2, n_harmonics + 1):
                # Amplitude segue distribuição 1/f (padrões naturais)
                amplitude = 0.3 / (h ** 0.8)  # Decaimento menos rápido para mais detalhes
                
                # Fase aleatória para variações naturais
                phase_cos = random.uniform(0, 2 * math.pi)
                phase_sin = random.uniform(0, 2 * math.pi)
                
                # Termos coseno afetam mais a forma geral
                cos_amplitude = amplitude * random.uniform(0.7, 1.0)
                harmonic_coeffs.append((cos_amplitude, h, phase_cos, 'cos'))
                
                # Termos seno adicionam assimetrias e detalhes
                sin_amplitude = amplitude * random.uniform(0.5, 0.9)
                harmonic_coeffs.append((sin_amplitude, h, phase_sin, 'sin'))
    
    # Determinar número ótimo de pontos com base na frequência mais alta
    max_freq = max(freq for _, freq, _, _ in harmonic_coeffs)
    # Teorema de amostragem de Nyquist: precisamos de pelo menos 2x pontos da frequência mais alta
    # Multiplicamos por 4 para maior suavidade visual
    n_points = max(50, 2 * int(max_freq) * 4)
    
    # Adicionar pontos suficientes para capturar todas as variações da forma
    points = []
    for i in range(n_points):
        theta = i * 2 * math.pi / n_points
        
        # Iniciar com círculo base
        r = radius
        
        # Aplicar decomposição de Fourier completa (senos e cossenos)
        for amplitude, frequency, phase, fn_type in harmonic_coeffs:
            if fn_type == 'cos':
                r += radius * amplitude * math.cos(frequency * theta + phase)
            else:  # 'sin'
                r += radius * amplitude * math.sin(frequency * theta + phase)
        
        # Converter para coordenadas cartesianas
        x = center_x + r * math.cos(theta)
        y = center_y + r * math.sin(theta)
        points.append((x, y))
    
    # Adicionar o primeiro ponto novamente para fechar o caminho
    points.append(points[0])
    
    # Algoritmo de simplificação de Douglas-Peucker para reduzir pontos mantendo qualidade visual
    def distance_point_to_line(p, line_start, line_end):
        if line_start == line_end:
            return math.sqrt((p[0] - line_start[0])**2 + (p[1] - line_start[1])**2)
        
        # Calcular distância perpendicular até o segmento de linha
        line_len_sq = (line_end[0] - line_start[0])**2 + (line_end[1] - line_start[1])**2
        t = max(0, min(1, ((p[0] - line_start[0]) * (line_end[0] - line_start[0]) + \
                            (p[1] - line_start[1]) * (line_end[1] - line_start[1])) / line_len_sq))
        
        proj_x = line_start[0] + t * (line_end[0] - line_start[0])
        proj_y = line_start[1] + t * (line_end[1] - line_start[1])
        
        return math.sqrt((p[0] - proj_x)**2 + (p[1] - proj_y)**2)
    
    def douglas_peucker(points, epsilon, start_idx, end_idx):
        # Encontrar o ponto mais distante do segmento de linha
        dmax = 0
        index = start_idx
        
        for i in range(start_idx + 1, end_idx):
            d = distance_point_to_line(points[i], points[start_idx], points[end_idx])
            if d > dmax:
                index = i
                dmax = d
        
        # Se a distância máxima > epsilon, simplificar recursivamente
        if dmax > epsilon:
            # Chamada recursiva para segmentos dos dois lados do ponto mais distante
            rec_results1 = douglas_peucker(points, epsilon, start_idx, index)
            rec_results2 = douglas_peucker(points, epsilon, index, end_idx)
            
            # Combinar resultados, evitando duplicar o ponto do meio
            return rec_results1[:-1] + rec_results2
        else:
            # Não precisa de pontos intermediários, apenas retornar pontos finais
            return [points[start_idx], points[end_idx]]
    
    # Aplicar simplificação se tivermos pontos suficientes
    if len(points) >= 4:
        # Ajustar epsilon com base no raio para nível de detalhe consistente
        # Menor epsilon = mais detalhes preservados
        epsilon = radius * 0.008
        
        # Manter o primeiro/último ponto idêntico para caminhos fechados
        simplified_points = douglas_peucker(points[:-1], epsilon, 0, len(points) - 2)
        # Fechar adequadamente o caminho
        simplified_points.append(simplified_points[0])
    else:
        simplified_points = points
    
    # Restaurar o estado aleatório se usamos uma semente
    if seed is not None:
        random.setstate(saved_state)
    
    # Criar caminho a partir dos pontos otimizados
    first = True
    for x, y in simplified_points:
        if first:
            path.move_to(x, y)
            first = False
        else:
            path.line_to(x, y)
    
    path.close_path()
    return path


def create_reusable_pattern(pattern_type: str, x: float, y: float, width: float, height: float,  \
                        pattern_config: Optional[Dict] = None, precision: int = 1) -> Tuple[ET.Element, ET.Element]:
    """
    Create a highly optimized reusable pattern with advanced semantic compression.Patterns are defined mathematically for maximum data efficiency and defined once
    but can be reused multiple times across the SVG for significant byte savings.
    
    Args:
        pattern_type: Type of pattern to create ("cloud", "grass", "water", "rock", "texture")
        x, y: Position coordinates for the pattern usage
        width, height: Size dimensions for the pattern usage
        pattern_config: Optional configuration parameters for the pattern
        precision: Decimal precision for coordinate values
        
    Returns:
        Tuple of (pattern_def, pattern_use) where pattern_def is the element to add to defs
        and pattern_use is the element referencing the pattern
    """
    config = pattern_config or {}
    
    # Create a unique ID based on pattern type and configuration hash
    # This ensures we only create one pattern definition even if used multiple times with same config
    config_hash = hash(str(sorted(config.items()))) if config else 0
    pattern_id = f"{pattern_type}_pattern_{abs(config_hash) % 10000}"
    
    if pattern_type == "cloud":
        # Cloud pattern optimization: use minimal geometry with maximum visual effect
        # Small pattern size reduces data requirements and works with tiling
        pattern_width = config.get('pattern_width', 60)
        pattern_height = config.get('pattern_height', 40)
        opacity = config.get('opacity', 0.8)
        color = config.get('color', "white")
        
        pattern = ET.Element("{%s}pattern" % SVGNS, attrib={ \
            "id": pattern_id,
            "patternUnits": "userSpaceOnUse", \
            "width": str(pattern_width), \
            "height": str(pattern_height), \
            "patternTransform": "rotate(5)",  # Slight rotation prevents obvious tiling artifacts
        })
        
        # Create cloud using minimal amount of data
        # Instead of many circles, use one path with quadratic Bézier curves
        # This is more compact than multiple separate elements
        cloud_path = Path(precision)
        
        # Calculate anchor points for cloud silhouette
        cx, cy = pattern_width/2, pattern_height/2
        r = min(pattern_width, pattern_height) * 0.35
        
        # Start at bottom of cloud
        cloud_path.move_to(cx - r*0.8, cy + r*0.3)
        
        # Define cloud shape with minimum number of Bézier curves
        # These points create the billowing effect without excessive detail
        curves = [
            # Left side - control point, endpoint
            [(cx - r*1.5, cy), (cx - r, cy - r*0.5)], \
            [(cx - r, cy - r*1.3), (cx - r*0.3, cy - r*0.8)], \
            [(cx, cy - r*1.2), (cx + r*0.3, cy - r*0.7)], \
            [(cx + r*0.8, cy - r), (cx + r, cy)], \
            [(cx + r*1.2, cy + r*0.5), (cx, cy + r*0.5)], \
            [(cx - r*0.5, cy + r*0.8), (cx - r*0.8, cy + r*0.3)]
        ]
        
        for control, end in curves:
            cloud_path.quad_curve_to(control[0], control[1], end[0], end[1])
            
        pattern.append(cloud_path.to_svg_element({ \
            "fill": color,
            "opacity": str(opacity)
        }))
        
    elif pattern_type == "grass":
        # Optimized grass pattern that balances detail with file size
        pattern_width = config.get('pattern_width', 30)
        pattern_height = config.get('pattern_height', 20)
        density = config.get('density', 7)  # Number of grass blades
        color = config.get('color', "#3a5f0b")
        variety = config.get('variety', 0.4)  # How much variation in grass blades
        
        pattern = ET.Element("{%s}pattern" % SVGNS, attrib={ \
            "id": pattern_id,
            "patternUnits": "userSpaceOnUse", \
            "width": str(pattern_width), \
            "height": str(pattern_height)
        })
        
        # Using a single path for all grass blades reduces file size significantly
        # vs. individual paths for each blade
        all_grass = Path(precision)
        
        # Create grass blades with mathematical distribution
        for i in range(density):
            # Distribute grass blades using golden ratio to avoid regular patterns
            # This looks more natural than evenly spaced blades
            x_pos = pattern_width * ((i * 0.618033988749895) % 1.0)
            
            # Vary height and shape
            height_var = pattern_height * (0.5 + 0.5 * variety * math.sin(i * 1.7))
            
            # Add slight curve to grass blade using quadratic Bézier
            bend = (random.random() - 0.5) * pattern_width * 0.15
            
            # Move to bottom of grass blade
            all_grass.move_to(x_pos, pattern_height)
            
            # Create curved blade
            all_grass.quad_curve_to(x_pos + bend, pattern_height - height_var/2, \
                                    x_pos + bend*0.5, pattern_height - height_var)
        
        pattern.append(all_grass.to_svg_element({ \
            "stroke": color,
            "stroke-width": "1", \
            "fill": "none"
        }))
    
    elif pattern_type == "water":
        # Water ripple pattern using minimal SVG for maximum effect
        pattern_width = config.get('pattern_width', 100)
        pattern_height = config.get('pattern_height', 50)
        color = config.get('color', "#0077be")
        opacity = config.get('opacity', 0.3)
        
        pattern = ET.Element("{%s}pattern" % SVGNS, attrib={ \
            "id": pattern_id,
            "patternUnits": "userSpaceOnUse", \
            "width": str(pattern_width), \
            "height": str(pattern_height), \
            "patternTransform": "rotate(5)"  # Slight rotation for more natural appearance
        })
        
        # Create water ripples with single sine wave path
        # More efficient than multiple elements
        water_path = Path(precision)
        
        # Start at left edge
        water_path.move_to(0, pattern_height/2)
        
        # Create sine wave with minimal points
        steps = 8  # Small number of steps keeps file size down
        amplitude = pattern_height * 0.2
        
        for i in range(1, steps + 1):
            x = i * pattern_width / steps
            y = pattern_height/2 + amplitude * math.sin(i * math.pi / 2)
            water_path.line_to(x, y)
        
        pattern.append(water_path.to_svg_element({ \
            "stroke": color,
            "stroke-width": "1.5", \
            "stroke-opacity": str(opacity), \
            "fill": "none"
        }))
    
    elif pattern_type == "rock":
        # Rock texture pattern using noise-like structures
        pattern_width = config.get('pattern_width', 40)
        pattern_height = config.get('pattern_height', 40)
        color1 = config.get('color1', "#555555")
        color2 = config.get('color2', "#333333")
        
        pattern = ET.Element("{%s}pattern" % SVGNS, attrib={ \
            "id": pattern_id,
            "patternUnits": "userSpaceOnUse", \
            "width": str(pattern_width), \
            "height": str(pattern_height)
        })
        
        # Add background
        background = Rectangle(0, 0, pattern_width, pattern_height, precision)
        pattern.append(background.to_svg_element({"fill": color1}))
        
        # Add rock texture using minimal line segments
        # Using controlled randomness with a mathematical basis
        rock_path = Path(precision)
        
        # Create cracks and texture lines that look like rock
        # Use golden ratio based distribution for natural-looking pattern
        for i in range(5):  # Just a few lines for efficiency
            phi = 1.618033988749895
            x1 = pattern_width * ((i * phi) % 1.0)
            y1 = pattern_height * (((i+1) * phi) % 1.0)
            x2 = pattern_width * (((i+2) * phi) % 1.0)
            y2 = pattern_height * (((i+3) * phi) % 1.0)
            
            rock_path.move_to(x1, y1)
            rock_path.line_to(x2, y2)
        
        pattern.append(rock_path.to_svg_element({ \
            "stroke": color2,
            "stroke-width": "0.7", \
            "stroke-linecap": "round"
        }))
    
    elif pattern_type == "texture":
        # Generic texture pattern that can be customized
        pattern_width = config.get('pattern_width', 20)
        pattern_height = config.get('pattern_height', 20)
        color = config.get('color', "#888888")
        opacity = config.get('opacity', 0.4)
        style = config.get('style', "dots")  # "dots", "lines", "grid", "noise"
        
        pattern = ET.Element("{%s}pattern" % SVGNS, attrib={ \
            "id": pattern_id,
            "patternUnits": "userSpaceOnUse", \
            "width": str(pattern_width), \
            "height": str(pattern_height)
        })
        
        if style == "dots":
            # Just use a few tiny circles for dots pattern
            for i in range(4):
                # Distribute using golden ratio
                cx = pattern_width * ((i * 0.618033988749895) % 1.0)
                cy = pattern_height * ((i * 0.618033988749895 * 1.618033988749895) % 1.0)
                r = min(pattern_width, pattern_height) * 0.08
                
                dot = Circle(cx, cy, r, precision)
                pattern.append(dot.to_svg_element({ \
                    "fill": color,
                    "opacity": str(opacity)
                }))
                
        elif style == "lines":
            # Diagonal lines are effective and simple
            lines_path = Path(precision)
            spacing = pattern_width / 4
            
            for i in range(-2, 6):  # Extra lines to ensure coverage
                lines_path.move_to(i * spacing, 0)
                lines_path.line_to(i * spacing + pattern_height, pattern_height)
            
            pattern.append(lines_path.to_svg_element({ \
                "stroke": color,
                "stroke-width": "0.5", \
                "opacity": str(opacity), \
                "fill": "none"
            }))
            
        elif style == "grid":
            # Simple grid pattern
            grid_path = Path(precision)
            
            # Horizontal lines
            for i in range(3):
                y = i * pattern_height / 2
                grid_path.move_to(0, y)
                grid_path.line_to(pattern_width, y)
            
            # Vertical lines
            for i in range(3):
                x = i * pattern_width / 2
                grid_path.move_to(x, 0)
                grid_path.line_to(x, pattern_height)
            
            pattern.append(grid_path.to_svg_element({ \
                "stroke": color,
                "stroke-width": "0.5", \
                "opacity": str(opacity), \
                "fill": "none"
            }))
            
        else:  # "noise" or default
            # Create noise-like texture using minimal SVG
            noise_path = Path(precision)
            
            # Use a minimal number of points distributed with controlled randomness
            for i in range(6):
                # Distribute using mathematical formula rather than pure randomness
                # This creates a deterministic but irregular pattern
                x = pattern_width * (0.5 + 0.5 * math.cos(i * 1.1))
                y = pattern_height * (0.5 + 0.5 * math.sin(i * 1.7))
                
                if i == 0:
                    noise_path.move_to(x, y)
                else:
                    noise_path.line_to(x, y)
            
            pattern.append(noise_path.to_svg_element({ \
                "stroke": color,
                "stroke-width": "0.4", \
                "opacity": str(opacity), \
                "fill": "none"
            }))
    
    else:
        # Default to simple dots pattern if unknown type
        pattern_width = config.get('pattern_width', 10)
        pattern_height = config.get('pattern_height', 10)
        
        pattern = ET.Element("{%s}pattern" % SVGNS, attrib={ \
            "id": pattern_id,
            "patternUnits": "userSpaceOnUse", \
            "width": str(pattern_width), \
            "height": str(pattern_height)
        })
        
        # Just a single centered circle
        dot = Circle(pattern_width/2, pattern_height/2, 1, precision)
        pattern.append(dot.to_svg_element({"fill": "#888888"}))
    
    # Create a rectangle that uses this pattern
    rect = Rectangle(x, y, width, height, precision)
    opacity_attr = {"fill": f"url(#{pattern_id})"}
    
    # Add opacity if specified in config
    if 'usage_opacity' in config:
        opacity_attr["opacity"] = str(config.get('usage_opacity'))
        
    return pattern, rect.to_svg_element(opacity_attr)


def create_svg_from_prompt(prompt: str, width: int = 800, height: int = 600, precision: int = 1) -> SVGDocument:
    """
    Create SVG illustration based on text prompt using advanced mathematical and geometric techniques.
    
    Args:
        prompt: Descriptive text for desired illustration
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
        
    Returns:
        SVG document object
    """
    # Initialize document
    doc = SVGDocument(width, height, precision)
    prompt = prompt.lower()  # Convert to lowercase for keyword matching
    
    # Advanced prompt analysis to extract scene features
    scene_features = extract_semantic_features(prompt)
    
    print(f"Prompt analysis: {scene_features}")
    
    # Initialize SVG optimizer for advanced compression techniques
    optimizer = SVGOptimizer(light_angle=compute_light_angle(scene_features))
    
    # Create the main scene graph using compositional generation
    generate_scene_graph(doc, scene_features, width, height, precision, optimizer)
    
    # Apply extreme optimization with perceptual preservation
    doc.extreme_optimize()
    
    return doc

def extract_semantic_features(prompt: str) -> Dict:
    """
    Extract rich semantic features from the prompt text using lexical-semantic analysis.
    
    Args:
        prompt: Descriptive text prompt
        
    Returns:
        Dictionary of extracted semantic features
    """
    prompt = prompt.lower()
    
    # Initialize feature extraction structure
    scene_features = {
        # Time and lighting
        'time_of_day': None,  # dawn, day, sunset, night
        'lighting': None,      # bright, dim, dramatic
        
        # Main environment
        'environment': None,   # mountains, ocean, forest, city, beach, valley, field
        
        # Water elements
        'water_features': [],  # lake, ocean, river, waves, reflections
        
        # Sky elements
        'sky_features': [],    # clouds, stars, aurora, moon, sun
        
        # Terrain elements
        'terrain_features': [], # mountains, beach, valley, rocks, snow, grass, flowers
        
        # Built elements
        'built_features': [],   # city, buildings, skyscrapers
        
        # Vegetation
        'vegetation': [],       # trees, forest, grass, flowers, palm
        
        # Additional features for richer context
        'materials': [],        # metal, wood, glass, stone
        'colors': [],           # blue, red, green, etc.
        'weather': None,        # clear, cloudy, rain, storm, snow
        'mood': None,           # calm, dramatic, peaceful, chaotic
        'season': None,         # spring, summer, fall, winter
        'perspective': None,    # low-angle, aerial, closeup
        
        # Raw prompt for advanced processing
        'raw_prompt': prompt
    }
    
    # Identify time/lighting characteristics
    if any(term in prompt for term in ['dawn', 'sunrise', 'morning']):
        scene_features['time_of_day'] = 'dawn'
    elif any(term in prompt for term in ['day', 'daylight', 'sunny', 'afternoon', 'noon']):
        scene_features['time_of_day'] = 'day'
    elif any(term in prompt for term in ['sunset', 'dusk', 'evening', 'twilight']):
        scene_features['time_of_day'] = 'sunset'
    elif any(term in prompt for term in ['night', 'dark', 'midnight', 'nighttime']):
        scene_features['time_of_day'] = 'night'
    
    # Identify lighting characteristics
    if any(term in prompt for term in ['bright', 'vivid', 'clear', 'sunny', 'brilliant']):
        scene_features['lighting'] = 'bright'
    elif any(term in prompt for term in ['dim', 'shadowy', 'muted', 'soft', 'subtle']):
        scene_features['lighting'] = 'dim'
    elif any(term in prompt for term in ['dramatic', 'contrast', 'stark', 'intense', 'dynamic']):
        scene_features['lighting'] = 'dramatic'
    elif any(term in prompt for term in ['warm', 'golden', 'amber']):
        scene_features['lighting'] = 'warm'
    elif any(term in prompt for term in ['cool', 'cold', 'blue', 'crisp']):
        scene_features['lighting'] = 'cool'
    
    # Identify main environment
    environments = { \
        'mountains': ['mountains', 'mountain', 'alpine', 'peak', 'summit', 'high altitude'],
        'ocean': ['ocean', 'sea', 'maritime', 'seaside', 'coast', 'coastal', 'shore', 'beach'], \
        'forest': ['forest', 'woodland', 'woods', 'jungle', 'grove'], \
        'city': ['city', 'urban', 'metropolis', 'downtown', 'cityscape', 'skyline'], \
        'beach': ['beach', 'shore', 'coast', 'sand'], \
        'valley': ['valley', 'canyon', 'ravine', 'gorge'], \
        'field': ['field', 'meadow', 'plain', 'grassland', 'prairie'], \
        'desert': ['desert', 'arid', 'sand', 'dune', 'barren'], \
        'lake': ['lake', 'pond', 'lagoon']
    }
    
    # Find the environment with the most matches
    env_scores = {env: 0 for env in environments}
    for env, terms in environments.items():
        for term in terms:
            if term in prompt:
                env_scores[env] += 1
    
    # Select environment with highest score
    if any(env_scores.values()):
        scene_features['environment'] = max(env_scores, key=env_scores.get)
    
    # Extract water features
    water_features = { \
        'lake': ['lake', 'pond', 'reservoir'],
        'ocean': ['ocean', 'sea', 'seas'], \
        'river': ['river', 'stream', 'creek', 'waterway'], \
        'waves': ['waves', 'wave', 'surf', 'swells'], \
        'reflections': ['reflection', 'reflections', 'mirror', 'reflected'], \
        'waterfall': ['waterfall', 'falls', 'cascade']
    }
    
    for feature, terms in water_features.items():
        if any(term in prompt for term in terms):
            scene_features['water_features'].append(feature)
    
    # Extract sky features
    sky_features = { \
        'clouds': ['cloud', 'clouds', 'cloudy', 'overcast'],
        'stars': ['star', 'stars', 'starry', 'stellar'], \
        'moon': ['moon', 'lunar', 'crescent'], \
        'sun': ['sun', 'sunshine', 'solar'], \
        'aurora': ['aurora', 'northern lights', 'aurora borealis', 'aurora australis'], \
        'rainbow': ['rainbow', 'spectrum']
    }
    
    for feature, terms in sky_features.items():
        if any(term in prompt for term in terms):
            scene_features['sky_features'].append(feature)

    # Extract terrain features
    terrain_features = { \
        'mountains': ['mountain', 'mountains', 'peak', 'summit'],
        'beach': ['beach', 'shore', 'sand'], \
        'valley': ['valley', 'vale', 'glen'], \
        'rocks': ['rock', 'rocks', 'rocky', 'boulder', 'stone'], \
        'snow': ['snow', 'snowy', 'snowfall', 'snowed'], \
        'grass': ['grass', 'grassy', 'lawn'], \
        'flowers': ['flower', 'flowers', 'floral', 'blossom'], \
        'cliff': ['cliff', 'cliffs', 'precipice', 'bluff'], \
        'canyon': ['canyon', 'gorge', 'ravine']
    }
    
    for feature, terms in terrain_features.items():
        if any(term in prompt for term in terms):
            scene_features['terrain_features'].append(feature)
    
    # Extract built features
    built_features = { \
        'buildings': ['building', 'buildings', 'structure', 'structures'],
        'skyscrapers': ['skyscraper', 'skyscrapers', 'tower', 'towers', 'high-rise'], \
        'houses': ['house', 'houses', 'home', 'homes', 'cottage', 'cabin'], \
        'bridge': ['bridge', 'bridges', 'span'], \
        'city': ['city', 'cities', 'urban', 'metropolis'], \
        'road': ['road', 'roads', 'street', 'avenue', 'highway'], \
        'temple': ['temple', 'shrine', 'sanctuary'], \
        'church': ['church', 'cathedral', 'chapel']
    }
    
    for feature, terms in built_features.items():
        if any(term in prompt for term in terms):
            scene_features['built_features'].append(feature)
    
    # Extract vegetation
    vegetation_features = { \
        'trees': ['tree', 'trees', 'oak', 'pine', 'elm', 'birch', 'evergreen'],
        'forest': ['forest', 'woods', 'woodland', 'jungle'], \
        'grass': ['grass', 'grassy', 'lawn', 'turf'], \
        'flowers': ['flower', 'flowers', 'blossom', 'bloom', 'petal'], \
        'palm': ['palm', 'palms', 'palm tree', 'coconut tree'], \
        'bush': ['bush', 'bushes', 'shrub', 'shrubs'], \
        'cacti': ['cactus', 'cacti', 'succulent']
    }
    
    for feature, terms in vegetation_features.items():
        if any(term in prompt for term in terms):
            scene_features['vegetation'].append(feature)
    
    # Extract materials
    materials = { \
        'metal': ['metal', 'metallic', 'steel', 'iron', 'aluminum', 'chrome'],
        'wood': ['wood', 'wooden', 'timber', 'log'], \
        'glass': ['glass', 'transparent', 'translucent'], \
        'stone': ['stone', 'rock', 'granite', 'marble', 'slate'], \
        'water': ['water', 'liquid', 'aquatic'], \
        'cloth': ['cloth', 'fabric', 'textile', 'canvas', 'silk']
    }
    
    for material, terms in materials.items():
        if any(term in prompt for term in terms):
            scene_features['materials'].append(material)
    
    # Extract colors
    colors = { \
        'blue': ['blue', 'azure', 'cyan', 'teal', 'turquoise', 'sapphire'],
        'red': ['red', 'crimson', 'scarlet', 'maroon', 'ruby'], \
        'green': ['green', 'emerald', 'jade', 'olive', 'lime'], \
        'yellow': ['yellow', 'amber', 'gold', 'blonde'], \
        'orange': ['orange', 'tangerine', 'peach'], \
        'purple': ['purple', 'violet', 'lavender', 'indigo', 'magenta'], \
        'pink': ['pink', 'rose', 'salmon', 'coral'], \
        'white': ['white', 'snow', 'ivory', 'pearl'], \
        'black': ['black', 'ebony', 'obsidian', 'midnight'], \
        'brown': ['brown', 'tan', 'beige', 'sepia', 'chocolate']
    }
    
    for color, terms in colors.items():
        if any(term in prompt for term in terms):
            scene_features['colors'].append(color)
    
    # Extract weather
    weather_types = { \
        'clear': ['clear', 'sunny', 'fair', 'cloudless'],
        'cloudy': ['cloudy', 'overcast', 'cloud', 'clouds'], \
        'rain': ['rain', 'rainy', 'raining', 'rainfall', 'drizzle', 'showers'], \
        'storm': ['storm', 'stormy', 'thunderstorm', 'lightning', 'thunder'], \
        'snow': ['snow', 'snowy', 'snowing', 'snowfall', 'snowflakes'], \
        'fog': ['fog', 'foggy', 'mist', 'misty', 'haze']
    }
    
    for weather, terms in weather_types.items():
        if any(term in prompt for term in terms):
            scene_features['weather'] = weather
            break
    
    # Extract mood
    mood_types = { \
        'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'quiet', 'still'],
        'dramatic': ['dramatic', 'epic', 'intense', 'powerful', 'magnificent'], \
        'cheerful': ['cheerful', 'happy', 'joyful', 'bright', 'playful', 'vibrant'], \
        'melancholic': ['melancholy', 'sad', 'somber', 'gloomy', 'moody', 'wistful'], \
        'mysterious': ['mysterious', 'enigmatic', 'strange', 'cryptic', 'magical'], \
        'chaotic': ['chaotic', 'wild', 'disorganized', 'turbulent', 'tempestuous']
    }
    
    for mood, terms in mood_types.items():
        if any(term in prompt for term in terms):
            scene_features['mood'] = mood
            break
    
    # Extract season
    season_types = { \
        'spring': ['spring', 'vernal', 'blossom', 'bloom'],
        'summer': ['summer', 'estival', 'summery', 'midsummer'], \
        'fall': ['fall', 'autumn', 'autumnal', 'harvest'], \
        'winter': ['winter', 'wintry', 'hibernal', 'frosty']
    }
    
    for season, terms in season_types.items():
        if any(term in prompt for term in terms):
            scene_features['season'] = season
            break
    
    # Extract perspective
    perspective_types = { \
        'aerial': ['aerial', 'birds-eye', 'overhead', 'top-down', 'bird view', 'bird\'s eye'],
        'low-angle': ['low-angle', 'from below', 'bottom-up', 'upward view'], \
        'eye-level': ['eye-level', 'straight-on', 'human perspective', 'direct'], \
        'closeup': ['closeup', 'close-up', 'macro', 'detail', 'upclose'], \
        'wide': ['wide', 'panoramic', 'panorama', 'wide-angle', 'sprawling', 'wide view']
    }
    
    for perspective, terms in perspective_types.items():
        if any(term in prompt for term in terms):
            scene_features['perspective'] = perspective
            break
    
    return scene_features

def compute_light_angle(scene_features: Dict) -> float:
    """
    Compute the optimal light angle based on scene features.
    
    Args:
        scene_features: Dictionary of semantic features
        
    Returns:
        Light angle in degrees
    """
    # Default angle (45 degrees) for standard lighting
    base_angle = 45.0
    
    # Adjust based on time of day
    if scene_features['time_of_day'] == 'dawn':
        base_angle = 15.0  # Low angle from sunrise
    elif scene_features['time_of_day'] == 'day':
        base_angle = 70.0  # High noon
    elif scene_features['time_of_day'] == 'sunset':
        base_angle = 160.0  # Setting sun from the west
    elif scene_features['time_of_day'] == 'night':
        base_angle = 230.0  # Moonlight or ambient
    
    # Adjust based on lighting type
    if scene_features['lighting'] == 'dramatic':
        # Dramatic lighting often comes from a side angle
        base_angle = (base_angle + 90) % 360
    
    return base_angle

def generate_scene_graph(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int, optimizer: SVGOptimizer) -> None:
    """
    Main compositional scene graph generator that orchestrates the creation of SVG elements
    based on semantic features extracted from the prompt.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
        optimizer: SVG optimizer instance for compression techniques
    """
    # Create scene layers in proper z-index order (back to front)
    
    # 1. Background gradient (sky/environment)
    generate_background(doc, scene_features, width, height, precision)
    
    # 2. Celestial elements (sun, moon, stars, etc.)
    generate_celestial_elements(doc, scene_features, width, height, precision)
    
    # 3. Atmospheric elements (clouds, fog, rain, etc.)
    generate_atmospheric_elements(doc, scene_features, width, height, precision)
    
    # 4. Background terrain (mountains, hills in distance)
    generate_background_terrain(doc, scene_features, width, height, precision)
    
    # 5. Mid-ground elements (landscape features)
    generate_midground_elements(doc, scene_features, width, height, precision)
    
    # 6. Water bodies (ocean, lake, river)
    generate_water_elements(doc, scene_features, width, height, precision)
    
    # 7. Foreground terrain elements
    generate_foreground_terrain(doc, scene_features, width, height, precision)
    
    # 8. Vegetation (trees, plants, flowers)
    generate_vegetation(doc, scene_features, width, height, precision)
    
    # 9. Built structures (buildings, bridges, etc.)
    generate_built_structures(doc, scene_features, width, height, precision)
    
    # 10. Foreground details (rocks, small elements)
    generate_foreground_details(doc, scene_features, width, height, precision)
    
    # 11. Apply global lighting effects
    apply_global_lighting(doc, scene_features, width, height, precision, optimizer)
    
    # 12. Apply final optimizations
    optimizer.optimize_document(doc)

def generate_background(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate the background sky gradient based on time of day and environmental features.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Determine gradient colors based on time of day and mood
    if scene_features['time_of_day'] == 'dawn':
        # Dawn sky with pinkish/orange gradients
        sky = LinearGradient("sky_gradient", 0, 0, 0, height * 0.7, precision)
        sky.add_stop(0, "#1A2456")  # Deep blue at top
        sky.add_stop(0.4, "#724e91")  # Purple transition
        sky.add_stop(0.7, "#EF7674")  # Pale pink
        sky.add_stop(1, "#F7D6BF")    # Light orange at horizon
    elif scene_features['time_of_day'] == 'sunset':
        # Sunset sky with deep orange/red gradients
        sky = LinearGradient("sky_gradient", 0, 0, 0, height * 0.7, precision)
        sky.add_stop(0, "#0c1445")    # Deep blue at top
        sky.add_stop(0.4, "#472E87")  # Deep purple
        sky.add_stop(0.6, "#FF6347")  # Tomato red
        sky.add_stop(0.8, "#FF8C42")  # Orange
        sky.add_stop(1, "#FFDFB8")    # Pale yellow at horizon
    elif scene_features['time_of_day'] == 'night':
        # Night sky with deep blues
        sky = LinearGradient("sky_gradient", 0, 0, 0, height * 0.7, precision)
        sky.add_stop(0, "#0A0E21")    # Nearly black at top
        sky.add_stop(0.7, "#0C1445")  # Deep blue
        sky.add_stop(1, "#1A2980")    # Dark blue at horizon
    else:  # 'day' or default
        # Basic blue daytime sky
        # Adjust colors based on weather
        if scene_features['weather'] == 'cloudy':
            sky = LinearGradient("sky_gradient", 0, 0, 0, height * 0.7, precision)
            sky.add_stop(0, "#4a6a8f")    # Grayish-blue at top
            sky.add_stop(0.5, "#8ab4d8")  # Lighter gray-blue in middle
            sky.add_stop(1, "#c6d4e2")    # Very light gray-blue at horizon
        elif scene_features['weather'] == 'storm':
            sky = LinearGradient("sky_gradient", 0, 0, 0, height * 0.7, precision)
            sky.add_stop(0, "#1c2331")    # Very dark gray-blue at top
            sky.add_stop(0.5, "#2d3d54")  # Medium dark gray in middle
            sky.add_stop(1, "#5f788c")    # Lighter gray at horizon
        else:  # clear day
            sky = LinearGradient("sky_gradient", 0, 0, 0, height * 0.7, precision)
            sky.add_stop(0, "#1a4d8c")    # Medium blue at top
            sky.add_stop(0.5, "#4a90e2")  # Light blue in middle
            sky.add_stop(1, "#c8e6ff")    # Very light blue at horizon
    
    # Apply mood modifiers to the gradient
    if scene_features['mood'] == 'dramatic':
        # Make colors more intense and high-contrast
        for i in range(len(sky.stops)):
            stop = sky.stops[i]
            color = stop[1]
            # Convert to HSL, increase saturation and adjust lightness for drama
            h, s, l = rgb_to_hsl(parse_hex_color(color))
            s = min(s * 1.3, 1.0)  # Increase saturation by 30% \
            l = max(0.1, min(l * 0.9, 0.9))  # Adjust lightness for more contrast
            sky.stops[i] = (stop[0], hsl_to_hex(h, s, l))
    elif scene_features['mood'] == 'calm' or scene_features['mood'] == 'peaceful':
        # Make colors more subtle and harmonious
        for i in range(len(sky.stops)):
            stop = sky.stops[i]
            color = stop[1]
            # Convert to HSL, decrease saturation for calmness
            h, s, l = rgb_to_hsl(parse_hex_color(color))
            s = s * 0.8  # Decrease saturation by 20% \
            l = min(l * 1.1, 0.95)  # Slightly increase lightness
            sky.stops[i] = (stop[0], hsl_to_hex(h, s, l))
    
    # Add the gradient definition to the document
    doc.add_definition(sky.to_svg_element())
    
    # Create the sky background rectangle
    sky_rect = Rectangle(0, 0, width, height).to_svg_element({ \
        "fill": "url(#sky_gradient)",
        "stroke": "none"
    })
    
    # Add to document
    doc.add_element(sky_rect)

# Color utility functions for parametric transformations
def parse_hex_color(hex_color: str) -> Tuple[float, float, float]:
    """
    Parse a hex color string (#RRGGBB) to RGB tuple (0-1 range).
    
    Args:
        hex_color: Hex color string (#RRGGBB)
        
    Returns:
        Tuple of (red, green, blue) values in 0-1 range
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)

def rgb_to_hsl(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert RGB (0-1 range) to HSL (0-1 range).
    
    Args:
        rgb: Tuple of (red, green, blue) values in 0-1 range
        
    Returns:
        Tuple of (hue, saturation, lightness) values in 0-1 range
    """
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    l = (max_val + min_val) / 2
    
    if max_val == min_val:
        h = s = 0  # achromatic
    else:
        d = max_val - min_val
        s = d / (2 - max_val - min_val) if l > 0.5 else d / (max_val + min_val)
        
        if max_val == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_val == g:
            h = (b - r) / d + 2
        else:  # max_val == b
            h = (r - g) / d + 4
        
        h /= 6
    
    return (h, s, l)

def hsl_to_rgb(hsl: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert HSL (0-1 range) to RGB (0-1 range).
    
    Args:
        hsl: Tuple of (hue, saturation, lightness) values in 0-1 range
        
    Returns:
        Tuple of (red, green, blue) values in 0-1 range
    """
    h, s, l = hsl
    
    if s == 0:
        r = g = b = l  # achromatic
    else:
        def hue_to_rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
    
    return (r, g, b)

def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """
    Convert RGB (0-1 range) to hex color string.
    
    Args:
        rgb: Tuple of (red, green, blue) values in 0-1 range
        
    Returns:
        Hex color string (#RRGGBB)
    """
    r, g, b = rgb
    r = int(r * 255)
    g = int(g * 255)
    b = int(b * 255)
    return f"#{r:02x}{g:02x}{b:02x}"

def hsl_to_hex(h: float, s: float, l: float) -> str:
    """
    Convert HSL (0-1 range) to hex color string.
    
    Args:
        h: Hue (0-1)
        s: Saturation (0-1)
        l: Lightness (0-1)
        
    Returns:
        Hex color string (#RRGGBB)
    """
    return rgb_to_hex(hsl_to_rgb((h, s, l)))

def generate_celestial_elements(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate celestial elements like sun, moon, and stars based on scene features.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    time_of_day = scene_features['time_of_day']
    
    # Add stars for night or sunset scenes
    if time_of_day == 'night' or 'stars' in scene_features['sky_features']:
        # Create a defs group for the stars
        star_defs = doc.create_group_element("star_defs", {"class": "star-definitions"})
        
        # Create a few different star shapes for variety
        for i in range(3):
            size = 1 + i * 0.5  # Different sizes
            # Path data for a simple star shape
            star_path = Path(f"star_{i}").M(0, -size).L(0.3*size, -0.3*size).L(size, -0.3*size)\
                .L(0.5*size, 0.3*size).L(0.7*size, size).L(0, 0.5*size).L(-0.7*size, size)\
                .L(-0.5*size, 0.3*size).L(-size, -0.3*size).L(-0.3*size, -0.3*size).Z()
            
            star_attrs = {"fill": "#FFFFFF", "stroke": "none"}
            star_defs.append(star_path.to_svg_element(star_attrs))
        
        doc.add_definition(star_defs)
        
        # Create a group for all stars
        stars_group = doc.create_group_element("stars", {"class": "stars-layer"})
        
        # Add stars with parametric distribution
        num_stars = 150 if 'stars' in scene_features['sky_features'] else 50
        for i in range(num_stars):
            # Use deterministic pseudo-random to ensure consistent generation
            x = (i * 17 % 97) / 97 * width
            y = (i * 23 % 89) / 89 * height * 0.7  # Only in the sky portion
            
            # Vary opacity for twinkling effect
            opacity = 0.5 + ((i * 13) % 51) / 100
            
            # Pick one of the star shapes
            star_type = i % 3
            
            # Use element reference for efficiency
            star = doc.create_element("use", { \
                "x": round(x, precision),
                "y": round(y, precision), \
                "href": f"#star_{star_type}", \
                "opacity": round(opacity, 2)
            })
            
            stars_group.append(star)
        
        doc.add_element(stars_group)
    
    # Add sun for day, dawn, or sunset scenes
    if time_of_day in ['day', 'dawn', 'sunset'] and ('sun' in scene_features['sky_features'] or time_of_day != 'night'):
        # Position sun based on time of day
        if time_of_day == 'dawn':
            sun_x = width * 0.2
            sun_y = height * 0.3
            sun_color = "#FFA726"  # Orange
            glow_color = "#FFECB3"  # Light amber
        elif time_of_day == 'sunset':
            sun_x = width * 0.8
            sun_y = height * 0.3
            sun_color = "#FF7043"  # Deep orange
            glow_color = "#FFCCBC"  # Light orange
        else:  # day
            sun_x = width * 0.7
            sun_y = height * 0.2
            sun_color = "#FFEB3B"  # Yellow
            glow_color = "#FFF9C4"  # Light yellow
        
        # Create sun gradient
        sun_gradient_id = "sun_gradient"
        sun_gradient = RadialGradient(sun_gradient_id, sun_x, sun_y, 0, sun_x, sun_y, 30, precision)
        sun_gradient.add_stop(0, sun_color)
        sun_gradient.add_stop(0.7, sun_color)
        sun_gradient.add_stop(1, glow_color)
        doc.add_definition(sun_gradient.to_svg_element())
        
        # Create sun glow filter for more realistic appearance
        sun_filter_id = "sun_glow"
        sun_filter = doc.create_element("filter", { \
            "id": sun_filter_id,
            "x": "-50%", \
            "y": "-50%", \
            "width": "200%", \
            "height": "200%"
        })
        
        # Add blur effect
        sun_filter.append(doc.create_element("feGaussianBlur", { \
            "in": "SourceGraphic",
            "stdDeviation": "5"
        }))
        
        # Add glow
        sun_filter.append(doc.create_element("feColorMatrix", { \
            "type": "matrix",
            "values": "1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 0.8 0"
        }))
        
        doc.add_definition(sun_filter)
        
        # Create sun circle
        sun_radius = 30 if time_of_day == 'day' else 25
        sun = doc.create_element("circle", { \
            "cx": round(sun_x, precision),
            "cy": round(sun_y, precision), \
            "r": sun_radius, \
            "fill": f"url(#{sun_gradient_id})", \
            "filter": f"url(#{sun_filter_id})"
        })
        
        doc.add_element(sun)
    
    # Add moon for night scenes
    if time_of_day == 'night' and ('moon' in scene_features['sky_features'] or not scene_features['sky_features']):
        # Position moon
        moon_x = width * 0.75
        moon_y = height * 0.25
        
        # Create moon gradient
        moon_gradient_id = "moon_gradient"
        moon_gradient = RadialGradient(moon_gradient_id, moon_x - 5, moon_y - 5, 0, moon_x, moon_y, 25, precision)
        moon_gradient.add_stop(0, "#FFFFFF")
        moon_gradient.add_stop(0.7, "#F5F5F5")
        moon_gradient.add_stop(1, "#E0E0E0")
        doc.add_definition(moon_gradient.to_svg_element())
        
        # Create moon glow filter
        moon_filter_id = "moon_glow"
        moon_filter = doc.create_element("filter", { \
            "id": moon_filter_id,
            "x": "-50%", \
            "y": "-50%", \
            "width": "200%", \
            "height": "200%"
        })
        
        # Add blur effect
        moon_filter.append(doc.create_element("feGaussianBlur", { \
            "in": "SourceGraphic",
            "stdDeviation": "3"
        }))
        
        doc.add_definition(moon_filter)
        
        # Create moon circle
        moon = doc.create_element("circle", { \
            "cx": round(moon_x, precision),
            "cy": round(moon_y, precision), \
            "r": 25, \
            "fill": f"url(#{moon_gradient_id})", \
            "filter": f"url(#{moon_filter_id})"
        })
        
        doc.add_element(moon)

def generate_atmospheric_elements(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate atmospheric elements like clouds, fog, and rain based on scene features.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Add clouds if it's in sky features or based on weather
    if 'clouds' in scene_features['sky_features'] or scene_features['weather'] in ['cloudy', 'storm', 'rain']:
        # Create cloud group
        cloud_group = doc.create_group_element("clouds", {"class": "cloud-layer"})
        
        # Define cloud types based on weather
        if scene_features['weather'] == 'storm':
            cloud_color = "#37474F"  # Dark gray
            cloud_count = 8
            cloud_size = 1.5
            cloud_opacity = 0.9
        elif scene_features['weather'] == 'cloudy':
            cloud_color = "#B0BEC5"  # Light gray
            cloud_count = 12
            cloud_size = 1.2
            cloud_opacity = 0.8
        elif scene_features['weather'] == 'rain':
            cloud_color = "#78909C"  # Medium gray
            cloud_count = 10
            cloud_size = 1.3
            cloud_opacity = 0.85
        else:  # Clear day with some clouds
            cloud_color = "#FFFFFF"  # White
            cloud_count = 6
            cloud_size = 1.0
            cloud_opacity = 0.7
        
        # Adjust based on time of day
        if scene_features['time_of_day'] == 'sunset':
            # Tint clouds with orange for sunset
            base_rgb = parse_hex_color(cloud_color)
            h, s, l = rgb_to_hsl(base_rgb)
            # Warm up the cloud color
            h = 0.08  # Orange-ish hue
            s = min(s + 0.1, 1.0)  # Increase saturation slightly
            cloud_color = hsl_to_hex(h, s, l)
        elif scene_features['time_of_day'] == 'night':
            # Make clouds darker for night
            base_rgb = parse_hex_color(cloud_color)
            h, s, l = rgb_to_hsl(base_rgb)
            l = max(l - 0.3, 0.1)  # Darken significantly
            cloud_color = hsl_to_hex(h, s, l)
        
        # Create cloud generator function for reusability
        def create_cloud(x, y, size):
            # Create cloud shape using overlapping circles
            cloud = doc.create_group_element(f"cloud_{x}_{y}", {"class": "cloud"})
            
            # Base size parameters
            base_radius = 20 * size
            
            # Create semi-random cloud shape with multiple circles
            circles = [ \
                (x, y, base_radius),
                (x + base_radius * 0.8, y - base_radius * 0.2, base_radius * 0.9), \
                (x + base_radius * 1.6, y, base_radius * 0.8), \
                (x + base_radius * 0.4, y + base_radius * 0.3, base_radius * 0.7), \
                (x + base_radius * 1.2, y + base_radius * 0.3, base_radius * 0.7)
            ]
            
            # Create filter for cloud softness
            cloud_filter_id = f"cloud_filter_{x}_{y}"  # Unique ID
            cloud_filter = doc.create_element("filter", { \
                "id": cloud_filter_id,
                "x": "-20%", \
                "y": "-20%", \
                "width": "140%", \
                "height": "140%"
            })
            
            # Add blur effect for soft edges
            cloud_filter.append(doc.create_element("feGaussianBlur", { \
                "in": "SourceGraphic",
                "stdDeviation": f"{2 * size}"
            }))
            
            doc.add_definition(cloud_filter)
            
            # Add cloud fill circles with gradients
            for i, (cx, cy, r) in enumerate(circles):
                # Create subtle radial gradient for each circle
                circle_gradient_id = f"cloud_grad_{x}_{y}_{i}"
                circle_gradient = RadialGradient(circle_gradient_id, cx, cy, 0, cx, cy, r, precision)
                
                # Slightly brighten or darken parts for texture
                base_rgb = parse_hex_color(cloud_color)
                h, s, l = rgb_to_hsl(base_rgb)
                
                # Add highlight and shadow
                circle_gradient.add_stop(0, hsl_to_hex(h, max(s - 0.1, 0), min(l + 0.1, 0.95)))
                circle_gradient.add_stop(0.7, cloud_color)
                circle_gradient.add_stop(1, hsl_to_hex(h, min(s + 0.05, 1.0), max(l - 0.1, 0.05)))
                
                doc.add_definition(circle_gradient.to_svg_element())
                
                # Create the cloud circle
                circle = doc.create_element("circle", { \
                    "cx": round(cx, precision),
                    "cy": round(cy, precision), \
                    "r": round(r, precision), \
                    "fill": f"url(#{circle_gradient_id})", \
                    "filter": f"url(#{cloud_filter_id})"
                })
                
                cloud.append(circle)
            
            return cloud
        
        # Add cloud distributions based on weather type
        for i in range(cloud_count):
            # Determine cloud position parameters
            coverage_factor = 0.8 if scene_features['weather'] in ['cloudy', 'storm'] else 0.5
            
            # Distribute clouds with controlled randomness
            x_offset = ((i * 17) % 101) / 100.0
            x = width * (0.1 + x_offset * coverage_factor)
            
            y_offset = ((i * 23) % 53) / 100.0
            y = height * (0.1 + y_offset * 0.3)  # Keep clouds in upper portion
            
            # Size variation for natural look
            size_var = 0.8 + ((i * 13) % 41) / 100.0
            cloud_size_actual = cloud_size * size_var
            
            # Create and add cloud
            cloud = create_cloud(x, y, cloud_size_actual)
            cloud.set_attribute("opacity", str(round(cloud_opacity, 2)))
            
            cloud_group.append(cloud)
        
        # Add the cloud group to the document
        doc.add_element(cloud_group)
    
    # Add rain if weather is rain or storm
    if scene_features['weather'] in ['rain', 'storm']:
        # Create rain group
        rain_group = doc.create_group_element("rain", {"class": "rain-layer"})
        
        # Define rain properties based on weather intensity
        rain_color = "#E3F2FD" if scene_features['time_of_day'] != 'night' else "#90CAF9"
        rain_count = 80 if scene_features['weather'] == 'storm' else 50
        rain_opacity = 0.4 if scene_features['weather'] == 'storm' else 0.3
        
        # Create a symbol for rain drops to optimize SVG size
        rain_symbol_id = "rain_drop"
        rain_symbol = doc.create_element("symbol", {"id": rain_symbol_id, "viewBox": "0 0 4 20"})
        
        # Rain drop path is a simple vertical line with a round cap
        rain_path = doc.create_element("line", { \
            "x1": "2",
            "y1": "0", \
            "x2": "2", \
            "y2": "20", \
            "stroke": rain_color, \
            "stroke-width": "2", \
            "stroke-linecap": "round"
        })
        
        rain_symbol.append(rain_path)
        doc.add_definition(rain_symbol)
        
        # Add rain drops with controlled randomness
        for i in range(rain_count):
            # Distribute rain across width
            x_offset = ((i * 19) % 97) / 97.0
            x = width * x_offset
            
            # Distribute rain vertically, avoiding the very top
            y_offset = ((i * 23) % 89) / 89.0
            y = height * (0.2 + y_offset * 0.6)  # Keep rain in middle portion
            
            # Vary length/scale
            length_var = 0.7 + ((i * 11) % 61) / 100.0
            
            # Create rain drop using the symbol
            rain_drop = doc.create_element("use", { \
                "href": f"#{rain_symbol_id}",
                "x": round(x, precision), \
                "y": round(y, precision), \
                "opacity": round(rain_opacity * (0.7 + ((i * 7) % 31) / 100.0), 2),  # Vary opacity
                "transform": f"scale(1, {length_var})"
            })
            
            rain_group.append(rain_drop)
        
        # Add the rain group to the document
        doc.add_element(rain_group)
    
    # Add fog if weather is fog
    if scene_features['weather'] == 'fog':
        # Create fog effect using a semi-transparent gradient overlay
        fog_overlay = Rectangle(0, height * 0.3, width, height * 0.5)
        
        # Create fog gradient
        fog_gradient_id = "fog_gradient"
        fog_gradient = LinearGradient(fog_gradient_id, 0, height * 0.3, 0, height * 0.8, precision)
        
        # White with varying opacity
        fog_gradient.add_stop(0, "#FFFFFF")
        fog_gradient.add_stop(0.3, "#FFFFFF")
        fog_gradient.add_stop(1, "#FFFFFF00")  # Transparent at bottom
        
        doc.add_definition(fog_gradient.to_svg_element())
        
        # Create fog filter for softness
        fog_filter_id = "fog_filter"
        fog_filter = doc.create_element("filter", { \
            "id": fog_filter_id,
            "x": "-10%", \
            "y": "-10%", \
            "width": "120%", \
            "height": "120%"
        })
        
        # Add blur effect
        fog_filter.append(doc.create_element("feGaussianBlur", { \
            "in": "SourceGraphic",
            "stdDeviation": "15"
        }))
        
        doc.add_definition(fog_filter)
        
        # Set fog overlay attributes
        fog_attrs = { \
            "fill": f"url(#{fog_gradient_id})",
            "opacity": "0.7", \
            "filter": f"url(#{fog_filter_id})"
        }
        
        # Add fog to the document
        doc.add_element(fog_overlay.to_svg_element(fog_attrs))
def generate_background_terrain(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate background terrain elements like distant mountains, hills, etc.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Create a group for all background terrain elements
    bg_terrain_group = doc.create_group_element("background_terrain", {"class": "bg-terrain-layer"})
    
    # Generate mountains if in terrain features
    if 'mountains' in scene_features['terrain_features'] or scene_features['environment'] == 'mountains':
        # Create mountain range with parametric variation
        mountain_count = 5  # Number of major peaks
        
        # Define mountain range parameters based on scene features
        mountain_height_base = height * 0.3  # Base height of mountains
        valley_depth = 0.4  # How deep the valleys go (0-1)
        
        # Adjust based on time of day and mood for color
        if scene_features['time_of_day'] == 'sunset':
            mountain_color = "#5D4037"  # Brown with sunset tint
            shadow_color = "#3E2723"    # Dark brown shadow
        elif scene_features['time_of_day'] == 'night':
            mountain_color = "#263238"  # Very dark blue-gray
            shadow_color = "#0D1B1E"    # Nearly black shadow
        elif scene_features['time_of_day'] == 'dawn':
            mountain_color = "#5D4037"  # Brown with sunrise tint
            shadow_color = "#3E2723"    # Dark brown shadow
        else:  # day
            mountain_color = "#795548"  # Medium brown
            shadow_color = "#4E342E"    # Dark brown shadow
        
        # Create the mountain range path with controlled randomness
        path_data = f"M 0,{height * 0.7} "  # Start at the left edge at horizon level
        
        # Generate peaks and valleys with coherent noise
        for i in range(mountain_count * 2 + 1):
            # Calculate x position
            x_pos = width * i / (mountain_count * 2)
            
            # Calculate height with alternating peaks and valleys
            if i % 2 == 0:  # Valley
                # Valleys have varying depths
                depth_var = 0.5 + ((i * 23) % 51) / 100.0
                y_pos = height * (0.7 - mountain_height_base * valley_depth * depth_var / height)
            else:  # Peak
                # Peaks have varying heights
                height_var = 0.7 + ((i * 17) % 61) / 100.0
                y_pos = height * (0.7 - mountain_height_base * height_var / height)
            
            # Add a control point for smoother mountains
            if i > 0:
                # Calculate control point for smooth curve
                ctrl_x = (path_x + x_pos) / 2
                ctrl_y = path_y + (y_pos - path_y) * ((i % 2) * 0.2 + 0.4)  # Different for peaks vs valleys
                
                # Add quadratic Bézier curve for smoother mountains
                path_data += f"Q {round(ctrl_x, precision)},{round(ctrl_y, precision)} {round(x_pos, precision)},{round(y_pos, precision)} "
            
            # Store current point for next control point calculation
            path_x, path_y = x_pos, y_pos
        
        # Close the path by extending to the right edge and bottom
        path_data += f"L {width},{height * 0.7} L {width},{height} L 0,{height} Z"
        
        # Create mountain range path element
        mountain_path = doc.create_element("path", { \
            "d": path_data,
            "fill": mountain_color, \
            "class": "mountain-range"
        })
        
        # Add mountains to the group
        bg_terrain_group.append(mountain_path)
        
        # Add mountain shadows for more realistic effect
        if scene_features['time_of_day'] in ['sunset', 'dawn', 'day']:
            # Determine shadow direction based on time of day
            shadow_offset_x = -30 if scene_features['time_of_day'] == 'sunset' else 30  # Negative for sunset (west)
            
            # Create mountain shadow group
            mountain_shadow_group = doc.create_group_element("mountain_shadows", {"class": "mountain-shadows"})
            
            # Add subtle shadows for each major peak
            for i in range(1, mountain_count * 2, 2):  # Only for peaks (odd indices)
                # Calculate peak position
                x_pos = width * i / (mountain_count * 2)
                height_var = 0.7 + ((i * 17) % 61) / 100.0
                y_pos = height * (0.7 - mountain_height_base * height_var / height)
                
                # Create shadow path based on peak position
                shadow_path_data = f"M {round(x_pos - 15, precision)},{round(y_pos + 10, precision)} "
                shadow_path_data += f"Q {round(x_pos + shadow_offset_x, precision)},{round(y_pos + 40, precision)} "
                shadow_path_data += f"{round(x_pos + 50, precision)},{round(height * 0.7, precision)} "
                shadow_path_data += f"L {round(x_pos - 20, precision)},{round(height * 0.7, precision)} Z"
                
                # Create shadow path element
                shadow_path = doc.create_element("path", { \
                    "d": shadow_path_data,
                    "fill": shadow_color, \
                    "opacity": "0.3", \
                    "class": "mountain-shadow"
                })
                
                mountain_shadow_group.append(shadow_path)
            
            # Add shadows group to terrain group
            bg_terrain_group.append(mountain_shadow_group)
    
    # Generate hills if in terrain features or environment is fields/meadows
    if any(terrain in scene_features['terrain_features'] for terrain in ['hills', 'valley']) or \
        scene_features['environment'] in ['field', 'valley']:
        # Create background hills
        hill_count = 7  # Number of hills
        
        # Define hill parameters
        hill_height_base = height * 0.15  # Base height of hills
        
        # Determine hill color based on scene features
        if 'grass' in scene_features['terrain_features']:
            # Grassy hills
            if scene_features['season'] == 'fall':
                hill_color = "#D7CCC8"  # Brown-tinted for autumn
            elif scene_features['season'] == 'winter':
                hill_color = "#ECEFF1"  # Light blue-gray for winter
            elif scene_features['season'] == 'spring':
                hill_color = "#81C784"  # Vibrant green for spring
            else:  # summer or unspecified
                hill_color = "#66BB6A"  # Medium green for summer
        else:
            # Default hill color
            hill_color = "#8D6E63"  # Brown
        
        # Adjust color based on time of day
        if scene_features['time_of_day'] == 'night':
            # Darken for night
            base_rgb = parse_hex_color(hill_color)
            h, s, l = rgb_to_hsl(base_rgb)
            hill_color = hsl_to_hex(h, s, l * 0.5)  # Darken by reducing lightness
        
        # Create hill path with smooth curves
        for i in range(hill_count):
            # Vary hill position and size for natural look
            x_offset = ((i * 19) % 73) / 100.0
            width_var = 0.8 + ((i * 13) % 51) / 100.0
            height_var = 0.7 + ((i * 17) % 41) / 100.0
            
            # Calculate hill dimensions
            hill_x = width * (-0.2 + x_offset * 1.4)  # Spread across width with overlap
            hill_width = width * 0.3 * width_var
            hill_height = hill_height_base * height_var
            
            # Calculate hill top position (on horizon line)
            hill_y = height * 0.7 - hill_height
            
            # Create elliptical hill using arc
            hill_path = Path().M(hill_x, height * 0.7)\
                .A(hill_width/2, hill_height, 0, 0, 1, hill_x + hill_width, height * 0.7)\
                .L(hill_x + hill_width, height).L(hill_x, height).Z()
            
            # Set hill attributes including opacity based on distance (for depth)
            opacity = 0.8 - i * 0.1  # Further hills are fainter
            hill_attrs = { \
                "fill": hill_color,
                "opacity": str(max(0.3, opacity)), \
                "class": "background-hill"
            }
            
            # Add hill to terrain group
            bg_terrain_group.append(hill_path.to_svg_element(hill_attrs))
    
    # Add the background terrain group to the document
    doc.add_element(bg_terrain_group)

def generate_water_elements(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate water elements like ocean, lakes, rivers, etc.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Check if we need to generate water elements
    has_water = len(scene_features['water_features']) > 0 or scene_features['environment'] in ['ocean', 'lake']
    if not has_water:
        return  # No water to generate
    
    # Create a group for all water elements
    water_group = doc.create_group_element("water_elements", {"class": "water-layer"})
    
    # Determine water type and position
    if 'ocean' in scene_features['water_features'] or scene_features['environment'] == 'ocean':
        # Generate ocean
        # Determine ocean position (typically bottom half of the image)
        ocean_y = height * 0.65  # Position of the ocean horizon
        
        # Create ocean gradient based on time of day and weather
        ocean_gradient_id = "ocean_gradient"
        ocean_gradient = LinearGradient(ocean_gradient_id, 0, ocean_y, 0, height, precision)
        
        # Determine ocean colors based on time of day
        if scene_features['time_of_day'] == 'sunset':
            # Sunset-colored ocean with reflections
            ocean_gradient.add_stop(0, "#1565C0")  # Deep blue at horizon
            ocean_gradient.add_stop(0.3, "#5C6BC0")  # Purple-blue
            ocean_gradient.add_stop(0.6, "#E65100")  # Orange reflection
            ocean_gradient.add_stop(1, "#0D47A1")  # Dark blue at bottom
        elif scene_features['time_of_day'] == 'night':
            # Dark nighttime ocean
            ocean_gradient.add_stop(0, "#0D47A1")  # Deep blue at horizon
            ocean_gradient.add_stop(0.5, "#1A237E")  # Darker blue
            ocean_gradient.add_stop(1, "#111B30")  # Almost black at bottom
        elif scene_features['time_of_day'] == 'dawn':
            # Dawn-colored ocean with pink/gold reflections
            ocean_gradient.add_stop(0, "#1E88E5")  # Blue at horizon
            ocean_gradient.add_stop(0.3, "#7986CB")  # Purple-blue
            ocean_gradient.add_stop(0.6, "#FF9800")  # Orange/gold reflection
            ocean_gradient.add_stop(1, "#1565C0")  # Deeper blue at bottom
        else:  # day
            # Daytime ocean colors
            if scene_features['weather'] == 'clear':
                ocean_gradient.add_stop(0, "#039BE5")  # Light blue at horizon
                ocean_gradient.add_stop(0.5, "#0288D1")  # Medium blue
                ocean_gradient.add_stop(1, "#01579B")  # Deep blue at bottom
            else:  # cloudy or stormy
                ocean_gradient.add_stop(0, "#546E7A")  # Gray-blue at horizon
                ocean_gradient.add_stop(0.5, "#455A64")  # Darker gray-blue
                ocean_gradient.add_stop(1, "#263238")  # Very dark at bottom
        
        # Add ocean gradient to definitions
        doc.add_definition(ocean_gradient.to_svg_element())
        
        # Create ocean rectangle
        ocean_rect = Rectangle(0, ocean_y, width, height - ocean_y).to_svg_element({ \
            "fill": f"url(#{ocean_gradient_id})",
            "class": "ocean-water"
        })
        
        # Add ocean to water group
        water_group.append(ocean_rect)
        
        # Add subtle wave patterns to the ocean
        wave_group = doc.create_group_element("ocean_waves", {"class": "wave-patterns"})
        
        # Generate several wave lines
        wave_count = 8
        for i in range(wave_count):
            # Calculate wave y position
            wave_y = ocean_y + (height - ocean_y) * i / wave_count
            
            # Create wave path with gentle curves
            wave_path = "M 0," + str(round(wave_y, precision))
            
            # Number of wave segments
            segments = 8
            for j in range(1, segments + 1):
                # Calculate segment position
                x = width * j / segments
                
                # Add controlled randomness to wave height
                wave_height = 2 + ((i * j * 7) % 8)  # 2-10 pixels
                
                # Alternate direction of waves
                if j % 2 == 0:
                    y = wave_y + wave_height
                else:
                    y = wave_y - wave_height
                
                # Add smooth curve to wave
                control_x = x - width / segments / 2
                control_y = wave_y + (y - wave_y) / 2
                
                wave_path += f" Q{round(control_x, precision)},{round(control_y, precision)} "
                wave_path += f"{round(x, precision)},{round(y, precision)}"
            
            # Create wave path element
            opacity = 0.15 - i * 0.015  # Decreasing opacity for deeper waves
            wave_elem = doc.create_element("path", { \
                "d": wave_path,
                "fill": "none", \
                "stroke": "#FFFFFF", \
                "stroke-width": "1", \
                "opacity": str(max(0.03, opacity))
            })
            
            wave_group.append(wave_elem)
        
        # Add waves to water group
        water_group.append(wave_group)
        
        # Add reflections if applicable to time of day
        if scene_features['time_of_day'] in ['sunset', 'dawn', 'day'] and 'reflections' in scene_features['water_features']:
            # Create reflection mask for the sky
            reflection_mask_id = "reflection_mask"
            reflection_mask = doc.create_element("linearGradient", { \
                "id": reflection_mask_id,
                "x1": "0%", \
                "y1": "0%", \
                "x2": "0%", \
                "y2": "100%"
            })
            
            # Create gradient stops for the mask
            reflection_mask.append(doc.create_element("stop", { \
                "offset": "0%",
                "stop-color": "#FFFFFF", \
                "stop-opacity": "0.5"
            }))
            
            reflection_mask.append(doc.create_element("stop", { \
                "offset": "100%",
                "stop-color": "#FFFFFF", \
                "stop-opacity": "0"
            }))
            
            doc.add_definition(reflection_mask)
            
            # Create a reflection of the sky on the water
            reflection_height = (height - ocean_y) * 0.5
            reflection_rect = Rectangle(0, ocean_y, width, reflection_height).to_svg_element({ \
                "fill": "url(#sky_gradient)",  # Use the sky gradient
                "mask": f"url(#{reflection_mask_id})", \
                "opacity": "0.4", \
                "transform": f"scale(1, -0.5) translate(0, {-2 * (ocean_y + reflection_height)})"
            })
            
            water_group.append(reflection_rect)
    
    # Generate lake if specified
    elif 'lake' in scene_features['water_features']:
        # Determine lake position (typically in lower half)
        lake_y = height * 0.7  # Top of the lake
        
        # Calculate lake dimensions
        lake_width = width * 0.8
        lake_height = height * 0.25
        lake_x = (width - lake_width) / 2  # Center horizontally
        
        # Create elliptical lake
        lake_path = Path().M(lake_x, lake_y)\
            .A(lake_width/2, lake_height/2, 0, 0, 1, lake_x + lake_width, lake_y)\
            .A(lake_width/2, lake_height/2, 0, 0, 0, lake_x, lake_y)\
            .Z()
        
        # Create lake gradient based on time of day
        lake_gradient_id = "lake_gradient"
        lake_gradient = RadialGradient(lake_gradient_id, \
                                    lake_x + lake_width/2,
                                    lake_y + lake_height/4, \
                                    0, \
                                    lake_x + lake_width/2, \
                                    lake_y + lake_height/4, \
                                    lake_width/2, \
                                    precision)
        
        # Determine lake colors based on time of day
        if scene_features['time_of_day'] == 'sunset':
            lake_gradient.add_stop(0, "#5C6BC0")  # Purple-blue
            lake_gradient.add_stop(0.7, "#1565C0")  # Deeper blue
            lake_gradient.add_stop(1, "#0D47A1")  # Dark blue at edges
        elif scene_features['time_of_day'] == 'night':
            lake_gradient.add_stop(0, "#1A237E")  # Deep blue
            lake_gradient.add_stop(0.7, "#121858")  # Darker blue
            lake_gradient.add_stop(1, "#0D1337")  # Almost black at edges
        else:  # day or dawn
            lake_gradient.add_stop(0, "#64B5F6")  # Light blue
            lake_gradient.add_stop(0.7, "#1E88E5")  # Medium blue
            lake_gradient.add_stop(1, "#1565C0")  # Deeper blue at edges
        
        # Add lake gradient to definitions
        doc.add_definition(lake_gradient.to_svg_element())
        
        # Create lake shape with gradient fill
        lake_elem = lake_path.to_svg_element({ \
            "fill": f"url(#{lake_gradient_id})",
            "class": "lake-water"
        })
        
        # Add lake to water group
        water_group.append(lake_elem)
        
        # Add subtle ripples to the lake
        ripple_group = doc.create_group_element("lake_ripples", {"class": "ripple-patterns"})
        
        # Generate several elliptical ripples
        ripple_count = 5
        for i in range(ripple_count):
            # Calculate ripple dimensions (progressively smaller)
            ripple_width = lake_width * (0.9 - i * 0.15)
            ripple_height = lake_height * (0.9 - i * 0.15)
            ripple_x = lake_x + (lake_width - ripple_width) / 2
            ripple_y = lake_y + (lake_height - ripple_height) / 2
            
            # Create elliptical ripple
            ripple_path = Path().M(ripple_x, ripple_y + ripple_height/2)\
                .A(ripple_width/2, ripple_height/2, 0, 0, 1, ripple_x + ripple_width, ripple_y + ripple_height/2)\
                .A(ripple_width/2, ripple_height/2, 0, 0, 0, ripple_x, ripple_y + ripple_height/2)
            
            # Create ripple element
            opacity = 0.1 - i * 0.015  # Decreasing opacity for inner ripples
            ripple_elem = ripple_path.to_svg_element({ \
                "fill": "none",
                "stroke": "#FFFFFF", \
                "stroke-width": "1", \
                "opacity": str(max(0.03, opacity))
            })
            
            ripple_group.append(ripple_elem)
        
        # Add ripples to water group
        water_group.append(ripple_group)
    
    # Generate river if specified
    elif 'river' in scene_features['water_features']:
        # Determine river path (winding through the landscape)
        river_start_x = width * 0.2  # Start from left side
        river_end_x = width * 0.8  # End at right side
        river_y = height * 0.7  # Horizon level
        
        # Create meandering river path
        river_path = Path().M(river_start_x, height)
        
        # Number of control points for the river
        control_points = 5
        for i in range(control_points):
            # Calculate control point position
            x1 = river_start_x + (river_end_x - river_start_x) * (i / control_points)
            x2 = river_start_x + (river_end_x - river_start_x) * ((i + 1) / control_points)
            
            # Add controlled randomness to river flow
            y_offset = ((i * 17) % 41) / 100.0 * height * 0.1
            y1 = river_y + y_offset
            y2 = river_y + ((i+1) * 23 % 37) / 100.0 * height * 0.1
            
            # Control points for smoother curves
            cx = (x1 + x2) / 2
            cy = river_y + height * 0.15  # Below the river line
            
            # Add cubic Bézier curve segment
            river_path.C(x1, y1, cx, cy, x2, y2)
        
        # Complete the river path back to the bottom
        river_path.L(river_end_x, height).L(river_start_x, height).Z()
        
        # Create river gradient
        river_gradient_id = "river_gradient"
        river_gradient = LinearGradient(river_gradient_id, 0, river_y, 0, height, precision)
        
        # Determine river colors based on time of day
        if scene_features['time_of_day'] == 'sunset':
            river_gradient.add_stop(0, "#1976D2")  # Blue with sunset tint
            river_gradient.add_stop(1, "#0D47A1")  # Deep blue at bottom
        elif scene_features['time_of_day'] == 'night':
            river_gradient.add_stop(0, "#0D47A1")  # Deep blue
            river_gradient.add_stop(1, "#0E1F31")  # Very dark blue
        else:  # day or dawn
            river_gradient.add_stop(0, "#2196F3")  # Medium blue
            river_gradient.add_stop(1, "#1565C0")  # Deeper blue
        
        # Add river gradient to definitions
        doc.add_definition(river_gradient.to_svg_element())
        
        # Create river element with gradient fill
        river_elem = river_path.to_svg_element({ \
            "fill": f"url(#{river_gradient_id})",
            "class": "river-water"
        })
        
        # Add river to water group
        water_group.append(river_elem)
    
    # Add the water group to the document
    doc.add_element(water_group)

def generate_midground_elements(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate midground elements like distant landscape features.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Create a group for all midground elements
    mid_group = doc.create_group_element("midground_elements", {"class": "midground-layer"})
    
    # Add landscape elements based on environment
    if scene_features['environment'] == 'forest':
        # Add distant tree line
        tree_line_y = height * 0.65  # At the horizon line
        
        # Create a treeline silhouette using a series of overlapping ellipses
        tree_count = 15
        tree_group = doc.create_group_element("distant_trees", {"class": "tree-line"})
        
        # Determine tree color based on season and time of day
        if scene_features['season'] == 'fall':
            tree_color = "#795548"  # Brown for autumn
        elif scene_features['season'] == 'winter':
            tree_color = "#546E7A"  # Grayish for winter
        else:  # spring or summer
            tree_color = "#2E7D32"  # Deep green
        
        # Adjust color for time of day
        if scene_features['time_of_day'] == 'night':
            # Darken trees for night
            base_rgb = parse_hex_color(tree_color)
            h, s, l = rgb_to_hsl(base_rgb)
            tree_color = hsl_to_hex(h, s, max(l * 0.4, 0.1))  # Significantly darker
        
        # Create treeline
        for i in range(tree_count):
            # Calculate tree position with variations
            tree_width = width * 0.08 + ((i * 13) % 11) / 100.0 * width * 0.05
            tree_height = height * 0.12 + ((i * 17) % 13) / 100.0 * height * 0.08
            
            # Position trees along the horizon with slight variations
            tree_x = width * i / tree_count - tree_width / 4
            tree_y = tree_line_y
            
            # Create tree crown using ellipse
            tree_crown = Ellipse(tree_x + tree_width/2, tree_y - tree_height/2, \
                                tree_width/2, tree_height/2).to_svg_element({
                "fill": tree_color, \
                "stroke": "none", \
                "class": "tree-silhouette"
            })
            
            tree_group.append(tree_crown)
            
            # Add tree trunk (small rectangle)
            trunk_width = tree_width * 0.1
            trunk_height = tree_height * 0.4
            trunk = Rectangle(tree_x + tree_width/2 - trunk_width/2, \
                            tree_y - trunk_height/2,
                            trunk_width, trunk_height).to_svg_element({ \
                "fill": "#5D4037",  # Brown
                "stroke": "none", \
                "class": "tree-trunk"
            })
            
            tree_group.append(trunk)
        
        # Add treeline to midground group
        mid_group.append(tree_group)
    
    elif scene_features['environment'] == 'beach':
        # Generate beach shoreline
        shore_y = height * 0.7  # Horizon level
        
        # Create sandy beach shape
        beach_path = Path().M(0, shore_y)
        
        # Create wavy shoreline
        segments = 8
        for i in range(segments + 1):
            x = width * i / segments
            
            # Add controlled randomness to shoreline
            wave_height = 2 + ((i * 7) % 6)  # 2-8 pixels
            y = shore_y + (wave_height if i % 2 == 0 else -wave_height)
            
            # Add curve points
            if i == 0:
                beach_path.L(x, y)
            else:
                # Add control point for smooth curve
                ctrl_x = x - width / segments / 2
                ctrl_y = y_prev + (y - y_prev) / 2
                beach_path.Q(ctrl_x, ctrl_y, x, y)
            
            y_prev = y
        
        # Complete the beach shape
        beach_path.L(width, height).L(0, height).Z()
        
        # Create beach gradient
        beach_gradient_id = "beach_gradient"
        beach_gradient = LinearGradient(beach_gradient_id, 0, shore_y, 0, height, precision)
        
        # Beach colors based on time of day
        if scene_features['time_of_day'] == 'sunset':
            beach_gradient.add_stop(0, "#E0E0E0")  # Light sand at shore
            beach_gradient.add_stop(0.3, "#FFE0B2")  # Warm sand
            beach_gradient.add_stop(1, "#FFCC80")  # Deeper sand color
        elif scene_features['time_of_day'] == 'night':
            beach_gradient.add_stop(0, "#9E9E9E")  # Grayish sand at shore
            beach_gradient.add_stop(0.3, "#757575")  # Darker gray
            beach_gradient.add_stop(1, "#616161")  # Deep gray
        else:  # day or dawn
            beach_gradient.add_stop(0, "#FAFAFA")  # White sand at shore
            beach_gradient.add_stop(0.3, "#FFF8E1")  # Light sand
            beach_gradient.add_stop(1, "#FFE0B2")  # Deeper sand color
        
        # Add beach gradient to definitions
        doc.add_definition(beach_gradient.to_svg_element())
        
        # Create beach element with gradient fill
        beach_elem = beach_path.to_svg_element({ \
            "fill": f"url(#{beach_gradient_id})",
            "class": "beach-sand"
        })
        
        # Add beach to midground group
        mid_group.append(beach_elem)
    
    # Add the midground group to the document
    doc.add_element(mid_group)

def generate_foreground_terrain(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate foreground terrain elements like ground, rocks, etc.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Create a group for foreground terrain
    fg_terrain_group = doc.create_group_element("foreground_terrain", {"class": "fg-terrain-layer"})
    
    # Generate ground based on environment
    # Add a base ground element to cover the bottom portion of the image
    ground_y = height * 0.75  # Starting position of ground
    
    # Determine ground color based on environment and season
    if scene_features['environment'] == 'forest':
        if scene_features['season'] == 'fall':
            ground_color = "#A1887F"  # Brown for autumn forest floor
        elif scene_features['season'] == 'winter':
            ground_color = "#ECEFF1"  # Light gray for snowy ground
        elif scene_features['season'] == 'spring':
            ground_color = "#8D6E63"  # Medium brown with hints of green
        else:  # summer
            ground_color = "#795548"  # Rich brown for summer forest floor
    elif scene_features['environment'] == 'beach':
        ground_color = "#FFF8E1"  # Light sand color
    elif scene_features['environment'] == 'mountains':
        ground_color = "#6D4C41"  # Dark brown for mountain terrain
    elif scene_features['environment'] == 'field':
        if 'grass' in scene_features['terrain_features']:
            ground_color = "#66BB6A"  # Green for grassy field
        else:
            ground_color = "#8D6E63"  # Brown for plain field
    else:  # Default ground
        ground_color = "#8D6E63"  # Medium brown
    
    # Adjust color based on time of day
    if scene_features['time_of_day'] == 'night':
        # Darken for night
        base_rgb = parse_hex_color(ground_color)
        h, s, l = rgb_to_hsl(base_rgb)
        ground_color = hsl_to_hex(h, s, max(l * 0.5, 0.1))  # Darker
    
    # Create ground element
    ground_path = Path().M(0, ground_y)
    
    # Create undulating ground with several control points
    control_points = 8
    for i in range(1, control_points + 1):
        x = width * i / control_points
        
        # Add controlled randomness to ground height
        height_var = ((i * 13) % 21) / 100.0 * height * 0.05
        y = ground_y - height_var if i % 2 == 0 else ground_y + height_var
        
        # Add curve points with control points for smooth ground
        if i == 1:
            ground_path.L(x, y)
        else:
            # Calculate control point for smooth curve
            ctrl_x = x - width / control_points / 2
            ctrl_y = prev_y + (y - prev_y) / 2
            
            # Add quadratic Bézier curve for smooth ground
            ground_path.Q(ctrl_x, ctrl_y, x, y)
        
        prev_y = y
    
    # Complete the ground shape
    ground_path.L(width, height).L(0, height).Z()
    
    # Create ground gradient for depth
    ground_gradient_id = "ground_gradient"
    ground_gradient = LinearGradient(ground_gradient_id, 0, ground_y, 0, height, precision)
    
    # Add color stops for gradient
    base_rgb = parse_hex_color(ground_color)
    h, s, l = rgb_to_hsl(base_rgb)
    
    # Lighter at the top, darker at the bottom for depth
    ground_gradient.add_stop(0, ground_color)  # Base color at top
    ground_gradient.add_stop(1, hsl_to_hex(h, s, max(l * 0.7, 0.05)))  # Darker at bottom
    
    # Add ground gradient to definitions
    doc.add_definition(ground_gradient.to_svg_element())
    
    # Create ground element with gradient fill
    ground_elem = ground_path.to_svg_element({ \
        "fill": f"url(#{ground_gradient_id})",
        "class": "ground"
    })
    
    # Add ground to foreground terrain group
    fg_terrain_group.append(ground_elem)
    
    # Add rocks if present in terrain_features
    if 'rocks' in scene_features['terrain_features']:
        # Create a rock group
        rock_group = doc.create_group_element("rocks", {"class": "rock-group"})
        
        # Generate several rocks with parametric variation
        rock_count = 8
        for i in range(rock_count):
            # Calculate rock position with variations
            x_offset = ((i * 17) % 97) / 97.0
            x = width * (0.1 + x_offset * 0.8)  # Spread across width
            
            # Determine rock's y position (on ground with variations)
            y_base = ground_y + ((i * 13) % 51) / 100.0 * height * 0.15  # Varying depths
            
            # Determine rock size with controlled randomness
            size_var = 0.5 + ((i * 19) % 71) / 100.0  # 0.5 to 1.2 size variation
            width_var = ((i * 23) % 41) / 100.0 * 0.3 + 0.7  # Width variation factor
            
            rock_width = width * 0.05 * size_var * width_var
            rock_height = height * 0.04 * size_var
            
            # Create rock shape as irregular polygon
            pts = []
            vertices = 6 + i % 4  # 6-9 vertices for varied shapes
            
            for j in range(vertices):
                # Calculate vertex position on ellipse with noise
                angle = 2 * math.pi * j / vertices
                radius_var = 0.8 + ((i * j * 11) % 41) / 100.0  # Radius variation
                
                pt_x = x + rock_width/2 * math.cos(angle) * radius_var
                pt_y = y_base - rock_height/2 * math.sin(angle) * radius_var
                
                pts.append((round(pt_x, precision), round(pt_y, precision)))
            
            # Create rock path
            rock_path = "M " + " L ".join([f"{x},{y}" for x, y in pts]) + " Z"
            
            # Determine rock color based on environment and lighting
            if scene_features['environment'] == 'beach':
                rock_color = "#9E9E9E"  # Gray
            elif scene_features['environment'] == 'mountains':
                rock_color = "#616161"  # Darker gray
            else:
                rock_color = "#757575"  # Medium gray
            
            # Adjust color based on time of day
            if scene_features['time_of_day'] == 'night':
                base_rgb = parse_hex_color(rock_color)
                h, s, l = rgb_to_hsl(base_rgb)
                rock_color = hsl_to_hex(h, s, max(l * 0.7, 0.1))  # Darker for night
            
            # Create rock element
            rock_elem = doc.create_element("path", { \
                "d": rock_path,
                "fill": rock_color, \
                "stroke": hsl_to_hex(*rgb_to_hsl(parse_hex_color(rock_color)), max(l * 0.5, 0.05)), \
                "stroke-width": "1", \
                "class": "rock"
            })
            
            rock_group.append(rock_elem)
        
        # Add rocks to foreground terrain group
        fg_terrain_group.append(rock_group)
    
    # Add the foreground terrain group to the document
    doc.add_element(fg_terrain_group)

def generate_vegetation(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate vegetation elements like trees, flowers, etc.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Check if we need to generate vegetation
    has_vegetation = len(scene_features['vegetation']) > 0
    if not has_vegetation:
        return  # No vegetation to generate
    
    # Create a group for all vegetation elements
    vegetation_group = doc.create_group_element("vegetation_elements", {"class": "vegetation-layer"})
    
    # Generate trees if present in vegetation list
    if 'trees' in scene_features['vegetation']:
        # Create tree group
        tree_group = doc.create_group_element("trees", {"class": "tree-group"})
        
        # Determine tree type and color based on environment and season
        if scene_features['environment'] == 'forest':
            # Dense forest trees
            if scene_features['season'] == 'fall':
                leaf_color = "#FF9800"  # Orange for autumn
                trunk_color = "#5D4037"  # Brown
            elif scene_features['season'] == 'winter':
                leaf_color = "#ECEFF1"  # Light gray (snow-covered or bare)
                trunk_color = "#5D4037"  # Brown
            elif scene_features['season'] == 'spring':
                leaf_color = "#81C784"  # Light green for spring
                trunk_color = "#5D4037"  # Brown
            else:  # summer
                leaf_color = "#2E7D32"  # Deep green for summer
                trunk_color = "#5D4037"  # Brown
        elif scene_features['environment'] == 'beach':
            # Palm trees
            leaf_color = "#43A047"  # Green for palm fronds
            trunk_color = "#8D6E63"  # Light brown for palm trunk
        else:  # Default trees
            leaf_color = "#388E3C"  # Medium green
            trunk_color = "#5D4037"  # Brown
        
        # Adjust color based on time of day
        if scene_features['time_of_day'] == 'night':
            # Darken colors for night
            leaf_rgb = parse_hex_color(leaf_color)
            leaf_h, leaf_s, leaf_l = rgb_to_hsl(leaf_rgb)
            leaf_color = hsl_to_hex(leaf_h, leaf_s, max(leaf_l * 0.4, 0.1))  # Significantly darker
            
            trunk_rgb = parse_hex_color(trunk_color)
            trunk_h, trunk_s, trunk_l = rgb_to_hsl(trunk_rgb)
            trunk_color = hsl_to_hex(trunk_h, trunk_s, max(trunk_l * 0.6, 0.1))  # Darker
        
        # Generate trees with parametric variation
        tree_count = 5
        for i in range(tree_count):
            # Calculate tree position with variations
            x_offset = ((i * 19) % 83) / 83.0
            x = width * (0.1 + x_offset * 0.8)  # Spread across width
            
            # Position trees in foreground
            y_base = height * (0.75 + ((i * 13) % 31) / 100.0 * 0.1)  # Varying positions near ground
            
            # Determine tree size with controlled randomness
            size_var = 0.8 + ((i * 17) % 41) / 100.0  # Size variation
            
            # Create tree based on environment
            if scene_features['environment'] == 'beach' or 'palm' in scene_features['vegetation']:
                # Create palm tree
                # Trunk
                trunk_width = width * 0.015 * size_var
                trunk_height = height * 0.15 * size_var
                trunk = Rectangle(x - trunk_width/2, y_base - trunk_height, trunk_width, trunk_height).to_svg_element({ \
                    "fill": trunk_color,
                    "stroke": "none", \
                    "class": "palm-trunk"
                })
                
                # Palm fronds as simplified fan
                frond_group = doc.create_group_element(f"palm_fronds_{i}", {"class": "palm-fronds"})
                
                frond_count = 7
                for j in range(frond_count):
                    # Calculate frond angle and length
                    angle = (j * 45 - 135) * math.pi / 180  # Spread in an arc
                    length_var = 0.7 + ((i * j * 11) % 61) / 100.0
                    frond_length = height * 0.12 * size_var * length_var
                    
                    # Calculate frond end point
                    end_x = x + math.cos(angle) * frond_length
                    end_y = y_base - trunk_height + math.sin(angle) * frond_length * 0.5
                    
                    # Create frond as curved path
                    ctrl_x = x + math.cos(angle) * frond_length * 0.5
                    ctrl_y = y_base - trunk_height + math.sin(angle) * frond_length * 0.25
                    
                    frond_path = Path().M(x, y_base - trunk_height).Q(ctrl_x, ctrl_y, end_x, end_y)  # Quadratic curve for bend
                    
                    # Create frond element
                    frond_elem = frond_path.to_svg_element({ \
                        "fill": "none",
                        "stroke": leaf_color, \
                        "stroke-width": str(3 + j % 3), \
                        "stroke-linecap": "round", \
                        "class": "palm-frond"
                    })
                    
                    frond_group.append(frond_elem)
                
                # Add trunk and fronds to tree group
                tree_element = doc.create_group_element(f"palm_tree_{i}", {"class": "palm-tree"})
                tree_element.append(trunk)
                tree_element.append(frond_group)
                
                tree_group.append(tree_element)
            else:
                # Create standard tree
                # Trunk
                trunk_width = width * 0.02 * size_var
                trunk_height = height * 0.12 * size_var
                trunk = Rectangle(x - trunk_width/2, y_base - trunk_height, trunk_width, trunk_height).to_svg_element({ \
                    "fill": trunk_color,
                    "stroke": "none", \
                    "class": "tree-trunk"
                })
                
                # Crown as overlapping ellipses
                crown_group = doc.create_group_element(f"tree_crown_{i}", {"class": "tree-crown"})
                
                # Create multiple ellipses for crown with slight offsets
                crown_width = width * 0.08 * size_var
                crown_height = height * 0.1 * size_var
                crown_base_y = y_base - trunk_height
                
                for j in range(3):  # Multiple layers of foliage
                    # Create offset for crown layers
                    offset_x = ((i * j * 7) % 31 - 15) / 100.0 * crown_width
                    offset_y = -j * crown_height * 0.3  # Stack vertically
                    
                    # Vary size slightly for each layer
                    size_factor = 0.9 - j * 0.1
                    layer_width = crown_width * size_factor
                    layer_height = crown_height * size_factor
                    
                    # Create crown layer
                    crown_layer = Ellipse( \
                        x + offset_x,
                        crown_base_y + offset_y, \
                        layer_width/2, \
                        layer_height/2
                    ).to_svg_element({ \
                        "fill": leaf_color,
                        "stroke": "none", \
                        "class": "tree-crown-layer"
                    })
                    
                    crown_group.append(crown_layer)
                
                # Add trunk and crown to tree group
                tree_element = doc.create_group_element(f"tree_{i}", {"class": "standard-tree"})
                tree_element.append(trunk)
                tree_element.append(crown_group)
                
                tree_group.append(tree_element)
        
        # Add tree group to vegetation group
        vegetation_group.append(tree_group)
    
    # Generate flowers if present in vegetation list
    if 'flowers' in scene_features['vegetation']:
        # Create flower group
        flower_group = doc.create_group_element("flowers", {"class": "flower-group"})
        
        # Determine flower colors based on scene features
        # Generate a small palette of flower colors
        flower_colors = []
        
        # Add specific colors if mentioned in the scene features
        if 'colors' in scene_features and scene_features['colors']:
            for color in scene_features['colors']:
                if color == 'red':
                    flower_colors.append("#E57373")  # Light red
                elif color == 'blue':
                    flower_colors.append("#64B5F6")  # Light blue
                elif color == 'yellow':
                    flower_colors.append("#FFF176")  # Light yellow
                elif color == 'purple':
                    flower_colors.append("#BA68C8")  # Light purple
                elif color == 'pink':
                    flower_colors.append("#F06292")  # Pink
                elif color == 'white':
                    flower_colors.append("#F5F5F5")  # White
        
        # If no specific colors, use default palette
        if not flower_colors:
            flower_colors = ["#E57373", "#64B5F6", "#FFF176", "#BA68C8", "#F06292", "#F5F5F5"]
        
        # Adjust colors based on time of day
        if scene_features['time_of_day'] == 'night':
            # Darken colors for night
            for i in range(len(flower_colors)):
                color = flower_colors[i]
                color_rgb = parse_hex_color(color)
                h, s, l = rgb_to_hsl(color_rgb)
                flower_colors[i] = hsl_to_hex(h, s, max(l * 0.5, 0.1))  # Darker
        
        # Create stem and leaf colors
        stem_color = "#388E3C"  # Green
        
        # Adjust for night
        if scene_features['time_of_day'] == 'night':
            stem_rgb = parse_hex_color(stem_color)
            h, s, l = rgb_to_hsl(stem_rgb)
            stem_color = hsl_to_hex(h, s, max(l * 0.4, 0.1))  # Darker
        
        # Generate flowers with parametric variation
        flower_count = 20
        for i in range(flower_count):
            # Calculate flower position with controlled randomness
            x_offset = ((i * 23) % 97) / 97.0
            x = width * (0.05 + x_offset * 0.9)  # Spread across width
            
            # Position flowers in foreground
            y_offset = ((i * 19) % 73) / 73.0
            y = height * (0.75 + y_offset * 0.15)  # Varying positions near ground
            
            # Determine flower size
            size_var = 0.5 + ((i * 13) % 51) / 100.0  # Size variation
            flower_size = width * 0.01 * size_var
            
            # Select flower color randomly from palette
            color_index = i % len(flower_colors)
            flower_color = flower_colors[color_index]
            
            # Create flower group
            flower_element = doc.create_group_element(f"flower_{i}", {"class": "flower"})
            
            # Create stem
            stem_height = height * 0.03 * (0.7 + ((i * 7) % 31) / 100.0)  # Varying heights
            stem = doc.create_element("line", { \
                "x1": round(x, precision),
                "y1": round(y, precision), \
                "x2": round(x, precision), \
                "y2": round(y - stem_height, precision), \
                "stroke": stem_color, \
                "stroke-width": "1", \
                "class": "flower-stem"
            })
            
            flower_element.append(stem)
            
            # Create flower based on index for variety
            if i % 3 == 0:
                # Simple circle flower
                flower_head = doc.create_element("circle", { \
                    "cx": round(x, precision),
                    "cy": round(y - stem_height, precision), \
                    "r": round(flower_size, precision), \
                    "fill": flower_color, \
                    "class": "flower-head"
                })
                
                # Add center
                flower_center = doc.create_element("circle", { \
                    "cx": round(x, precision),
                    "cy": round(y - stem_height, precision), \
                    "r": round(flower_size * 0.3, precision), \
                    "fill": "#FBC02D",  # Yellow center
                    "class": "flower-center"
                })
                
                flower_element.append(flower_head)
                flower_element.append(flower_center)
            elif i % 3 == 1:
                # Petal flower (simple petals)
                petal_count = 5
                for j in range(petal_count):
                    # Calculate petal position
                    angle = j * (2 * math.pi / petal_count)
                    petal_x = x + math.cos(angle) * flower_size
                    petal_y = y - stem_height + math.sin(angle) * flower_size
                    
                    # Create petal
                    petal = doc.create_element("circle", { \
                        "cx": round(petal_x, precision),
                        "cy": round(petal_y, precision), \
                        "r": round(flower_size * 0.7, precision), \
                        "fill": flower_color, \
                        "class": "flower-petal"
                    })
                    
                    flower_element.append(petal)
                
                # Add center
                flower_center = doc.create_element("circle", { \
                    "cx": round(x, precision),
                    "cy": round(y - stem_height, precision), \
                    "r": round(flower_size * 0.5, precision), \
                    "fill": "#FBC02D",  # Yellow center
                    "class": "flower-center"
                })
                
                flower_element.append(flower_center)
            else:
                # Star-shaped flower
                points = []
                point_count = 8
                
                for j in range(point_count * 2):
                    # Alternate inner and outer points
                    radius = flower_size if j % 2 == 0 else flower_size * 0.4
                    angle = j * math.pi / point_count
                    
                    pt_x = x + math.cos(angle) * radius
                    pt_y = (y - stem_height) + math.sin(angle) * radius
                    
                    points.append((round(pt_x, precision), round(pt_y, precision)))
                
                # Create star path
                star_path = "M " + " L ".join([f"{x},{y}" for x, y in points]) + " Z"
                
                # Create flower element
                flower_head = doc.create_element("path", { \
                    "d": star_path,
                    "fill": flower_color, \
                    "class": "flower-head"
                })
                
                # Add center
                flower_center = doc.create_element("circle", { \
                    "cx": round(x, precision),
                    "cy": round(y - stem_height, precision), \
                    "r": round(flower_size * 0.3, precision), \
                    "fill": "#FBC02D",  # Yellow center
                    "class": "flower-center"
                })
                
                flower_element.append(flower_head)
                flower_element.append(flower_center)
            
            # Add flower element to group
            flower_group.append(flower_element)
        
        # Add flower group to vegetation group
        vegetation_group.append(flower_group)
    
    # Add the vegetation group to the document
    doc.add_element(vegetation_group)

def generate_built_structures(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate built structures like buildings, bridges, etc.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Check if we need to generate built structures
    has_buildings = len(scene_features['built_features']) > 0
    if not has_buildings:
        return  # No structures to generate
    
    # Create a group for all built elements
    built_group = doc.create_group_element("built_structures", {"class": "built-layer"})
    
    # Generate buildings if present in built features
    if any(feature in scene_features['built_features'] for feature in ['buildings', 'city', 'skyscrapers']):
        # Create buildings group
        buildings_group = doc.create_group_element("buildings", {"class": "building-group"})
        
        # Determine building parameters based on features
        is_city = 'city' in scene_features['built_features']
        is_skyscrapers = 'skyscrapers' in scene_features['built_features']
        
        # Building count and positioning
        building_count = 12 if is_city else 6
        horizon_y = height * 0.7  # Horizon level
        skyline_width = width * (0.8 if is_city else 0.6)
        skyline_start_x = (width - skyline_width) / 2
        
        # Determine building colors based on time of day and materials
        if 'metal' in scene_features['materials'] or 'glass' in scene_features['materials']:
            # Modern/glass buildings
            if scene_features['time_of_day'] == 'night':
                # Night: dark with lit windows
                building_colors = ["#263238", "#1A237E", "#1B1F30"]
                window_opacity = 0.9  # Brighter windows at night
            elif scene_features['time_of_day'] == 'sunset':
                # Sunset: buildings with orange reflection
                building_colors = ["#455A64", "#37474F", "#5D4037"]
                window_opacity = 0.6
            else:  # day or dawn
                # Day: reflective glass buildings
                building_colors = ["#546E7A", "#607D8B", "#78909C"]
                window_opacity = 0.4
        else:
            # Standard buildings
            if scene_features['time_of_day'] == 'night':
                building_colors = ["#212121", "#263238", "#1F1F1F"]
                window_opacity = 0.9
            elif scene_features['time_of_day'] == 'sunset':
                building_colors = ["#4E342E", "#5D4037", "#6D4C41"]
                window_opacity = 0.6
            else:  # day or dawn
                building_colors = ["#616161", "#757575", "#9E9E9E"]
                window_opacity = 0.4
        
        # Generate buildings with parametric variation
        for i in range(building_count):
            # Calculate building position along skyline
            x_offset = i / building_count
            building_x = skyline_start_x + x_offset * skyline_width
            
            # Determine building height with variations
            if is_skyscrapers:
                # Taller buildings for skyscrapers
                building_height = random.uniform(height * 0.4, height * 0.6)
            else:
                # Shorter buildings for regular skyline
                building_height = random.uniform(height * 0.2, height * 0.4)
            
            # Position buildings with some randomness but mainly spaced out
            buffer = width * 0.05  # 5% buffer on edges
            available_width = width - 2 * buffer
            segment_width = available_width / building_count
            base_x = buffer + i * segment_width + (segment_width - building_width) / 2
            x_jitter = segment_width * 0.1 * random.uniform(-1, 1)  # Add slight randomness
            x_pos = base_x + x_jitter
            
            # Buildings sit on the ground
            y_pos = height - building_height - height * 0.05  # Slight offset from bottom
            
            # Create the building node with properties
            building_properties = { \
                "style": "architectural_style",
                "window_pattern": random.choice(["grid", "stripe"]), \
                "time_of_day": scene_features['time_of_day'], \
                "detail_level": random.uniform(0.6, 1.0), \
                "floors": max(2, int(building_height / 40))
            }
            
            # Add materials if specified
            if 'materials' in scene_features:
                building_properties["materials"] = scene_features['materials']
                # Generate color based on material for the first material
                if scene_features['materials'][0]:
                    building_properties["color"] = self._generate_material_color(scene_features['materials'][0])
            
            # Add architectural features
            if 'architectural_features' in scene_features:
                building_properties["architectural_features"] = scene_features['architectural_features']
                
                # Add some variation to buildings by randomly removing some features
                # so not all buildings look identical
                if len(scene_features['architectural_features']) > 1 and random.random() > 0.7:  # 30% chance to vary
                    # Copy list to avoid modifying the original
                    varied_features = scene_features['architectural_features'].copy()
                    # Remove a random feature, but keep at least one
                    features_to_remove = random.randint(1, min(len(varied_features) - 1, 2))
                    for _ in range(features_to_remove):
                        if len(varied_features) > 1:  # Always keep at least one feature
                            varied_features.remove(random.choice(varied_features))
                    building_properties["architectural_features"] = varied_features
            
            building = scene_graph.create_node( \
                "building",
                SceneNodeBounds(x_pos, y_pos, building_width, building_height), \
                building_properties
            )
            
            # Add building to group
            buildings_group.append(building)
            
            window_spacing_x = window_width * 1.5
            window_spacing_y = window_height * 1.5
            
            # Create windows pattern
            for row in range(min(window_rows, 15)):  # Limit max rows to avoid too many elements
                for col in range(min(window_cols, 8)):  # Limit max columns
                    # Calculate window position
                    window_x = building_x + (col + 0.5) * window_spacing_x
                    window_y = horizon_y - building_height + (row + 0.5) * window_spacing_y
                    
                    # Some randomness to window lighting
                    is_lit = (row * col + i) % 3 != 0 if scene_features['time_of_day'] == 'night' else (row * col) % 5 == 0
                    window_color = "#FFECB3" if is_lit else "#E0E0E0"
                    opacity = window_opacity if is_lit else 0.3
                    
                    # Create window element
                    window = Rectangle( \
                        window_x,
                        window_y, \
                        window_width, \
                        window_height
                    ).to_svg_element({ \
                        "fill": window_color,
                        "opacity": str(opacity), \
                        "class": "window"
                    })
                    
                    window_group.append(window)
            
            # Add windows to building group
            buildings_group.append(window_group)
            
            # Store building info for next building
            prev_building_x = building_x
            prev_building_width = building_width
        
        # Add buildings to built structures group
        built_group.append(buildings_group)
    
    # Generate bridge if present in built features
    if 'bridge' in scene_features['built_features'] and any(feature in scene_features['water_features'] for feature in ['river', 'lake', 'ocean']):
        # Create bridge
        horizon_y = height * 0.7  # Horizon level
        bridge_width = width * 0.25
        bridge_x = (width - bridge_width) / 2
        bridge_height = height * 0.01  # Thickness
        
        # Determine bridge y-position based on water features
        if 'river' in scene_features['water_features']:
            bridge_y = horizon_y + height * 0.1  # Lower over river
        else:
            bridge_y = horizon_y + height * 0.05  # Higher over lake/ocean
        
        # Create bridge group
        bridge_group = doc.create_group_element("bridge", {"class": "bridge-structure"})
        
        # Bridge deck (main horizontal part)
        bridge_deck = Rectangle( \
            bridge_x,
            bridge_y, \
            bridge_width, \
            bridge_height
        ).to_svg_element({ \
            "fill": "#78909C",  # Gray
            "stroke": "#455A64", \
            "stroke-width": "0.5", \
            "class": "bridge-deck"
        })
        
        bridge_group.append(bridge_deck)
        
        # Bridge supports
        support_count = 3
        for i in range(support_count):
            support_x = bridge_x + bridge_width * i / (support_count - 1)
            support_width = bridge_width * 0.02
            support_height = height * 0.1
            
            support = Rectangle( \
                support_x - support_width / 2,
                bridge_y, \
                support_width, \
                support_height
            ).to_svg_element({ \
                "fill": "#546E7A",  # Darker gray
                "stroke": "#37474F", \
                "stroke-width": "0.5", \
                "class": "bridge-support"
            })
            
            bridge_group.append(support)
        
        # Add bridge railing
        railing_height = height * 0.01
        
        railing_top = Rectangle( \
            bridge_x,
            bridge_y - railing_height, \
            bridge_width, \
            railing_height / 2
        ).to_svg_element({ \
            "fill": "#455A64",  # Dark gray
            "class": "bridge-railing"
        })
        
        bridge_group.append(railing_top)
        
        # Add bridge to built structures group
        built_group.append(bridge_group)
    
    # Add built structures group to document
    doc.add_element(built_group)

def generate_foreground_details(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int) -> None:
    """
    Generate small foreground details for added realism.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    """
    # Create a group for foreground details
    details_group = doc.create_group_element("foreground_details", {"class": "details-layer"})
    
    # Add small rocks or pebbles in foreground
    if 'rocks' in scene_features['terrain_features'] or scene_features['environment'] in ['beach', 'mountains']:
        pebble_group = doc.create_group_element("pebbles", {"class": "pebble-group"})
        
        # Determine pebble color based on environment
        if scene_features['environment'] == 'beach':
            pebble_color = "#E0E0E0"  # Light gray for beach pebbles
        else:
            pebble_color = "#9E9E9E"  # Medium gray for standard pebbles
        
        # Adjust for night
        if scene_features['time_of_day'] == 'night':
            pebble_rgb = parse_hex_color(pebble_color)
            h, s, l = rgb_to_hsl(pebble_rgb)
            pebble_color = hsl_to_hex(h, s, max(l * 0.6, 0.1))  # Darker
        
        # Generate pebbles with parametric distribution
        pebble_count = 30
        for i in range(pebble_count):
            # Calculate pebble position with controlled randomness
            x_offset = ((i * 29) % 101) / 101.0
            x = width * (0.1 + x_offset * 0.8)  # Spread across width
            
            # Position pebbles in extreme foreground
            y_offset = ((i * 23) % 53) / 53.0
            y = height * (0.85 + y_offset * 0.12)  # Near bottom of image
            
            # Determine pebble size with variations
            size_var = 0.5 + ((i * 17) % 51) / 100.0  # Size variation
            pebble_radius = width * 0.005 * size_var
            
            # Create pebble as small ellipse
            pebble = Ellipse(x, y, pebble_radius, pebble_radius * 0.7).to_svg_element({ \
                "fill": pebble_color,
                "class": "pebble"
            })
            
            pebble_group.append(pebble)
        
        # Add pebbles to details group
        details_group.append(pebble_group)
    
    # Add footprints or other human traces if in scene features
    if 'footprints' in scene_features.get('details', []) or 'path' in scene_features['terrain_features']:
        footprint_group = doc.create_group_element("footprints", {"class": "human-trace"})
        
        # Create a meandering path in the foreground
        path_y = height * 0.85  # Path in foreground
        path_width = width * 0.05
        
        # Create curved path
        footpath = Path(precision)
        
        # Start path from left side
        footpath.move_to(width * 0.1, path_y)
        
        # Create a natural-looking path with curves
        footpath.curve_to( \
            width * 0.3, path_y - height * 0.02,  # First control point
            width * 0.5, path_y + height * 0.01,  # Second control point
            width * 0.7, path_y - height * 0.015  # End point
        )
        
        footpath.curve_to( \
            width * 0.8, path_y - height * 0.025,  # First control point
            width * 0.85, path_y,                 # Second control point
            width * 0.9, path_y + height * 0.01   # End point
        )
        
        # Create path element
        footpath_elem = footpath.to_svg_element({ \
            "fill": "none",
            "stroke": "#A1887F",  # Light brown
            "stroke-width": str(path_width), \
            "stroke-linecap": "round", \
            "stroke-opacity": "0.5", \
            "class": "footpath"
        })
        
        footprint_group.append(footpath_elem)
        details_group.append(footprint_group)
    
    # Add the details group to the document
    doc.add_element(details_group)

def apply_global_lighting(doc: SVGDocument, scene_features: Dict, width: int, height: int, precision: int, optimizer: SVGOptimizer) -> None:
    """
    Apply global lighting effects to enhance realism.
    
    Args:
        doc: SVG document to populate
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
        optimizer: SVG optimizer instance for compression techniques
    """
    # Create lighting effects based on time of day and mood
    if scene_features['time_of_day'] in ['sunset', 'dawn']:
        # Add warm color overlay for sunset/dawn
        color = "#FF9800" if scene_features['time_of_day'] == 'sunset' else "#FFCC80"  # Orange for sunset, lighter for dawn
        opacity = 0.15 if scene_features['time_of_day'] == 'sunset' else 0.1  # Stronger for sunset
        
        # Create overlay
        overlay = Rectangle(0, 0, width, height).to_svg_element({ \
            "fill": color,
            "opacity": str(opacity), \
            "class": "lighting-overlay"
        })
        
        doc.add_element(overlay)
        
        # Add light rays from sun/horizon for dramatic effect
        if scene_features.get('mood') == 'dramatic':
            # Create light rays group
            rays_group = doc.create_group_element("light_rays", {"class": "sun-rays"})
            
            # Determine sun position (general area)
            sun_x = width * (0.2 if scene_features['time_of_day'] == 'dawn' else 0.8)
            sun_y = height * 0.3
            
            # Create rays emanating from sun
            ray_count = 6
            for i in range(ray_count):
                # Calculate ray angle
                angle = math.pi/4 + i * math.pi/6  # Spread rays downward
                
                # Determine ray length and width with variations
                length_var = 0.7 + ((i * 19) % 51) / 100.0
                ray_length = height * 0.6 * length_var
                
                # Calculate ray end point
                end_x = sun_x + math.cos(angle) * ray_length
                end_y = sun_y + math.sin(angle) * ray_length
                
                # Create ray path
                ray_width = width * 0.03 * (0.6 + ((i * 13) % 51) / 100.0)  # Varying widths
                
                # Create ray as triangle
                ray_path = Path(precision).M(sun_x, sun_y)\
                    .L(sun_x + math.cos(angle - 0.05) * ray_length, sun_y + math.sin(angle - 0.05) * ray_length)\
                    .L(sun_x + math.cos(angle + 0.05) * ray_length, sun_y + math.sin(angle + 0.05) * ray_length)\
                    .Z()
                
                # Create ray element
                ray_color = "#FFCC80" if scene_features['time_of_day'] == 'sunset' else "#FFF9C4"  # Orange for sunset, yellow for dawn
                ray_elem = ray_path.to_svg_element({ \
                    "fill": ray_color,
                    "opacity": "0.3", \
                    "class": "sun-ray"
                })
                
                rays_group.append(ray_elem)
            
            # Add rays to document
            doc.add_element(rays_group)
    
    elif scene_features['time_of_day'] == 'night':
        # Add dark blue overlay for night
        night_overlay = Rectangle(0, 0, width, height).to_svg_element({ \
            "fill": "#0D47A1",
            "opacity": "0.2", \
            "class": "night-overlay"
        })
        
        doc.add_element(night_overlay)
        
        # Add vignette effect for night scenes
        vignette_id = "vignette_gradient"
        vignette_gradient = RadialGradient(vignette_id, width/2, height/2, 0, width/2, height/2, width, precision)
        vignette_gradient.add_stop(0, "#00000000")  # Transparent in center
        vignette_gradient.add_stop(0.7, "#00000000")  # Transparent at 70% \
        vignette_gradient.add_stop(1, "#000000")  # Black at edges
        
        doc.add_definition(vignette_gradient.to_svg_element())
        
        vignette = Rectangle(0, 0, width, height).to_svg_element({ \
            "fill": f"url(#{vignette_id})",
            "opacity": "0.7", \
            "class": "vignette-effect"
        })
        
        doc.add_element(vignette)
    
    # Apply atmospheric perspective for depth
    if scene_features['environment'] in ['mountains', 'forest', 'valley'] and scene_features.get('perspective') != 'closeup':
        # Add atmospheric haze in the distance
        atmo_id = "atmospheric_haze"
        atmo_gradient = LinearGradient(atmo_id, 0, 0, 0, height * 0.7, precision)
        
        # Determine haze color based on time of day
        if scene_features['time_of_day'] == 'sunset':
            atmo_gradient.add_stop(0, "#FF9800")  # Orange at top
            atmo_gradient.add_stop(1, "#FFFFFF00")  # Transparent at bottom
        elif scene_features['time_of_day'] == 'night':
            atmo_gradient.add_stop(0, "#0D47A1")  # Dark blue at top
            atmo_gradient.add_stop(1, "#FFFFFF00")  # Transparent at bottom
        else:  # day or dawn
            atmo_gradient.add_stop(0, "#B3E5FC")  # Light blue at top
            atmo_gradient.add_stop(1, "#FFFFFF00")  # Transparent at bottom
        
        doc.add_definition(atmo_gradient.to_svg_element())
        
        atmo_haze = Rectangle(0, 0, width, height * 0.7).to_svg_element({ \
            "fill": f"url(#{atmo_id})",
            "opacity": "0.3", \
            "class": "atmospheric-haze"
        })
        
        doc.add_element(atmo_haze)
    
    # Apply final optimizations with the optimizer
    optimizer.optimize_svg_paths(doc)
    optimizer.optimize_colors(doc)
    
def generate_scene_graph(scene_features: Dict, width: int, height: int, precision: int = 2) -> SVGDocument:
    """
    Main function to create the SVG scene graph based on semantic features
    
    Args:
        scene_features: Dictionary of semantic features extracted from prompt
        width: SVG width in pixels
        height: SVG height in pixels
        precision: Decimal precision for coordinates
    
    Returns:
        Complete SVG document with all scene elements
    """
    # Create optimizer to apply SVG compression techniques
    optimizer = SVGOptimizer()
    
    # Create the SVG document
    doc = SVGDocument(width, height)
    
    # Apply compositional generation pipeline where each layer is built on extracted semantics
    # Each function populates the SVG document with specific elements
    
    # 1. Generate background (sky, day/night, gradients)
    generate_background(doc, scene_features, width, height, precision)
    
    # 2. Generate atmospheric elements (clouds, fog, rain)
    generate_atmospheric_elements(doc, scene_features, width, height, precision)
    
    # 3. Generate background terrain (mountains, hills in distance)
    generate_background_terrain(doc, scene_features, width, height, precision)
    
    # 4. Generate water bodies (oceans, lakes, rivers)
    generate_water_elements(doc, scene_features, width, height, precision)
    
    # 5. Generate midground elements (midground terrain, hills)
    generate_midground_elements(doc, scene_features, width, height, precision)
    
    # 6. Generate foreground terrain (ground, rocks)
    generate_foreground_terrain(doc, scene_features, width, height, precision)
    
    # 7. Generate vegetation (trees, flowers, grass)
    generate_vegetation(doc, scene_features, width, height, precision)
    
    # 8. Generate built structures (buildings, bridges)
    generate_built_structures(doc, scene_features, width, height, precision)
    
    # 9. Generate small foreground details for added realism
    generate_foreground_details(doc, scene_features, width, height, precision)
    
    # 10. Apply global lighting effects
    apply_global_lighting(doc, scene_features, width, height, precision, optimizer)
    
    return doc

def create_svg_from_prompt(prompt: str, width: int = 800, height: int = 600, max_size_kb: int = 10, precision: int = 2) -> str:
    """
    Create an SVG illustration based on a text prompt
    
    Args:
        prompt: Text prompt describing the desired scene
        width: Width of SVG in pixels
        height: Height of SVG in pixels
        max_size_kb: Maximum size in KB for the final SVG
        precision: Decimal precision for coordinates
        
    Returns:
        SVG code as string
    """
    # Extract semantic features from prompt
    scene_features = extract_semantic_features(prompt)
    
    # Generate the scene graph using parametric and compositional architecture
    svg_doc = generate_scene_graph(scene_features, width, height, precision)
    
    # Convert to string
    svg_string = svg_doc.to_string()
    
    # Ensure the SVG is under the size limit
    svg_size_kb = len(svg_string.encode('utf-8')) / 1024
    
    if svg_size_kb > max_size_kb:
        # Apply more aggressive optimization techniques if exceeding max size
        optimized_doc = optimize_svg_for_size(svg_doc, max_size_kb)
        svg_string = optimized_doc.to_string()
    
    return svg_string

def optimize_svg_for_size(svg_doc: SVGDocument, max_size_kb: int) -> SVGDocument:
    """
    Apply more aggressive optimization techniques to reduce file size
    
    Args:
        svg_doc: SVG document to optimize
        max_size_kb: Target maximum size in KB
    
    Returns:
        Optimized SVG document
    """
    # Create optimizer with more aggressive settings
    optimizer = SVGOptimizer()
    optimizer.create_limited_palette(num_colors=5)  # Reduce color palette
    
    # Apply path optimizations with higher precision loss tolerance
    optimizer.optimize_svg_paths(svg_doc)
    
    # Apply color optimizations
    optimizer.optimize_colors(svg_doc)
    
    # For now, we'll skip the additional optimizations as they require further implementation
    # In a future update, we can implement consolidate_paths and remove_metadata
    
    # If still too large, simplify scene graph by removing less important elements
    current_size = len(svg_doc.to_string().encode('utf-8')) / 1024
    if current_size > max_size_kb:
        # Remove details layer if exists
        details_layer = svg_doc.find_element_by_id("foreground_details")
        if details_layer:
            svg_doc.remove_element(details_layer)
    
    return svg_doc
    
def main():
    """
    Main function to generate an SVG illustration based on command line arguments
    """
    parser = argparse.ArgumentParser(description='Generate an SVG illustration from a text prompt')
    parser.add_argument('prompt', type=str, help='Text prompt describing the desired scene')
    parser.add_argument('--width', type=int, default=800, help='Width of SVG in pixels')
    parser.add_argument('--height', type=int, default=600, help='Height of SVG in pixels')
    parser.add_argument('--output', type=str, default='output.svg', help='Output file name')
    parser.add_argument('--max-size', type=int, default=10, help='Maximum size in KB')
    parser.add_argument('--precision', type=int, default=2, help='Decimal precision for coordinates')
    
    args = parser.parse_args()
    
    # Create SVG from prompt using the new compositional architecture
    svg_string = create_svg_from_prompt( \
        prompt=args.prompt,
        width=args.width, \
        height=args.height, \
        max_size_kb=args.max_size, \
        precision=args.precision
    )
    
    # Save to file
    with open(args.output, 'w') as f:
        f.write(svg_string)
    
    print(f"SVG illustration saved to {args.output}")
    print(f"File size: {len(svg_string.encode('utf-8')) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
class SVGConstraints:
    """Defines constraints for validating SVG documents.

    Attributes
    ----------
    max_svg_size : int, default=10000
        Maximum allowed size of an SVG file in bytes.
    allowed_elements : dict[str, set[str]]
        Mapping of the allowed elements to the allowed attributes of each element.
    """

    max_svg_size: int = 10000
    allowed_elements: Dict[str, set] = field(
        default_factory=lambda: {
            'common': {
                'id',
                'clip-path',
                'clip-rule',
                'color',
                'color-interpolation',
                'color-interpolation-filters',
                'color-rendering',
                'display',
                'fill',
                'fill-opacity',
                'fill-rule',
                'filter',
                'flood-color',
                'flood-opacity',
                'lighting-color',
                'marker-end',
                'marker-mid',
                'marker-start',
                'mask',
                'opacity',
                'paint-order',
                'stop-color',
                'stop-opacity',
                'stroke',
                'stroke-dasharray',
                'stroke-dashoffset',
                'stroke-linecap',
                'stroke-linejoin',
                'stroke-miterlimit',
                'stroke-opacity',
                'stroke-width',
                'transform',
            },
            'svg': {
                'width',
                'height',
                'viewBox',
                'preserveAspectRatio',
                'xmlns',
            },
            'g': {'viewBox'},
            'defs': set(),
            'title': set(),
            'desc': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox',
                'preserveAspectRatio',
                'refX',
                'refY',
                'markerUnits',
                'markerWidth',
                'markerHeight',
                'orient',
            },
            'pattern': {
                'viewBox',
                'preserveAspectRatio',
                'x',
                'y',
                'width',
                'height',
                'patternUnits',
                'patternContentUnits',
                'patternTransform',
                'href',
            },
            'linearGradient': {
                'x1',
                'x2',
                'y1',
                'y2',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'radialGradient': {
                'cx',
                'cy',
                'r',
                'fx',
                'fy',
                'fr',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'stop': {'offset'},
            'filter': {
                'x',
                'y',
                'width',
                'height',
                'filterUnits',
                'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feColorMatrix': {'result', 'in', 'type', 'values'},
            'feComposite': {
                'result',
                'style',
                'in',
                'in2',
                'operator',
                'k1',
                'k2',
                'k3',
                'k4',
            },
            'feFlood': {'result'},
            'feGaussianBlur': {
                'result',
                'in',
                'stdDeviation',
                'edgeMode',
            },
            'feMerge': {
                'result',
                'x',
                'y',
                'width',
                'height',
                'result',
            },
            'feMergeNode': {'result', 'in'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'feTurbulence': {
                'result',
                'baseFrequency',
                'numOctaves',
                'seed',
                'stitchTiles',
                'type',
            },
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }
    )

    def validate_svg(self, svg_code: str) -> bool:
        """Validates an SVG string against a set of predefined constraints.

        Parameters
        ----------
        svg_code : str
            The SVG string to validate.

        Returns
        -------
        bool
            True if validation passed, False otherwise
        """
        try:
            # Check file size
            if len(svg_code.encode('utf-8')) > self.max_svg_size:
                logger.warning('SVG exceeds allowed size')
                return False

            # Parse XML
            if SECURE_XML:
                tree = safe_ET.fromstring(
                    svg_code.encode('utf-8'),
                    forbid_dtd=True,
                    forbid_entities=True,
                    forbid_external=True,
                )
            else:
                tree = safe_ET.fromstring(svg_code.encode('utf-8'))

            elements = set(self.allowed_elements.keys())

            # Check elements and attributes
            for element in tree.iter():
                # Check for disallowed elements
                tag_name = element.tag.split('}')[-1]
                if tag_name not in elements:
                    logger.warning(f'Disallowed SVG element: {tag_name}')
                    return False

                # Check attributes
                for attr, attr_value in element.attrib.items():
                    # Check for disallowed attributes
                    attr_name = attr.split('}')[-1]
                    if (
                        attr_name not in self.allowed_elements[tag_name]
                        and attr_name not in self.allowed_elements['common']
                    ):
                        logger.warning(f'Disallowed SVG attribute: {attr_name}')
                        return False

                    # Check for embedded data
                    if 'data:' in attr_value.lower():
                        logger.warning('Embedded data not allowed in SVG')
                        return False
                    if ';base64' in attr_value:
                        logger.warning('Base64 encoded content not allowed in SVG')
                        return False

                    # Check that href attributes are internal references
                    if attr_name == 'href':
                        if not attr_value.startswith('#'):
                            logger.warning(
                                f'Invalid href attribute in <{tag_name}>. Only internal references (starting with "#") are allowed.'
                            )
                            return False
            
            return True
        except Exception as e:
            logger.error(f"SVG validation error: {e}")
            return False

    def sanitize_svg(self, svg_code: str) -> str:
        """Attempts to sanitize an SVG string by removing disallowed elements and attributes.
        
        Parameters
        ----------
        svg_code : str
            The SVG string to sanitize
            
        Returns
        -------
        str
            The sanitized SVG string
        """
        try:
            # Parse XML
            if SECURE_XML:
                tree = safe_ET.fromstring(
                    svg_code.encode('utf-8'),
                    forbid_dtd=True,
                    forbid_entities=True,
                    forbid_external=True,
                )
            else:
                tree = safe_ET.fromstring(svg_code.encode('utf-8'))
                
            elements_to_remove = []
            attributes_to_remove = {}
            
            elements = set(self.allowed_elements.keys())
            
            # First pass: identify elements and attributes to remove
            for element in tree.iter():
                tag_name = element.tag.split('}')[-1]
                
                # Check if element is allowed
                if tag_name not in elements:
                    elements_to_remove.append(element)
                    continue
                    
                # Check attributes
                attrs_to_remove = []
                for attr, attr_value in element.attrib.items():
                    attr_name = attr.split('}')[-1]
                    
                    # Check if attribute is allowed
                    if (attr_name not in self.allowed_elements[tag_name] and 
                        attr_name not in self.allowed_elements['common']):
                        attrs_to_remove.append(attr)
                    
                    # Check for embedded data or external references
                    elif 'data:' in attr_value.lower() or ';base64' in attr_value:
                        attrs_to_remove.append(attr)
                    
                    # Check href attributes
                    elif attr_name == 'href' and not attr_value.startswith('#'):
                        attrs_to_remove.append(attr)
                        
                if attrs_to_remove:
                    attributes_to_remove[element] = attrs_to_remove
            
            # Second pass: remove disallowed attributes
            for element, attrs in attributes_to_remove.items():
                for attr in attrs:
                    element.attrib.pop(attr, None)
            
            # Third pass: remove disallowed elements (in reverse to avoid tree modification issues)
            for element in reversed(elements_to_remove):
                parent = tree.find('./..', element)
                if parent is not None:
                    parent.remove(element)
            
            # Convert back to string
            sanitized_svg = ET.tostring(tree, encoding='unicode')
            
            # Check size after sanitization
            if len(sanitized_svg.encode('utf-8')) > self.max_svg_size:
                # If still too large, truncate
                logger.warning("SVG still exceeds size limit after sanitization, truncating")
                tree = ET.fromstring("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"10\" height=\"10\"></svg>")
                return ET.tostring(tree, encoding='unicode')
                
            return sanitized_svg
            
        except Exception as e:
            logger.error(f"SVG sanitization error: {e}")
            # Return a minimal valid SVG
            return '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
# Try to import tqdm, but provide fallback if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
class PromptToSVGEngine:
    """Engine for generating SVG illustrations from text prompts.
    
    This engine processes natural language descriptions and produces
    corresponding SVG illustrations with realistic visual elements.
    """
    
    def __init__(self, width=800, height=600, detail_level="medium"):
        """Initialize the SVG generation engine.
        
        Args:
            width: Width of the SVG canvas in pixels
            height: Height of the SVG canvas in pixels
            detail_level: Level of detail for the generated SVG ("low", "medium", "high")
        """
        self.width = width
        self.height = height
        self.detail_level = detail_level
        self.material_lib = MaterialTextureLibrary()
        self.pattern_gen = PatternGenerator()
    
    def generate_svg(self, prompt):
        """Generate an SVG illustration from a text prompt.
        
        Args:
            prompt: Text description of the desired illustration
            
        Returns:
            SVG content as a string
        """
        # Create SVG root element
        svg_root = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(self.width),
            "height": str(self.height),
            "viewBox": f"0 0 {self.width} {self.height}"
        })
        
        # Add title based on prompt
        title = ET.SubElement(svg_root, "title")
        title.text = prompt
        
        # Add defs section for gradients, patterns, etc.
        defs = ET.SubElement(svg_root, "defs")
        
        # Create simple background gradient
        bg_gradient = ET.SubElement(defs, "linearGradient", {
            "id": "sky_gradient",
            "x1": "0%",
            "y1": "0%",
            "x2": "0%",
            "y2": "100%"
        })
        
        # Add gradient stops for sky
        ET.SubElement(bg_gradient, "stop", {
            "offset": "0%",
            "stop-color": "#87CEEB"
        })
        ET.SubElement(bg_gradient, "stop", {
            "offset": "100%",
            "stop-color": "#1E90FF"
        })
        
        # Create water gradient
        water_gradient = ET.SubElement(defs, "linearGradient", {
            "id": "water_gradient",
            "x1": "0%",
            "y1": "0%",
            "x2": "0%",
            "y2": "100%"
        })
        
        # Add gradient stops for water
        ET.SubElement(water_gradient, "stop", {
            "offset": "0%",
            "stop-color": "#4682B4"
        })
        ET.SubElement(water_gradient, "stop", {
            "offset": "100%",
            "stop-color": "#000080"
        })
        
        # Add background
        ET.SubElement(svg_root, "rect", {
            "x": "0",
            "y": "0",
            "width": str(self.width),
            "height": str(self.height),
            "fill": "url(#sky_gradient)"
        })
        
        # Simple semantic parsing of the prompt
        has_mountain = "mountain" in prompt.lower()
        has_forest = any(x in prompt.lower() for x in ["forest", "tree", "trees"])
        has_water = any(x in prompt.lower() for x in ["lake", "ocean", "sea", "river"])
        has_sun = "sun" in prompt.lower()
        has_buildings = any(x in prompt.lower() for x in ["city", "building", "skyline", "house"])
        
        # Ground
        ground_height = self.height * 0.3
--
class PromptToSVGEngine:
    """Engine for generating SVG illustrations from text prompts.
    
    This engine processes natural language descriptions and produces
    corresponding SVG illustrations with realistic visual elements.
    """
    
    def __init__(self, width=800, height=600, detail_level="medium"):
        """Initialize the SVG generation engine.
        
        Args:
            width: Width of the SVG canvas in pixels
            height: Height of the SVG canvas in pixels
            detail_level: Level of detail for the generated SVG ("low", "medium", "high")
        """
        self.width = width
        self.height = height
        self.detail_level = detail_level
        self.material_lib = MaterialTextureLibrary()
        self.pattern_gen = PatternGenerator()
    
    def generate_svg(self, prompt):
        """Generate an SVG illustration from a text prompt.
        
        Args:
            prompt: Text description of the desired illustration
            
        Returns:
            SVG content as a string
        """
        # Create SVG root element
        svg_root = ET.Element("svg", {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": str(self.width),
            "height": str(self.height),
            "viewBox": f"0 0 {self.width} {self.height}"
        })
        
        # Add title based on prompt
        title = ET.SubElement(svg_root, "title")
        title.text = prompt
        
        # Add defs section for gradients, patterns, etc.
        defs = ET.SubElement(svg_root, "defs")
        
        # Create simple background gradient
        bg_gradient = ET.SubElement(defs, "linearGradient", {
            "id": "sky_gradient",
            "x1": "0%",
            "y1": "0%",
            "x2": "0%",
            "y2": "100%"
        })
        
        # Add gradient stops for sky
        ET.SubElement(bg_gradient, "stop", {
            "offset": "0%",
            "stop-color": "#87CEEB"
        })
        ET.SubElement(bg_gradient, "stop", {
            "offset": "100%",
            "stop-color": "#1E90FF"
        })
        
        # Create water gradient
        water_gradient = ET.SubElement(defs, "linearGradient", {
            "id": "water_gradient",
            "x1": "0%",
            "y1": "0%",
            "x2": "0%",
            "y2": "100%"
        })
        
        # Add gradient stops for water
        ET.SubElement(water_gradient, "stop", {
            "offset": "0%",
            "stop-color": "#4682B4"
        })
        ET.SubElement(water_gradient, "stop", {
            "offset": "100%",
            "stop-color": "#000080"
        })
        
        # Add background
        ET.SubElement(svg_root, "rect", {
            "x": "0",
            "y": "0",
            "width": str(self.width),
            "height": str(self.height),
            "fill": "url(#sky_gradient)"
        })
        
        # Simple semantic parsing of the prompt
        has_mountain = "mountain" in prompt.lower()
        has_forest = any(x in prompt.lower() for x in ["forest", "tree", "trees"])
        has_water = any(x in prompt.lower() for x in ["lake", "ocean", "sea", "river"])
        has_sun = "sun" in prompt.lower()
        has_buildings = any(x in prompt.lower() for x in ["city", "building", "skyline", "house"])
        
        # Ground
        ground_height = self.height * 0.3
def save_svg(svg_content, filename):
    """Save SVG content to a file.
    
    Args:
        svg_content: SVG content as a string
        filename: Path to save the SVG file
    """
    # Create parent directories if they don't exist
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Write SVG content to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print(f"Saved SVG to {filename}")


if __name__ == "__main__":
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate material samples
    print("Generating material samples...")
    create_material_sample_svg()
    create_pattern_sample_svg()
    print("Done! Check the output directory for sample SVGs.")
    
    # Example prompts
    examples = [
--
def save_svg(svg_content, filename):
    """Save SVG content to a file with validation and sanitization.
    
    Args:
        svg_content: SVG content as a string
        filename: Path to save the SVG file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    # Check if svg_content is None or empty
    if svg_content is None:
        logger.warning(f"Cannot save to {filename}: SVG content is None")
        return False
    
    # Create SVG validator instance
    svg_validator = SVGConstraints()
    
    # Validate the SVG
    is_valid = svg_validator.validate_svg(svg_content)
    
    # If invalid, try to sanitize
    if not is_valid:
        logger.warning(f"SVG validation failed for {filename}. Attempting to sanitize...")
        svg_content = svg_validator.sanitize_svg(svg_content)
        
        # Validate again after sanitization
        if not svg_validator.validate_svg(svg_content):
            logger.error(f"Failed to sanitize SVG for {filename}. Using minimal SVG.")
            svg_content = '<svg xmlns="http://www.w3.org/2000/svg" width="50" height="50"><text x="10" y="30" fill="red">Validation Error</text></svg>'
    
