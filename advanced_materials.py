#!/usr/bin/env python3
"""
Hyper-Realistic SVG Generator
=============================

A unified, monolithic implementation for generating hyper-realistic SVG illustrations 
from text prompts through a sophisticated mathematical, graphical and semantic approach.

This integrated system implements a multi-stage pipeline:
1. LEXICAL-SEMANTIC INTERPRETATION - Parsing text into structured semantics
2. CONCEPTUAL MAPPING TO VISUAL CONSTRUCTS - Creating an intermediate scene representation
3. PROCEDURAL SVG GENERATION - Transforming scene graph into SVG primitives with PBR materials
4. VALIDATION AND SANITIZATION - Ensuring SVG safety and constraints
5. CORRECTION AND SEMANTIC FEEDBACK - Auto-correcting for completeness and coherence

Architecture highlights:
- Unified type system with strong validation
- Hybrid semantic parsing combining rule-based and ML approaches
- PBR-style material and texture generation
- Spatial partitioning for complex scene optimization
- Multi-stage rendering pipeline with progressive refinement
"""

import argparse
import colorsys
import json
import logging
import math
import os
import random
import re
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from enum import Enum
from math import sin, cos, radians, pi, sqrt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set, Union, Callable
from typing import Literal, NamedTuple, TypedDict, Final, cast
from typing import ForwardRef, ClassVar, Protocol, Iterator
import xml.etree.ElementTree as ET
from xml.dom import minidom
import functools
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field

# Try to import defusedxml for secure XML parsing
try:
    from defusedxml import ElementTree as safe_ET
    SECURE_XML = True
except ImportError:
    # Fall back to standard ElementTree
    import xml.etree.ElementTree as safe_ET
    SECURE_XML = False
    logging.warning("defusedxml not available, using standard ElementTree instead. This is less secure.")

# SVG Security Validation Module
@dataclass(frozen=True)
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
    TQDM_AVAILABLE = False
    # Create a simple fallback tqdm function
    def tqdm(iterable, **kwargs):
        if "desc" in kwargs:
            print(f"\n{kwargs['desc']}")
        return iterable
    
    # Create a fallback write function for tqdm
    tqdm.write = print

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('svg_generation.log')
    ]
)
logger = logging.getLogger('svg_generator')

try:
    from pydantic import BaseModel, Field, validator, root_validator
    from pydantic import Extra, ValidationError, create_model
except ImportError:
    # Fallback for pure-Python implementation with no pydantic dependency
    class DummyBaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    BaseModel = DummyBaseModel
    Field = lambda *args, **kwargs: None
    validator = lambda *args, **kwargs: lambda func: func
    root_validator = lambda *args, **kwargs: lambda func: func
    Extra = type('Extra', (), {'forbid': None})
    ValidationError = Exception
    create_model = lambda *args, **kwargs: type('DummyModel', (DummyBaseModel,), {})


def prettify_svg(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


class PatternGenerator:
    """Generate SVG patterns for use in materials and textures.
    
    This class provides methods to create various SVG patterns such as grids,
    stripes, dots, and other texture elements that can be used as fills
    for SVG elements.
    """
    
    def __init__(self):
        """Initialize the pattern generator."""
        self.patterns = {}
        self.pattern_counter = 0
    
    def _create_pattern_element(self, width, height, pattern_units="userSpaceOnUse"):
        """Create a base pattern element.
        
        Args:
            width: Width of the pattern
            height: Height of the pattern
            pattern_units: Units for the pattern (userSpaceOnUse or objectBoundingBox)
            
        Returns:
            Tuple of (pattern_id, pattern_element)
        """
        pattern_id = f"pattern_{self.pattern_counter}"
        self.pattern_counter += 1
        
        pattern = ET.Element("pattern", {
            "id": pattern_id,
            "width": str(width),
            "height": str(height),
            "patternUnits": pattern_units
        })
        
        return pattern_id, pattern
    
    def generate_grid_pattern(self, cell_width=10.0, cell_height=10.0, 
                             line_width=1.0, line_color="#000000",
                             background_color="#FFFFFF", opacity=1.0):
        """Generate a grid pattern.
        
        Args:
            cell_width: Width of each grid cell
            cell_height: Height of each grid cell
            line_width: Width of grid lines
            line_color: Color of grid lines
            background_color: Background color of the grid
            opacity: Overall opacity of the pattern
            
        Returns:
            Pattern ID that can be referenced in SVG fill attributes
        """
        pattern_id, pattern = self._create_pattern_element(cell_width, cell_height)
        
        # Add background rectangle if needed
        if background_color:
            ET.SubElement(pattern, "rect", {
                "x": "0",
                "y": "0",
                "width": str(cell_width),
                "height": str(cell_height),
                "fill": background_color
            })
        
        # Add horizontal line
        ET.SubElement(pattern, "line", {
            "x1": "0",
            "y1": "0",
            "x2": str(cell_width),
            "y2": "0",
            "stroke": line_color,
            "stroke-width": str(line_width),
            "opacity": str(opacity)
        })
        
        # Add vertical line
        ET.SubElement(pattern, "line", {
            "x1": "0",
            "y1": "0",
            "x2": "0",
            "y2": str(cell_height),
            "stroke": line_color,
            "stroke-width": str(line_width),
            "opacity": str(opacity)
        })
        
        self.patterns[pattern_id] = pattern
        return pattern_id
    
    def generate_stripe_pattern(self, stripe_width=5.0, stripe_spacing=5.0,
                               stripe_color="#000000", background_color="#FFFFFF",
                               angle=0.0):
        """Generate a striped pattern.
        
        Args:
            stripe_width: Width of each stripe
            stripe_spacing: Spacing between stripes
            stripe_color: Color of the stripes
            background_color: Background color
            angle: Rotation angle in degrees
            
        Returns:
            Pattern ID that can be referenced in SVG fill attributes
        """
        pattern_width = stripe_width + stripe_spacing
        pattern_id, pattern = self._create_pattern_element(pattern_width, pattern_width)
        
        # Add background rectangle
        if background_color:
            ET.SubElement(pattern, "rect", {
                "x": "0",
                "y": "0",
                "width": str(pattern_width),
                "height": str(pattern_width),
                "fill": background_color
            })
        
        # Create stripe rectangle
        stripe = ET.SubElement(pattern, "rect", {
            "x": "0",
            "y": "0",
            "width": str(stripe_width),
            "height": str(pattern_width),
            "fill": stripe_color
        })
        
        # Apply rotation if needed
        if angle != 0:
            # Calculate center point of pattern for rotation
            cx = pattern_width / 2
            cy = pattern_width / 2
            transform = f"rotate({angle} {cx} {cy})"
            stripe.set("transform", transform)
        
        self.patterns[pattern_id] = pattern
        return pattern_id
    
    def generate_dot_pattern(self, dot_radius=2.0, spacing_x=10.0, spacing_y=10.0,
                           dot_color="#000000", background_color="#FFFFFF"):
        """Generate a pattern of dots.
        
        Args:
            dot_radius: Radius of each dot
            spacing_x: Horizontal spacing between dot centers
            spacing_y: Vertical spacing between dot centers
            dot_color: Color of the dots
            background_color: Background color
            
        Returns:
            Pattern ID that can be referenced in SVG fill attributes
        """
        pattern_id, pattern = self._create_pattern_element(spacing_x, spacing_y)
        
        # Add background rectangle
        if background_color:
            ET.SubElement(pattern, "rect", {
                "x": "0",
                "y": "0",
                "width": str(spacing_x),
                "height": str(spacing_y),
                "fill": background_color
            })
        
        # Add dot at center of cell
        cx = spacing_x / 2
        cy = spacing_y / 2
        
        ET.SubElement(pattern, "circle", {
            "cx": str(cx),
            "cy": str(cy),
            "r": str(dot_radius),
            "fill": dot_color
        })
        
        self.patterns[pattern_id] = pattern
        return pattern_id
    
    def generate_window_pattern(self, width=40.0, height=60.0, divisions_x=2,
                              divisions_y=3, frame_width=2.0, frame_color="#884400",
                              glass_color="#AADDFF", glass_opacity=0.5):
        """Generate a window pattern with multiple panes.
        
        Args:
            width: Overall width of the window
            height: Overall height of the window
            divisions_x: Number of horizontal divisions (panes)
            divisions_y: Number of vertical divisions (panes)
            frame_width: Width of the window frame
            frame_color: Color of the window frame
            glass_color: Color of the glass
            glass_opacity: Opacity of the glass
            
        Returns:
            Pattern ID that can be referenced in SVG fill attributes
        """
        pattern_id, pattern = self._create_pattern_element(width, height)
        
        # Add outer frame
        ET.SubElement(pattern, "rect", {
            "x": "0",
            "y": "0",
            "width": str(width),
            "height": str(height),
            "fill": frame_color
        })
        
        # Calculate pane dimensions
        pane_width = (width - frame_width * (divisions_x + 1)) / divisions_x
        pane_height = (height - frame_width * (divisions_y + 1)) / divisions_y
        
        # Add window panes
        for row in range(divisions_y):
            for col in range(divisions_x):
                x = frame_width + col * (pane_width + frame_width)
                y = frame_width + row * (pane_height + frame_width)
                
                ET.SubElement(pattern, "rect", {
                    "x": str(x),
                    "y": str(y),
                    "width": str(pane_width),
                    "height": str(pane_height),
                    "fill": glass_color,
                    "opacity": str(glass_opacity)
                })
        
        self.patterns[pattern_id] = pattern
        return pattern_id
    
    def generate_brick_pattern(self, brick_width=30.0, brick_height=15.0,
                            mortar_width=2.0, brick_color="#AA4444",
                            mortar_color="#CCCCCC"):
        """Generate a brick pattern.
        
        Args:
            brick_width: Width of each brick
            brick_height: Height of each brick
            mortar_width: Width of mortar between bricks
            brick_color: Color of the bricks
            mortar_color: Color of the mortar
            
        Returns:
            Pattern ID that can be referenced in SVG fill attributes
        """
        # Calculate pattern dimensions to include a full row of bricks with mortar
        pattern_width = brick_width + mortar_width
        pattern_height = 2 * (brick_height + mortar_width)  # Two rows for offset pattern
        
        pattern_id, pattern = self._create_pattern_element(pattern_width, pattern_height)
        
        # Add mortar background
        ET.SubElement(pattern, "rect", {
            "x": "0",
            "y": "0",
            "width": str(pattern_width),
            "height": str(pattern_height),
            "fill": mortar_color
        })
        
        # Add first row brick
        ET.SubElement(pattern, "rect", {
            "x": "0",
            "y": "0",
            "width": str(brick_width),
            "height": str(brick_height),
            "fill": brick_color
        })
        
        # Add second row brick (offset)
        ET.SubElement(pattern, "rect", {
            "x": str(mortar_width / 2),  # Offset by half
            "y": str(brick_height + mortar_width),
            "width": str(brick_width),
            "height": str(brick_height),
            "fill": brick_color
        })
        
        self.patterns[pattern_id] = pattern
        return pattern_id
    
    def get_pattern_defs(self):
        """Get all pattern definitions for inclusion in SVG defs.
        
        Returns:
            List of pattern elements
        """
        return list(self.patterns.values())
    
    def add_patterns_to_defs(self, defs_element):
        """Add all generated patterns to an SVG defs element.
        
        Args:
            defs_element: SVG defs element to add patterns to
            
        Returns:
            None
        """
        for pattern in self.patterns.values():
            defs_element.append(pattern)


class MaterialTextureLibrary:
    """Library of procedurally generated material textures for SVG rendering.
    
    This class provides methods to generate realistic-looking material textures
    such as wood, stone, metal, etc. for use in SVG elements.
    """
    
    def __init__(self):
        """Initialize the material texture library."""
        self.pattern_generator = PatternGenerator()
    
    def _lighten_color(self, hex_color, factor=0.2):
        """Lighten a hex color by a factor.
        
        Args:
            hex_color: Hex color code (e.g., '#FF0000')
            factor: Factor to lighten by (0.0-1.0)
            
        Returns:
            Lightened hex color
        """
        # Remove hash if present
        hex_color = hex_color.lstrip('#')
        
        # Convert hex to RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        # Lighten
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def _darken_color(self, hex_color, factor=0.2):
        """Darken a hex color by a factor.
        
        Args:
            hex_color: Hex color code (e.g., '#FF0000')
            factor: Factor to darken by (0.0-1.0)
            
        Returns:
            Darkened hex color
        """
        # Remove hash if present
        hex_color = hex_color.lstrip('#')
        
        # Convert hex to RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        
        # Darken
        r = max(0, int(r * (1 - factor)))
        g = max(0, int(g * (1 - factor)))
        b = max(0, int(b * (1 - factor)))
        
        # Convert back to hex
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def get_material_texture(self, material_type, x, y, width, height, color, options=None):
        """Get SVG elements for a specific material texture.
        
        Args:
            material_type: Type of material (wood, stone, metal, etc.)
            x: X coordinate of the texture placement
            y: Y coordinate of the texture placement
            width: Width of the texture
            height: Height of the texture
            color: Base color for the material
            options: Optional parameters specific to the material type
            
        Returns:
            List of SVG elements representing the material texture
        """
        options = options or {}
        
        # Map material type to generator function
        material_generators = {
            "wood": self._generate_wood_texture,
            "stone": self._generate_stone_texture,
            "marble": self._generate_marble_texture,
            "metal": self._generate_metal_texture,
            "glass": self._generate_glass_texture,
            "brick": self._generate_brick_texture,
            "concrete": self._generate_concrete_texture,
            "water": self._generate_water_texture
        }
        
        # Get the appropriate generator or use a default
        generator = material_generators.get(material_type.lower(), self._generate_default_texture)
        
        # Generate the texture
        return generator(x, y, width, height, color, options)
    
    def _generate_default_texture(self, x, y, width, height, color, options):
        """Generate a default solid color texture.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base color
            options: Additional options (unused)
            
        Returns:
            List of SVG elements
        """
        # Just a basic rectangle with the specified color
        rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color
        })
        
        return [rect]
    
    def _generate_wood_texture(self, x, y, width, height, color, options):
        """Generate a wood texture.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base wood color
            options: Additional options like grain_count
            
        Returns:
            List of SVG elements
        """
        elements = []
        
        # Get wood-specific options with defaults
        grain_count = options.get("grain_count", 10)
        grain_color = options.get("grain_color", self._darken_color(color, 0.3))
        grain_opacity = options.get("grain_opacity", 0.7)
        
        # Base rectangle
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color
        })
        elements.append(base_rect)
        
        # Add wood grain lines
        for i in range(grain_count):
            # Calculate position with slight randomness
            grain_x = x + (i * width / grain_count) + random.uniform(-2, 2)
            
            # Create a grain line
            grain = ET.Element("line", {
                "x1": str(grain_x),
                "y1": str(y),
                "x2": str(grain_x),
                "y2": str(y + height),
                "stroke": grain_color,
                "stroke-width": str(random.uniform(1, 3)),
                "opacity": str(grain_opacity * random.uniform(0.7, 1.0))
            })
            elements.append(grain)
        
        return elements
    
    def _generate_stone_texture(self, x, y, width, height, color, options):
        """Generate a stone texture.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base stone color
            options: Additional options like type (granite, etc.)
            
        Returns:
            List of SVG elements
        """
        elements = []
        stone_type = options.get("type", "granite")
        
        # Base rectangle
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color
        })
        elements.append(base_rect)
        
        if stone_type == "granite":
            # Add speckles for granite
            speckle_count = int(width * height / 100)  # Density of speckles
            
            for _ in range(speckle_count):
                speck_x = x + random.uniform(0, width)
                speck_y = y + random.uniform(0, height)
                speck_radius = random.uniform(0.5, 2.0)
                
                # Randomly choose lighter or darker speckles
                if random.random() > 0.5:
                    speck_color = self._lighten_color(color, random.uniform(0.1, 0.3))
                else:
                    speck_color = self._darken_color(color, random.uniform(0.1, 0.3))
                
                speck = ET.Element("circle", {
                    "cx": str(speck_x),
                    "cy": str(speck_y),
                    "r": str(speck_radius),
                    "fill": speck_color,
                    "opacity": str(random.uniform(0.5, 0.9))
                })
                elements.append(speck)
        
        return elements
    
    def _generate_marble_texture(self, x, y, width, height, color, options):
        """Generate a marble texture with veins.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base marble color
            options: Additional options
            
        Returns:
            List of SVG elements
        """
        elements = []
        
        # Base rectangle
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color
        })
        elements.append(base_rect)
        
        # Create marble veins
        vein_count = random.randint(3, 6)
        
        for _ in range(vein_count):
            # Start and end points for the vein
            start_x = x + random.uniform(0, width * 0.3)
            start_y = y + random.uniform(0, height)
            
            # Control points for Bezier curve
            cx1 = x + random.uniform(width * 0.3, width * 0.6)
            cy1 = y + random.uniform(0, height)
            
            cx2 = x + random.uniform(width * 0.4, width * 0.7)
            cy2 = y + random.uniform(0, height)
            
            end_x = x + random.uniform(width * 0.7, width)
            end_y = y + random.uniform(0, height)
            
            # Path data for a cubic Bezier curve
            path_data = f"M {start_x} {start_y} C {cx1} {cy1}, {cx2} {cy2}, {end_x} {end_y}"
            
            # Create vein
            vein_color = self._lighten_color(color, random.uniform(0.1, 0.2))
            vein = ET.Element("path", {
                "d": path_data,
                "stroke": vein_color,
                "stroke-width": str(random.uniform(1, 3)),
                "fill": "none",
                "opacity": str(random.uniform(0.3, 0.7))
            })
            elements.append(vein)
        
        return elements
    
    def _generate_metal_texture(self, x, y, width, height, color, options):
        """Generate a metal texture with reflection effects.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base metal color
            options: Additional options like type (steel, gold, etc.)
            
        Returns:
            List of SVG elements
        """
        elements = []
        metal_type = options.get("type", "steel")
        
        # Adjust color based on metal type
        if metal_type == "gold":
            color = "#FFD700"  # Gold color
        elif metal_type == "copper":
            color = "#B87333"  # Copper color
        
        # Base rectangle
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color
        })
        elements.append(base_rect)
        
        # Generate gradient ID
        gradient_id = f"metal_gradient_{int(x)}_{int(y)}"
        
        # Create gradient element
        gradient = ET.Element("linearGradient", {
            "id": gradient_id,
            "x1": "0%",
            "y1": "0%",
            "x2": "100%",
            "y2": "100%"
        })
        
        # Add gradient stops
        ET.SubElement(gradient, "stop", {
            "offset": "0%",
            "stop-color": "#FFFFFF",
            "stop-opacity": "0.7"
        })
        
        ET.SubElement(gradient, "stop", {
            "offset": "20%",
            "stop-color": "#FFFFFF",
            "stop-opacity": "0.0"
        })
        
        ET.SubElement(gradient, "stop", {
            "offset": "100%",
            "stop-color": "#000000",
            "stop-opacity": "0.1"
        })
        
        elements.append(gradient)
        
        # Add reflection overlay
        overlay = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": f"url(#{gradient_id})"
        })
        elements.append(overlay)
        
        return elements
    
    def _generate_glass_texture(self, x, y, width, height, color, options):
        """Generate a glass texture with transparency and reflection.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base glass color
            options: Additional options like opacity
            
        Returns:
            List of SVG elements
        """
        elements = []
        opacity = options.get("opacity", 0.6)
        
        # Base rectangle with opacity
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color,
            "opacity": str(opacity)
        })
        elements.append(base_rect)
        
        # Add highlight/reflection
        highlight = ET.Element("path", {
            "d": f"M {x} {y} L {x + width * 0.8} {y} L {x + width * 0.6} {y + height * 0.2} L {x} {y + height * 0.1} Z",
            "fill": "#FFFFFF",
            "opacity": "0.3"
        })
        elements.append(highlight)
        
        # Add subtle border
        border = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "stroke": "#FFFFFF",
            "stroke-width": "0.5",
            "stroke-opacity": "0.5",
            "fill": "none"
        })
        elements.append(border)
        
        return elements
    
    def _generate_brick_texture(self, x, y, width, height, color, options):
        """Generate a brick texture.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base brick color
            options: Additional options
            
        Returns:
            List of SVG elements
        """
        elements = []
        
        # Get brick-specific options
        brick_height = options.get("brick_height", 20)
        brick_width = options.get("brick_width", 60)
        mortar_width = options.get("mortar_width", 5)
        mortar_color = options.get("mortar_color", "#CCCCCC")
        
        # Base rectangle (mortar color)
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": mortar_color
        })
        elements.append(base_rect)
        
        # Calculate rows and columns
        rows = int(height / (brick_height + mortar_width))
        cols = int(width / (brick_width + mortar_width))
        
        # Add bricks with alternating pattern
        for row in range(rows + 1):
            for col in range(cols + 1):
                # Offset alternate rows for realistic pattern
                offset_x = (brick_width + mortar_width) / 2 if row % 2 == 1 else 0
                
                brick_x = x + col * (brick_width + mortar_width) + offset_x
                brick_y = y + row * (brick_height + mortar_width)
                
                # Vary brick color slightly for realism
                brick_color = self._darken_color(color, random.uniform(0, 0.2)) if random.random() > 0.5 else self._lighten_color(color, random.uniform(0, 0.1))
                
                # Only render bricks that would be visible
                if (brick_x < x + width and brick_y < y + height and 
                    brick_x + brick_width > x and brick_y + brick_height > y):
                    
                    # Adjust dimensions for bricks at the edges
                    actual_width = min(brick_width, x + width - brick_x)
                    actual_height = min(brick_height, y + height - brick_y)
                    
                    if actual_width > 0 and actual_height > 0:
                        brick = ET.Element("rect", {
                            "x": str(max(x, brick_x)),
                            "y": str(max(y, brick_y)),
                            "width": str(actual_width),
                            "height": str(actual_height),
                            "fill": brick_color
                        })
                        elements.append(brick)
        
        return elements
    
    def _generate_concrete_texture(self, x, y, width, height, color, options):
        """Generate a concrete texture.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base concrete color
            options: Additional options
            
        Returns:
            List of SVG elements
        """
        elements = []
        
        # Base rectangle
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color
        })
        elements.append(base_rect)
        
        # Add noise/specks for concrete texture
        speck_count = int(width * height / 70)  # More dense than stone
        
        for _ in range(speck_count):
            speck_x = x + random.uniform(0, width)
            speck_y = y + random.uniform(0, height)
            speck_radius = random.uniform(0.1, 1.0)
            
            # Randomly choose color variation
            variation = random.uniform(-0.1, 0.1)
            if variation >= 0:
                speck_color = self._lighten_color(color, variation)
            else:
                speck_color = self._darken_color(color, -variation)
            
            speck = ET.Element("circle", {
                "cx": str(speck_x),
                "cy": str(speck_y),
                "r": str(speck_radius),
                "fill": speck_color,
                "opacity": str(random.uniform(0.3, 0.8))
            })
            elements.append(speck)
        
        # Add a few cracks
        for _ in range(2):
            start_x = x + random.uniform(0, width)
            start_y = y + random.uniform(0, height)
            
            # Create a zigzag path for the crack
            points = []
            current_x, current_y = start_x, start_y
            
            for _ in range(5):  # 5 segments in the crack
                next_x = current_x + random.uniform(-width/10, width/10)
                next_y = current_y + random.uniform(-height/10, height/10)
                
                # Keep within bounds
                next_x = max(x, min(x + width, next_x))
                next_y = max(y, min(y + height, next_y))
                
                points.append((next_x, next_y))
                current_x, current_y = next_x, next_y
            
            # Construct path data
            path_data = f"M {start_x} {start_y}"
            for px, py in points:
                path_data += f" L {px} {py}"
            
            crack = ET.Element("path", {
                "d": path_data,
                "stroke": self._darken_color(color, 0.3),
                "stroke-width": "0.5",
                "fill": "none",
                "opacity": "0.7"
            })
            elements.append(crack)
        
        return elements
    
    def _generate_water_texture(self, x, y, width, height, color, options):
        """Generate a water texture with waves.
        
        Args:
            x, y, width, height: Position and dimensions
            color: Base water color
            options: Additional options like wave_height
            
        Returns:
            List of SVG elements
        """
        elements = []
        
        # Get water-specific options
        wave_height = options.get("wave_height", 5)
        
        # Base water rectangle
        base_rect = ET.Element("rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": color,
            "opacity": "0.8"
        })
        elements.append(base_rect)
        
        # Create wave layers
        for layer in range(3):
            # Different parameters for each layer
            layer_opacity = 0.2 - (layer * 0.05)
            layer_color = self._lighten_color(color, 0.1 * (layer + 1))
            y_offset = y + (layer * height / 5)
            
            # Calculate wave points
            wave_points = []
            segments = 10
            
            for i in range(segments + 1):
                segment_x = x + (i * width / segments)
                segment_y = y_offset + math.sin(i * 2 * math.pi / segments) * wave_height
                wave_points.append((segment_x, segment_y))
            
            # Construct the wave path
            path_data = f"M {wave_points[0][0]} {wave_points[0][1]}"
            
            for i in range(1, len(wave_points)):
                path_data += f" L {wave_points[i][0]} {wave_points[i][1]}"
            
            # Close the path
            path_data += f" L {x + width} {y + height} L {x} {y + height} Z"
            
            # Create the wave path element
            wave = ET.Element("path", {
                "d": path_data,
                "fill": layer_color,
                "opacity": str(layer_opacity)
            })
            elements.append(wave)
            
        # Add a few white caps/highlights
        for _ in range(3):
            cap_x = x + random.uniform(width * 0.2, width * 0.8)
            cap_y = y + random.uniform(height * 0.2, height * 0.4)
            cap_radius = random.uniform(2, 5)
            
            cap = ET.Element("circle", {
                "cx": str(cap_x),
                "cy": str(cap_y),
                "r": str(cap_radius),
                "fill": "#FFFFFF",
                "opacity": "0.3"
            })
            elements.append(cap)
        
        return elements


def create_material_sample_svg():
    """Create an SVG showcasing different material textures."""
    # Initialize components
    material_lib = MaterialTextureLibrary()
    pattern_gen = PatternGenerator()
    
    # Create SVG root element
    svg_root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": "800",
        "height": "600",
        "viewBox": "0 0 800 600"
    })
    
    # Add defs section for patterns
    defs = ET.SubElement(svg_root, "defs")
    
    # Add title
    title = ET.SubElement(svg_root, "title")
    title.text = "Material Texture Samples"
    
    # Add background
    bg = ET.SubElement(svg_root, "rect", {
        "x": "0",
        "y": "0",
        "width": "800",
        "height": "600",
        "fill": "#F0F0F0"
    })
    
    # Define materials to showcase
    materials = [
        {"name": "wood", "color": "#A05A2C", "options": {"grain_count": 8}},
        {"name": "stone", "color": "#808080", "options": {"type": "granite"}},
        {"name": "marble", "color": "#F0F0F0", "options": {}},
        {"name": "metal", "color": "#C0C0C0", "options": {"type": "steel"}},
        {"name": "glass", "color": "#AADDFF", "options": {"opacity": 0.6}},
        {"name": "brick", "color": "#AA4444", "options": {}},
        {"name": "concrete", "color": "#BBBBBB", "options": {}},
        {"name": "water", "color": "#0088CC", "options": {"wave_height": 4.0}}
    ]
    
    # Add material samples
    margin = 50
    columns = 2
    row_height = 200
    col_width = 350
    
    # Create heading
    heading = ET.SubElement(svg_root, "text", {
        "x": "400",
        "y": "40",
        "text-anchor": "middle",
        "font-family": "Arial",
        "font-size": "24",
        "font-weight": "bold"
    })
    heading.text = "SVG Material Texture Library"
    
    # Create description
    desc = ET.SubElement(svg_root, "text", {
        "x": "400",
        "y": "70",
        "text-anchor": "middle",
        "font-family": "Arial",
        "font-size": "14"
    })
    desc.text = "Procedurally generated textures for realistic SVG rendering"
    
    # Add material samples
    for i, material in enumerate(materials):
        row = i // columns
        col = i % columns
        
        x = margin + col * col_width
        y = margin + row * row_height + 80  # Offset for heading
        width = col_width - margin
        height = row_height - margin
        
        # Add material name
        label = ET.SubElement(svg_root, "text", {
            "x": str(x + width/2),
            "y": str(y - 10),
            "text-anchor": "middle",
            "font-family": "Arial",
            "font-size": "16",
            "font-weight": "bold"
        })
        label.text = material["name"].title()
        
        # Add sample rectangle
        rect = ET.SubElement(svg_root, "rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "stroke": "#000000",
            "stroke-width": "1"
        })
        
        # Generate material textures
        material_elements = material_lib.get_material_texture(
            material["name"],
            x,
            y,
            width,
            height,
            material["color"],
            material["options"]
        )
        
        # Add material elements to SVG
        for element in material_elements:
            svg_root.append(element)
    
    # Save to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / "material_samples.svg"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prettify_svg(svg_root))
        
    print(f"Saved material samples to {filename}")


def create_pattern_sample_svg():
    """Create an SVG showcasing different pattern generators."""
    # Initialize pattern generator
    pattern_gen = PatternGenerator()
    
    # Create SVG root element
    svg_root = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "width": "800",
        "height": "600",
        "viewBox": "0 0 800 600"
    })
    
    # Add defs section for patterns
    defs = ET.SubElement(svg_root, "defs")
    
    # Create patterns
    grid_pattern_id = pattern_gen.generate_grid_pattern(
        cell_width=20.0,
        cell_height=20.0,
        line_width=1.0,
        line_color="#000000",
        background_color="#FFFFFF",
        opacity=0.8
    )
    
    stripe_pattern_id = pattern_gen.generate_stripe_pattern(
        stripe_width=10.0,
        stripe_spacing=10.0,
        stripe_color="#3366CC",
        background_color="#FFFFFF",
        angle=45.0
    )
    
    dot_pattern_id = pattern_gen.generate_dot_pattern(
        dot_radius=3.0,
        spacing_x=15.0,
        spacing_y=15.0,
        dot_color="#CC3366",
        background_color="#FFFFCC"
    )
    
    window_pattern_id = pattern_gen.generate_window_pattern(
        width=40.0,
        height=60.0,
        divisions_x=2,
        divisions_y=3,
        frame_width=2.0,
        frame_color="#333333",
        glass_color="#AADDFF"
    )
    
    brick_pattern_id = pattern_gen.generate_brick_pattern(
        brick_width=30.0,
        brick_height=15.0,
        mortar_width=2.0,
        brick_color="#AA4444",
        mortar_color="#CCCCCC"
    )
    
    # Add pattern elements to defs
    for pattern in pattern_gen.get_pattern_defs():
        defs.append(pattern)
    
    # Add background
    bg = ET.SubElement(svg_root, "rect", {
        "x": "0",
        "y": "0",
        "width": "800",
        "height": "600",
        "fill": "#F0F0F0"
    })
    
    # Define patterns to showcase
    patterns = [
        {"name": "Grid Pattern", "id": grid_pattern_id},
        {"name": "Stripe Pattern", "id": stripe_pattern_id},
        {"name": "Dot Pattern", "id": dot_pattern_id},
        {"name": "Window Pattern", "id": window_pattern_id},
        {"name": "Brick Pattern", "id": brick_pattern_id}
    ]
    
    # Add pattern samples
    margin = 50
    columns = 2
    row_height = 200
    col_width = 350
    
    # Create heading
    heading = ET.SubElement(svg_root, "text", {
        "x": "400",
        "y": "40",
        "text-anchor": "middle",
        "font-family": "Arial",
        "font-size": "24",
        "font-weight": "bold"
    })
    heading.text = "SVG Pattern Generator"
    
    # Create description
    desc = ET.SubElement(svg_root, "text", {
        "x": "400",
        "y": "70",
        "text-anchor": "middle",
        "font-family": "Arial",
        "font-size": "14"
    })
    desc.text = "Reusable pattern definitions for SVG illustrations"
    
    # Add pattern samples
    for i, pattern in enumerate(patterns):
        row = i // columns
        col = i % columns
        
        x = margin + col * col_width
        y = margin + row * row_height + 80  # Offset for heading
        width = col_width - margin
        height = row_height - margin
        
        # Add pattern name
        label = ET.SubElement(svg_root, "text", {
            "x": str(x + width/2),
            "y": str(y - 10),
            "text-anchor": "middle",
            "font-family": "Arial",
            "font-size": "16",
            "font-weight": "bold"
        })
        label.text = pattern["name"]
        
        # Add sample rectangle with pattern fill
        rect = ET.SubElement(svg_root, "rect", {
            "x": str(x),
            "y": str(y),
            "width": str(width),
            "height": str(height),
            "fill": f"url(#{pattern['id']})",
            "stroke": "#000000",
            "stroke-width": "1"
        })
    
    # Save to file
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / "pattern_samples.svg"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(prettify_svg(svg_root))
        
    print(f"Saved pattern samples to {filename}")


# Define the PromptToSVGEngine class here, before it's used
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
        ground_y = self.height - ground_height
        
        ET.SubElement(svg_root, "rect", {
            "x": "0",
            "y": str(ground_y),
            "width": str(self.width),
            "height": str(ground_height),
            "fill": "#228B22"  # Forest green for ground
        })
        
        # Add mountains if mentioned
        if has_mountain:
            self._add_mountains(svg_root)
        
        # Add water (lake/ocean) if mentioned
        if has_water:
            self._add_water(svg_root)
        
        # Add forest/trees if mentioned
        if has_forest:
            self._add_forest(svg_root)
        
        # Add sun if mentioned
        if has_sun:
            self._add_sun(svg_root)
        
        # Add buildings if mentioned
        if has_buildings:
            self._add_buildings(svg_root)
            
        # Convert to string
        return prettify_svg(svg_root)
    
    def _add_mountains(self, svg_root):
        """Add mountains to the scene."""
        # Create 2-3 mountains
        mountain_count = random.randint(2, 3)
        base_y = self.height * 0.7  # Ground level
        
        for i in range(mountain_count):
            # Randomize mountain position and size
            peak_height = random.uniform(self.height * 0.3, self.height * 0.5)
            mountain_width = random.uniform(self.width * 0.2, self.width * 0.4)
            x_pos = (i * self.width / mountain_count) + random.uniform(-self.width * 0.1, self.width * 0.1)
            
            # Ensure the mountain is within canvas bounds
            x_pos = max(0, min(x_pos, self.width - mountain_width))
            
            # Create mountain path (triangular shape)
            path_data = f"M{x_pos},{base_y} "
            path_data += f"L{x_pos + mountain_width/2},{base_y - peak_height} "
            path_data += f"L{x_pos + mountain_width},{base_y} Z"
            
            # Color based on height (taller mountains are darker)
            color_value = int(180 - (peak_height / self.height) * 100)
            mountain_color = f"#{color_value:02x}{color_value:02x}{color_value:02x}"
            
            # Add mountain to SVG
            ET.SubElement(svg_root, "path", {
                "d": path_data,
                "fill": mountain_color,
                "stroke": "#000000",
                "stroke-width": "1",
                "stroke-opacity": "0.3"
            })
            
            # Add snow cap if mountain is tall enough
            if peak_height > self.height * 0.4:
                snow_height = peak_height * 0.2
                snow_path = f"M{x_pos + mountain_width*0.25},{base_y - peak_height + snow_height} "
                snow_path += f"L{x_pos + mountain_width/2},{base_y - peak_height} "
                snow_path += f"L{x_pos + mountain_width*0.75},{base_y - peak_height + snow_height} Z"
                
                ET.SubElement(svg_root, "path", {
                    "d": snow_path,
                    "fill": "#FFFFFF",
                    "stroke": "#CCCCCC",
                    "stroke-width": "0.5"
                })
    
    def _add_water(self, svg_root):
        """Add a lake or ocean to the scene."""
        # Determine if it's a lake or ocean
        is_lake = random.choice([True, False])
        base_y = self.height * 0.7  # Ground level
        
        if is_lake:
            # Create an elliptical lake
            lake_width = self.width * 0.3
            lake_height = self.height * 0.1
            lake_x = random.uniform(self.width * 0.1, self.width * 0.6)
            lake_y = base_y - lake_height / 2
            
            ET.SubElement(svg_root, "ellipse", {
                "cx": str(lake_x + lake_width/2),
                "cy": str(lake_y + lake_height/2),
                "rx": str(lake_width/2),
                "ry": str(lake_height/2),
                "fill": "url(#water_gradient)",
                "stroke": "#4682B4",
                "stroke-width": "1"
            })
        else:
            # Create an ocean (rectangle at the bottom)
            ocean_height = self.height * 0.2
            ocean_y = self.height - ocean_height
            
            ET.SubElement(svg_root, "rect", {
                "x": "0",
                "y": str(ocean_y),
                "width": str(self.width),
                "height": str(ocean_height),
                "fill": "url(#water_gradient)"
            })
            
            # Add a few waves
            for i in range(5):
                wave_y = ocean_y + random.uniform(0, ocean_height * 0.3)
                wave_width = self.width * 0.2
                wave_x = i * (self.width / 5) - wave_width / 2
                
                ET.SubElement(svg_root, "path", {
                    "d": f"M{wave_x},{wave_y} Q{wave_x + wave_width/2},{wave_y - 5} {wave_x + wave_width},{wave_y}",
                    "fill": "none",
                    "stroke": "#FFFFFF",
                    "stroke-width": "2",
                    "stroke-opacity": "0.3"
                })
    
    def _add_forest(self, svg_root):
        """Add a forest with multiple trees."""
        tree_count = 15  # Number of trees
        base_y = self.height * 0.7  # Ground level
        forest_width = self.width * 0.8
        forest_start_x = (self.width - forest_width) / 2
        
        for i in range(tree_count):
            # Randomize tree position and size
            tree_x = forest_start_x + random.uniform(0, forest_width)
            tree_height = random.uniform(self.height * 0.1, self.height * 0.2)
            tree_width = tree_height * 0.6
            
            # Calculate tree position
            trunk_width = tree_width * 0.2
            trunk_height = tree_height * 0.4
            trunk_x = tree_x - trunk_width / 2
            trunk_y = base_y - trunk_height
            
            # Add tree trunk (rectangle)
            ET.SubElement(svg_root, "rect", {
                "x": str(trunk_x),
                "y": str(trunk_y),
                "width": str(trunk_width),
                "height": str(trunk_height),
                "fill": "#8B4513",  # Saddle brown
                "stroke": "#422109",
                "stroke-width": "0.5"
            })
            
            # Add foliage (circles or triangles, depending on tree type)
            if random.random() > 0.3:  # 70% chance of circular foliage
                # Circular foliage (deciduous tree)
                foliage_radius = tree_width / 2
                foliage_cx = tree_x
                foliage_cy = trunk_y - foliage_radius * 0.8
                
                # Vary the green color
                green_value = random.randint(100, 180)
                foliage_color = f"#00{green_value:02x}00"
                
                ET.SubElement(svg_root, "circle", {
                    "cx": str(foliage_cx),
                    "cy": str(foliage_cy),
                    "r": str(foliage_radius),
                    "fill": foliage_color,
                    "stroke": "#006400",  # Dark green
                    "stroke-width": "0.5"
                })
            else:
                # Triangular foliage (coniferous tree)
                triangle_height = tree_height * 0.7
                triangle_width = tree_width
                triangle_x = tree_x - triangle_width / 2
                triangle_y = base_y - trunk_height - triangle_height
                
                # Path for triangular shape
                path_data = f"M{triangle_x},{triangle_y + triangle_height} "
                path_data += f"L{triangle_x + triangle_width/2},{triangle_y} "
                path_data += f"L{triangle_x + triangle_width},{triangle_y + triangle_height} Z"
                
                ET.SubElement(svg_root, "path", {
                    "d": path_data,
                    "fill": "#006400",  # Dark green
                    "stroke": "#004d00",
                    "stroke-width": "0.5"
                })
    
    def _add_sun(self, svg_root):
        """Add a sun to the scene."""
        # Sun position (usually top right or top left)
        if random.random() > 0.5:
            sun_x = self.width * 0.85  # Right side
        else:
            sun_x = self.width * 0.15  # Left side
        
        sun_y = self.height * 0.15
        sun_radius = self.width * 0.06
        
        # Add sun circle
        ET.SubElement(svg_root, "circle", {
            "cx": str(sun_x),
            "cy": str(sun_y),
            "r": str(sun_radius),
            "fill": "#FFD700",  # Gold
            "stroke": "#FFA500",  # Orange
            "stroke-width": "2"
        })
        
        # Add sun rays
        for i in range(8):
            angle = i * 45  # 8 rays evenly spaced
            ray_length = sun_radius * 0.7
            
            # Calculate ray end position
            end_x = sun_x + ray_length * math.cos(math.radians(angle))
            end_y = sun_y + ray_length * math.sin(math.radians(angle))
            
            # Create ray
            ET.SubElement(svg_root, "line", {
                "x1": str(sun_x),
                "y1": str(sun_y),
                "x2": str(end_x),
                "y2": str(end_y),
                "stroke": "#FFD700",
                "stroke-width": "3",
                "stroke-linecap": "round"
            })
    
    def _add_buildings(self, svg_root):
        """Add buildings to the scene."""
        building_count = random.randint(3, 6)
        base_y = self.height * 0.7  # Ground level
        skyline_width = self.width * 0.6
        skyline_start_x = (self.width - skyline_width) / 2
        
        building_positions = []
        for i in range(building_count):
            # Randomize building position and size
            building_width = random.uniform(self.width * 0.05, self.width * 0.12)
            building_height = random.uniform(self.height * 0.15, self.height * 0.3)
            building_x = skyline_start_x + (i * skyline_width / building_count)
            
            # Add some randomness to x position, but avoid overlaps
            if i > 0:
                prev_x_end = building_positions[-1][0] + building_positions[-1][2]
                building_x = max(building_x, prev_x_end + 10)
            
            # Make sure the building is within canvas
            building_x = min(building_x, self.width - building_width)
            
            # Building position and dimensions (x, y, width, height)
            building_positions.append((building_x, base_y - building_height, building_width, building_height))
            
            # Choose a building color (grayscale)
            color_value = random.randint(100, 200)
            building_color = f"#{color_value:02x}{color_value:02x}{color_value:02x}"
            
            # Add building rectangle
            ET.SubElement(svg_root, "rect", {
                "x": str(building_x),
                "y": str(base_y - building_height),
                "width": str(building_width),
                "height": str(building_height),
                "fill": building_color,
                "stroke": "#000000",
                "stroke-width": "1"
            })
            
            # Add windows
            window_width = building_width * 0.15
            window_height = window_width * 1.5
            window_margin = building_width * 0.1
            windows_per_row = max(1, int((building_width - window_margin) / (window_width + window_margin)))
            windows_per_column = max(2, int((building_height - window_margin) / (window_height + window_margin)))
            
            for row in range(windows_per_column):
                for col in range(windows_per_row):
                    window_x = building_x + window_margin + col * (window_width + window_margin)
                    window_y = (base_y - building_height) + window_margin + row * (window_height + window_margin)
                    
                    # Only add window if it fits within the building
                    if (window_x + window_width <= building_x + building_width and 
                        window_y + window_height <= base_y):
                        
                        # Random window color (yellow/white for lights)
                        if random.random() > 0.3:  # 70% chance of light on
                            window_color = "#FFFF88" if random.random() > 0.5 else "#FFFFFF"
                        else:
                            window_color = "#333333"  # Dark window (light off)
                        
                        ET.SubElement(svg_root, "rect", {
                            "x": str(window_x),
                            "y": str(window_y),
                            "width": str(window_width),
                            "height": str(window_height),
                            "fill": window_color,
                            "stroke": "#000000",
                            "stroke-width": "0.5"
                        })


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
        {
            "name": "mountain_landscape",
            "prompt": "A mountain landscape with forest and lake at sunset",
            "width": 800,
            "height": 600
        },
        {
            "name": "cityscape_night",
            "prompt": "A modern cityscape at night with skyscrapers",
            "width": 800,
            "height": 600
        },
        {
            "name": "abstract_composition",
            "prompt": "Abstract geometric composition with blue and orange shapes",
            "width": 800,
            "height": 600
        },
        {
            "name": "seascape_lighthouse",
            "prompt": "A seascape with lighthouse on rocky shore",
            "width": 1200,
            "height": 600
        }
    ]
    
    # Initialize default engine for standard dimensions
    engine = PromptToSVGEngine(width=800, height=600)
    
    # Generate and save each example
    for example in examples:
        print(f"\nGenerating: {example['prompt']}")
        
        # Create engine with specific dimensions if needed
        if example["width"] != 800 or example["height"] != 600:
            custom_engine = PromptToSVGEngine(width=example["width"], height=example["height"])
            svg_code = custom_engine.generate_svg(example["prompt"])
        else:
            svg_code = engine.generate_svg(example["prompt"])
        
        # Save the result
        filename = output_dir / f"{example['name']}.svg"
        save_svg(svg_code, filename)
        
        # Print stats
        svg_size_kb = len(svg_code) / 1024
        print(f"SVG size: {svg_size_kb:.2f}KB")


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
        ground_y = self.height - ground_height
        
        ET.SubElement(svg_root, "rect", {
            "x": "0",
            "y": str(ground_y),
            "width": str(self.width),
            "height": str(ground_height),
            "fill": "#228B22"  # Forest green for ground
        })
        
        # Add mountains if mentioned
        if has_mountain:
            self._add_mountains(svg_root)
        
        # Add water (lake/ocean) if mentioned
        if has_water:
            self._add_water(svg_root)
        
        # Add forest/trees if mentioned
        if has_forest:
            self._add_forest(svg_root)
        
        # Add sun if mentioned
        if has_sun:
            self._add_sun(svg_root)
        
        # Add buildings if mentioned
        if has_buildings:
            self._add_buildings(svg_root)
            
        # Convert to string
        return prettify_svg(svg_root)
    
    def _add_mountains(self, svg_root):
        """Add mountains to the scene."""
        # Create 2-3 mountains
        mountain_count = random.randint(2, 3)
        base_y = self.height * 0.7  # Ground level
        
        for i in range(mountain_count):
            # Randomize mountain position and size
            peak_height = random.uniform(self.height * 0.3, self.height * 0.5)
            mountain_width = random.uniform(self.width * 0.2, self.width * 0.4)
            x_pos = (i * self.width / mountain_count) + random.uniform(-self.width * 0.1, self.width * 0.1)
            
            # Ensure the mountain is within canvas bounds
            x_pos = max(0, min(x_pos, self.width - mountain_width))
            
            # Create mountain path (triangular shape)
            path_data = f"M{x_pos},{base_y} "
            path_data += f"L{x_pos + mountain_width/2},{base_y - peak_height} "
            path_data += f"L{x_pos + mountain_width},{base_y} Z"
            
            # Color based on height (taller mountains are darker)
            color_value = int(180 - (peak_height / self.height) * 100)
            mountain_color = f"#{color_value:02x}{color_value:02x}{color_value:02x}"
            
            # Add mountain to SVG
            ET.SubElement(svg_root, "path", {
                "d": path_data,
                "fill": mountain_color,
                "stroke": "#000000",
                "stroke-width": "1",
                "stroke-opacity": "0.3"
            })
            
            # Add snow cap if mountain is tall enough
            if peak_height > self.height * 0.4:
                snow_height = peak_height * 0.2
                snow_path = f"M{x_pos + mountain_width*0.25},{base_y - peak_height + snow_height} "
                snow_path += f"L{x_pos + mountain_width/2},{base_y - peak_height} "
                snow_path += f"L{x_pos + mountain_width*0.75},{base_y - peak_height + snow_height} Z"
                
                ET.SubElement(svg_root, "path", {
                    "d": snow_path,
                    "fill": "#FFFFFF",
                    "stroke": "#CCCCCC",
                    "stroke-width": "0.5"
                })
    
    def _add_water(self, svg_root):
        """Add a lake or ocean to the scene."""
        # Determine if it's a lake or ocean
        is_lake = random.choice([True, False])
        base_y = self.height * 0.7  # Ground level
        
        if is_lake:
            # Create an elliptical lake
            lake_width = self.width * 0.3
            lake_height = self.height * 0.1
            lake_x = random.uniform(self.width * 0.1, self.width * 0.6)
            lake_y = base_y - lake_height / 2
            
            ET.SubElement(svg_root, "ellipse", {
                "cx": str(lake_x + lake_width/2),
                "cy": str(lake_y + lake_height/2),
                "rx": str(lake_width/2),
                "ry": str(lake_height/2),
                "fill": "url(#water_gradient)",
                "stroke": "#4682B4",
                "stroke-width": "1"
            })
        else:
            # Create an ocean (rectangle at the bottom)
            ocean_height = self.height * 0.2
            ocean_y = self.height - ocean_height
            
            ET.SubElement(svg_root, "rect", {
                "x": "0",
                "y": str(ocean_y),
                "width": str(self.width),
                "height": str(ocean_height),
                "fill": "url(#water_gradient)"
            })
            
            # Add a few waves
            for i in range(5):
                wave_y = ocean_y + random.uniform(0, ocean_height * 0.3)
                wave_width = self.width * 0.2
                wave_x = i * (self.width / 5) - wave_width / 2
                
                ET.SubElement(svg_root, "path", {
                    "d": f"M{wave_x},{wave_y} Q{wave_x + wave_width/2},{wave_y - 5} {wave_x + wave_width},{wave_y}",
                    "fill": "none",
                    "stroke": "#FFFFFF",
                    "stroke-width": "2",
                    "stroke-opacity": "0.3"
                })
    
    def _add_forest(self, svg_root):
        """Add a forest with multiple trees."""
        tree_count = 15  # Number of trees
        base_y = self.height * 0.7  # Ground level
        forest_width = self.width * 0.8
        forest_start_x = (self.width - forest_width) / 2
        
        for i in range(tree_count):
            # Randomize tree position and size
            tree_x = forest_start_x + random.uniform(0, forest_width)
            tree_height = random.uniform(self.height * 0.1, self.height * 0.2)
            tree_width = tree_height * 0.6
            
            # Calculate tree position
            trunk_width = tree_width * 0.2
            trunk_height = tree_height * 0.4
            trunk_x = tree_x - trunk_width / 2
            trunk_y = base_y - trunk_height
            
            # Add tree trunk (rectangle)
            ET.SubElement(svg_root, "rect", {
                "x": str(trunk_x),
                "y": str(trunk_y),
                "width": str(trunk_width),
                "height": str(trunk_height),
                "fill": "#8B4513",  # Saddle brown
                "stroke": "#422109",
                "stroke-width": "0.5"
            })
            
            # Add foliage (circles or triangles, depending on tree type)
            if random.random() > 0.3:  # 70% chance of circular foliage
                # Circular foliage (deciduous tree)
                foliage_radius = tree_width / 2
                foliage_cx = tree_x
                foliage_cy = trunk_y - foliage_radius * 0.8
                
                # Vary the green color
                green_value = random.randint(100, 180)
                foliage_color = f"#00{green_value:02x}00"
                
                ET.SubElement(svg_root, "circle", {
                    "cx": str(foliage_cx),
                    "cy": str(foliage_cy),
                    "r": str(foliage_radius),
                    "fill": foliage_color,
                    "stroke": "#006400",  # Dark green
                    "stroke-width": "0.5"
                })
            else:
                # Triangular foliage (coniferous tree)
                triangle_height = tree_height * 0.7
                triangle_width = tree_width
                triangle_x = tree_x - triangle_width / 2
                triangle_y = base_y - trunk_height - triangle_height
                
                # Path for triangular shape
                path_data = f"M{triangle_x},{triangle_y + triangle_height} "
                path_data += f"L{triangle_x + triangle_width/2},{triangle_y} "
                path_data += f"L{triangle_x + triangle_width},{triangle_y + triangle_height} Z"
                
                ET.SubElement(svg_root, "path", {
                    "d": path_data,
                    "fill": "#006400",  # Dark green
                    "stroke": "#004d00",
                    "stroke-width": "0.5"
                })
    
    def _add_sun(self, svg_root):
        """Add a sun to the scene."""
        # Sun position (usually top right or top left)
        if random.random() > 0.5:
            sun_x = self.width * 0.85  # Right side
        else:
            sun_x = self.width * 0.15  # Left side
        
        sun_y = self.height * 0.15
        sun_radius = self.width * 0.06
        
        # Add sun circle
        ET.SubElement(svg_root, "circle", {
            "cx": str(sun_x),
            "cy": str(sun_y),
            "r": str(sun_radius),
            "fill": "#FFD700",  # Gold
            "stroke": "#FFA500",  # Orange
            "stroke-width": "2"
        })
        
        # Add sun rays
        for i in range(8):
            angle = i * 45  # 8 rays evenly spaced
            ray_length = sun_radius * 0.7
            
            # Calculate ray end position
            end_x = sun_x + ray_length * math.cos(math.radians(angle))
            end_y = sun_y + ray_length * math.sin(math.radians(angle))
            
            # Create ray
            ET.SubElement(svg_root, "line", {
                "x1": str(sun_x),
                "y1": str(sun_y),
                "x2": str(end_x),
                "y2": str(end_y),
                "stroke": "#FFD700",
                "stroke-width": "3",
                "stroke-linecap": "round"
            })
    
    def _add_buildings(self, svg_root):
        """Add buildings to the scene."""
        building_count = random.randint(3, 6)
        base_y = self.height * 0.7  # Ground level
        skyline_width = self.width * 0.6
        skyline_start_x = (self.width - skyline_width) / 2
        
        building_positions = []
        for i in range(building_count):
            # Randomize building position and size
            building_width = random.uniform(self.width * 0.05, self.width * 0.12)
            building_height = random.uniform(self.height * 0.15, self.height * 0.3)
            building_x = skyline_start_x + (i * skyline_width / building_count)
            
            # Add some randomness to x position, but avoid overlaps
            if i > 0:
                prev_x_end = building_positions[-1][0] + building_positions[-1][2]
                building_x = max(building_x, prev_x_end + 10)
            
            # Make sure the building is within canvas
            building_x = min(building_x, self.width - building_width)
            
            # Building position and dimensions (x, y, width, height)
            building_positions.append((building_x, base_y - building_height, building_width, building_height))
            
            # Choose a building color (grayscale)
            color_value = random.randint(100, 200)
            building_color = f"#{color_value:02x}{color_value:02x}{color_value:02x}"
            
            # Add building rectangle
            ET.SubElement(svg_root, "rect", {
                "x": str(building_x),
                "y": str(base_y - building_height),
                "width": str(building_width),
                "height": str(building_height),
                "fill": building_color,
                "stroke": "#000000",
                "stroke-width": "1"
            })
            
            # Add windows
            window_width = building_width * 0.15
            window_height = window_width * 1.5
            window_margin = building_width * 0.1
            windows_per_row = max(1, int((building_width - window_margin) / (window_width + window_margin)))
            windows_per_column = max(2, int((building_height - window_margin) / (window_height + window_margin)))
            
            for row in range(windows_per_column):
                for col in range(windows_per_row):
                    window_x = building_x + window_margin + col * (window_width + window_margin)
                    window_y = (base_y - building_height) + window_margin + row * (window_height + window_margin)
                    
                    # Only add window if it fits within the building
                    if (window_x + window_width <= building_x + building_width and 
                        window_y + window_height <= base_y):
                        
                        # Random window color (yellow/white for lights)
                        if random.random() > 0.3:  # 70% chance of light on
                            window_color = "#FFFF88" if random.random() > 0.5 else "#FFFFFF"
                        else:
                            window_color = "#333333"  # Dark window (light off)
                        
                        ET.SubElement(svg_root, "rect", {
                            "x": str(window_x),
                            "y": str(window_y),
                            "width": str(window_width),
                            "height": str(window_height),
                            "fill": window_color,
                            "stroke": "#000000",
                            "stroke-width": "0.5"
                        })



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
    
    # Create parent directories if they don't exist
    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Write SVG content to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        print(f"Saved SVG to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving SVG to {filename}: {e}")
        return False

# Google Cloud Dataflow and Vertex AI Integration

try:
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    from apache_beam.options.value_provider import ValueProvider, StaticValueProvider, NestedValueProvider
    from apache_beam.io import fileio
    from google.cloud import storage
    from google.cloud import aiplatform
    CLOUD_DEPS_INSTALLED = True
except ImportError:
    CLOUD_DEPS_INSTALLED = False
    print("Google Cloud dependencies not installed. Cloud features will be disabled.")
    print("To enable, install: pip install apache-beam[gcp] google-cloud-storage google-cloud-aiplatform")


# First check if Apache Beam is installed
try:
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    BEAM_AVAILABLE = True
except ImportError:
    print("Apache Beam not installed. Using local execution mode only.")
    BEAM_AVAILABLE = False
    
    # Create dummy classes for Beam components
    class PipelineOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            
        def view_as(self, cls):
            return self
    
    # Simple DoFn replacement that can be subclassed
    class DoFn:
        def process(self, element):
            pass
    
    # Dummy beam module with basic components for local execution
    class beam:
        class Pipeline:
            def __init__(self, options=None):
                self.options = options
                self.transforms = []
            
            def run(self):
                return PipelineResult()
        
        class PipelineResult:
            def wait_until_finish(self):
                pass
        
        class Create:
            def __init__(self, values):
                self.values = values
        
        class Map:
            def __init__(self, fn):
                self.fn = fn
        
        class ParDo:
            def __init__(self, fn):
                self.fn = fn
        
        DoFn = DoFn
        
        class io:
            class fileio:
                class WriteToFiles:
                    def __init__(self, *args, **kwargs):
                        pass
            
class SVGGeneratorOptions(PipelineOptions):
    """Pipeline options for SVG generation with Dataflow."""
    
    @classmethod
    def _add_argparse_args(cls, parser):
        # Runtime parameters for Dataflow template
        parser.add_value_provider_argument(
            '--prompt', 
            help='Description of the scene to generate')
        parser.add_value_provider_argument(
            '--output_bucket', 
            help='GCS bucket to save SVGs, e.g., gs://my-bucket/outputs/')
        parser.add_value_provider_argument(
            '--style', 
            default='realistic',
            help='Style of rendering: realistic, abstract, minimalist')
        parser.add_value_provider_argument(
            '--detail_level', 
            default='medium',
            help='Level of detail: low, medium, high')
        parser.add_value_provider_argument(
            '--enhance_with_vertex_ai',
            default='false',
            help='Whether to enhance output using Vertex AI')


class ParsePromptFn(beam.DoFn):
    """Parse a text prompt into semantic elements."""
    
    def __init__(self, prompt_provider):
        self.prompt_provider = prompt_provider
    
    def process(self, element):
        if not self.prompt_provider.is_accessible():
            raise ValueError("Prompt parameter not accessible. This function requires runtime values.")
            
        prompt = self.prompt_provider.get()
        
        # Here we would use semantic parsing from the original code
        # For now we'll use a simple representation
        scene_elements = {
            'prompt': prompt,
            'objects': self._extract_objects(prompt),
            'style': self._extract_style(prompt)
        }
        
        yield scene_elements
    
    def _extract_objects(self, prompt):
        """Extract key objects from the prompt."""
        objects = []
        keywords = ['mountain', 'lake', 'tree', 'forest', 'building', 'sky', 'sun',
                   'river', 'ocean', 'city', 'house', 'clouds', 'road']
        
        for keyword in keywords:
            if keyword in prompt.lower():
                objects.append(keyword)
                
        return objects
    
    def _extract_style(self, prompt):
        """Extract style information from the prompt."""
        styles = {
            'abstract': ['abstract', 'geometric', 'cubist', 'modern'],
            'realistic': ['realistic', 'natural', 'photo', 'detailed'],
            'minimalist': ['minimalist', 'simple', 'clean', 'minimal']
        }
        
        for style, keywords in styles.items():
            for keyword in keywords:
                if keyword in prompt.lower():
                    return style
                    
        return 'realistic'  # Default style


class BuildSceneFn(beam.DoFn):
    """Build a scene structure from semantic elements."""
    
    def __init__(self, style_provider, detail_level_provider):
        self.style_provider = style_provider
        self.detail_level_provider = detail_level_provider
    
    def process(self, scene_elements):
        if self.style_provider.is_accessible() and self.detail_level_provider.is_accessible():
            style_override = self.style_provider.get()
            detail_override = self.detail_level_provider.get()
            
            if style_override.lower() not in ['default', '']:
                scene_elements['style'] = style_override
                
            scene_elements['detail_level'] = detail_override
        
        # Create a scene structure that can be used by the SVG generator
        scene_structure = {
            'canvas': {'width': 800, 'height': 600},
            'objects': [],
            'style': scene_elements['style'],
            'detail_level': scene_elements.get('detail_level', 'medium')
        }
        
        # Add objects to the scene based on extracted elements
        for obj in scene_elements['objects']:
            if obj == 'mountain':
                scene_structure['objects'].append({
                    'type': 'mountain',
                    'position': {'x': 400, 'y': 400},
                    'size': {'width': 300, 'height': 200}
                })
            elif obj == 'lake':
                scene_structure['objects'].append({
                    'type': 'water',
                    'position': {'x': 400, 'y': 500},
                    'size': {'width': 350, 'height': 100}
                })
            elif obj in ['tree', 'forest']:
                scene_structure['objects'].append({
                    'type': 'forest',
                    'position': {'x': 300, 'y': 450},
                    'size': {'width': 200, 'height': 150}
                })
            # Add more object types as needed
        
        yield scene_structure


class GenerateSVGFn(beam.DoFn):
    """Generate SVG based on scene structure."""
    
    def process(self, scene_structure):
        # Create an instance of our SVG generator
        engine = PromptToSVGEngine(
            width=scene_structure['canvas']['width'],
            height=scene_structure['canvas']['height'],
            detail_level=scene_structure['detail_level']
        )
        
        # Generate SVG
        svg_content = engine.generate_svg(scene_structure['prompt'])
        
        # Output SVG with metadata
        yield {
            'svg_content': svg_content,
            'filename': f"scene_{hash(scene_structure['prompt'])}.svg",
            'metadata': scene_structure
        }


class EnhanceWithVertexAIFn(beam.DoFn):
    """Optional enhancement using Vertex AI."""
    
    def __init__(self, enhance_provider):
        self.enhance_provider = enhance_provider
        self.vertex_client = None
    
    def setup(self):
        # Initialize Vertex AI client during worker setup
        if CLOUD_DEPS_INSTALLED and self.enhance_provider.get().lower() == 'true':
            aiplatform.init()
            self.vertex_client = True
    
    def process(self, element):
        if not self.enhance_provider.is_accessible() or self.enhance_provider.get().lower() != 'true':
            # Skip enhancement if not requested
            yield element
            return
            
        if not CLOUD_DEPS_INSTALLED or not self.vertex_client:
            print("Vertex AI enhancement requested but dependencies not available")
            yield element
            return
            
        # Here we would send the SVG to Vertex AI for enhancement
        # For this example we're just adding a comment to indicate it was processed
        svg_content = element['svg_content']
        enhanced_svg = svg_content.replace('</svg>', 
                                      '<!-- Enhanced with Vertex AI -->\n</svg>')
        
        enhanced_element = dict(element)
        enhanced_element['svg_content'] = enhanced_svg
        enhanced_element['enhanced'] = True
        
        yield enhanced_element


class SaveSVGFn(beam.DoFn):
    """Save SVG to a file or GCS bucket."""
    
    def __init__(self, output_bucket_provider):
        self.output_bucket_provider = output_bucket_provider
        self.gcs_client = None
    
    def setup(self):
        # Initialize GCS client during worker setup if running in cloud
        if CLOUD_DEPS_INSTALLED:
            try:
                self.gcs_client = storage.Client()
            except Exception as e:
                print(f"Warning: Could not initialize GCS client: {e}")
    
    def process(self, element):
        svg_content = element['svg_content']
        filename = element['filename']
        
        if self.output_bucket_provider.is_accessible():
            # We have a cloud storage path
            output_path = self.output_bucket_provider.get()
            
            if output_path.startswith('gs://') and self.gcs_client:
                # Upload to GCS
                bucket_name = output_path.replace('gs://', '').split('/')[0]
                blob_prefix = '/'.join(output_path.replace('gs://', '').split('/')[1:])
                
                if not blob_prefix.endswith('/'):
                    blob_prefix += '/'
                
                full_path = f"{blob_prefix}{filename}"
                
                try:
                    bucket = self.gcs_client.bucket(bucket_name)
                    blob = bucket.blob(full_path)
                    blob.upload_from_string(svg_content)
                    print(f"Saved SVG to gs://{bucket_name}/{full_path}")
                    yield {'status': 'success', 'path': f"gs://{bucket_name}/{full_path}"}
                    return
                except Exception as e:
                    print(f"Error saving to GCS: {e}")
        
        # Local file storage fallback
        from pathlib import Path
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
            
        print(f"Saved SVG to {output_path}")
        yield {'status': 'success', 'path': str(output_path)}


def run_dataflow_pipeline(argv=None):
    """Execute the SVG generation pipeline on Dataflow."""
    if not CLOUD_DEPS_INSTALLED:
        print("Cannot run Dataflow pipeline without Google Cloud dependencies.")
        return
    
    pipeline_options = PipelineOptions(argv)
    svg_options = pipeline_options.view_as(SVGGeneratorOptions)
    
    with beam.Pipeline(options=pipeline_options) as p:
        (p 
         | 'Create Trigger' >> beam.Create([1])  # Singleton to trigger processing
         | 'Parse Prompt' >> beam.ParDo(ParsePromptFn(svg_options.prompt))
         | 'Build Scene' >> beam.ParDo(BuildSceneFn(svg_options.style, svg_options.detail_level))
         | 'Generate SVG' >> beam.ParDo(GenerateSVGFn())
         | 'Enhance with Vertex AI' >> beam.ParDo(EnhanceWithVertexAIFn(svg_options.enhance_with_vertex_ai))
         | 'Save Output' >> beam.ParDo(SaveSVGFn(svg_options.output_bucket))
        )


def train_vertex_model(bucket, prompts_file, model_name='svg-generator'):
    """Train a model on Vertex AI to enhance SVG generation."""
    if not CLOUD_DEPS_INSTALLED:
        print("Cannot train model without Google Cloud dependencies.")
        return
    
    try:
        # Initialize Vertex AI
        aiplatform.init()
        
        # This is a placeholder for actual model training
        # In a real implementation, you would:
        # 1. Load training data (prompts and SVG pairs)
        # 2. Create a dataset on Vertex AI
        # 3. Train a model using AutoML or custom training
        # 4. Deploy the model for prediction
        
        print(f"Training model '{model_name}' with data from {prompts_file} in bucket {bucket}")
        print("This is a placeholder for actual Vertex AI model training.")
        print("In a real implementation, you would create a custom training job or use AutoML.")
        
        return {
            "model_name": model_name,
            "status": "success",
            "message": "Model training simulation complete"
        }
    except Exception as e:
        print(f"Error training Vertex AI model: {e}")
        return {"status": "error", "message": str(e)}


# Extend the main function to handle cloud options
if __name__ == "__main__":
    import argparse
    
    # Record the start time for measuring performance
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="SVG Generator with Google Cloud integration")
    parser.add_argument('--cloud', action='store_true', help='Run in Google Cloud mode')
    parser.add_argument('--train', action='store_true', help='Train a model on Vertex AI')
    parser.add_argument('--bucket', type=str, help='GCS bucket for outputs or training data')
    parser.add_argument('--prompts', type=str, help='File containing training prompts')
    parser.add_argument('--project', type=str, help='Google Cloud project ID')
    parser.add_argument('--region', type=str, default='us-central1', help='Google Cloud region')
    
    # Parse known args to separate from potential Dataflow args
    args, remaining_args = parser.parse_known_args()
    
    if args.cloud:
        if args.train and CLOUD_DEPS_INSTALLED:
            print("Training model on Vertex AI...")
            result = train_vertex_model(args.bucket, args.prompts)
            print(f"Training result: {result}")
        elif CLOUD_DEPS_INSTALLED:
            # Add required Dataflow args
            dataflow_args = remaining_args + [
                f'--project={args.project}',
                f'--region={args.region}',
                '--runner=DataflowRunner'
            ]
            print("Running SVG generation pipeline on Dataflow...")
            run_dataflow_pipeline(dataflow_args)
        else:
            print("Google Cloud dependencies not installed. Cannot run in cloud mode.")
    else:
        # Run the original local code
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Generate material and pattern samples
        print("Generating material samples...")
        material_svg = create_material_sample_svg()
        save_svg(material_svg, output_dir / "material_samples.svg")

        pattern_svg = create_pattern_sample_svg()
        save_svg(pattern_svg, output_dir / "pattern_samples.svg")

        print("Done! Check the output directory for sample SVGs.")

        # Generate example SVGs using the text-to-SVG engine
        engine = PromptToSVGEngine(width=800, height=600)

        # Example prompts
        prompts = [
            "A mountain landscape with forest and lake at sunset",
            "A modern cityscape at night with skyscrapers",
            "Abstract geometric composition with blue and orange shapes",
            "A seascape with lighthouse on rocky shore"
        ]

        # Enhanced example prompts with your provided test prompts
        test_prompts = [
            "a purple forest at dusk",
            "gray wool coat with a faux fur collar",
            "a lighthouse overlooking the ocean",
            "burgundy corduroy pants with patch pockets and silver buttons",
            "orange corduroy overalls",
            "a purple silk scarf with tassel trim",
            "a green lagoon under a cloudy sky",
            "crimson rectangles forming a chaotic grid",
            "purple pyramids spiraling around a bronze cone",
            "magenta trapezoids layered on a transluscent silver sheet",
            "a snowy plain",
            "black and white checkered pants",
            "a starlit night over snow-covered peaks",
            "khaki triangles and azure crescents",
            "a maroon dodecahedron interwoven with teal threads"
        ]
        
        # Combine original prompts with test prompts
        all_prompts = prompts + test_prompts
        
        # Statistics collection
        generation_stats = {
            "time": [],
            "size": [],
            "elements": []
        }
        
        # Use tqdm to create a progress bar
        print("\nStarting SVG generation with progress tracking...")
        
        # Create a progress bar with rich statistics
        for prompt in tqdm(all_prompts, desc="Generating SVGs", unit="svg"):
            # Measure generation time
            start_time_svg = time.time()
            
            # Generate the SVG
            svg_content = engine.generate_svg(prompt)
            
            # Calculate generation time
            gen_time = time.time() - start_time_svg
            generation_stats["time"].append(gen_time)
            
            # Create a safe filename
            filename = prompt.lower().replace(" ", "_")[:30] + ".svg"
            
            # Save the SVG
            save_svg(svg_content, output_dir / filename)
            
            # Collect statistics
            svg_size = len(svg_content)/1024  # Size in KB
            generation_stats["size"].append(svg_size)
            element_count = svg_content.count('<')  # Simple element count estimation
            generation_stats["elements"].append(element_count)
            
            # Update the progress bar description with current stats
            tqdm.write(f"Generated: {prompt[:30]}... | Size: {svg_size:.2f}KB | Time: {gen_time:.2f}s | Elements: {element_count}")
        
        # Print summary statistics
        avg_time = sum(generation_stats["time"]) / len(generation_stats["time"])
        avg_size = sum(generation_stats["size"]) / len(generation_stats["size"])
        avg_elements = sum(generation_stats["elements"]) / len(generation_stats["elements"])
        
        print(f"\nSVG Generation Summary:")
        print(f"Average generation time: {avg_time:.2f}s")
        print(f"Average SVG size: {avg_size:.2f}KB")
        print(f"Average element count: {avg_elements:.1f}")
        print(f"Total SVGs generated: {len(all_prompts)}")
        
        # Save statistics as JSON
        stats_file = output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump({
                "prompts": all_prompts,
                "statistics": generation_stats,
                "summary": {
                    "avg_time": avg_time,
                    "avg_size": avg_size,
                    "avg_elements": avg_elements,
                    "total_svgs": len(all_prompts)
                }
            }, f, indent=2)
        print(f"Statistics saved to {stats_file}")
        
    
    # Print runtime stats
    print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")
