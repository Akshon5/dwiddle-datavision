import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import json
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from src.inference import QuickDrawPredictor

REPORTS_DIR = 'reports'


def generate_fresh_sketches():
    """
    Constructs 48 completely fresh, diverse sketches (6 per class) on a 300x300 canvas.
    Varies orientation, stroke width (4-14 px), position, and structural style.
    """
    sketches = []
    
    def canvas():
        return Image.new('RGBA', (300, 300), (255, 255, 255, 255))
        
    # =========================================================================
    # 1. AIRPLANE (6 sketches)
    # =========================================================================
    # 1.1 Commercial Jet (Bold closed wings)
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([135, 30, 165, 260], outline=(0, 0, 0, 255), width=7)
    d.polygon([(140, 120), (30, 160), (40, 185), (140, 155)], outline=(0, 0, 0, 255), width=6)
    d.polygon([(160, 120), (270, 160), (260, 185), (160, 155)], outline=(0, 0, 0, 255), width=6)
    d.polygon([(140, 230), (80, 255), (90, 270), (140, 250)], outline=(0, 0, 0, 255), width=5)
    d.polygon([(160, 230), (220, 255), (210, 270), (160, 250)], outline=(0, 0, 0, 255), width=5)
    sketches.append(('airplane', 'Commercial jet with swept polygon wings', img))

    # 1.2 Cross Fuselage Propeller Plane
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([130, 45, 170, 245], outline=(0, 0, 0, 255), width=8)
    d.rectangle([45, 115, 255, 155], outline=(0, 0, 0, 255), width=8)
    d.rectangle([95, 220, 205, 245], outline=(0, 0, 0, 255), width=6)
    sketches.append(('airplane', 'Bold cross fuselage plane', img))

    # 1.3 Diagonal Fighter Jet
    img = canvas()
    d = ImageDraw.Draw(img)
    d.polygon([(240, 50), (220, 70), (60, 230), (50, 250), (70, 240), (230, 80)], outline=(0, 0, 0, 255), width=6)
    d.polygon([(180, 120), (90, 90), (120, 150)], outline=(0, 0, 0, 255), width=6)
    d.polygon([(150, 150), (180, 210), (210, 180)], outline=(0, 0, 0, 255), width=6)
    sketches.append(('airplane', 'Diagonal banking delta jet', img))

    # 1.4 Horizontal Glider
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([40, 135, 260, 165], outline=(0, 0, 0, 255), width=7)
    d.polygon([(135, 140), (150, 40), (165, 40), (150, 140)], outline=(0, 0, 0, 255), width=6)
    d.polygon([(135, 160), (150, 260), (165, 260), (150, 160)], outline=(0, 0, 0, 255), width=6)
    d.line([(55, 115), (75, 150)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('airplane', 'Horizontal glider with wide wingspan', img))

    # 1.5 Cartoon Biplane
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([80, 120, 220, 170], outline=(0, 0, 0, 255), width=8)
    d.line([(50, 100), (250, 100)], fill=(0, 0, 0, 255), width=8)
    d.line([(50, 190), (250, 190)], fill=(0, 0, 0, 255), width=8)
    d.line([(100, 100), (100, 190)], fill=(0, 0, 0, 255), width=5)
    d.line([(200, 100), (200, 190)], fill=(0, 0, 0, 255), width=5)
    sketches.append(('airplane', 'Biplane with struts', img))

    # 1.6 Thin Line Wing Plane (Stress Test)
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([140, 50, 160, 250], outline=(0, 0, 0, 255), width=5)
    d.line([(40, 140), (260, 140)], fill=(0, 0, 0, 255), width=5)
    d.line([(100, 230), (200, 230)], fill=(0, 0, 0, 255), width=4)
    sketches.append(('airplane', 'Thin line wings (stress case)', img))

    # =========================================================================
    # 2. APPLE (6 sketches)
    # =========================================================================
    # 2.1 Standard Round Apple
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([70, 75, 230, 235], outline=(0, 0, 0, 255), width=8)
    d.line([(150, 75), (155, 35)], fill=(0, 0, 0, 255), width=7)
    d.arc([155, 35, 195, 75], start=180, end=360, fill=(0, 0, 0, 255), width=6)
    sketches.append(('apple', 'Standard round apple with leaf', img))

    # 2.2 Bitten Apple
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([65, 70, 225, 230], outline=(0, 0, 0, 255), width=8)
    d.arc([185, 110, 245, 170], start=90, end=270, fill=(255, 255, 255, 255), width=18)
    d.line([(145, 70), (145, 35)], fill=(0, 0, 0, 255), width=7)
    sketches.append(('apple', 'Apple with prominent bite taken out', img))

    # 2.3 Heart Double-Lobe Apple
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([60, 70, 160, 170], start=180, end=360, fill=(0, 0, 0, 255), width=7)
    d.arc([140, 70, 240, 170], start=180, end=360, fill=(0, 0, 0, 255), width=7)
    d.arc([60, 120, 240, 250], start=0, end=180, fill=(0, 0, 0, 255), width=7)
    d.line([(150, 70), (150, 30)], fill=(0, 0, 0, 255), width=7)
    sketches.append(('apple', 'Heart-shaped double-lobed apple', img))

    # 2.4 Broad Stalk Apple
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([50, 80, 250, 240], outline=(0, 0, 0, 255), width=9)
    d.line([(150, 80), (160, 40)], fill=(0, 0, 0, 255), width=8)
    sketches.append(('apple', 'Broad thick-outline apple', img))

    # 2.5 Small Off-Center Apple
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([140, 130, 230, 220], outline=(0, 0, 0, 255), width=6)
    d.line([(185, 130), (190, 95)], fill=(0, 0, 0, 255), width=5)
    sketches.append(('apple', 'Small off-center apple', img))

    # 2.6 Apple with Double Leaf
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([70, 85, 230, 235], outline=(0, 0, 0, 255), width=7)
    d.line([(150, 85), (150, 45)], fill=(0, 0, 0, 255), width=6)
    d.arc([150, 45, 185, 75], start=180, end=360, fill=(0, 0, 0, 255), width=5)
    d.arc([115, 45, 150, 75], start=180, end=360, fill=(0, 0, 0, 255), width=5)
    sketches.append(('apple', 'Apple with symmetrical double leaf', img))

    # =========================================================================
    # 3. AXE (6 sketches)
    # =========================================================================
    # 3.1 Standard Woodcutter Axe
    img = canvas()
    d = ImageDraw.Draw(img)
    d.line([(150, 50), (150, 260)], fill=(0, 0, 0, 255), width=10)
    d.polygon([(150, 50), (240, 70), (230, 130), (150, 105)], outline=(0, 0, 0, 255), width=8)
    sketches.append(('axe', 'Standard vertical woodcutter axe', img))

    # 3.2 Double-Bladed Battle Axe
    img = canvas()
    d = ImageDraw.Draw(img)
    d.line([(150, 40), (150, 260)], fill=(0, 0, 0, 255), width=9)
    d.polygon([(150, 50), (230, 60), (220, 120), (150, 100)], outline=(0, 0, 0, 255), width=7)
    d.polygon([(150, 50), (70, 60), (80, 120), (150, 100)], outline=(0, 0, 0, 255), width=7)
    sketches.append(('axe', 'Double-headed medieval battle axe', img))

    # 3.3 Diagonal Chopping Axe
    img = canvas()
    d = ImageDraw.Draw(img)
    d.line([(70, 240), (210, 80)], fill=(0, 0, 0, 255), width=10)
    d.polygon([(210, 80), (255, 50), (275, 95), (220, 115)], outline=(0, 0, 0, 255), width=8)
    sketches.append(('axe', 'Angled diagonal chopping axe', img))

    # 3.4 Broad Hand Hatchet
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([135, 75, 165, 245], outline=(0, 0, 0, 255), width=8)
    d.polygon([(165, 75), (240, 65), (250, 150), (165, 125)], outline=(0, 0, 0, 255), width=8)
    sketches.append(('axe', 'Broad-bladed camp hatchet', img))

    # 3.5 Curved Handle Axe
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([100, 50, 200, 260], start=260, end=380, fill=(0, 0, 0, 255), width=10)
    d.polygon([(150, 50), (235, 60), (230, 125), (155, 110)], outline=(0, 0, 0, 255), width=8)
    sketches.append(('axe', 'Axe with ergonomic curved handle', img))

    # 3.6 Left-Facing Blade Axe
    img = canvas()
    d = ImageDraw.Draw(img)
    d.line([(160, 50), (160, 260)], fill=(0, 0, 0, 255), width=9)
    d.polygon([(160, 50), (70, 70), (80, 130), (160, 105)], outline=(0, 0, 0, 255), width=8)
    sketches.append(('axe', 'Left-facing wedge blade axe', img))

    # =========================================================================
    # 4. BICYCLE (6 sketches)
    # =========================================================================
    # 4.1 Classic Road Bike
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([40, 140, 120, 220], outline=(0, 0, 0, 255), width=8)
    d.ellipse([180, 140, 260, 220], outline=(0, 0, 0, 255), width=8)
    d.line([(80, 180), (150, 180)], fill=(0, 0, 0, 255), width=7)
    d.line([(80, 180), (125, 115)], fill=(0, 0, 0, 255), width=7)
    d.line([(150, 180), (210, 105)], fill=(0, 0, 0, 255), width=7)
    d.line([(125, 115), (195, 115)], fill=(0, 0, 0, 255), width=7)
    d.line([(190, 100), (225, 100)], fill=(0, 0, 0, 255), width=7)
    sketches.append(('bicycle', 'Classic road bike with diamond frame', img))

    # 4.2 Thick Mountain Bike Wheels
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([30, 130, 125, 225], outline=(0, 0, 0, 255), width=12)
    d.ellipse([175, 130, 270, 225], outline=(0, 0, 0, 255), width=12)
    d.polygon([(78, 178), (145, 178), (195, 128), (125, 128)], outline=(0, 0, 0, 255), width=8)
    d.line([(185, 110), (220, 110)], fill=(0, 0, 0, 255), width=8)
    sketches.append(('bicycle', 'Mountain bike with bold knobby tires', img))

    # 4.3 Compact Commuter Bike
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([60, 160, 130, 230], outline=(0, 0, 0, 255), width=7)
    d.ellipse([170, 160, 240, 230], outline=(0, 0, 0, 255), width=7)
    d.line([(95, 195), (155, 195)], fill=(0, 0, 0, 255), width=6)
    d.line([(95, 195), (130, 140)], fill=(0, 0, 0, 255), width=6)
    d.line([(130, 140), (190, 140)], fill=(0, 0, 0, 255), width=6)
    d.line([(155, 195), (200, 130)], fill=(0, 0, 0, 255), width=6)
    d.line([(190, 120), (215, 120)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('bicycle', 'Compact city commuter bike', img))

    # 4.4 Spoked Wheel Bicycle
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([45, 145, 115, 215], outline=(0, 0, 0, 255), width=7)
    d.line([(45, 180), (115, 180)], fill=(0, 0, 0, 255), width=4)
    d.line([(80, 145), (80, 215)], fill=(0, 0, 0, 255), width=4)
    d.ellipse([185, 145, 255, 215], outline=(0, 0, 0, 255), width=7)
    d.line([(185, 180), (255, 180)], fill=(0, 0, 0, 255), width=4)
    d.line([(220, 145), (220, 215)], fill=(0, 0, 0, 255), width=4)
    d.line([(80, 180), (150, 180)], fill=(0, 0, 0, 255), width=6)
    d.line([(80, 180), (130, 120)], fill=(0, 0, 0, 255), width=6)
    d.line([(150, 180), (210, 110)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('bicycle', 'Bicycle with cross-spokes', img))

    # 4.5 Angled/Tilted Bicycle
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([50, 130, 120, 200], outline=(0, 0, 0, 255), width=8)
    d.ellipse([180, 160, 250, 230], outline=(0, 0, 0, 255), width=8)
    d.line([(85, 165), (155, 180)], fill=(0, 0, 0, 255), width=7)
    d.line([(85, 165), (130, 110)], fill=(0, 0, 0, 255), width=7)
    d.line([(155, 180), (215, 120)], fill=(0, 0, 0, 255), width=7)
    sketches.append(('bicycle', 'Angled sports bike', img))

    # 4.6 Minimalist Bicycle
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([50, 150, 110, 210], outline=(0, 0, 0, 255), width=6)
    d.ellipse([190, 150, 250, 210], outline=(0, 0, 0, 255), width=6)
    d.line([(80, 180), (220, 180)], fill=(0, 0, 0, 255), width=6)
    d.line([(140, 180), (140, 120)], fill=(0, 0, 0, 255), width=6)
    d.line([(220, 180), (210, 110)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('bicycle', 'Minimalist linear bike', img))

    # =========================================================================
    # 5. CANDLE (6 sketches)
    # =========================================================================
    # 5.1 Tall Pillar Candle
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([125, 95, 175, 250], outline=(0, 0, 0, 255), width=7)
    d.line([(150, 95), (150, 75)], fill=(0, 0, 0, 255), width=5)
    d.ellipse([140, 40, 160, 75], outline=(0, 0, 0, 255), width=6)
    sketches.append(('candle', 'Tall pillar with teardrop flame', img))

    # 5.2 Candle on Holder Dish
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([135, 90, 165, 230], outline=(0, 0, 0, 255), width=7)
    d.ellipse([142, 50, 158, 85], outline=(0, 0, 0, 255), width=6)
    d.ellipse([90, 220, 210, 250], outline=(0, 0, 0, 255), width=7)
    d.arc([80, 210, 110, 250], start=90, end=270, fill=(0, 0, 0, 255), width=6)
    sketches.append(('candle', 'Candle in ceramic saucer with handle', img))

    # 5.3 Wide Melting Candle
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([110, 115, 190, 245], outline=(0, 0, 0, 255), width=8)
    d.line([(125, 115), (125, 165)], fill=(0, 0, 0, 255), width=6)
    d.line([(175, 115), (175, 155)], fill=(0, 0, 0, 255), width=6)
    d.ellipse([140, 60, 160, 110], outline=(0, 0, 0, 255), width=7)
    sketches.append(('candle', 'Chunky candle with wax drips', img))

    # 5.4 Slender Taper Candle
    img = canvas()
    d = ImageDraw.Draw(img)
    d.polygon([(140, 90), (135, 255), (165, 255), (160, 90)], outline=(0, 0, 0, 255), width=6)
    d.ellipse([143, 40, 157, 85], outline=(0, 0, 0, 255), width=6)
    sketches.append(('candle', 'Tapered dining candle', img))

    # 5.5 Short Votive Candle
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([105, 160, 195, 240], outline=(0, 0, 0, 255), width=7)
    d.line([(150, 160), (150, 135)], fill=(0, 0, 0, 255), width=5)
    d.ellipse([138, 95, 162, 135], outline=(0, 0, 0, 255), width=6)
    sketches.append(('candle', 'Short squat votive candle', img))

    # 5.6 Burning Wick Stick
    img = canvas()
    d = ImageDraw.Draw(img)
    d.line([(150, 85), (150, 250)], fill=(0, 0, 0, 255), width=10)
    d.ellipse([140, 45, 160, 85], outline=(0, 0, 0, 255), width=6)
    sketches.append(('candle', 'Minimalist single-stem taper candle', img))

    # =========================================================================
    # 6. DOOR (6 sketches)
    # =========================================================================
    # 6.1 Standard Rectangle with Doorknob
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([90, 45, 210, 260], outline=(0, 0, 0, 255), width=8)
    d.ellipse([180, 150, 195, 165], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Classic rectangular door with knob', img))

    # 6.2 Four-Panel Door
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([80, 40, 220, 260], outline=(0, 0, 0, 255), width=7)
    d.rectangle([98, 58, 145, 132], outline=(0, 0, 0, 255), width=5)
    d.rectangle([155, 58, 202, 132], outline=(0, 0, 0, 255), width=5)
    d.rectangle([98, 150, 145, 235], outline=(0, 0, 0, 255), width=5)
    d.rectangle([155, 150, 202, 235], outline=(0, 0, 0, 255), width=5)
    d.ellipse([185, 140, 196, 151], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Detailed panelled entry door', img))

    # 6.3 Arched Castle Door
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([80, 40, 220, 160], start=180, end=360, fill=(0, 0, 0, 255), width=8)
    d.line([(80, 100), (80, 260)], fill=(0, 0, 0, 255), width=8)
    d.line([(220, 100), (220, 260)], fill=(0, 0, 0, 255), width=8)
    d.line([(80, 260), (220, 260)], fill=(0, 0, 0, 255), width=8)
    d.ellipse([185, 170, 200, 185], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Arched wooden fortress door', img))

    # 6.4 Open Doorway in Perspective
    img = canvas()
    d = ImageDraw.Draw(img)
    d.polygon([(90, 50), (170, 70), (170, 240), (90, 260)], outline=(0, 0, 0, 255), width=7)
    d.line([(170, 70), (210, 50)], fill=(0, 0, 0, 255), width=6)
    d.line([(210, 50), (210, 260)], fill=(0, 0, 0, 255), width=6)
    d.ellipse([155, 150, 165, 160], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Open doorway with perspective frame', img))

    # 6.5 Double French Doors
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([70, 45, 230, 255], outline=(0, 0, 0, 255), width=7)
    d.line([(150, 45), (150, 255)], fill=(0, 0, 0, 255), width=6)
    d.ellipse([135, 150, 145, 160], fill=(0, 0, 0, 255))
    d.ellipse([155, 150, 165, 160], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Double blast doors with twin handles', img))

    # 6.6 Door with Frame Outline
    img = canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([70, 35, 230, 265], outline=(0, 0, 0, 255), width=5)
    d.rectangle([90, 50, 210, 260], outline=(0, 0, 0, 255), width=8)
    d.ellipse([185, 150, 200, 165], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Door with outer casing frame', img))

    # =========================================================================
    # 7. KEY (6 sketches)
    # =========================================================================
    # 7.1 Classic Skeleton Key (Horizontal, 2 teeth)
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([45, 120, 115, 185], outline=(0, 0, 0, 255), width=8)
    d.ellipse([68, 142, 92, 163], outline=(0, 0, 0, 255), width=5)
    d.line([(115, 152), (245, 152)], fill=(0, 0, 0, 255), width=8)
    d.line([(210, 152), (210, 185)], fill=(0, 0, 0, 255), width=7)
    d.line([(235, 152), (235, 180)], fill=(0, 0, 0, 255), width=7)
    sketches.append(('key', 'Horizontal skeleton key with dual teeth', img))

    # 7.2 Modern Serrated House Key
    img = canvas()
    d = ImageDraw.Draw(img)
    d.polygon([(50, 110), (105, 95), (120, 150), (60, 165)], outline=(0, 0, 0, 255), width=7)
    d.ellipse([75, 122, 95, 142], outline=(0, 0, 0, 255), width=4)
    d.line([(120, 130), (245, 130)], fill=(0, 0, 0, 255), width=8)
    d.line([(170, 130), (170, 150)], fill=(0, 0, 0, 255), width=6)
    d.line([(190, 130), (190, 160)], fill=(0, 0, 0, 255), width=6)
    d.line([(215, 130), (215, 145)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('key', 'Modern serrated house key', img))

    # 7.3 Diagonal Key
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([50, 50, 110, 110], outline=(0, 0, 0, 255), width=7)
    d.line([(105, 105), (235, 235)], fill=(0, 0, 0, 255), width=8)
    d.line([(190, 190), (215, 165)], fill=(0, 0, 0, 255), width=7)
    d.line([(215, 215), (235, 195)], fill=(0, 0, 0, 255), width=7)
    sketches.append(('key', 'Diagonal key with angled teeth', img))

    # 7.4 Oval Head Key
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([45, 105, 125, 195], outline=(0, 0, 0, 255), width=8)
    d.line([(125, 150), (255, 150)], fill=(0, 0, 0, 255), width=9)
    d.rectangle([215, 150, 245, 185], fill=(0, 0, 0, 255))
    sketches.append(('key', 'Oval head key with block tooth', img))

    # 7.5 Vertical Key (Testing orientation sensitivity)
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([120, 35, 180, 95], outline=(0, 0, 0, 255), width=7)
    d.line([(150, 95), (150, 255)], fill=(0, 0, 0, 255), width=8)
    d.line([(150, 210), (185, 210)], fill=(0, 0, 0, 255), width=7)
    d.line([(150, 240), (180, 240)], fill=(0, 0, 0, 255), width=7)
    sketches.append(('key', 'Vertical skeleton key (orientation test)', img))

    # 7.6 Tri-Notch Antique Key
    img = canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([40, 120, 100, 180], outline=(0, 0, 0, 255), width=7)
    d.line([(100, 150), (250, 150)], fill=(0, 0, 0, 255), width=8)
    d.line([(180, 150), (180, 175)], fill=(0, 0, 0, 255), width=6)
    d.line([(205, 150), (205, 175)], fill=(0, 0, 0, 255), width=6)
    d.line([(230, 150), (230, 180)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('key', 'Triple-notched dungeon key', img))

    # =========================================================================
    # 8. PARACHUTE (6 sketches)
    # =========================================================================
    # 8.1 Classic Hemispherical Dome
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([50, 35, 250, 160], start=180, end=360, fill=(0, 0, 0, 255), width=8)
    d.line([(50, 97), (250, 97)], fill=(0, 0, 0, 255), width=7)
    d.line([(50, 97), (150, 235)], fill=(0, 0, 0, 255), width=5)
    d.line([(100, 97), (150, 235)], fill=(0, 0, 0, 255), width=5)
    d.line([(200, 97), (150, 235)], fill=(0, 0, 0, 255), width=5)
    d.line([(250, 97), (150, 235)], fill=(0, 0, 0, 255), width=5)
    d.ellipse([145, 235, 155, 250], fill=(0, 0, 0, 255))
    sketches.append(('parachute', 'Classic round dome with jumper dot', img))

    # 8.2 Scalloped Canopy with Cargo Box
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([60, 45, 240, 165], start=180, end=360, fill=(0, 0, 0, 255), width=8)
    d.arc([60, 95, 120, 125], start=0, end=180, fill=(0, 0, 0, 255), width=5)
    d.arc([120, 95, 180, 125], start=0, end=180, fill=(0, 0, 0, 255), width=5)
    d.arc([180, 95, 240, 125], start=0, end=180, fill=(0, 0, 0, 255), width=5)
    d.line([(60, 110), (150, 225)], fill=(0, 0, 0, 255), width=5)
    d.line([(240, 110), (150, 225)], fill=(0, 0, 0, 255), width=5)
    d.rectangle([140, 225, 160, 245], outline=(0, 0, 0, 255), width=5)
    sketches.append(('parachute', 'Scalloped canopy carrying cargo box', img))

    # 8.3 Wind-Blown Tilted Parachute
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([80, 30, 260, 140], start=160, end=340, fill=(0, 0, 0, 255), width=8)
    d.line([(85, 95), (250, 75)], fill=(0, 0, 0, 255), width=6)
    d.line([(85, 95), (135, 230)], fill=(0, 0, 0, 255), width=5)
    d.line([(165, 85), (135, 230)], fill=(0, 0, 0, 255), width=5)
    d.line([(250, 75), (135, 230)], fill=(0, 0, 0, 255), width=5)
    sketches.append(('parachute', 'Wind-drift tilted parachute', img))

    # 8.4 Rectangular Ram-Air Paraglider
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([40, 55, 260, 130], start=180, end=360, fill=(0, 0, 0, 255), width=8)
    d.line([(40, 92), (260, 92)], fill=(0, 0, 0, 255), width=7)
    d.line([(60, 92), (150, 225)], fill=(0, 0, 0, 255), width=5)
    d.line([(120, 92), (150, 225)], fill=(0, 0, 0, 255), width=5)
    d.line([(180, 92), (150, 225)], fill=(0, 0, 0, 255), width=5)
    d.line([(240, 92), (150, 225)], fill=(0, 0, 0, 255), width=5)
    sketches.append(('parachute', 'Wide rectangular paraglider wing', img))

    # 8.5 Heavy Drop Parachute with Payload
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([55, 40, 245, 155], start=180, end=360, fill=(0, 0, 0, 255), width=9)
    d.line([(55, 98), (245, 98)], fill=(0, 0, 0, 255), width=8)
    d.line([(70, 98), (150, 215)], fill=(0, 0, 0, 255), width=5)
    d.line([(230, 98), (150, 215)], fill=(0, 0, 0, 255), width=5)
    d.rectangle([130, 215, 170, 255], fill=(0, 0, 0, 255))
    sketches.append(('parachute', 'Heavy military cargo parachute', img))

    # 8.6 Minimalist Line Parachute
    img = canvas()
    d = ImageDraw.Draw(img)
    d.arc([60, 50, 240, 170], start=180, end=360, fill=(0, 0, 0, 255), width=6)
    d.line([(60, 110), (150, 230)], fill=(0, 0, 0, 255), width=4)
    d.line([(240, 110), (150, 230)], fill=(0, 0, 0, 255), width=4)
    sketches.append(('parachute', 'Minimalist wireframe parachute', img))

    return sketches


def run_fresh_evaluation(threshold=0.40):
    predictor = QuickDrawPredictor()
    sketches = generate_fresh_sketches()
    
    print("=" * 70)
    print(f"STAGE 5 FRESH-SKETCH EVALUATION BENCHMARK")
    print(f"Total sketches: {len(sketches)} (6 per class across 8 classes)")
    print(f"Confidence cutoff: {threshold*100:.0f}%")
    print("=" * 70)
    
    results = []
    class_stats = {c: {'total': 0, 'correct': 0, 'accepted_correct': 0, 'accepted_wrong': 0, 'rejected': 0, 'confidences': []} for c in predictor.class_names}
    
    processed_gallery = []
    
    for i, (true_class, desc, canvas_img) in enumerate(sketches, 1):
        pred_class, conf, top3, meta = predictor.predict_canvas(canvas_img)
        
        is_correct = (pred_class == true_class)
        is_accepted = (conf >= threshold)
        
        stats = class_stats[true_class]
        stats['total'] += 1
        stats['confidences'].append(conf)
        
        if is_correct:
            stats['correct'] += 1
            if is_accepted:
                stats['accepted_correct'] += 1
            else:
                stats['rejected'] += 1
        else:
            if is_accepted:
                stats['accepted_wrong'] += 1
            else:
                stats['rejected'] += 1
                
        status = "CORRECT" if (is_correct and is_accepted) else ("REJECTED_LOW_CONF" if not is_accepted else "MISCLASSIFIED")
        
        results.append({
            'id': i,
            'true_class': true_class,
            'description': desc,
            'predicted_class': pred_class,
            'confidence': round(conf, 4),
            'top3': top3,
            'is_correct': is_correct,
            'is_accepted': is_accepted,
            'status': status
        })
        
        processed_gallery.append((true_class, pred_class, conf, is_correct, is_accepted, meta['processed_28x28']))
        
        symbol = "✓" if is_correct else "✗"
        print(f"[{i:02d}/48] {symbol} {true_class:10} -> {pred_class:10} ({conf*100:5.1f}%) | [{status:17}] | {desc}")
        
    # Overall summary
    total_sketches = len(sketches)
    total_correct = sum(s['correct'] for s in class_stats.values())
    total_accepted_correct = sum(s['accepted_correct'] for s in class_stats.values())
    total_accepted_wrong = sum(s['accepted_wrong'] for s in class_stats.values())
    total_rejected = sum(s['rejected'] for s in class_stats.values())
    
    print("\n" + "=" * 70)
    print("PER-CLASS EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Class':12} | {'Tested':6} | {'Correct':7} | {'Accuracy':8} | {'Accepted/Corr':13} | {'Accepted/Wrong':14} | {'Mean Conf':9}")
    print("-" * 75)
    
    for c in sorted(predictor.class_names):
        st = class_stats[c]
        acc = st['correct'] / st['total'] * 100
        mean_c = np.mean(st['confidences']) * 100
        print(f"{c:12} | {st['total']:6} | {st['correct']:7} | {acc:7.1f}% | {st['accepted_correct']:13} | {st['accepted_wrong']:14} | {mean_c:8.1f}%")
        
    print("-" * 75)
    raw_acc = total_correct / total_sketches * 100
    print(f"TOTAL:       {total_sketches} sketches | {total_correct} correct ({raw_acc:.1f}% raw accuracy)")
    print(f"Accepted & Correct:        {total_accepted_correct} ({total_accepted_correct/total_sketches*100:.1f}%)")
    print(f"Accepted but Incorrect:    {total_accepted_wrong} ({total_accepted_wrong/total_sketches*100:.1f}%)")
    print(f"Rejected (< {threshold*100:.0f}% threshold):  {total_rejected} ({total_rejected/total_sketches*100:.1f}%)")
    
    # Save gallery
    save_gallery(processed_gallery)
    
    # Save machine-readable JSON
    output_json = {
        'total_sketches': total_sketches,
        'overall_raw_accuracy': round(raw_acc, 2),
        'accepted_correct': total_accepted_correct,
        'accepted_wrong': total_accepted_wrong,
        'rejected': total_rejected,
        'threshold': threshold,
        'per_class_summary': {
            c: {
                'total': class_stats[c]['total'],
                'correct': class_stats[c]['correct'],
                'accuracy': round(class_stats[c]['correct'] / class_stats[c]['total'] * 100, 1),
                'mean_confidence': round(float(np.mean(class_stats[c]['confidences'])), 4)
            } for c in sorted(predictor.class_names)
        },
        'trials': results
    }
    
    json_path = os.path.join(REPORTS_DIR, 'stage5_fresh_evaluation.json')
    with open(json_path, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"\n[OUTPUT] Saved fresh evaluation data to {json_path}")
    
    return output_json


def save_gallery(processed_gallery):
    n = len(processed_gallery)
    cols = 8
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.4))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < n:
            true_c, pred_c, conf, is_corr, is_acc, img28 = processed_gallery[i]
            # Invert for visual viewing: white bg, black strokes
            axes[i].imshow(255 - img28, cmap='gray')
            color = 'darkgreen' if is_corr else 'darkred'
            axes[i].set_title(f"True: {true_c}\nPred: {pred_c}\n({conf*100:.0f}%)", fontsize=8, color=color, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
            
    plt.tight_layout()
    gallery_path = os.path.join(REPORTS_DIR, 'stage5_fresh_sketches_gallery.png')
    plt.savefig(gallery_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved fresh sketches gallery to {gallery_path}")


if __name__ == '__main__':
    run_fresh_evaluation(threshold=0.40)
