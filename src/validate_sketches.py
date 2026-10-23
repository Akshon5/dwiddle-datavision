import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

import json
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from src.model import QuickDrawCNN
from src.preprocess import preprocess_canvas_image

REPORTS_DIR = 'reports'
MODELS_DIR = 'models'


def create_sketches():
    """
    Creates 32 diverse, multi-stroke synthetic human sketches across all 8 classes
    simulating canvas drawings with varied stroke widths (4-9px), sizes, and positions.
    """
    sketches = []
    
    def new_canvas():
        return Image.new('RGBA', (300, 300), (255, 255, 255, 255))
        
    # --- 1. APPLE (4 variations) ---
    # Apple 1: Classic round apple with stem and leaf
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([70, 80, 230, 240], outline=(0, 0, 0, 255), width=6)
    d.line([(150, 80), (160, 40)], fill=(0, 0, 0, 255), width=6)
    d.arc([160, 40, 190, 70], start=180, end=360, fill=(0, 0, 0, 255), width=5)
    sketches.append(('apple', 'Classic round with stem and leaf', img))
    
    # Apple 2: Apple with bite taken out
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([60, 70, 220, 230], outline=(0, 0, 0, 255), width=7)
    # Erase bite by drawing white arc
    d.arc([180, 110, 240, 170], start=90, end=270, fill=(255, 255, 255, 255), width=16)
    d.line([(140, 70), (145, 35)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('apple', 'Apple with bite taken out', img))
    
    # Apple 3: Heart/indented base apple
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.arc([60, 70, 160, 170], start=180, end=360, fill=(0, 0, 0, 255), width=6)
    d.arc([140, 70, 240, 170], start=180, end=360, fill=(0, 0, 0, 255), width=6)
    d.arc([60, 120, 240, 250], start=0, end=180, fill=(0, 0, 0, 255), width=6)
    d.line([(150, 70), (150, 30)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('apple', 'Heart-shaped double lobe apple', img))
    
    # Apple 4: Small off-center apple
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([140, 130, 240, 230], outline=(0, 0, 0, 255), width=5)
    d.line([(190, 130), (195, 95)], fill=(0, 0, 0, 255), width=5)
    sketches.append(('apple', 'Small off-center apple', img))
    
    # --- 2. AIRPLANE (4 variations) ---
    # Airplane 1: Jet with wings, fuselage, and tail
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([135, 40, 165, 250], outline=(0, 0, 0, 255), width=6)  # fuselage
    d.line([(30, 140), (270, 140)], fill=(0, 0, 0, 255), width=7)     # wings
    d.line([(90, 230), (210, 230)], fill=(0, 0, 0, 255), width=6)     # horizontal stabilizer
    sketches.append(('airplane', 'Vertical jet with swept wings', img))
    
    # Airplane 2: Horizontal airliner
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([40, 135, 260, 165], outline=(0, 0, 0, 255), width=6)
    d.line([(140, 50), (160, 250)], fill=(0, 0, 0, 255), width=7)
    d.line([(45, 110), (65, 150)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('airplane', 'Horizontal airliner', img))
    
    # Airplane 3: Diagonal ascending jet
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.line([(50, 250), (240, 60)], fill=(0, 0, 0, 255), width=8)
    d.line([(90, 110), (210, 210)], fill=(0, 0, 0, 255), width=7)
    d.line([(60, 220), (90, 250)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('airplane', 'Diagonal climbing jet', img))
    
    # Airplane 4: Cartoon cross-wing plane
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([130, 50, 170, 240], outline=(0, 0, 0, 255), width=6)
    d.rectangle([50, 120, 250, 160], outline=(0, 0, 0, 255), width=6)
    d.rectangle([100, 220, 200, 240], outline=(0, 0, 0, 255), width=5)
    sketches.append(('airplane', 'Blocky cartoon cross plane', img))

    # --- 3. AXE (4 variations) ---
    # Axe 1: Straight handle + wedge blade
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.line([(150, 50), (150, 260)], fill=(0, 0, 0, 255), width=8)   # handle
    d.polygon([(150, 50), (240, 70), (230, 120), (150, 100)], outline=(0, 0, 0, 255), width=6)  # blade
    sketches.append(('axe', 'Vertical handle with right wedge blade', img))
    
    # Axe 2: Double-sided battle axe
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.line([(150, 40), (150, 260)], fill=(0, 0, 0, 255), width=7)
    d.polygon([(150, 50), (230, 60), (220, 110), (150, 95)], outline=(0, 0, 0, 255), width=5)
    d.polygon([(150, 50), (70, 60), (80, 110), (150, 95)], outline=(0, 0, 0, 255), width=5)
    sketches.append(('axe', 'Double-sided battle axe', img))
    
    # Axe 3: Diagonal woodchopping axe
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.line([(70, 240), (210, 80)], fill=(0, 0, 0, 255), width=8)
    d.polygon([(210, 80), (250, 50), (270, 90), (220, 110)], outline=(0, 0, 0, 255), width=6)
    sketches.append(('axe', 'Diagonal chopping axe', img))
    
    # Axe 4: Thick hatchet
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([135, 70, 165, 250], outline=(0, 0, 0, 255), width=6)
    d.polygon([(165, 70), (230, 60), (240, 140), (165, 120)], outline=(0, 0, 0, 255), width=6)
    sketches.append(('axe', 'Broad-bladed hand hatchet', img))

    # --- 4. BICYCLE (4 variations) ---
    # Bicycle 1: Standard two wheels + frame
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([40, 150, 110, 220], outline=(0, 0, 0, 255), width=6)  # back wheel
    d.ellipse([190, 150, 260, 220], outline=(0, 0, 0, 255), width=6) # front wheel
    d.line([(75, 185), (145, 185)], fill=(0, 0, 0, 255), width=6)    # bottom frame
    d.line([(75, 185), (120, 120)], fill=(0, 0, 0, 255), width=6)    # seat post
    d.line([(145, 185), (200, 110)], fill=(0, 0, 0, 255), width=6)   # front fork
    d.line([(120, 120), (190, 120)], fill=(0, 0, 0, 255), width=6)   # top tube
    d.line([(180, 105), (210, 105)], fill=(0, 0, 0, 255), width=6)   # handlebars
    sketches.append(('bicycle', 'Classic frame with two wheels and handlebars', img))
    
    # Bicycle 2: Large wheels bicycle
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([30, 130, 120, 220], outline=(0, 0, 0, 255), width=7)
    d.ellipse([180, 130, 270, 220], outline=(0, 0, 0, 255), width=7)
    d.polygon([(75, 175), (140, 175), (190, 130), (120, 130)], outline=(0, 0, 0, 255), width=6)
    d.line([(180, 110), (210, 110)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('bicycle', 'Large wheel road bike', img))
    
    # Bicycle 3: Penny-farthing / asymmetric wheels
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([130, 60, 260, 190], outline=(0, 0, 0, 255), width=6)  # big front
    d.ellipse([50, 150, 100, 200], outline=(0, 0, 0, 255), width=5)   # small rear
    d.line([(75, 175), (195, 125)], fill=(0, 0, 0, 255), width=6)
    d.line([(175, 80), (210, 80)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('bicycle', 'Asymmetric wheel cruiser', img))
    
    # Bicycle 4: Off-center compact bike
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([80, 160, 140, 220], outline=(0, 0, 0, 255), width=5)
    d.ellipse([180, 160, 240, 220], outline=(0, 0, 0, 255), width=5)
    d.line([(110, 190), (170, 190)], fill=(0, 0, 0, 255), width=5)
    d.line([(110, 190), (140, 140)], fill=(0, 0, 0, 255), width=5)
    d.line([(140, 140), (200, 140)], fill=(0, 0, 0, 255), width=5)
    d.line([(170, 190), (210, 130)], fill=(0, 0, 0, 255), width=5)
    sketches.append(('bicycle', 'Compact bike in bottom canvas', img))

    # --- 5. CANDLE (4 variations) ---
    # Candle 1: Tall pillar candle with flame
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([125, 100, 175, 250], outline=(0, 0, 0, 255), width=6)  # body
    d.line([(150, 100), (150, 80)], fill=(0, 0, 0, 255), width=4)        # wick
    d.ellipse([140, 45, 160, 80], outline=(0, 0, 0, 255), width=5)       # flame
    sketches.append(('candle', 'Tall pillar with teardrop flame', img))
    
    # Candle 2: Candle on a base dish
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([135, 90, 165, 230], outline=(0, 0, 0, 255), width=6)
    d.ellipse([142, 50, 158, 85], outline=(0, 0, 0, 255), width=5)
    d.ellipse([90, 220, 210, 250], outline=(0, 0, 0, 255), width=6)      # base dish
    sketches.append(('candle', 'Candle in holder dish', img))
    
    # Candle 3: Fat melting candle with drips
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([110, 120, 190, 240], outline=(0, 0, 0, 255), width=6)
    d.line([(125, 120), (125, 160)], fill=(0, 0, 0, 255), width=5)       # wax drip
    d.line([(175, 120), (175, 150)], fill=(0, 0, 0, 255), width=5)
    d.ellipse([140, 65, 160, 115], outline=(0, 0, 0, 255), width=5)
    sketches.append(('candle', 'Wide candle with wax drips', img))
    
    # Candle 4: Slender tapered candle
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.line([(140, 90), (135, 250)], fill=(0, 0, 0, 255), width=5)
    d.line([(160, 90), (165, 250)], fill=(0, 0, 0, 255), width=5)
    d.line([(135, 250), (165, 250)], fill=(0, 0, 0, 255), width=5)
    d.ellipse([143, 40, 157, 85], outline=(0, 0, 0, 255), width=5)
    sketches.append(('candle', 'Slender taper candle', img))

    # --- 6. DOOR (4 variations) ---
    # Door 1: Standard rectangle with doorknob
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([90, 50, 210, 260], outline=(0, 0, 0, 255), width=7)
    d.ellipse([180, 150, 195, 165], fill=(0, 0, 0, 255))               # knob
    sketches.append(('door', 'Classic rectangular door with knob', img))
    
    # Door 2: 4-Panel door
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.rectangle([80, 40, 220, 260], outline=(0, 0, 0, 255), width=6)
    d.rectangle([100, 60, 145, 130], outline=(0, 0, 0, 255), width=4)
    d.rectangle([155, 60, 200, 130], outline=(0, 0, 0, 255), width=4)
    d.rectangle([100, 150, 145, 230], outline=(0, 0, 0, 255), width=4)
    d.rectangle([155, 150, 200, 230], outline=(0, 0, 0, 255), width=4)
    d.ellipse([185, 140, 195, 150], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Detailed panel door', img))
    
    # Door 3: Arched wooden door
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.arc([80, 40, 220, 160], start=180, end=360, fill=(0, 0, 0, 255), width=7)
    d.line([(80, 100), (80, 260)], fill=(0, 0, 0, 255), width=7)
    d.line([(220, 100), (220, 260)], fill=(0, 0, 0, 255), width=7)
    d.line([(80, 260), (220, 260)], fill=(0, 0, 0, 255), width=7)
    d.ellipse([185, 170, 200, 185], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Arched fantasy door', img))
    
    # Door 4: Open door in perspective
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.polygon([(90, 50), (170, 70), (170, 240), (90, 260)], outline=(0, 0, 0, 255), width=6)
    d.line([(170, 70), (210, 50)], fill=(0, 0, 0, 255), width=5)
    d.line([(210, 50), (210, 260)], fill=(0, 0, 0, 255), width=5)
    d.ellipse([155, 150, 165, 160], fill=(0, 0, 0, 255))
    sketches.append(('door', 'Perspective open door', img))

    # --- 7. KEY (4 variations) ---
    # Key 1: Round head + shaft + 2 teeth
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([50, 120, 110, 180], outline=(0, 0, 0, 255), width=6)  # head
    d.ellipse([70, 140, 90, 160], outline=(0, 0, 0, 255), width=4)   # inner hole
    d.line([(110, 150), (240, 150)], fill=(0, 0, 0, 255), width=7)   # shaft
    d.line([(210, 150), (210, 180)], fill=(0, 0, 0, 255), width=6)   # tooth 1
    d.line([(235, 150), (235, 175)], fill=(0, 0, 0, 255), width=6)   # tooth 2
    sketches.append(('key', 'Horizontal skeleton key with teeth', img))
    
    # Key 2: Vertical key
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([120, 40, 180, 100], outline=(0, 0, 0, 255), width=6)
    d.line([(150, 100), (150, 250)], fill=(0, 0, 0, 255), width=7)
    d.line([(150, 210), (180, 210)], fill=(0, 0, 0, 255), width=6)
    d.line([(150, 240), (175, 240)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('key', 'Vertical skeleton key', img))
    
    # Key 3: Diagonal car/house key
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.polygon([(60, 60), (110, 50), (120, 100), (70, 110)], outline=(0, 0, 0, 255), width=6)
    d.line([(115, 105), (230, 220)], fill=(0, 0, 0, 255), width=7)
    d.line([(180, 170), (200, 150)], fill=(0, 0, 0, 255), width=6)
    d.line([(210, 200), (230, 180)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('key', 'Diagonal house key', img))
    
    # Key 4: Oval head key
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.ellipse([50, 110, 120, 190], outline=(0, 0, 0, 255), width=6)
    d.line([(120, 150), (250, 150)], fill=(0, 0, 0, 255), width=8)
    d.line([(220, 150), (220, 185)], fill=(0, 0, 0, 255), width=6)
    sketches.append(('key', 'Oval head key with single thick notch', img))

    # --- 8. PARACHUTE (4 variations) ---
    # Parachute 1: Classic dome canopy with lines and person dot
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.arc([50, 40, 250, 160], start=180, end=360, fill=(0, 0, 0, 255), width=7) # dome
    d.line([(50, 100), (250, 100)], fill=(0, 0, 0, 255), width=6)                 # base line
    d.line([(50, 100), (150, 230)], fill=(0, 0, 0, 255), width=4)                 # cord 1
    d.line([(100, 100), (150, 230)], fill=(0, 0, 0, 255), width=4)                # cord 2
    d.line([(200, 100), (150, 230)], fill=(0, 0, 0, 255), width=4)                # cord 3
    d.line([(250, 100), (150, 230)], fill=(0, 0, 0, 255), width=4)                # cord 4
    d.ellipse([145, 230, 155, 245], fill=(0, 0, 0, 255))                          # load dot
    sketches.append(('parachute', 'Classic dome canopy with converging cords', img))
    
    # Parachute 2: Scalloped edge canopy
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.arc([60, 50, 240, 170], start=180, end=360, fill=(0, 0, 0, 255), width=7)
    d.arc([60, 95, 120, 125], start=0, end=180, fill=(0, 0, 0, 255), width=4)
    d.arc([120, 95, 180, 125], start=0, end=180, fill=(0, 0, 0, 255), width=4)
    d.arc([180, 95, 240, 125], start=0, end=180, fill=(0, 0, 0, 255), width=4)
    d.line([(60, 110), (150, 220)], fill=(0, 0, 0, 255), width=4)
    d.line([(240, 110), (150, 220)], fill=(0, 0, 0, 255), width=4)
    d.rectangle([140, 220, 160, 240], outline=(0, 0, 0, 255), width=4)
    sketches.append(('parachute', 'Scalloped canopy with cargo box', img))
    
    # Parachute 3: Tilted wind-drift parachute
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.arc([80, 30, 260, 140], start=160, end=340, fill=(0, 0, 0, 255), width=6)
    d.line([(85, 95), (250, 75)], fill=(0, 0, 0, 255), width=5)
    d.line([(85, 95), (130, 230)], fill=(0, 0, 0, 255), width=4)
    d.line([(160, 85), (130, 230)], fill=(0, 0, 0, 255), width=4)
    d.line([(250, 75), (130, 230)], fill=(0, 0, 0, 255), width=4)
    sketches.append(('parachute', 'Angled wind-drift parachute', img))
    
    # Parachute 4: Wide ram-air / rectangular paraglider
    img = new_canvas()
    d = ImageDraw.Draw(img)
    d.arc([40, 60, 260, 130], start=180, end=360, fill=(0, 0, 0, 255), width=7)
    d.line([(40, 95), (260, 95)], fill=(0, 0, 0, 255), width=6)
    d.line([(60, 95), (150, 220)], fill=(0, 0, 0, 255), width=4)
    d.line([(120, 95), (150, 220)], fill=(0, 0, 0, 255), width=4)
    d.line([(180, 95), (150, 220)], fill=(0, 0, 0, 255), width=4)
    d.line([(240, 95), (150, 220)], fill=(0, 0, 0, 255), width=4)
    sketches.append(('parachute', 'Wide paraglider canopy', img))
    
    return sketches


def run_validation(confidence_threshold=0.40):
    model_path = os.path.join(MODELS_DIR, 'baseline_cnn.pt')
    mapping_path = os.path.join(MODELS_DIR, 'label_mapping.json')
    
    with open(mapping_path) as f:
        label_mapping = json.load(f)
    rev_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [rev_mapping[i] for i in range(len(rev_mapping))]
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model = QuickDrawCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    sketches = create_sketches()
    print(f"Loaded {len(sketches)} newly drawn canvas sketches across all 8 classes.")
    
    results = []
    correct_count = 0
    accepted_correct = 0
    accepted_wrong = 0
    rejected_count = 0
    
    processed_images = []
    
    for i, (true_class, desc, canvas_pil) in enumerate(sketches):
        tensor, meta = preprocess_canvas_image(canvas_pil)
        if tensor is None:
            results.append({
                'id': i + 1,
                'true_class': true_class,
                'description': desc,
                'pred_class': 'EMPTY',
                'confidence': 0.0,
                'top3': [],
                'is_correct': False,
                'is_accepted': False,
                'status': 'REJECTED_EMPTY'
            })
            continue
            
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            
        top3_indices = np.argsort(probs)[::-1][:3]
        top_idx = top3_indices[0]
        pred_class = class_names[top_idx]
        conf = float(probs[top_idx])
        
        top3 = [(class_names[idx], float(probs[idx])) for idx in top3_indices]
        
        is_correct = (pred_class == true_class)
        is_accepted = (conf >= confidence_threshold)
        
        if is_correct:
            correct_count += 1
            if is_accepted:
                accepted_correct += 1
            else:
                rejected_count += 1
        else:
            if is_accepted:
                accepted_wrong += 1
            else:
                rejected_count += 1
                
        results.append({
            'id': i + 1,
            'true_class': true_class,
            'description': desc,
            'pred_class': pred_class,
            'confidence': conf,
            'top3': top3,
            'is_correct': is_correct,
            'is_accepted': is_accepted,
            'status': 'CORRECT' if (is_correct and is_accepted) else ('REJECTED_LOW_CONF' if not is_accepted else 'MISCLASSIFIED')
        })
        
        processed_images.append((true_class, pred_class, conf, is_correct, is_accepted, meta['processed_28x28'], canvas_pil))
        
        mark = "✓" if is_correct else "✗"
        thresh_mark = "ACCEPTED" if is_accepted else "REJECTED (<40%)"
        print(f"[{i+1:02d}/32] {mark} True: {true_class:10} | Pred: {pred_class:10} ({conf*100:5.1f}%) | [{thresh_mark}] | {desc}")
        
    raw_accuracy = correct_count / len(sketches)
    print("\n" + "=" * 65)
    print("CANVAS PROTOTYPE VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Total sketches tested:     {len(sketches)}")
    print(f"Raw Correct Predictions:   {correct_count} / {len(sketches)} ({raw_accuracy*100:.2f}%)")
    print(f"Accepted & Correct:        {accepted_correct}")
    print(f"Accepted but Incorrect:    {accepted_wrong}")
    print(f"Rejected by Threshold:     {rejected_count}")
    
    # Save gallery of all 32 tested sketches
    save_validation_gallery(processed_images)
    
    return results, raw_accuracy


def save_validation_gallery(processed_images):
    n = len(processed_images)
    cols = 8
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < n:
            true_c, pred_c, conf, is_corr, is_acc, img28, canvas_pil = processed_images[i]
            # Show preprocessed 28x28 inverted (white bg, black ink for visual clarity)
            axes[i].imshow(255 - img28, cmap='gray')
            color = 'darkgreen' if is_corr else 'darkred'
            axes[i].set_title(f"True: {true_c}\nPred: {pred_c}\n({conf*100:.0f}%)", fontsize=8, color=color, fontweight='bold')
            axes[i].axis('off')
        else:
            axes[i].axis('off')
            
    plt.tight_layout()
    gallery_path = os.path.join(REPORTS_DIR, 'canvas_validation_gallery.png')
    plt.savefig(gallery_path, dpi=200)
    plt.close()
    print(f"[PLOT] Saved canvas validation gallery to {gallery_path}")


if __name__ == '__main__':
    run_validation(confidence_threshold=0.40)
