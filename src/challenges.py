import os
import json
from dataclasses import dataclass, field
from typing import List, Optional

MODELS_DIR = 'models'
LABEL_PATH_8 = os.path.join(MODELS_DIR, 'label_mapping.json')
LABEL_PATH_20 = os.path.join(MODELS_DIR, 'label_mapping_20.json')
DATA_LABEL_PATH_20 = os.path.join('data/expanded_20', 'label_mapping.json')


@dataclass
class Challenge:
    id: str
    title: str
    scenario: str
    prompt: str
    accepted_classes: List[str]
    success_message: str
    failure_hint: str
    difficulty: str
    progressive_hints: List[str] = field(default_factory=list)
    riddle: Optional[str] = None
    intended_answer: Optional[str] = None



# -------------------------------------------------------------
# ORIGINAL 8 CHALLENGES (PRESERVED FOR BACKWARD COMPATIBILITY)
# -------------------------------------------------------------
CHALLENGES_8: List[Challenge] = [
    Challenge(
        id="room_1_cell",
        title="Room 1: The Armored Cell",
        scenario=(
            "You awaken inside a reinforced containment vault. Cold concrete surrounds you, and the heavy "
            "steel security gate is locked tight with a heavy brass deadbolt."
        ),
        prompt="Draw a KEY or a DOOR to unlock or breach your way out of the cell.",
        accepted_classes=["key", "door"],
        success_message=(
            "The deadbolt clicks and turns! The heavy gate creaks open, revealing the dim facility corridor beyond."
        ),
        failure_hint="Draw an object with a round head, long shaft, and distinct notched teeth (or a clean doorway).",
        difficulty="Introductory",
        progressive_hints=[
            "Look closely at the brass lock mechanism.",
            "You need an item with teeth to turn the lock tumblers.",
            "Draw a horizontal key with a round bow on the left and notched teeth on the right."
        ]
    ),
    Challenge(
        id="room_2_corridor",
        title="Room 2: The Collapsed Timberway",
        scenario=(
            "The hallway is choked by fallen ceiling timbers, tangled wooden scaffolding, and heavy wooden beams. "
            "You cannot squeeze through the narrow gaps."
        ),
        prompt="Draw an AXE to chop through the fallen timbers and clear a passage.",
        accepted_classes=["axe"],
        success_message=(
            "With powerful swings, the heavy wedge blade splinters the fallen timbers, carving a clear walkway!"
        ),
        failure_hint="Draw a straight vertical handle with a distinct triangular or wedge-shaped cutting head at the top.",
        difficulty="Introductory",
        progressive_hints=[
            "A sharp edge mounted on a lever can sever timber.",
            "Woodcutters use this heavy wedge tool.",
            "Draw a straight handle with a prominent triangular blade."
        ]
    ),
    Challenge(
        id="room_3_blackout",
        title="Room 3: The Pitch-Black Basement",
        scenario=(
            "The emergency power grid has failed. You step into an abyss of absolute darkness. "
            "You hear grinding gears and open pit shafts ahead, but you cannot take another step blindly."
        ),
        prompt="Draw a CANDLE to ignite a steady flame and illuminate the dark room.",
        accepted_classes=["candle"],
        success_message=(
            "A warm yellow glow fills the chamber! The flickering flame reveals the treacherous floor pits and safe path."
        ),
        failure_hint="Draw a vertical wax pillar topped with a small wick and a teardrop-shaped flame.",
        difficulty="Moderate",
        progressive_hints=[
            "You need a steady portable source of illumination.",
            "Wax and a wick produce light without electricity.",
            "Draw a vertical cylindrical candle with a teardrop flame."
        ]
    ),
    Challenge(
        id="room_4_tunnel",
        title="Room 4: The High-Speed Transit Tunnel",
        scenario=(
            "A yellow toxic gas begins hissing from fractured pipe vents along an extended quarter-mile underground "
            "maintenance trackway! Running on foot will take too long—you must move fast before your breath runs out."
        ),
        prompt="Draw a BICYCLE to race down the transit tunnel before the gas engulfs you.",
        accepted_classes=["bicycle"],
        success_message=(
            "Pedaling with frantic speed, you outrun the toxic mist and burst through the pressurized exit hatch!"
        ),
        failure_hint="Draw two separate circular wheels connected by a triangular frame and handlebars.",
        difficulty="Moderate",
        progressive_hints=[
            "Two wheels and human pedals can cover ground quickly.",
            "Think of a common two-wheeled road vehicle.",
            "Draw two circular wheels with a triangle frame connecting them."
        ]
    ),
    Challenge(
        id="room_5_quarantine",
        title="Room 5: The Medical Quarantine",
        scenario=(
            "Your energy is depleted and exhaustion is setting in. An automated biosensor at the quarantine airlock "
            "demands nutritional organic sustenance before releasing the biometric lock."
        ),
        prompt="Draw an APPLE to restore your vital energy and satisfy the biometric scanner.",
        accepted_classes=["apple"],
        success_message=(
            "The crisp fruit replenishes your strength, and the scanner registers organic bio-signature, unlocking the door!"
        ),
        failure_hint="Draw a rounded body with a small stem or leaf at the top.",
        difficulty="Introductory",
        progressive_hints=[
            "The scanner requires an organic orchard fruit.",
            "A round red fruit with a stem.",
            "Draw a round fruit with a small stem and leaf at the top."
        ]
    ),
    Challenge(
        id="room_6_airlock",
        title="Room 6: The Security Checkpoint",
        scenario=(
            "Security alarms are blaring! Lockdown protocols have sealed the main blast gate with an electromagnetic "
            "override panel. You need access through the security barrier."
        ),
        prompt="Draw a KEY or a DOOR to bypass the security checkpoint.",
        accepted_classes=["key", "door"],
        success_message=(
            "The security barrier disengages with a hiss of pneumatic pressure, granting access to the rooftop ladder!"
        ),
        failure_hint="Draw a skeleton key with distinct teeth or a rectangular door with a handle.",
        difficulty="Moderate",
        progressive_hints=[
            "Override the security entryway.",
            "Either an entryway portal or the tool that unlocks it.",
            "Draw a key horizontally or a clear rectangular door."
        ]
    ),
    Challenge(
        id="room_7_rooftop",
        title="Room 7: The Rooftop Chasm",
        scenario=(
            "You reach the high observation tower, but the bridge to the airfield has collapsed into a 200-meter canyon! "
            "The facility self-destruct countdown echoes across the speakers—your only way is to leap into the open abyss!"
        ),
        prompt="Draw a PARACHUTE to leap from the rooftop and glide safely across the gorge.",
        accepted_classes=["parachute"],
        success_message=(
            "The wide canopy catches the thermal updraft! You float smoothly across the gorge and land at the airstrip."
        ),
        failure_hint="Draw a wide curved hemispherical dome with lines converging downward to a harness point.",
        difficulty="Challenging",
        progressive_hints=[
            "Gravity will be fatal without aerodynamic drag.",
            "A silk canopy overhead slows a descent.",
            "Draw a wide dome with cords converging downward."
        ]
    ),
    Challenge(
        id="room_8_runway",
        title="Room 8: The Final Runway",
        scenario=(
            "You have reached the remote island airstrip. The perimeter gates are crumbling, and the only path to "
            "freedom and survival lies on the empty runway."
        ),
        prompt="Draw an AIRPLANE to fire up the engines and escape the facility into the open sky!",
        accepted_classes=["airplane"],
        success_message=(
            "Engines roar to life! The aircraft accelerates down the runway, lifts into the dawn sky, and leaves the facility behind forever!"
        ),
        failure_hint="Draw a fuselage with wide wings extending on both sides and a tail fin.",
        difficulty="Challenging",
        progressive_hints=[
            "An island airstrip needs a winged flight machine.",
            "Cruciform wings, fuselage, and tail.",
            "Draw an airplane with a long central fuselage and wide perpendicular wings."
        ]
    )
]


# -------------------------------------------------------------
# EXPANDED 20 CHALLENGES (FULL FACILITY ESCAPE ADVENTURE)
# -------------------------------------------------------------
CHALLENGES_20: List[Challenge] = [
    # Sector 1
    Challenge(
        id="sector_01_cell",
        title="Sector 1: The Armored Cell",
        scenario=(
            "You awaken inside a reinforced subterranean containment cell. Emergency klaxons sound, and the heavy "
            "steel security gate is locked tight with a mechanical deadbolt."
        ),
        prompt="Draw a KEY or a DOOR to breach containment and escape the holding cell.",
        accepted_classes=["key", "door"],
        success_message="The deadbolt clicks and disengages! The heavy gate swings open into the dim corridor.",
        failure_hint="Draw a horizontal key with a round bow and downward teeth, or a rectangular door with a knob.",
        difficulty="Introductory",
        progressive_hints=[
            "Examine the heavy mechanical lock cylinder.",
            "A notched metal instrument turns the deadbolt.",
            "Draw a key horizontally with a circular head on the left and teeth pointing downward."
        ]
    ),
    # Sector 2
    Challenge(
        id="sector_02_timberway",
        title="Sector 2: The Collapsed Timberway",
        scenario=(
            "The corridor is blocked by fallen structural beams, shattered ceiling scaffolding, and heavy wooden barricades."
        ),
        prompt="Draw an AXE to chop through the fallen timbers and clear a passage.",
        accepted_classes=["axe"],
        success_message="The wedge blade cuts through the dry timber, clearing an open path ahead!",
        failure_hint="Draw a straight vertical handle with a triangular or wedge cutting head at the top.",
        difficulty="Introductory",
        progressive_hints=[
            "Heavy wood blocks the corridor.",
            "A woodcutter's tool with a wedge blade.",
            "Draw a straight handle with a bold triangular blade at the top."
        ]
    ),
    # Sector 3
    Challenge(
        id="sector_03_blackout",
        title="Sector 3: The Pitch-Black Basement",
        scenario=(
            "The electrical sub-station has suffered total failure. Pitch darkness envelops the machinery room, "
            "with open maintenance trenches underfoot."
        ),
        prompt="Draw a CANDLE to illuminate the treacherous machinery room.",
        accepted_classes=["candle"],
        success_message="A steady yellow flame fills the room with light, revealing the safe walkway across the trenches!",
        failure_hint="Draw a vertical wax pillar with a wick and a teardrop flame at the top.",
        difficulty="Introductory",
        progressive_hints=[
            "Electric power is out, but fire provides illumination.",
            "A wax cylinder with a wick.",
            "Draw a vertical rectangular candle with a teardrop flame on top."
        ]
    ),
    # Sector 4
    Challenge(
        id="sector_04_transit",
        title="Sector 4: The High-Speed Transit Conduit",
        scenario=(
            "A toxic gas leak triggers a 3-minute automatic conduit lockdown! A 500-meter straight underground "
            "railway tube stretches ahead—running on foot is too slow to beat the closing blast shutters."
        ),
        prompt="Draw a BICYCLE to pedal down the transit tube before the lockdown shutters seal.",
        accepted_classes=["bicycle"],
        success_message="Pedaling frantically, you speed down the tracks and glide under the descending blast shutter just in time!",
        failure_hint="Draw two circular wheels connected by a triangular frame with handlebars.",
        difficulty="Moderate",
        progressive_hints=[
            "Human pedal power provides rapid land transit.",
            "Two wheels and a frame.",
            "Draw two round wheels with connecting frame bars and handlebars."
        ]
    ),
    # Sector 5
    Challenge(
        id="sector_05_quarantine",
        title="Sector 5: The Medical Quarantine",
        scenario=(
            "An automated biosecurity gate monitors the quarantine foyer. The metabolic analyzer demands "
            "an organic nutritional sample to verify bio-viability before disengaging the lockdown."
        ),
        prompt="Draw an APPLE to satisfy the nutritional biosensor and unlock the gate.",
        accepted_classes=["apple"],
        success_message="The scanner verifies organic bio-matter, releasing the magnetic seals on the quarantine gate!",
        failure_hint="Draw a round fruit with an indentation, stem, and leaf at the top.",
        difficulty="Introductory",
        progressive_hints=[
            "The scanner requests common orchard fruit.",
            "A crisp round fruit with a top stem.",
            "Draw a round apple shape with a top stem and leaf."
        ]
    ),
    # Sector 6
    Challenge(
        id="sector_06_archives",
        title="Sector 6: The Research Archives",
        scenario=(
            "An encrypted security terminal blocks the laboratory corridor. A prompt requests the primary "
            "facility operating cipher printed in the chief engineer's technical protocol manual."
        ),
        prompt="Draw a BOOK to provide the cipher reference manual and override the terminal.",
        accepted_classes=["book"],
        success_message="The cipher codes match the terminal database! The encryption locks release immediately.",
        failure_hint="Draw an open book with two rectangular page spreads, or a bound volume with a spine.",
        difficulty="Moderate",
        progressive_hints=[
            "The engineer's cipher is bound between printed pages.",
            "A bound publication with a spine and pages.",
            "Draw two open rectangular pages side-by-side or a rectangular book with a binding."
        ]
    ),
    # Sector 7
    Challenge(
        id="sector_07_breaker",
        title="Sector 7: The High-Voltage Breaker",
        scenario=(
            "A severed hydraulic valve has flooded the floor with electrified coolant. The emergency circuit breaker "
            "is mounted high on the ceiling, 3 meters above your reach."
        ),
        prompt="Draw a CHAIR to stand on and flip the overhead circuit breaker.",
        accepted_classes=["chair"],
        success_message="Standing firmly on the seat, you reach the overhead breaker and shut off the current!",
        failure_hint="Draw a side or 3D view of a chair with a seat, vertical backrest, and 4 legs.",
        difficulty="Moderate",
        progressive_hints=[
            "You need sturdy elevated furniture to gain height.",
            "Something you sit on that has a backrest and legs.",
            "Draw an L-profile backrest and seat supported by four vertical legs."
        ]
    ),
    # Sector 8
    Challenge(
        id="sector_08_reactor",
        title="Sector 8: The Reactor Synchronizer",
        scenario=(
            "The containment reactor's coolant valves must be triggered at an exact synchronized second. "
            "Without an accurate analog timepiece, improper valve timing will trigger a catastrophic meltdown."
        ),
        prompt="Draw a CLOCK to synchronize the countdown and vent the reactor core safely.",
        accepted_classes=["clock"],
        success_message="The countdown synchronizes precisely to the second! Steam vents safely as temperatures normalize.",
        failure_hint="Draw a circle with internal hour and minute hands pointing at an angle.",
        difficulty="Moderate",
        progressive_hints=[
            "You need an instrument that measures hours, minutes, and seconds.",
            "A circular dial with rotating hands.",
            "Draw a round face containing two distinct angled hands."
        ]
    ),
    # Sector 9
    Challenge(
        id="sector_09_tap",
        title="Sector 9: The Neutralizer Dispenser",
        scenario=(
            "A vat of corrosive alkaline acid is leaking across the doorway. An automated dispensary offers liquid "
            "neutralizing agent, but you need a handheld vessel to catch and carry the liquid."
        ),
        prompt="Draw a CUP to collect the chemical neutralizer from the dispensing tap.",
        accepted_classes=["cup"],
        success_message="You fill the vessel with neutralizing agent and douse the corrosive spill, clearing the threshold!",
        failure_hint="Draw a cylindrical mug or cup with an attached side loop handle.",
        difficulty="Moderate",
        progressive_hints=[
            "You need a small drinking vessel with a handle.",
            "A coffee mug or cup to hold liquid.",
            "Draw a vertical cup body with an open top and a distinct loop handle on the side."
        ]
    ),
    # Sector 10
    Challenge(
        id="sector_10_reservoir",
        title="Sector 10: The Flooded Aquarium Reservoir",
        scenario=(
            "A shattered coolant tank has flooded an entire chamber 5 meters deep. An underwater pipe hatch "
            "must be visually inspected by an autonomous aquatic sensor drone."
        ),
        prompt="Draw a FISH to activate the biomimetic submersible reconnaissance drone.",
        accepted_classes=["fish"],
        success_message="The biomimetic drone swims through the murky depths, disengaging the submerged pipe latch!",
        failure_hint="Draw a horizontal oval body with a triangular tail fin and an eye.",
        difficulty="Moderate",
        progressive_hints=[
            "An aquatic organism that swims underwater.",
            "Gills, fins, and a triangular tail.",
            "Draw a horizontal oval with a pointed triangular tail fin on the end."
        ]
    ),
    # Sector 11
    Challenge(
        id="sector_11_hydroponics",
        title="Sector 11: The Botanical Hydroponics",
        scenario=(
            "Airborne spore toxins have entered the ventilation grid. The hydroponics synthesis station "
            "requires a natural botanical specimen to extract neutralizing pollen enzymes."
        ),
        prompt="Draw a FLOWER to provide the botanical catalyst and synthesize the air filter.",
        accepted_classes=["flower"],
        success_message="The synthesis station distills the botanical enzyme, cleansing the atmosphere in the sector!",
        failure_hint="Draw a center circle surrounded by radiating petals and a stem.",
        difficulty="Moderate",
        progressive_hints=[
            "A blooming botanical plant with colorful petals.",
            "Petals radiating around a central core.",
            "Draw a central round circle surrounded by distinct petal loops and a downward stem."
        ]
    ),
    # Sector 12
    Challenge(
        id="sector_12_blastgate",
        title="Sector 12: The High-Security Airlock",
        scenario=(
            "You reach the transition barrier separating the research sector from the surface tunnels. "
            "A biometric security portal has locked down with magnetic deadbolts."
        ),
        prompt="Draw a DOOR or a KEY to open the reinforced security airlock.",
        accepted_classes=["door", "key"],
        success_message="The security airlock uncouples its magnetic locks, sliding open smoothly!",
        failure_hint="Draw a tall rectangular door with a knob, or a horizontal skeleton key.",
        difficulty="Moderate",
        progressive_hints=[
            "Access the passageway beyond the wall.",
            "Draw either the gateway itself or the unlocking implement.",
            "Draw a clean rectangular door with a handle, or a horizontal key with teeth."
        ]
    ),
    # Sector 13
    Challenge(
        id="sector_13_tripwire",
        title="Sector 13: The Explosive Demolition Grid",
        scenario=(
            "The corridor is laced with taut explosive tripwires connected to high-voltage detonator blasting caps. "
            "To disarm the sequence, you must sever the main detonation cord cleanly."
        ),
        prompt="Draw SCISSORS to snip the emergency detonation cord and defuse the charges.",
        accepted_classes=["scissors"],
        success_message="With a sharp snip, the crossed blades sever the wire! The detonation timer shuts off.",
        failure_hint="Draw two crossed blades pivoting at a center pin with two circular finger loops.",
        difficulty="Challenging",
        progressive_hints=[
            "A hand tool with two pivoting blades used for cutting.",
            "Shears with two finger holes.",
            "Draw an X-shaped crossed blade profile with two round handle loops at the bottom."
        ]
    ),
    # Sector 14
    Challenge(
        id="sector_14_floorgrid",
        title="Sector 14: The Electrified Floor Plates",
        scenario=(
            "High-voltage electrical conduits have shorted out onto a 20-meter stretch of damp metal floor plates. "
            "Walking across barefoot or in conducting gear will deliver a lethal shock."
        ),
        prompt="Draw a SHOE to equip rubber-soled insulating footwear and traverse the floor safely.",
        accepted_classes=["shoe"],
        success_message="Your heavy rubber soles insulate against the current, allowing safe transit across the energized plates!",
        failure_hint="Draw a horizontal shoe profile with a sole, heel, and ankle opening.",
        difficulty="Challenging",
        progressive_hints=[
            "Footwear with thick rubber soles insulates against electricity.",
            "A sneaker or boot worn on the foot.",
            "Draw an L-shaped side profile showing a flat sole, heel, and upper opening."
        ]
    ),
    # Sector 15
    Challenge(
        id="sector_15_stellar",
        title="Sector 15: The Astrological Vault",
        scenario=(
            "The complex was built atop an ancient subterranean observatory. An ancient mechanical iris lock "
            "requires a 5-pointed celestial emblem to align its radial gear train."
        ),
        prompt="Draw a STAR to align the astrological gear lock and open the vault.",
        accepted_classes=["star"],
        success_message="The 5-pointed emblem settles into the recess! Heavy brass gears turn and the iris gate opens.",
        failure_hint="Draw a clean 5-pointed geometric star polygon.",
        difficulty="Challenging",
        progressive_hints=[
            "A celestial body seen in the night sky.",
            "A geometric shape with five pointed arms.",
            "Draw a classic 5-pointed star with symmetrical sharp points."
        ]
    ),
    # Sector 16
    Challenge(
        id="sector_16_courtyard",
        title="Sector 16: The Courtyard Antenna",
        scenario=(
            "You reach an interior open-air ventilation courtyard. Facility surveillance drones patrol the perimeter; "
            "you must camouflage your emergency radio beacon as natural vegetation."
        ),
        prompt="Draw a TREE or FLOWER to disguise the transmission mast among the courtyard greenery.",
        accepted_classes=["tree", "flower"],
        success_message="The antenna blends seamlessly into the courtyard foliage! Drones pass overhead without detecting the signal.",
        failure_hint="Draw a vertical trunk with a rounded or cloud-like leafy canopy on top.",
        difficulty="Challenging",
        progressive_hints=[
            "A large perennial woody plant.",
            "A brown trunk supporting green leafy canopy.",
            "Draw a vertical trunk extending up into a bushy, cloud-like canopy."
        ]
    ),
    # Sector 17
    Challenge(
        id="sector_17_acid",
        title="Sector 17: The Acid Deluge Shaft",
        scenario=(
            "Overhead chemical tanks have ruptured, raining down continuous droplets of concentrated nitric acid. "
            "To traverse the 30-meter catwalk, you need an overhead shield to deflect the corrosive spray."
        ),
        prompt="Draw an UMBRELLA to shield yourself from the falling chemical deluge.",
        accepted_classes=["umbrella"],
        success_message="The canopy deflects the corrosive droplets harmlessly away as you cross the catwalk!",
        failure_hint="Draw a curved dome canopy supported by a straight shaft with a hooked J-handle.",
        difficulty="Challenging",
        progressive_hints=[
            "A portable canopy used to protect against falling rain.",
            "A dome top with a J-shaped handle below.",
            "Draw a wide curved dome with a single central vertical rod ending in a hooked handle."
        ]
    ),
    # Sector 18
    Challenge(
        id="sector_18_radar",
        title="Sector 18: The Surface Radar Terminal",
        scenario=(
            "You access the perimeter radar station overlooking the island. A long-range scanner requires you "
            "to designate the coordinates of the safe extraction rendezvous shelter."
        ),
        prompt="Draw a HOUSE to lock radar telemetry onto the abandoned surface safehouse.",
        accepted_classes=["house"],
        success_message="Radar coordinates locked! The automated telemetry locks onto the island shelter outpost.",
        failure_hint="Draw a square base with a triangular pitched roof, door, and windows.",
        difficulty="Challenging",
        progressive_hints=[
            "A residential building offering shelter and walls.",
            "A square structure with a triangular roof.",
            "Draw a square facade with a peaked triangular gabled roof."
        ]
    ),
    # Sector 19
    Challenge(
        id="sector_19_precipice",
        title="Sector 19: The Rooftop Precipice",
        scenario=(
            "You burst onto the highest rooftop observation deck. The perimeter bridge has collapsed into a 250-meter "
            "abyss separating the complex from the island runway! Self-destruct klaxons count down—you must leap!"
        ),
        prompt="Draw a PARACHUTE to leap from the precipice and glide across the chasm.",
        accepted_classes=["parachute"],
        success_message="The canopy catches the gale-force updraft! You glide gracefully across the canyon and touch down on the tarmac.",
        failure_hint="Draw a wide hemispherical canopy with cords converging downward to a harness point.",
        difficulty="Challenging",
        progressive_hints=[
            "A fabric canopy designed to slow fall rate from heights.",
            "Hemispheric dome with suspension cords.",
            "Draw a wide curved dome with lines converging down toward a center weight."
        ]
    ),
    # Sector 20
    Challenge(
        id="sector_20_airfield",
        title="Sector 20: The Island Airstrip",
        scenario=(
            "You stand on the island airfield tarmac as the facility foundation collapses into rubble behind you! "
            "The final escape aircraft sits on the runway—fuel lines connected, engines waiting for ignition!"
        ),
        prompt="Draw an AIRPLANE to fire up the dual turboprops and soar to freedom!",
        accepted_classes=["airplane"],
        success_message="Turboprops roar to life! The aircraft surges down the runway, lifts into the golden dawn sky, and leaves the facility behind forever!",
        failure_hint="Draw a long central fuselage with wide wings extending horizontally and a vertical tail fin.",
        difficulty="Challenging",
        progressive_hints=[
            "A fixed-wing powered aircraft.",
            "Fuselage, wide wings, and tail fin.",
            "Draw a classic aircraft with a long central body, wide horizontal wings, and a tail."
        ]
    )
]


# -------------------------------------------------------------
# 15 PLAYABLE DRAWING RIDDLES (3 DIFFICULTY MODES: 5 EACH)
# -------------------------------------------------------------

RIDDLES_EASY: List[Challenge] = [
    Challenge(
        id="riddle_easy_01_candle",
        title="The Shrinking Storyteller",
        scenario=(
            "I speak without a voice, and the more I tell, the less of me remains. "
            "I can brighten a room, but a little breath can silence me. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["candle"],
        success_message="Correct! The answer is a candle—brightening the room until silence takes its flame.",
        failure_hint="Draw a vertical wax pillar with a flickering teardrop flame on top.",
        difficulty="Easy",
        progressive_hints=[
            "It provides light when the power fails.",
            "It has a wick and slowly melts down as it burns.",
            "Draw a tall vertical cylinder topped with a small teardrop flame."
        ],
        riddle=(
            "I speak without a voice, and the more I tell, the less of me remains. "
            "I can brighten a room, but a little breath can silence me. What am I?"
        ),
        intended_answer="candle"
    ),
    Challenge(
        id="riddle_easy_02_umbrella",
        title="The Portable Roof",
        scenario=(
            "I open up when the sky breaks down. I have ribs but no bones, "
            "and I protect you without ever stepping outside myself. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["umbrella"],
        success_message="Correct! The answer is an umbrella—shielding you from the falling storm.",
        failure_hint="Draw a curved dome canopy with a central shaft and a hooked J-handle.",
        difficulty="Easy",
        progressive_hints=[
            "You carry it when dark rain clouds gather.",
            "It opens into a dome canopy supported by metal ribs.",
            "Draw a curved mushroom-like dome with a straight handle ending in a J-hook."
        ],
        riddle=(
            "I open up when the sky breaks down. I have ribs but no bones, "
            "and I protect you without ever stepping outside myself. What am I?"
        ),
        intended_answer="umbrella"
    ),
    Challenge(
        id="riddle_easy_03_fish",
        title="The Silent Swimmer",
        scenario=(
            "I travel without feet, breathe without lungs, and wear my own shimmering coat. "
            "Take me out of my world, and my journey ends. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["fish"],
        success_message="Correct! The answer is a fish—gliding silently beneath the ocean currents.",
        failure_hint="Draw a streamlined oval body tapering to a triangular tail fin.",
        difficulty="Easy",
        progressive_hints=[
            "An aquatic creature that lives underwater.",
            "It breathes through gills and swims with fins.",
            "Draw an oval body with a triangular tail fin on one side and a round eye."
        ],
        riddle=(
            "I travel without feet, breathe without lungs, and wear my own shimmering coat. "
            "Take me out of my world, and my journey ends. What am I?"
        ),
        intended_answer="fish"
    ),
    Challenge(
        id="riddle_easy_04_bicycle",
        title="The Balanced Traveller",
        scenario=(
            "I have two circles but no face, a chain but no prisoner, "
            "and I can carry you far without ever taking a step. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["bicycle"],
        success_message="Correct! The answer is a bicycle—powered by two rolling wheels and pedal power.",
        failure_hint="Draw two circular wheels connected by a triangular frame and handlebars.",
        difficulty="Easy",
        progressive_hints=[
            "A two-wheeled vehicle you ride by pedaling.",
            "It has handlebars, pedals, and a metal chain.",
            "Draw two circular wheels with a triangle frame connecting them and a handlebar on top."
        ],
        riddle=(
            "I have two circles but no face, a chain but no prisoner, "
            "and I can carry you far without ever taking a step. What am I?"
        ),
        intended_answer="bicycle"
    ),
    Challenge(
        id="riddle_easy_05_book",
        title="The Quiet World",
        scenario=(
            "I can take you to places you have never visited and introduce you to people who never existed. "
            "I stay perfectly still while your mind travels. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["book"],
        success_message="Correct! The answer is a book—opening gateways to limitless imagination.",
        failure_hint="Draw an open book with curved pages or a rectangular closed cover.",
        difficulty="Easy",
        progressive_hints=[
            "Bound pages filled with written stories and knowledge.",
            "You read it page by page to journey through other worlds.",
            "Draw an open book showing two curved pages spreading out from a center spine."
        ],
        riddle=(
            "I can take you to places you have never visited and introduce you to people who never existed. "
            "I stay perfectly still while your mind travels. What am I?"
        ),
        intended_answer="book"
    )
]

RIDDLES_MEDIUM: List[Challenge] = [
    Challenge(
        id="riddle_med_01_clock",
        title="The Relentless Circle",
        scenario=(
            "I revisit the same places every day, yet never return to yesterday. "
            "Everyone watches me move, but no one can ask me to wait. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["clock"],
        success_message="Correct! The answer is a clock—counting the unyielding passage of seconds and hours.",
        failure_hint="Draw a circular outline with clock hands pointing outward from the center.",
        difficulty="Medium",
        progressive_hints=[
            "It hangs on the wall or sits on a bedside table to tell time.",
            "Its hands tick relentlessly around a circular dial numbered 1 to 12.",
            "Draw a circle with two hands pointing outward from the center."
        ],
        riddle=(
            "I revisit the same places every day, yet never return to yesterday. "
            "Everyone watches me move, but no one can ask me to wait. What am I?"
        ),
        intended_answer="clock"
    ),
    Challenge(
        id="riddle_med_02_key",
        title="The Tiny Negotiator",
        scenario=(
            "I have teeth but cannot bite. I can end a stranger's visit or begin a friend's, "
            "yet I never decide who is welcome. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["key"],
        success_message="Correct! The answer is a key—turning tumblers to open or seal every door.",
        failure_hint="Draw horizontally: a round bow/head, a long straight shaft, and notched teeth.",
        difficulty="Medium",
        progressive_hints=[
            "A metal tool used to lock and unlock doors.",
            "It has notched teeth designed to turn inside a deadbolt.",
            "Draw a circular loop on the left, a horizontal shaft, and two teeth pointing down on the right."
        ],
        riddle=(
            "I have teeth but cannot bite. I can end a stranger's visit or begin a friend's, "
            "yet I never decide who is welcome. What am I?"
        ),
        intended_answer="key"
    ),
    Challenge(
        id="riddle_med_03_flower",
        title="The Colourful Clock",
        scenario=(
            "I can mark a season without numbers, attract visitors without a sign, "
            "and offer beauty before I ever offer fruit. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["flower"],
        success_message="Correct! The answer is a flower—blossoming to announce the arrival of spring.",
        failure_hint="Draw a center circular disc surrounded by radiant petals and a stem below.",
        difficulty="Medium",
        progressive_hints=[
            "A plant bloom that heralds springtime and attracts bees and butterflies.",
            "It blooms with colorful petals before seeds or fruit develop.",
            "Draw a small center circle surrounded by 5 or 6 oval petals and a stem."
        ],
        riddle=(
            "I can mark a season without numbers, attract visitors without a sign, "
            "and offer beauty before I ever offer fruit. What am I?"
        ),
        intended_answer="flower"
    ),
    Challenge(
        id="riddle_med_04_shoe",
        title="The Loyal Follower",
        scenario=(
            "I follow you everywhere but never choose the route. "
            "I touch the ground so you don't have to, and I'm happiest when I fit you perfectly. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["shoe"],
        success_message="Correct! The answer is a shoe—protecting every step of your journey.",
        failure_hint="Draw a horizontal shoe profile with a flat sole, heel, and ankle collar.",
        difficulty="Medium",
        progressive_hints=[
            "An article of footwear worn on your foot.",
            "It protects the sole of your foot when you walk on rough ground.",
            "Draw an L-shaped side view of a sneaker or shoe with a flat sole, heel, and ankle opening."
        ],
        riddle=(
            "I follow you everywhere but never choose the route. "
            "I touch the ground so you don't have to, and I'm happiest when I fit you perfectly. What am I?"
        ),
        intended_answer="shoe"
    ),
    Challenge(
        id="riddle_med_05_house",
        title="The Keeper of Secrets",
        scenario=(
            "I have a face that strangers see, rooms that guests may enter, "
            "and a door that can keep the whole world out. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["house"],
        success_message="Correct! The answer is a house—sheltering lives within its welcoming walls.",
        failure_hint="Draw a square base with a triangular pitched roof and a central door.",
        difficulty="Medium",
        progressive_hints=[
            "A building where people live, sleep, and find shelter.",
            "It has walls, a pitched roof, windows, and an entrance door.",
            "Draw a square wall box with a triangular pointed roof and a rectangular door."
        ],
        riddle=(
            "I have a face that strangers see, rooms that guests may enter, "
            "and a door that can keep the whole world out. What am I?"
        ),
        intended_answer="house"
    )
]

RIDDLES_HARD: List[Challenge] = [
    Challenge(
        id="riddle_hard_01_scissors",
        title="The Double-Edged Decision",
        scenario=(
            "I need two partners to do my job, but they spend their time moving against each other. "
            "Bring them together and something whole becomes two. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["scissors"],
        success_message="Correct! The answer is scissors—crossing paired blades to divide and slice.",
        failure_hint="Draw two crossed diagonal blades with two round finger loop handles.",
        difficulty="Hard",
        progressive_hints=[
            "A cutting instrument with two pivoting blades.",
            "You put your thumb and finger through two loops to shear paper or cloth.",
            "Draw an X shape where the bottom half has two circular finger loops and top half has two blades."
        ],
        riddle=(
            "I need two partners to do my job, but they spend their time moving against each other. "
            "Bring them together and something whole becomes two. What am I?"
        ),
        intended_answer="scissors"
    ),
    Challenge(
        id="riddle_hard_02_parachute",
        title="The Art of Falling Slowly",
        scenario=(
            "My purpose is to make a fall last longer. I catch the air, turn speed into patience, "
            "and help a traveller meet the ground gently. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["parachute"],
        success_message="Correct! The answer is a parachute—billowing fabric capturing the wind to cushion descent.",
        failure_hint="Draw an arched dome canopy with suspension lines converging downward.",
        difficulty="Hard",
        progressive_hints=[
            "Used by skydivers to slow down their descent from an airplane.",
            "A large fabric dome that catches air resistance.",
            "Draw a broad curved half-circle canopy with lines angling down to a small point or basket."
        ],
        riddle=(
            "My purpose is to make a fall last longer. I catch the air, turn speed into patience, "
            "and help a traveller meet the ground gently. What am I?"
        ),
        intended_answer="parachute"
    ),
    Challenge(
        id="riddle_hard_03_airplane",
        title="The Sky's Contradiction",
        scenario=(
            "I am heavier than the air I cross, yet I can stay above it. "
            "I carry people over oceans without swimming, and my wings are not for flapping. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["airplane"],
        success_message="Correct! The answer is an airplane—soaring across oceans with fixed-wing thrust.",
        failure_hint="Draw a central fuselage with perpendicular wings and a vertical tail fin.",
        difficulty="Hard",
        progressive_hints=[
            "A heavy metal aircraft that flies high through the clouds.",
            "It has a long fuselage, two fixed wings, and a tail fin.",
            "Draw a cross shape: long body, two wide wings sticking out horizontally, and a tail fin at the back."
        ],
        riddle=(
            "I am heavier than the air I cross, yet I can stay above it. "
            "I carry people over oceans without swimming, and my wings are not for flapping. What am I?"
        ),
        intended_answer="airplane"
    ),
    Challenge(
        id="riddle_hard_04_tree",
        title="The Upside-Down Kingdom",
        scenario=(
            "My feet are hidden, my arms reach for the sky, and every year I may dress in a different way. "
            "Birds can call me home, though I never build a house. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["tree"],
        success_message="Correct! The answer is a tree—deep roots anchoring branches that reach for the sky.",
        failure_hint="Draw a vertical trunk supporting a rounded, bushy cloud-like leafy canopy.",
        difficulty="Hard",
        progressive_hints=[
            "A woody plant with roots deep underground and branches in the air.",
            "Its leaves change color with the seasons, and birds build nests in its boughs.",
            "Draw a vertical trunk at the bottom topped by a bushy, rounded cloud-like canopy."
        ],
        riddle=(
            "My feet are hidden, my arms reach for the sky, and every year I may dress in a different way. "
            "Birds can call me home, though I never build a house. What am I?"
        ),
        intended_answer="tree"
    ),
    Challenge(
        id="riddle_hard_05_axe",
        title="The Woodcutter's Paradox",
        scenario=(
            "I make my mark by separating what nature joined. "
            "The harder I meet a tree, the smaller its trunk becomes, while my own shape stays almost the same. What am I?"
        ),
        prompt="Solve the riddle by drawing your answer on the canvas!",
        accepted_classes=["axe"],
        success_message="Correct! The answer is an axe—cleaving timber with a sharpened wedge blade.",
        failure_hint="Draw a straight vertical handle with a triangular wedge blade attached at one side of the top.",
        difficulty="Hard",
        progressive_hints=[
            "A tool used by lumberjacks to chop down trees and split firewood.",
            "A long wooden handle fitted with a sharp heavy metal blade.",
            "Draw a straight vertical handle with a sharp triangular or wedge blade affixed to the top side."
        ],
        riddle=(
            "I make my mark by separating what nature joined. "
            "The harder I meet a tree, the smaller its trunk becomes, while my own shape stays almost the same. What am I?"
        ),
        intended_answer="axe"
    )
]

RIDDLES_15: List[Challenge] = RIDDLES_EASY + RIDDLES_MEDIUM + RIDDLES_HARD

RIDDLES_BY_DIFFICULTY: dict = {
    "easy": RIDDLES_EASY,
    "medium": RIDDLES_MEDIUM,
    "hard": RIDDLES_HARD
}


def get_riddles(difficulty: Optional[str] = None) -> List[Challenge]:
    """
    Returns riddle challenges based on difficulty ('easy', 'medium', 'hard').
    If difficulty is None or 'all', returns all 15 riddles.
    """
    if difficulty is None or difficulty.lower() == "all":
        return RIDDLES_15
    diff_key = difficulty.lower().strip()
    if diff_key not in RIDDLES_BY_DIFFICULTY:
        raise ValueError(f"Invalid difficulty '{difficulty}'. Expected one of: {list(RIDDLES_BY_DIFFICULTY.keys())}")
    return RIDDLES_BY_DIFFICULTY[diff_key]


def validate_challenges(version: int = 20) -> bool:
    """
    Validates all accepted classes against the appropriate model label mapping.
    Raises ValueError if any accepted class is unsupported by the model.
    """
    label_path = LABEL_PATH_20 if version == 20 else LABEL_PATH_8
    if not os.path.exists(label_path):
        if version == 20 and os.path.exists(DATA_LABEL_PATH_20):
            label_path = DATA_LABEL_PATH_20
        else:
            raise FileNotFoundError(f"Label mapping file not found at {label_path}")
        
    with open(label_path, 'r') as f:
        mapping = json.load(f)
        
    valid_classes = set(mapping.keys())
    challenges_to_check = CHALLENGES_20 if version == 20 else CHALLENGES_8
    
    for ch in challenges_to_check:
        for acc in ch.accepted_classes:
            if acc not in valid_classes:
                raise ValueError(
                    f"Challenge '{ch.id}' contains unsupported class '{acc}'. "
                    f"Must be one of {valid_classes}"
                )

    if version == 20:
        for r_ch in RIDDLES_15:
            for acc in r_ch.accepted_classes:
                if acc not in valid_classes:
                    raise ValueError(
                        f"Riddle '{r_ch.id}' contains unsupported class '{acc}'. "
                        f"Must be one of {valid_classes}"
                    )
    return True


# Run validation at module import time for both 8 and 20 versions
validate_challenges(version=8)
validate_challenges(version=20)

# Default challenges list set to 20
CHALLENGES: List[Challenge] = CHALLENGES_20


def get_challenges(version: Optional[int] = None, difficulty: Optional[str] = None) -> List[Challenge]:
    """
    Returns the list of challenges.
    - If difficulty is provided ('easy', 'medium', 'hard'): returns the 5 riddles for that difficulty.
    - If version == 8: returns CHALLENGES_8 (legacy facility).
    - If version == 20: returns CHALLENGES_20 (expanded facility).
    - If version is None: defaults to 8 for backward compatibility with legacy test assertions.
    """
    if difficulty is not None:
        return get_riddles(difficulty)
    if version is None:
        env_ver = os.environ.get("DRAW_MODEL_VERSION", "")
        version = int(env_ver) if env_ver in ("8", "20") else 8
    return CHALLENGES_20 if version == 20 else CHALLENGES_8


def get_challenge(index: int, version: Optional[int] = None, difficulty: Optional[str] = None) -> Challenge:
    """Returns the challenge at the specified index."""
    ch_list = get_challenges(version=version, difficulty=difficulty)
    if index < 0 or index >= len(ch_list):
        raise IndexError(f"Challenge index {index} out of range (0..{len(ch_list)-1})")
    return ch_list[index]

