#!/usr/bin/env python3
"""
Generate the Chev Chelis mascot SVG.

Usage:
    python3 assets/mascot/generate_chev.py
    python3 assets/mascot/generate_chev.py -o out.svg
"""
from __future__ import annotations
import argparse
from pathlib import Path

SVG = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="60 120 660 390" width="800" height="473">
  <title>Chev Chelis</title>
  <desc>Chev Chelis, a turtle riding a mountain bike up a steep hill.</desc>

  <defs>
    <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4a90d9"/>
      <stop offset="70%" stop-color="#a0cfff"/>
      <stop offset="100%" stop-color="#d4ebff"/>
    </linearGradient>
    <linearGradient id="mtnFar" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#7a8fa8"/>
      <stop offset="100%" stop-color="#9aacbf"/>
    </linearGradient>
    <linearGradient id="mtnNear" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4a6a3a"/>
      <stop offset="100%" stop-color="#5a8a4a"/>
    </linearGradient>
    <linearGradient id="shellGrad" x1="0.2" y1="0" x2="0.8" y2="1">
      <stop offset="0%" stop-color="#4a8c3f"/>
      <stop offset="100%" stop-color="#2d6b24"/>
    </linearGradient>
    <linearGradient id="frameGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#e85d26"/>
      <stop offset="100%" stop-color="#c43e10"/>
    </linearGradient>
    <linearGradient id="headbandGrad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#e83030"/>
      <stop offset="100%" stop-color="#ff5555"/>
    </linearGradient>
    <radialGradient id="wheelHub" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0%" stop-color="#999"/>
      <stop offset="100%" stop-color="#444"/>
    </radialGradient>
    <radialGradient id="sunGlow" cx="0.5" cy="0.5" r="0.5">
      <stop offset="0%" stop-color="#fff8e0"/>
      <stop offset="60%" stop-color="#ffee80"/>
      <stop offset="100%" stop-color="#ffd83000"/>
    </radialGradient>
    <filter id="spokeBlur">
      <feGaussianBlur in="SourceGraphic" stdDeviation="2"/>
    </filter>
  </defs>

  <!-- Background -->
  <rect x="60" y="120" width="660" height="390" fill="url(#skyGrad)"/>

  <!-- Sun -->
  <circle cx="650" cy="140" r="28" fill="url(#sunGlow)"/>
  <circle cx="650" cy="140" r="15" fill="#fff8d0"/>

  <!-- Clouds (puffy clusters) -->
  <g opacity="0.55" fill="#fff">
    <!-- Left cloud -->
    <ellipse cx="180" cy="140" rx="40" ry="14"/>
    <ellipse cx="210" cy="134" rx="32" ry="12"/>
    <ellipse cx="155" cy="136" rx="28" ry="10"/>
    <ellipse cx="195" cy="145" rx="25" ry="9"/>
    <!-- Right cloud -->
    <ellipse cx="430" cy="132" rx="35" ry="13"/>
    <ellipse cx="458" cy="127" rx="28" ry="11"/>
    <ellipse cx="410" cy="128" rx="24" ry="9"/>
  </g>

  <!-- Far mountains -->
  <polygon points="60,310 130,220 200,260 290,175 380,260 450,200 530,265 610,185 700,240 720,220 720,340 60,340" fill="url(#mtnFar)" opacity="0.6"/>
  <!-- Snow caps -->
  <polygon points="288,175 302,205 276,205" fill="#fff" opacity="0.7"/>
  <polygon points="608,185 622,215 596,215" fill="#fff" opacity="0.6"/>
  <!-- Mid mountains -->
  <polygon points="60,340 120,280 190,310 270,260 360,310 430,275 520,320 590,280 670,310 720,295 720,370 60,370" fill="#5a7a5a" opacity="0.5"/>

  <!-- Hillside -->
  <polygon points="60,428 60,510 720,510 720,314 640,330 560,350 480,372 400,392 320,406 240,416 160,422" fill="url(#mtnNear)"/>

  <!-- Road -->
  <polygon points="60,442 60,466 720,320 720,298" fill="#666"/>
  <g stroke="#ddd" stroke-width="1.5" stroke-dasharray="14,10" opacity="0.4">
    <line x1="60" y1="454" x2="720" y2="309"/>
  </g>

  <!-- Speed lines -->
  <g opacity="0.3" stroke="#4488cc" stroke-width="2.5" stroke-linecap="round">
    <line x1="150" y1="275" x2="260" y2="254"/>
    <line x1="135" y1="295" x2="270" y2="268"/>
    <line x1="155" y1="315" x2="275" y2="290"/>
    <line x1="140" y1="335" x2="265" y2="310"/>
    <line x1="160" y1="355" x2="280" y2="330"/>
    <line x1="145" y1="375" x2="270" y2="352"/>
    <line x1="155" y1="395" x2="268" y2="372"/>
  </g>
  <g opacity="0.18" stroke="#66aadd" stroke-width="1.5" stroke-linecap="round">
    <line x1="165" y1="285" x2="272" y2="262"/>
    <line x1="148" y1="325" x2="268" y2="300"/>
    <line x1="168" y1="365" x2="280" y2="340"/>
  </g>

  <!-- ===== SCENE (tilted uphill) ===== -->
  <g id="scene" transform="rotate(-10, 430, 400)">

    <!-- Dust clouds -->
    <g id="dustCloud" opacity="0.35" fill="#c8b898">
      <circle cx="195" cy="418" r="25"/>
      <circle cx="168" cy="408" r="20"/>
      <circle cx="145" cy="415" r="16"/>
      <circle cx="222" cy="410" r="18"/>
      <circle cx="130" cy="420" r="12"/>
      <circle cx="185" cy="400" r="14"/>
    </g>

    <!--
      LAYERING ORDER (back to front):
        1. Wheels (behind everything)
        2. Chain + chainring + rear cog
        3. Left leg (far side, behind frame)
        4. Left crank + pedal (behind frame)
        5. Frame (covers left leg partially)
        6. Right crank + pedal (in front of frame)
        7. Shell/body
        8. Right leg (near side, in front)
        9. Arms, watch, head
        10. Right pedal re-drawn on top of right leg
    -->

    <!-- WHEELS (chunky mountain bike tires) -->
    <g transform="translate(270, 410)">
      <circle cx="0" cy="0" r="65" fill="none" stroke="#222" stroke-width="10"/>
      <circle cx="0" cy="0" r="65" fill="none" stroke="#333" stroke-width="7"/>
      <!-- Tread texture -->
      <circle cx="0" cy="0" r="65" fill="none" stroke="#3a3a3a" stroke-width="3" stroke-dasharray="6,5"/>
      <g stroke="#aaa" stroke-width="1" filter="url(#spokeBlur)">
        <line x1="0" y1="-58" x2="0" y2="58"/>
        <line x1="-58" y1="0" x2="58" y2="0"/>
        <line x1="-41" y1="-41" x2="41" y2="41"/>
        <line x1="41" y1="-41" x2="-41" y2="41"/>
        <line x1="-20" y1="-55" x2="20" y2="55"/>
        <line x1="20" y1="-55" x2="-20" y2="55"/>
      </g>
      <circle cx="0" cy="0" r="8" fill="url(#wheelHub)"/>
    </g>

    <g transform="translate(540, 410)">
      <circle cx="0" cy="0" r="65" fill="none" stroke="#222" stroke-width="10"/>
      <circle cx="0" cy="0" r="65" fill="none" stroke="#333" stroke-width="7"/>
      <circle cx="0" cy="0" r="65" fill="none" stroke="#3a3a3a" stroke-width="3" stroke-dasharray="6,5"/>
      <g stroke="#aaa" stroke-width="1" filter="url(#spokeBlur)">
        <line x1="0" y1="-58" x2="0" y2="58"/>
        <line x1="-58" y1="0" x2="58" y2="0"/>
        <line x1="-41" y1="-41" x2="41" y2="41"/>
        <line x1="41" y1="-41" x2="-41" y2="41"/>
        <line x1="-20" y1="-55" x2="20" y2="55"/>
        <line x1="20" y1="-55" x2="-20" y2="55"/>
      </g>
      <circle cx="0" cy="0" r="8" fill="url(#wheelHub)"/>
    </g>

    <!-- Chain + gears (behind frame) -->
    <line x1="270" y1="414" x2="370" y2="402" stroke="#888" stroke-width="2.5" stroke-dasharray="4,3"/>
    <line x1="270" y1="406" x2="370" y2="394" stroke="#888" stroke-width="2.5" stroke-dasharray="4,3"/>
    <circle cx="370" cy="398" r="24" fill="none" stroke="#777" stroke-width="3"/>
    <circle cx="270" cy="410" r="11" fill="none" stroke="#777" stroke-width="2"/>

    <!-- LEFT LEG (far side, behind frame) -->
    <g id="leftLeg">
      <path d="M350,332 C342,352 336,362 340,368" stroke="#3d8a35" stroke-width="30" fill="none" stroke-linecap="round"/>
      <!-- Sneaker sitting ON the pedal -->
      <g transform="translate(342, 365)">
        <ellipse cx="0" cy="0" rx="20" ry="11" fill="#1a4fa0"/>
        <path d="M-16,5 Q0,12 18,5" stroke="#fff" stroke-width="2" fill="none"/>
      </g>
    </g>

    <!-- Left crank + pedal (behind frame) -->
    <line x1="370" y1="398" x2="342" y2="380" stroke="#555" stroke-width="6" stroke-linecap="round"/>
    <!-- Pedal sits UNDER the left foot -->
    <rect x="328" y="372" width="28" height="10" rx="3" fill="#666" stroke="#444" stroke-width="1.5"/>

    <!-- FRAME (drawn ON TOP of left leg, partially covering it) -->
    <g id="frame" stroke="url(#frameGrad)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round" fill="none">
      <line x1="370" y1="398" x2="355" y2="310"/>
      <line x1="355" y1="310" x2="500" y2="300"/>
      <line x1="500" y1="300" x2="370" y2="398"/>
      <line x1="370" y1="398" x2="270" y2="410"/>
      <line x1="355" y1="310" x2="270" y2="410"/>
      <line x1="500" y1="300" x2="540" y2="410"/>
    </g>

    <!-- Bottom bracket (on top of frame) -->
    <circle cx="370" cy="398" r="11" fill="#666" stroke="#555" stroke-width="2"/>

    <!-- Right crank (in front of frame, pointing to right pedal) -->
    <line x1="370" y1="398" x2="412" y2="418" stroke="#555" stroke-width="6" stroke-linecap="round"/>

    <!-- Seat -->
    <line x1="355" y1="310" x2="350" y2="280" stroke="#555" stroke-width="4" stroke-linecap="round"/>
    <ellipse cx="348" cy="275" rx="26" ry="7" fill="#333"/>

    <!-- Handlebars -->
    <line x1="500" y1="300" x2="516" y2="272" stroke="#555" stroke-width="4" stroke-linecap="round"/>
    <path d="M508,268 Q520,256 532,264 Q540,272 536,284" stroke="#444" stroke-width="5" fill="none" stroke-linecap="round"/>
    <path d="M508,268 Q512,260 518,258" stroke="#e83030" stroke-width="6" fill="none" stroke-linecap="round"/>
    <path d="M532,268 Q538,274 536,282" stroke="#e83030" stroke-width="6" fill="none" stroke-linecap="round"/>

    <!-- SHELL + BODY -->
    <g id="body">
      <ellipse cx="395" cy="272" rx="88" ry="68" fill="url(#shellGrad)" stroke="#2a5a20" stroke-width="3"/>
      <g stroke="#2a5a20" stroke-width="1.5" fill="none" opacity="0.3">
        <polygon points="395,228 418,242 418,266 395,280 372,266 372,242"/>
        <polygon points="420,216 442,228 442,250 420,262 408,250 408,228"/>
        <polygon points="370,216 382,228 382,250 370,262 348,250 348,228"/>
        <polygon points="444,238 460,250 460,270 444,282 430,270 430,250"/>
        <polygon points="346,238 360,250 360,270 346,282 332,270 332,250"/>
        <polygon points="410,272 428,282 428,300 410,308 398,300 398,282"/>
        <polygon points="380,272 392,282 392,300 380,308 362,300 362,282"/>
      </g>
      <ellipse cx="400" cy="328" rx="44" ry="16" fill="#c8e090" stroke="#9abf60" stroke-width="1.5"/>
    </g>

    <!-- TAIL -->
    <path d="M308,295 Q292,302 286,295 Q282,288 290,285 Q298,283 308,289" fill="#5aad55" stroke="#3d8a35" stroke-width="2"/>

    <!-- RIGHT LEG (near side, in front of everything) -->
    <g id="rightLeg">
      <path d="M388,330 C396,356 404,384 412,406" stroke="#4a9a44" stroke-width="34" fill="none" stroke-linecap="round"/>
      <!-- Sneaker sitting ON the pedal -->
      <g transform="translate(414, 408)">
        <ellipse cx="0" cy="0" rx="22" ry="12" fill="#2266cc"/>
        <path d="M-18,6 Q0,13 20,6" stroke="#fff" stroke-width="2" fill="none"/>
        <path d="M-8,0 Q0,-5 14,-1" stroke="#fff" stroke-width="1.5" fill="none"/>
      </g>
    </g>

    <!-- RIGHT PEDAL (drawn BELOW right foot so foot sits on it) -->
    <rect x="398" y="416" width="28" height="10" rx="3" fill="#666" stroke="#444" stroke-width="1.5"/>

    <!-- LEFT ARM (far side) -->
    <g id="leftArm">
      <path d="M455,278 C475,270 496,265 512,268" stroke="#3d8a35" stroke-width="24" fill="none" stroke-linecap="round"/>
      <ellipse cx="516" cy="270" rx="13" ry="11" fill="#4a9a44"/>
    </g>

    <!-- RIGHT ARM (near side) -->
    <g id="rightArm">
      <path d="M450,290 C472,284 498,278 524,278" stroke="#4a9a44" stroke-width="28" fill="none" stroke-linecap="round"/>
      <ellipse cx="528" cy="280" rx="15" ry="13" fill="#5aad55" stroke="#3d8a35" stroke-width="2"/>
      <path d="M520,272 Q528,264 536,271" stroke="#3d8a35" stroke-width="3" fill="none" stroke-linecap="round"/>
      <path d="M521,288 Q529,294 537,288" stroke="#3d8a35" stroke-width="3" fill="none" stroke-linecap="round"/>
    </g>

    <!-- SPORTS WATCH (on top of right forearm, with band) -->
    <g id="sportsWatch" transform="translate(488, 284)">
      <!-- Watch band (wraps around wrist) -->
      <rect x="-10" y="-18" width="20" height="8" rx="2" fill="#333"/>
      <rect x="-10" y="10" width="20" height="8" rx="2" fill="#333"/>
      <!-- Watch face -->
      <rect x="-10" y="-10" width="20" height="20" rx="3" fill="#222" stroke="#555" stroke-width="1.5"/>
      <rect x="-7" y="-7" width="14" height="14" rx="1.5" fill="#0a2a1a"/>
      <text x="0" y="1" text-anchor="middle" font-family="monospace" font-size="7" font-weight="bold" fill="#44ff88">12:34</text>
      <text x="0" y="6.5" text-anchor="middle" font-family="monospace" font-size="4" fill="#88ffaa">AVG</text>
      <!-- Side button -->
      <rect x="10" y="-3" width="3" height="6" rx="1" fill="#555"/>
    </g>

    <!-- HEAD -->
    <g id="head">
      <ellipse cx="462" cy="240" rx="22" ry="18" fill="#5aad55"/>
      <ellipse cx="494" cy="218" rx="42" ry="36" fill="#5aad55"/>

      <!-- Headband -->
      <path d="M456,206 Q474,188 498,190 Q518,192 532,208" stroke="url(#headbandGrad)" stroke-width="8" fill="none" stroke-linecap="round"/>

      <!-- Glasses -->
      <circle cx="482" cy="218" r="15" fill="none" stroke="#333" stroke-width="2.5"/>
      <circle cx="514" cy="216" r="14" fill="none" stroke="#333" stroke-width="2.5"/>
      <line x1="497" y1="217" x2="500" y2="216" stroke="#333" stroke-width="2.5"/>
      <path d="M467,218 Q460,214 457,208" stroke="#333" stroke-width="2.5" fill="none"/>

      <!-- Eyes -->
      <circle cx="482" cy="216" r="8" fill="#fff"/>
      <circle cx="485" cy="215" r="5" fill="#1a3a10"/>
      <circle cx="487" cy="213" r="2" fill="#fff"/>
      <circle cx="514" cy="214" r="7.5" fill="#fff"/>
      <circle cx="517" cy="213" r="4.8" fill="#1a3a10"/>
      <circle cx="519" cy="211" r="2" fill="#fff"/>

      <!-- Brows -->
      <path d="M472,205 Q480,200 490,203" stroke="#2a5a20" stroke-width="3" fill="none" stroke-linecap="round"/>
      <path d="M506,202 Q514,199 524,203" stroke="#2a5a20" stroke-width="3" fill="none" stroke-linecap="round"/>

      <!-- Mouth -->
      <path d="M498,234 Q508,244 526,236" stroke="#2a5a20" stroke-width="2.5" fill="none" stroke-linecap="round"/>

      <!-- Nostril -->
      <circle cx="530" cy="222" r="2" fill="#3a7a30"/>
    </g>

    <!-- Front wheel dust -->
    <g opacity="0.25" fill="#c8b898">
      <circle cx="508" cy="435" r="12"/>
      <circle cx="490" cy="430" r="8"/>
      <circle cx="525" cy="432" r="9"/>
    </g>

  </g>

</svg>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Chev Chelis mascot SVG")
    default_out = Path(__file__).parent / "chev.svg"
    parser.add_argument("-o", "--output", type=Path, default=default_out)
    args = parser.parse_args()
    args.output.write_text(SVG.lstrip(), encoding="utf-8")
    print(f"Wrote {len(SVG)} chars to {args.output}")


if __name__ == "__main__":
    main()
