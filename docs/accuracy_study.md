# FirePath accuracy study

20 fires dropped at random on campus (x 80–850 m, y -140–350 m, seed 2026), scenario `campus_watch`, fire width 40 m.  Every photo went through the live YOLO26 fire model; the geometry only saw the AI's boxes.

The camera images are **simulated**: a drawn campus with a real flame photograph pasted in where the geometry says it belongs.  That is what makes the error measurable — we know exactly where each fire really was.

## Summary

- Located: **20 / 20** fires (100%)
- Error: median **1.1 m**, mean 1.1 m, 90th percentile 1.7 m, worst 2.7 m
- True fire inside the reported region: 20 / 20
- Median region size: 0.17 ha
- Cameras-agree check (χ², 95%, box edges assumed to wobble 6 px): passed 14 / 14 runs with 3+ cameras (about 95% expected if the wobble is right)

## Every fire

| # | placed at (m) | flame photo | cameras used | found at (m) | error (m) | region (ha) | inside region | note |
|---|---|---|---|---|---|---|---|---|
| 1 | (171.7, 106.2) | holika_opendraft | 2 | (171.8, 106.9) | 0.7 | 0.172 | yes | |
| 2 | (474.1, 281.4) | campfire_hameltion | 3 | (473.8, 280.4) | 1.1 | 0.157 | yes | |
| 3 | (159.0, -30.6) | flame_waleedtariq | 2 | (160.7, -30.6) | 1.7 | 0.251 | yes | |
| 4 | (542.8, 132.7) | holika_opendraft | 3 | (543.0, 132.8) | 0.3 | 0.139 | yes | |
| 5 | (683.2, 128.4) | campfire_hameltion | 3 | (682.7, 129.0) | 0.8 | 0.164 | yes | |
| 6 | (642.5, 236.4) | flame_waleedtariq | 3 | (641.3, 237.2) | 1.4 | 0.198 | yes | |
| 7 | (658.3, 147.4) | holika_opendraft | 3 | (659.3, 147.7) | 1.0 | 0.123 | yes | |
| 8 | (264.7, 161.0) | campfire_hameltion | 2 | (266.4, 161.5) | 1.8 | 0.174 | yes | |
| 9 | (165.3, 260.2) | flame_waleedtariq | 2 | (166.0, 259.3) | 1.1 | 0.192 | yes | |
| 10 | (426.2, 259.2) | holika_opendraft | 3 | (426.6, 259.2) | 0.4 | 0.134 | yes | |
| 11 | (607.6, 192.9) | campfire_hameltion | 3 | (606.8, 193.7) | 1.1 | 0.157 | yes | |
| 12 | (241.9, -16.8) | flame_waleedtariq | 3 | (242.9, -16.3) | 1.1 | 0.229 | yes | |
| 13 | (834.5, 315.4) | holika_opendraft | 3 | (835.2, 315.8) | 0.8 | 0.172 | yes | |
| 14 | (700.0, 349.8) | campfire_hameltion | 3 | (698.2, 347.8) | 2.7 | 0.181 | yes | |
| 15 | (475.5, -102.6) | flame_waleedtariq | 2 | (475.9, -101.9) | 0.8 | 0.191 | yes | |
| 16 | (342.1, 133.2) | holika_opendraft | 3 | (342.6, 133.1) | 0.5 | 0.145 | yes | |
| 17 | (304.9, 84.1) | campfire_hameltion | 3 | (304.8, 85.1) | 1.0 | 0.174 | yes | |
| 18 | (754.9, 204.9) | flame_waleedtariq | 3 | (754.6, 206.0) | 1.1 | 0.201 | yes | |
| 19 | (524.6, -127.9) | holika_opendraft | 2 | (525.1, -129.4) | 1.6 | 0.155 | yes | |
| 20 | (627.4, 38.1) | campfire_hameltion | 3 | (626.7, 38.6) | 0.9 | 0.165 | yes | |
