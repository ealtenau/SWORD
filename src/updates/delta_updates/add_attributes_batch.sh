#!/bin/bash
python src/updates/delta_updates/1_attach_attributes_deltas.py AS data/inputs/Deltas/delta_updates/AS/to-SWORD_06-23-2025_Irrawaddy/Hydrologic_network/
python src/updates/delta_updates/1_attach_attributes_deltas.py AS data/inputs/Deltas/delta_updates/AS/to-SWORD_07-08-2025_Mekong/Hydrologic_network/
python src/updates/delta_updates/1_attach_attributes_deltas.py OC data/inputs/Deltas/delta_updates/OC/to-SWORD_07-08-2025_Mahakam/Hydrologic_network/
python src/updates/delta_updates/1_attach_attributes_deltas.py SA data/inputs/Deltas/delta_updates/SA/to-SWORD_06-18-2025_Amazon/Hydrologic_network/
python src/updates/delta_updates/1_attach_attributes_deltas.py SA data/inputs/Deltas/delta_updates/SA/to-SWORD_07-22-2025_Parana/Hydrologic_network/
