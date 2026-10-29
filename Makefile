# Convenience targets. Isaac Sim steps use scripts/isaac_python.sh (set ISAACSIM_PATH).
ISAAC := bash scripts/isaac_python.sh
ARGS ?=

.PHONY: help test urdf import inspect-gripper inspect-nova5 assemble inspect-assembled demo demo-headless plot clean-generated

help:
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

test: ## pure-Python unit tests (no Isaac Sim needed)
	PYTHONPATH=src python3 -m unittest discover -s tests -v

urdf: ## clone Dobot description + write assets/urdf/nova5.urdf
	bash scripts/fetch_nova5_description.sh

import: ## URDF -> USD (Isaac Sim URDF importer)
	$(ISAAC) scripts/import_nova5.py $(ARGS)

inspect-gripper: ## list bodies/joints of the Robotiq 2F-85 asset
	$(ISAAC) scripts/inspect_usd.py gripper $(ARGS)

inspect-nova5: ## list bodies/joints of the imported Nova5
	$(ISAAC) scripts/inspect_usd.py nova5 $(ARGS)

assemble: ## build assets/usd/nova5_2f85_tactile.usd (+ .meta.json)
	$(ISAAC) scripts/assemble_robot.py $(ARGS)

inspect-assembled: ## summary of the final asset incl. pad frames
	$(ISAAC) scripts/inspect_usd.py assembled --pads $(ARGS)

demo: ## grasp demo with GUI
	$(ISAAC) scripts/run_grasp_demo.py $(ARGS)

demo-headless: ## grasp demo, headless, log to logs/
	$(ISAAC) scripts/run_grasp_demo.py --headless $(ARGS)

plot: ## plot the newest log in logs/
	python3 scripts/plot_tactile_log.py $$(ls -t logs/*.npz | head -1)

clean-generated: ## remove generated URDF/USD/logs (keeps third_party clone)
	rm -rf assets/urdf/*.urdf assets/usd/nova5 assets/usd/*.usd assets/usd/*.meta.json logs
