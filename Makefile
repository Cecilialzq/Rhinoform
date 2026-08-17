PYTHON ?= python3
REPRO_CONFIG ?=
REPRO_CONFIG_ARG = $(if $(REPRO_CONFIG),--config $(REPRO_CONFIG),)

.PHONY: test verify replay preflight

test:
	$(PYTHON) -m pytest -q

verify:
	$(PYTHON) -m tools.reproduce verify $(REPRO_CONFIG_ARG)

replay:
	$(PYTHON) -m tools.reproduce replay

preflight:
	$(PYTHON) -m tools.reproduce preflight $(REPRO_CONFIG_ARG) --full-data-hash
