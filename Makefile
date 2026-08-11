PYTHON ?= python3
REPRO_CONFIG ?= reproduction.local.toml

.PHONY: test verify verify-assets preflight

test:
	$(PYTHON) -m pytest -q

verify:
	$(PYTHON) -m tools.reproduce verify --config $(REPRO_CONFIG)

verify-assets:
	$(PYTHON) -m tools.reproduce verify-assets --config $(REPRO_CONFIG)

preflight:
	$(PYTHON) -m tools.reproduce preflight --config $(REPRO_CONFIG) --full-data-hash
