.PHONY: verify-fixtures verify-kl-fixture regenerate-kl-fixture \
        verify-calibration-fixture regenerate-calibration-fixture

FIXTURE_DIR := SteeringKit/Tests/SteeringKitTests/Fixtures
KL_FIXTURE := $(FIXTURE_DIR)/kl_golden.json
CALIBRATION_FIXTURE := $(FIXTURE_DIR)/calibration_golden.json

verify-fixtures: verify-kl-fixture verify-calibration-fixture

verify-kl-fixture:
	python3 Scripts/generate_kl_fixture.py --check $(KL_FIXTURE)

regenerate-kl-fixture:
	python3 Scripts/generate_kl_fixture.py --output $(KL_FIXTURE)

# Derived from the committed Phase 6 calibration packets, which are read-only evidence. If this
# check fails, the fixture drifted from the packets -- not the other way around.
verify-calibration-fixture:
	python3 Scripts/generate_calibration_fixture.py --check $(CALIBRATION_FIXTURE)

regenerate-calibration-fixture:
	python3 Scripts/generate_calibration_fixture.py --output $(CALIBRATION_FIXTURE)
