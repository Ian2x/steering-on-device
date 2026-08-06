.PHONY: verify-kl-fixture regenerate-kl-fixture

KL_FIXTURE := SteeringKit/Tests/SteeringKitTests/Fixtures/kl_golden.json

verify-kl-fixture:
	python3 Scripts/generate_kl_fixture.py --check $(KL_FIXTURE)

regenerate-kl-fixture:
	python3 Scripts/generate_kl_fixture.py --output $(KL_FIXTURE)
