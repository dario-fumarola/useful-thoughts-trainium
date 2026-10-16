.PHONY: check bootstrap doctor verify
check:
	bash scripts/run_checks.sh
bootstrap:
	bash scripts/start_here.sh
doctor:
	PYTHONPATH=. python -m weekend_thinking.hardware --out local/instance_doctor.json
verify:
	python scripts/verify_package.py
