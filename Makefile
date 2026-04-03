# ── Load platform settings ──────────────────────────────────────────────────
# '-include' means: silently skip if site.env does not exist yet
-include site.env
export

# Fallbacks (used only when site.env is missing)
PYTHON      ?= python3
VASP_STD    ?= ~/BIN/vasp_std
VASP_NCL    ?= ~/BIN/vasp_ncl
VASP_GAM    ?= ~/BIN/vasp_gam
MPI_LAUNCH  ?= mpirun -np
MPI_NP      ?= 1
WANNIER90_X ?= wannier90.x

.PHONY: run snaps setup clean help

run:
	$(PYTHON) vasp-gui.py

snaps:
	$(PYTHON) sc-snaps-gui.py

setup:
	@if [ -f site.env ]; then \
		echo "site.env already exists — edit it to change settings"; \
	else \
		cp site.env.example site.env; \
		echo "Created site.env — edit it for your system, then run: make run"; \
	fi

clean:
	@rm -f instructions.txt POSCAR potcar_choices.json
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

help:
	@echo "make setup  — create site.env from template (first-time setup)"
	@echo "make run    — start the VASP Workflow GUI  (port 5001)"
	@echo "make snaps  — start the SC-Snaps GUI       (port 5050)"
	@echo "make clean  — remove generated temp files"
	@echo ""
	@echo "Platform settings (edit site.env):"
	@echo "  PYTHON      = $(PYTHON)"
	@echo "  VASP_STD    = $(VASP_STD)"
	@echo "  VASP_NCL    = $(VASP_NCL)"
	@echo "  VASP_GAM    = $(VASP_GAM)"
	@echo "  POTCAR DIR  = $(VASP_POTCAR_DIR)"
	@echo "  MPI         = $(MPI_LAUNCH) $(MPI_NP)"
	@echo "  WANNIER90   = $(WANNIER90_X)"
