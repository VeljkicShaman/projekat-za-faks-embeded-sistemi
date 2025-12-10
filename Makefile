# ===================================================================
# --- MAKEFILE ---
# ===================================================================

# --- Podesavanja za SystemC i C++ ---

SYSTEMC_HOME ?= /usr/local/systemc

SC_VERSION = 2.3.3
SC_TAR = systemc-$(SC_VERSION).tar.gz
SC_URL = https://www.accellera.org/images/downloads/standards/systemc/$(SC_TAR)
SC_LOCAL_DIR = $(PWD)/systemc-local

CXX = g++

CXXFLAGS = -O3 -DSC_INCLUDE_FX -I. -Iheader

SC_INCLUDE = -I$(SYSTEMC_HOME)/include
SC_LIB = -L$(SYSTEMC_HOME)/lib-linux64 -Wl,-rpath=$(SYSTEMC_HOME)/lib-linux64 -lsystemc

# --- Podesavanja za Python ---
PYTHON_VENV = venv/bin/python3
PYTHON_SYS = python3

ifneq ("$(wildcard $(PYTHON_VENV))","")
    PYTHON = $(PYTHON_VENV)
else
    PYTHON = $(PYTHON_SYS)
endif

PYBIND_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes)

PYBIND_FLAGS = -shared -fPIC

# --- Fajlovi ---

LIB_NAME = multihead_attention_algorithm.so

REF_EXE = reference_sim
SC_EXE = systemc_sim

OUT_SC = izlaz_multihead_systemc.txt
OUT_CPP = izlaz_cpp.txt

#TARGETS

.PHONY: all help run_app verify clean install_deps install_systemc

all: help

help:
	@echo "------------------------------------------------------------------"
	@echo "DOSTUPNE KOMANDE:"
	@echo "  make run_app        -> 1. Kompajlira C++ biblioteku(pybind_wrapper) za Python"
	@echo "                         2. Pokrece final_app.py (generise matrice)"
	@echo ""
	@echo "  make verify         -> 1. Kompajlira i pokrece C++ Referencu(multihead_module)"
	@echo "                         2. Kompajlira i pokrece SystemC fajl"
	@echo "                         3. Uporedjuje rezultate (compare.py)"
	@echo ""
	@echo "  make install_deps   -> Instalira Python biblioteke"
	@echo "  make install_systemc-> Skida i kompajlira SystemC lokalno (ako fali)"
	@echo "  make clean          -> Brise sve generisane fajlove"
	@echo ""
	@echo " Ako SystemC nije u /usr/local/systemc, pokreni sa:"
	@echo "  make SYSTEMC_HOME=/putanja/do/systemc verify"
	@echo "------------------------------------------------------------------"

# --- GLAVNI PROGRAM ---
run_app:
	@echo "=================================================="
	@echo "[1/2] Kompajliranje C++ biblioteke za Python..."
	@echo "=================================================="
	rm -f $(LIB_NAME)
	$(CXX) $(CXXFLAGS) $(PYBIND_FLAGS) $(PYBIND_INCLUDES) pybind_wrapper.cpp multihead_module.cpp -o $(LIB_NAME)
	
	@echo "=================================================="
	@echo "[2/2] Pokretanje glavnog programa..."
	@echo "=================================================="
	$(PYTHON) final_app.py

# --- VERIFIKACIJA ---
verify:
	@echo "=================================================="
	@echo "[1/3] Kompajliranje i pokretanje C++ reference..."
	@echo "=================================================="
	$(CXX) $(CXXFLAGS) multihead_module.cpp -o $(REF_EXE)
	./$(REF_EXE)
	
	@echo ""
	@echo "=================================================="
	@echo "[2/3] Kompajliranje i pokretanje SystemC fajla..."
	@echo "=================================================="
	@if [ ! -d "$(SYSTEMC_HOME)" ]; then \
		echo "GRESKA: SystemC nije pronadjen na $(SYSTEMC_HOME)"; \
		echo "Ako nemate SystemC, pokrenite: make install_systemc"; \
		exit 1; \
	fi
	$(CXX) $(CXXFLAGS) $(SC_INCLUDE) testbench.cpp $(SC_LIB) -o $(SC_EXE)
	./$(SC_EXE)
	
	@echo ""
	@echo "=================================================="
	@echo "[3/3] POREDJENJE REZULTATA..."
	@echo "=================================================="
	$(PYTHON) compare.py $(OUT_SC) $(OUT_CPP)

# --- INSTALACIJA BIBLIOTEKA ---
install_deps:
	@echo "Provera/kreiranje Python virtualnog okruženja..."
	if [ ! -d "venv" ]; then \
		python3 -m venv venv; \
	fi
	@echo "Instaliranje Python biblioteka..."
	$(PYTHON) -m pip install numpy torch==2.9.1 transformers==4.35.0 sounddevice soundfile pybind11

install_systemc:
	@echo "=================================================="
	@echo "Preuzimanje i instalacija SystemC (Lokalno)..."
	@echo "=================================================="
	wget $(SC_URL)
	tar -xzf $(SC_TAR)
	mkdir -p $(SC_LOCAL_DIR)
	cd systemc-$(SC_VERSION) && mkdir -p objdir && cd objdir && \
	../configure --prefix=$(SC_LOCAL_DIR) --disable-async-updates CXX=$(CXX) && \
	make -j$(shell nproc) && \
	make install
	@echo ""
	@echo ">>> SystemC je instaliran u: $(SC_LOCAL_DIR)"
	@echo ">>> Sada pokreni verifikaciju sa:"
	@echo "    make SYSTEMC_HOME=$(SC_LOCAL_DIR) verify"

# --- CISCENJE ---
clean:
	@echo "Brisanje svih generisanih fajlova..."
	rm -f $(LIB_NAME) $(REF_EXE) $(SC_EXE) $(OUT_SC) $(OUT_CPP)
	rm -f *.o *.so
	@echo "Cisto."