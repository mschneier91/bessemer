# Helper makefile used by cmake/FindMFEM.cmake to extract MFEM's build flags.
#
# This spack MFEM ships only a Makefile-style config.mk (no MFEMConfig.cmake), and
# its flag variables reference other make variables (e.g. $(MFEM_INC_DIR),
# $(MFEM_EXT_LIBS)). We let *make* do the expansion and echo the final strings,
# rather than re-implementing that expansion in CMake.
#
# Usage: make -s -f mfem_query.mk MFEM_CONFIG_MK=<path/to/config.mk> <target>

include $(MFEM_CONFIG_MK)

print-incflags:
	@echo $(MFEM_INCFLAGS)

print-libs:
	@echo $(MFEM_LIBS)

print-cxx:
	@echo $(MFEM_CXX)

.PHONY: print-incflags print-libs print-cxx
