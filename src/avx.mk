ifndef AVX
        AVX_DEF=$(shell $(CC) -Wno-unused-command-line-argument -mavx2 -dM -E - < /dev/null | egrep "AVX2" | sort)
ifeq ($(findstring AVX2,$(AVX_DEF)),AVX2)
        AVX=on
        AVX513_DEF=$(shell $(CC) -Wno-unused-command-line-argument -mavx512f -dM -E - < /dev/null | egrep "AVX512" | sort)
ifeq ($(findstring AVX2,$(AVX_DEF)),AVX2)
        HAS_AVX512=true
endif
endif
endif

