// SPDX-License-Identifier: MIT
#include "celero/Celero.h"

BASELINE(SpGEMM, CSR_CSR, 30, 1000000)
{
  celero::DoNotOptimizeAway(0);
}

BENCHMARK(SpGEMM, CSR_IIDX, 0, 0)
{
  celero::DoNotOptimizeAway(0);
}

BENCHMARK(SpGEMM, RSB_RSB, 0, 0)
{
  celero::DoNotOptimizeAway(0);
}
