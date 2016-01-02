extern bool isMonteCarloOk();
extern bool isMonteCarloMultiGPUOk();

bool useMonteCarlo();
bool useMonteCarloMultiGPU();

bool useMonteCarlo() {
  return isMonteCarloOk();
}

bool useMonteCarloMultiGPU() {
  return isMonteCarloMultiGPUOk();
}
