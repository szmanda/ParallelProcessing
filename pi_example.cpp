/*
$ gcc pi_serial.cpp -o pi_serial

$ ./pi_serial
Wartosc liczby PI wynosi  3.141592653592
Czas przetwarzania wynosi 15.125000 sekund

*/

#include "pi_example.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace PiExample;

double PiExample::compute(long long noSteps, int noThreads)
{
	omp_set_num_threads(noThreads);
	double x, pi, sum = 0.0;
	int i;
	double step = 1. / (double)noSteps;
	#pragma omp parallel for reduction(+:sum) private(i, x)
	for (i = 0; i < noSteps; i++)
	{
		x = (i + .5) * step;
		sum += 4.0 / (1. + x * x);
	}
	pi = sum * step;

	return pi;
}

double PiExample::computeSingle(int noSteps) {
	double x, pi, sum = 0.0;
	int i;
	double step = 1. / (double)noSteps;
	for (i = 0; i < noSteps; i++)
	{
		x = (i + .5) * step;
		sum += 4.0 / (1. + x * x);
	}
	pi = sum * step;
	return pi;
}
