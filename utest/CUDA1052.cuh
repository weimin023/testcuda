#pragma once


__global__ void scan(const int *customers, const int *grumpy, int n, int *result);
__global__ void scan2(const int *customers, const int *grumpy, int n, const int window, int *result);

int maxSatisfied(std::vector<int>& customers, std::vector<int>& grumpy, int minutes);
