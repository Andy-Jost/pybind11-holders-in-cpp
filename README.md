# CUDA Core Handle Utilities

This repository contains experimental code and examples related to managing CUDA resources in Python and C++.

## Overview

The primary design goal is to provide **referentially transparent handles** for CUDA resources. These handles:

- Encapsulate ownership and lifetime of resources such as streams, contexts, and device pointers.
- Can be passed safely across Python and C++ boundaries.
- Ensure resources remain valid while handles exist, without global lifetime tracking.

The full design rationale, examples, and alternative implementation strategies are documented in detail.

## Detailed Design Document

Please see the full design document for an in-depth explanation, example implementations, and discussion of alternative approaches:

[CUDA Core Handle Design Document](https://docs.google.com/document/d/1rEQwOH8wjju_ibT9x-h16Cgw813Zlk2p0EGLanvEVzo/edit?usp=sharing)

## Disclaimer

This repository contains experimental code for internal exploration and is not intended as a production-ready library.
