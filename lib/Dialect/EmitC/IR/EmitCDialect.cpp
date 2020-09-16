//===- EmitCOps.cpp - MLIR EmitC ops implementation -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the EmitC dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "emitc/Dialect/EmitC/EmitCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

void emitc::EmitCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "emitc/Dialect/EmitC/EmitC.cpp.inc"
      >();
  allowUnknownTypes();
  allowUnknownOperations();
}

#define GET_OP_CLASSES
#include "emitc/Dialect/EmitC/EmitC.cpp.inc"
