//===- StdToEmitC.cpp - std to EmitC conversion ---------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for lowering std dialect to EmitC dialect.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "emitc/Dialect/EmitC/Conversion/Passes.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::emitc;

namespace {

/// Convert `std.index_cast` into an `emitc.call` operation.
class IndexCastOpConversion : public OpConversionPattern<IndexCastOp> {
  using OpConversionPattern<IndexCastOp>::OpConversionPattern;

public:
  IndexCastOpConversion(MLIRContext *ctx)
      : OpConversionPattern<IndexCastOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(IndexCastOp indexCastOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::standard::index_cast");

    ArrayAttr args;
    Type resultType = indexCastOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(indexCastOp,
                                               indexCastOp.getType(), callee,
                                               args, templateArgs, operands);

    return success();
  }
};

/// Convert `std.splat` into an `emitc.call` operation.
class SplatOpConversion : public OpConversionPattern<SplatOp> {
  using OpConversionPattern<SplatOp>::OpConversionPattern;

public:
  SplatOpConversion(MLIRContext *ctx) : OpConversionPattern<SplatOp>(ctx) {}

private:
  LogicalResult
  matchAndRewrite(SplatOp splatOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr callee = rewriter.getStringAttr("emitc::standard::splat");

    ArrayAttr args;
    Type resultType = splatOp.getResult().getType();
    ArrayAttr templateArgs = rewriter.getArrayAttr({TypeAttr::get(resultType)});

    rewriter.replaceOpWithNewOp<emitc::CallOp>(
        splatOp, splatOp.getType(), callee, args, templateArgs, operands);

    return success();
  }
};
} // namespace

void populateStdToEmitcPatterns(MLIRContext *ctx, RewritePatternSet &patterns) {
  patterns.add<IndexCastOpConversion>(ctx);
  patterns.add<SplatOpConversion>(ctx);
}

namespace {

struct ConvertStdToEmitCPass
    : public ConvertStdToEmitCBase<ConvertStdToEmitCPass> {
  /// Perform the lowering to EmitC dialect.
  void runOnFunction() override {

    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();
    target.addLegalDialect<StandardOpsDialect>();
    target.addIllegalOp<IndexCastOp>();
    target.addIllegalOp<SplatOp>();

    RewritePatternSet patterns(&getContext());
    populateStdToEmitcPatterns(&getContext(), patterns);

    if (failed(
            applyPartialConversion(getFunction(), target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<FunctionPass> mlir::emitc::createConvertStdToEmitCPass() {
  return std::make_unique<ConvertStdToEmitCPass>();
}
