// Copyright 2020 LMNT, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

#include "device_ptr.h"
#include "haste.h"

using haste::v0::egru::ForwardPass;
using haste::v0::egru::BackwardPass;
using std::string;

using Tensor1 = Eigen::Tensor<float, 1>;
using Tensor2 = Eigen::Tensor<float, 2>;
using Tensor3 = Eigen::Tensor<float, 3>;

constexpr int BATCH_SIZE = 1;
constexpr int SEQUENCE_LEN = 3;
constexpr int HIDDEN_DIMS = 1;
constexpr int INPUT_DIMS = 1;

static cublasHandle_t g_blas_handle;

class ScopeTimer {
  public:
    ScopeTimer(const string& msg) : msg_(msg) {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
      cudaDeviceSynchronize();
      cudaEventRecord(start_);
    }

    ~ScopeTimer() {
      float elapsed_ms;
      cudaEventRecord(stop_);
      cudaEventSynchronize(stop_);
      cudaEventElapsedTime(&elapsed_ms, start_, stop_);
      printf("%s %fms\n", msg_.c_str(), elapsed_ms);
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }

  private:
    string msg_;
    cudaEvent_t start_, stop_;
};

void EgruInference(
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor1& thr,
    const Tensor3& x) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  // Copy weights over to GPU.
  device_ptr<Tensor2> W_dev(W);
  device_ptr<Tensor2> R_dev(R);
  device_ptr<Tensor1> bx_dev(bx);
  device_ptr<Tensor1> br_dev(br);
  device_ptr<Tensor3> x_dev(x);
  device_ptr<Tensor1> thr_dev(thr);

  device_ptr<Tensor2> y_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor2> o_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);

  device_ptr<Tensor2> trace_dev(time_steps * batch_size * hidden_size);

  h_dev.zero();
  trace_dev.zero();

  ScopeTimer t("Inference:");

  ForwardPass<float> forward = ForwardPass<float>(
      false,  // training
      batch_size,
      input_size,
      hidden_size,
      g_blas_handle);

  forward.Run(
      time_steps,
      W_dev.data,
      R_dev.data,
      bx_dev.data,
      br_dev.data,
      x_dev.data,
      h_dev.data,
      y_dev.data,
      nullptr,
      o_dev.data,
      thr_dev.data,
      tmp_Wx_dev.data,
      tmp_Rh_dev.data,
      trace_dev.data,
      0.0f,
      nullptr);
}

void EgruTrain(
    const Tensor2& W,
    const Tensor2& R,
    const Tensor1& bx,
    const Tensor1& br,
    const Tensor1& thr,
    const Tensor3& x,
    const Tensor3& dy_new,
    const Tensor3& dh_new,
    const Tensor3& do_new,
    const Tensor3& dtrs,
    const Tensor1& dampening_factor) {
  const int time_steps = x.dimension(2);
  const int batch_size = x.dimension(1);
  const int input_size = x.dimension(0);
  const int hidden_size = R.dimension(1);

  // Copy weights over to GPU.
  device_ptr<Tensor2> W_dev(W);
  device_ptr<Tensor2> R_dev(R);
  device_ptr<Tensor1> bx_dev(bx);
  device_ptr<Tensor1> br_dev(br);
  device_ptr<Tensor3> x_dev(x);
  device_ptr<Tensor3> dh_new_dev(dh_new);
  device_ptr<Tensor3> dy_new_dev(dy_new);
  device_ptr<Tensor3> do_new_dev(do_new);
  device_ptr<Tensor3> dtrs_new_dev(dtrs);
  device_ptr<Tensor1> thr_dev(thr);
  device_ptr<Tensor1> dampening_factor_dev(dampening_factor);

  device_ptr<Tensor2> y_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor2> h_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor2> o_dev((time_steps + 1) * batch_size * hidden_size);
  device_ptr<Tensor3> tmp_Wx_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor2> tmp_Rh_dev(batch_size * hidden_size * 3);
  device_ptr<Tensor3> v_dev(time_steps * batch_size * hidden_size * 5);

  device_ptr<Tensor2> trace_dev(time_steps * batch_size * hidden_size);

  h_dev.zero();
  y_dev.zero();
  trace_dev.zero();

  {
    ScopeTimer t("Train forward:");
    ForwardPass<float> forward = ForwardPass<float>(
        true,  // training
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    forward.Run(
        time_steps,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        x_dev.data,
        h_dev.data,
        y_dev.data,
        v_dev.data,
        o_dev.data,
        thr_dev.data,
        tmp_Wx_dev.data,
        tmp_Rh_dev.data,
        trace_dev.data,
        0.0f,
        nullptr);
  }

  device_ptr<Tensor3> dx_dev(time_steps * batch_size * input_size);
  device_ptr<Tensor2> dW_dev(input_size * hidden_size * 3);
  device_ptr<Tensor2> dR_dev(hidden_size * hidden_size * 3);
  device_ptr<Tensor1> dbx_dev(hidden_size * 3);
  device_ptr<Tensor1> dbr_dev(hidden_size * 3);
  device_ptr<Tensor1> dthr_dev(hidden_size);
  device_ptr<Tensor2> dy_dev(batch_size * hidden_size);
  device_ptr<Tensor2> dh_dev(batch_size * hidden_size);
  device_ptr<Tensor3> dp_dev(time_steps * batch_size * hidden_size * 3);
  device_ptr<Tensor3> dq_dev(time_steps * batch_size * hidden_size * 3);

  {
    ScopeTimer t("Train backward:");
    BackwardPass<float> backward(
        batch_size,
        input_size,
        hidden_size,
        g_blas_handle);

    backward.Run(
        time_steps,
        dampening_factor_dev.data,
        W_dev.data,
        R_dev.data,
        bx_dev.data,
        br_dev.data,
        thr_dev.data,
        x_dev.data,
        y_dev.data,
        h_dev.data,
        v_dev.data,
        dy_new_dev.data,
        dh_new_dev.data,
        do_new_dev.data,
        dx_dev.data,
        dW_dev.data,
        dR_dev.data,
        dbx_dev.data,
        dbr_dev.data,
        dthr_dev.data,
        dy_dev.data,
        dh_dev.data,
        dtrs_new_dev.data,
        dp_dev.data,
        dq_dev.data,
        nullptr);
  }
}

int main() {
  srand(time(0));

  cublasCreate(&g_blas_handle);

  // Weights.
  Tensor2 W(HIDDEN_DIMS * 3, INPUT_DIMS);
  Tensor2 R(HIDDEN_DIMS * 3, HIDDEN_DIMS);
  Tensor1 bx(HIDDEN_DIMS * 3);
  Tensor1 br(HIDDEN_DIMS * 3);
  Tensor1 thr(HIDDEN_DIMS);
  Tensor1 dampening_factor(1);

  // Input.
  Tensor3 x(INPUT_DIMS, BATCH_SIZE, SEQUENCE_LEN);

  // Gradients from upstream layers.
  Tensor3 dh(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);
  Tensor3 dy(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);
  Tensor3 dtrs(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);
  Tensor3 dout_gate(HIDDEN_DIMS, BATCH_SIZE, SEQUENCE_LEN + 1);

  W.setRandom();
  R.setRandom();
  bx.setRandom();
  br.setRandom();
  thr.setRandom();
  x.setRandom();
  dh.setRandom();
  dout_gate.setRandom();
  dampening_factor.setConstant(0.7f);

  EgruInference(W, R, bx, br, thr, x);
  EgruTrain(W, R, bx, br, thr, x, dy, dh, dout_gate, dtrs, dampening_factor);

  cublasDestroy(g_blas_handle);

  return 0;
}
