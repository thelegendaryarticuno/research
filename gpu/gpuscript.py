# opencl_train_logreg.py
import numpy as np
import pyopencl as cl

KERNEL_SRC = r"""
__kernel void sigmoid_dot(
    __global const float* X,   // N*D
    __global const float* W,   // D
    __global float* out,       // N
    const uint N, const uint D
) {
    int i = get_global_id(0);
    if (i < N) {
        float z = 0.0f;
        for (int j = 0; j < D; j++) {
            z += X[i*D + j] * W[j];
        }
        // sigmoid
        out[i] = 1.0f / (1.0f + exp(-z));
    }
}

__kernel void grad_update(
    __global const float* X,     // N*D
    __global const float* y,     // N
    __global const float* preds, // N
    __global float* W,           // D
    const float lr,
    const uint N, const uint D
) {
    int j = get_global_id(0); // weight index
    if (j < D) {
        float grad = 0.0f;
        for (int i = 0; i < N; i++) {
            grad += (preds[i] - y[i]) * X[i*D + j];
        }
        grad /= (float)N;
        W[j] -= lr * grad;
    }
}
"""

def make_ctx_queue():
    plats = cl.get_platforms()
    for p in plats:
        for d in p.get_devices():
            if d.type & cl.device_type.GPU:
                print(f"[OpenCL] Using {p.name} / {d.name}")
                ctx = cl.Context([d])
                queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
                return ctx, queue
    raise RuntimeError("No GPU device found!")

def train_logreg(N=2000, D=128, epochs=200000000, lr=0.1):
    ctx, queue = make_ctx_queue()
    prg = cl.Program(ctx, KERNEL_SRC).build()

    # Random dataset (binary classification)
    X = np.random.randn(N, D).astype(np.float32)
    true_w = np.random.randn(D).astype(np.float32)
    logits = X @ true_w
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(np.float32)

    W = np.zeros(D, dtype=np.float32)
    preds = np.zeros(N, dtype=np.float32)

    mf = cl.mem_flags
    buf_X = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
    buf_y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y)
    buf_W = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)
    buf_preds = cl.Buffer(ctx, mf.READ_WRITE, size=preds.nbytes)

    for epoch in range(epochs):
        # forward pass
        evt = prg.sigmoid_dot(queue, (N,), None, buf_X, buf_W, buf_preds, np.uint32(N), np.uint32(D))
        evt.wait()

        # backward/update
        evt2 = prg.grad_update(queue, (D,), None, buf_X, buf_y, buf_preds, buf_W,
                               np.float32(lr), np.uint32(N), np.uint32(D))
        evt2.wait()

        # monitor loss
        cl.enqueue_copy(queue, preds, buf_preds).wait()
        loss = -np.mean(y*np.log(preds+1e-6) + (1-y)*np.log(1-preds+1e-6))
        print(f"[Epoch {epoch+1}] loss={loss:.4f}")

    cl.enqueue_copy(queue, W, buf_W).wait()
    print("[Done] Final weights norm:", np.linalg.norm(W))

if __name__ == "__main__":
    train_logreg()
