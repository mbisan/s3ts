import torch

@torch.jit.script
def dtw_compute(dist_tensor: torch.Tensor, grad_tensor: torch.Tensor, w: float) -> None:
    for i in range(1, dist_tensor.shape[2]):
        for j in range(1, dist_tensor.shape[3]):
            # elements has shape (3, n, k)
            elements = torch.stack([w * dist_tensor[:, :, i, j-1], dist_tensor[:, :, i-1, j], w * dist_tensor[:, :, i-1, j-1]], dim=0)

            value, id = torch.min(elements, dim=0) # shape (n, k)

            dist_tensor[:,:, i, j] += value

            grad_tensor[id==0][:, :, i, j] += w * grad_tensor[id==0][:, :, i, j-1]

@torch.jit.script
def dtw_compute_faster(dist_tensor: torch.Tensor, grad_tensor: torch.Tensor, w: float) -> None:
    for i in range(1, dist_tensor.shape[0]):
        for j in range(1, dist_tensor.shape[1]):
            # elements has shape (3, n, k)
            value, id = torch.stack([w * dist_tensor[i, j-1], dist_tensor[i-1, j], w * dist_tensor[i-1, j-1]], dim=0).min(dim=0) # shape (n, k)

            dist_tensor[i, j] += value

            grad_tensor[id==0][:, :, i, j] += w * grad_tensor[id==0][:, :, i, j-1]

@torch.jit.script
def dtw_compute_by_index(dist_tensor: torch.Tensor, grad_tensor: torch.Tensor, w: float, n: int, s: int) -> None:
    for i in range(1, dist_tensor.shape[2]):
        for j in range(1, dist_tensor.shape[3]):
            elements = torch.stack([w * dist_tensor[n, s, i, j-1], dist_tensor[n, s, i-1, j], w * dist_tensor[n, s, i-1, j-1]], dim=0)

            value, id = torch.min(elements, dim=0)

            dist_tensor[n, s, i, j] += value

            grad_tensor[id==0][n, s, i, j] += w * grad_tensor[id==0][n, s, i, j-1]

@torch.jit.script
def torch_dtw_fast(x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5):
    # shape of x (n, dim, x_len) y (m, dim, y_len)    
    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    # compute dtw
    DTW = euc_d.clone()
    DTW[:,:,0,:] = torch.cumsum(DTW[:,:,0,:], dim=2)
    DTW[:,:,:,0] = torch.cumsum(DTW[:,:,:,0], dim=2)

    # p_diff contains the partial derivatives of DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))
    p_diff = p_diff / torch.sqrt(euc_d[:,:, None, :, :] + eps)


    dtw_compute(DTW, p_diff, w)
    # futures : List[torch.jit.Future[None]] = []
    # for n in range(DTW.shape[0]):
    #     for s in range(DTW.shape[1]):
    #         futures.append(torch.jit.fork(dtw_compute_by_index, DTW, p_diff, w, n, s))

    # for future in futures:
    #     torch.jit.wait(future)

    return DTW.sqrt(), p_diff

@torch.jit.script
def torch_dtw_faster(x: torch.Tensor, y: torch.Tensor, w: float, eps: float = 1e-5):
    # shape of x (n, dim, x_len) y (m, dim, y_len)

    # performs convolution-like operation, for each kernel the DF
    # (of shape (kernel_size, T)) is computed, then summed across channels
    # x has shape (batch, c, time_dimension)

    # compute pairwise diffs (squared)
    p_diff = x[:,None,:,None,:] - y[None,:,:,:,None] # shape (n, n_kernel, d, Kernel_size, T)
    euc_d = torch.square(p_diff).sum(2) # shape (n, n_kernel, kernel_size, T)

    # p_diff contains the partial derivatives of DTW[n, k, i, j] wrt K[k, d, i] (dims (n, k, d, i, j))
    p_diff = p_diff / torch.sqrt(euc_d[:,:, None, :, :] + eps)

    # compute dtw
    euc_d[:,:,0,:] = torch.cumsum(euc_d[:,:,0,:], dim=2)
    euc_d[:,:,:,0] = torch.cumsum(euc_d[:,:,:,0], dim=2)

    # rearrange dims
    DTW = torch.permute(euc_d, (2, 3, 0, 1)).contiguous()

    dtw_compute_faster(DTW, p_diff, w)

    # recover dimensions
    DTW = torch.permute(DTW, (2, 3, 0, 1)).contiguous()

    return DTW.sqrt(), p_diff

class torch_dtw(torch.autograd.Function):

    @staticmethod
    def forward(x, y, w):
        DTW, p_diff = torch_dtw_faster(x, y, w)
        return DTW, p_diff
    
    @staticmethod
    def setup_context(ctx, inputs, output):
        DTW, p_diff = output
        ctx.save_for_backward(p_diff)
    
    @staticmethod
    def backward(ctx, dtw_grad, p_diff_grad):
        p_diff, = ctx.saved_tensors
        mult = (p_diff * dtw_grad[:,:,None,:,:])
        return mult.mean(dim=(1, 3)), mult.mean(dim=(0, 4)), None