import torch
# from IPython import embed
# import pdb

def lr_crop_index(n, N, D, base_size, overlap):
    n_end = D if n == N - 1 else (n + 1) * base_size + overlap
    n_start = n_end - base_size - overlap
    return n_start, n_end


def hr_crop_index(n, N, D, Dmod, base_size, overlap):
    if n == 0:
        n_start = 0
        n_end = (n + 1) * base_size + (overlap // 2 if N > 1 else overlap)
        pn_start = n_start
        pn_end = n_end
    elif n == N - 1:
        n_end = D
        pn_end = base_size + overlap
        if Dmod > overlap:
            n_start = n_end - Dmod
            pn_start = pn_end - Dmod
        else:
            n_start = n_end - base_size - overlap // 2
            pn_start = overlap // 2
    else:
        n_start = n * base_size + overlap // 2
        n_end = n_start + base_size
        pn_start = overlap // 2
        pn_end = pn_start + base_size
    return n_start, n_end, pn_start, pn_end


def forward_crop(x, model, continous=False, use_ddim=False, lq_size=40, scale=4, overlap=5):

    # Assert the required dimension.
    assert lq_size == 40 , "Default patch size of LR images during training and validation should be {}.".format(lq_size)
    assert overlap == 5 , "Default overlap of patches during validation should be {}.".format(overlap)

    # Prepare for the image crops.
    # print(x.shape)
    base_size = lq_size - overlap
    lr = x['inp']
    B, C, H, W = lr.shape
    # print(x.shape)

    I = H // base_size
    Hmod = H % base_size
    if Hmod > overlap:
        I += 1

    J = W // base_size
    Wmod = W % base_size
    if Wmod > overlap:
        J += 1

    # print(I, Hmod, J, Wmod)
    data = {
        'inp': lr,
        'coord': x['coord'],
        'cell': x['cell'],
        'gt': x['gt'],
        "scaler": x['scaler']} 
    # Crop the entire image into 64 x 64 patches. Concatenate the crops along the batch dimension.
    x_crops = []
    for i in range(I):
        i_start, i_end = lr_crop_index(i, I, H, base_size, overlap)
        for j in range(J):
            j_start, j_end = lr_crop_index(j, J, W, base_size, overlap)
            x_crop = lr[:, :, i_start: i_end, j_start: j_end]
            data['inp'] = x_crop
            x_crop = model.super_resolution(data, continous, use_ddim)
            x_crops.append(x_crop)
            
    x_crops = torch.stack(x_crops)
    

    # Calculate the enlarged dimension.
    H, W = H * scale, W * scale
    Hmod, Wmod = Hmod * scale, Wmod * scale
    base_size, overlap = base_size * scale, overlap * scale
    # print(H, W, Hmod, Wmod, base_size, overlap)
    # print('Second')
    # Convert the SR crops to an entire image
    if len(x_crops.shape) == 4:
        x = torch.zeros(B, C, H, W)
        for i in range(I):
            i_start, i_end, pi_start, pi_end = hr_crop_index(i, I, H, Hmod, base_size, overlap)
            for j in range(J):
                j_start, j_end, pj_start, pj_end = hr_crop_index(j, J, W, Wmod, base_size, overlap)
                # print(i_start, i_end, j_start, j_end)
                # print(pi_start, pi_end, pj_start, pj_end)
                B_start = B * (i * J + j)
                B_end = B_start + B
                # print(B_start, B_end)
                x[:, :, i_start: i_end, j_start: j_end] \
                    = x_crops[B_start: B_end, :, pi_start: pi_end, pj_start: pj_end]

    return x



