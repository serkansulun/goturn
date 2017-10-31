using Images, Knet, MAT, ArgParse, JLD

function get_data(set)
  # returns list of training data
  # written for compatibility with ALOV dataset

  # every element of the data gives us a sample input, vector of 10 elements
  # 1st element is filename of previous frame, 5th element is filename of current
  # frame. rest are corresponding bbox coordinates (x1 y1 x2 y2)

  # because of the memory leak while reading .jpeg files, all images are previously
  # converted to .jld format

  data = Any[]
  categories = readdir(string("dataset/",set,"/annotations"))
  for i=1:length(categories)
    videos = readdir(string("dataset/",set,"/annotations/",categories[i]))
    for j=1:length(videos)
      f = open(string("dataset/",set,"/annotations/",categories[i],"/",videos[j]))
      frames = readlines(f)
      for k=1:length(frames)-1
        fr_pre = split(frames[k])
        fname_pre = string(fr_pre[1],".jld")
        for p=length(fr_pre[1])+1:8
          fname_pre = string("0",fname_pre)
        end
        fr_cur = split(frames[k+1])
        fname_cur = string(fr_cur[1],".jld")
        for p=length(fr_cur[1])+1:8
          fname_cur = string("0",fname_cur)
        end
        fullname_pre = string("dataset/",set,"/images/",categories[i],"/",videos[j][1:end-4],"/",fname_pre)
        fullname_cur = string("dataset/",set,"/images/",categories[i],"/",videos[j][1:end-4],"/",fname_cur)
        x_pre = [parse(Float32,fr_pre[2]) parse(Float32,fr_pre[4]) parse(Float32,fr_pre[6]) parse(Float32,fr_pre[8])]
        y_pre = [parse(Float32,fr_pre[3]) parse(Float32,fr_pre[5]) parse(Float32,fr_pre[7]) parse(Float32,fr_pre[9])]
        x1_pre = minimum(x_pre)
        x2_pre = maximum(x_pre)
        y1_pre = minimum(y_pre)
        y2_pre = maximum(y_pre)
        x_cur = [parse(Float32,fr_cur[2]) parse(Float32,fr_cur[4]) parse(Float32,fr_cur[6]) parse(Float32,fr_cur[8])]
        y_cur = [parse(Float32,fr_cur[3]) parse(Float32,fr_cur[5]) parse(Float32,fr_cur[7]) parse(Float32,fr_cur[9])]
        x1_cur = minimum(x_cur)
        x2_cur = maximum(x_cur)
        y1_cur = minimum(y_cur)
        y2_cur = maximum(y_cur)
        samp = [fullname_pre x1_pre y1_pre x2_pre y2_pre fullname_cur x1_cur y1_cur x2_cur y2_cur]
        push!(data,samp)
      end
    end
  end
  shuffle!(data)
  return data
end



function get_minibatch(trn_data_list,bs,naug,i)
  # Pre-allocate
  target = Array(Float32,resiz,resiz,3,bs)
  search = Array(Float32,resiz,resiz,3,bs)
  label = Array(Float32,4,bs)
  box_pre = Array(Float32,4,bs)
  box_cur = Array(Float32,4,bs)
  imsize = Array(Int64,3,bs)
  augmented = Array(Bool,1,bs)

  k = 1 # index of samples put into MB
  while k-1<bs && i-1<length(trn_data_list)

    # put true sample
    im_pre,im_cur,box_pre[:,k],box_cur[:,k] = get_imandbox(trn_data_list[i]) # get images and boxes
    # get target and search regions, and labels
    target[:,:,:,k],search[:,:,:,k],label[:,k] = get_inputs(im_pre,im_cur,box_pre[:,k],box_cur[:,k])
    imsize[:,k] = collect(size(im_pre))
    augmented[k] = false
    if !discardsample(label[:,k]) # adds true sample
      k = k + 1
      j = 1 # index of augmentation trials
      while k-1<bs && j-1<naug
        # put augmented sample
        im_pre,im_cur,box_pre[:,k],box_cur[:,k] = get_imandbox(trn_data_list[i])
        target[:,:,:,k],search[:,:,:,k],label[:,k] = get_aug_inputs(im_pre,im_cur,box_pre[:,k],box_cur[:,k])
        imsize[:,k] = collect(size(im_pre))
        augmented[k] = true
        j = j+1
        if !discardsample(label[:,k]) # adds augmented sample
          k = k+1
        end
      end
    end
    i = i+1
  end
  # when the end of dataset is reached,
  # make sure to clear pre-allocated stuff that isn't filled
  if i>length(trn_data_list)
    target = target[:,:,:,1:k-1]
    search = search[:,:,:,1:k-1]
    label = label[:,1:k-1]
    box_pre = box_pre[:,1:k-1]
    box_cur = box_cur[:,1:k-1]
    imsize = imsize[:,1:k-1]
    augmented = augmented[1:k-1]
  end

  target = convert(atype,target)
  search = convert(atype,search)
  label = convert(atype,label)
  return (target,search,label),(box_pre,box_cur,imsize,augmented),i
end

function get_imandbox(samp)
  #= data list is a vector of samples, and samples are strings of filenames
  and array coordinates. this function uses that information and provides images
  and a vector of coordinates seperately=#
  im_pre = load(samp[1])
  im_pre = im_pre["im"]
  im_cur = load(samp[6])
  im_cur = im_cur["im"]
  box_pre = ([samp[2]; samp[3] ;samp[4]; samp[5]])
  box_cur = ([samp[7] ;samp[8]; samp[9] ;samp[10]])
  return im_pre,im_cur,box_pre,box_cur
end

function get_inputs(im_pre,im_cur,box_pre,box_cur)
  #= takes images and coordinates, returns cropped target and search regions,
  and coordinates relative to search region (label) =#

  # crop and (if necessary) pad
  target = croppad(im_pre,box_pre)
  search = croppad(im_cur,box_pre)
  # transform the bbox relative to the search region
  label = bbox_transform(size(im_cur),box_pre,box_cur)
  # resize regions to a fixed size, and transform label accordingly
  target_rs = imresize(target,(resiz,resiz))
  search_rs = imresize(search,(resiz,resiz))
  label_rs = resize_label(label,size(search_rs),size(search))
  return target_rs, search_rs, label_rs
end

function get_aug_inputs(im_pre,im_cur,box_pre,box_cur)
  # provides augmented inputs
  # only difference is previous frame's bbox is shifted and scaled (augmented)
  box_aug = shiftscale(box_pre)
  target = croppad(im_pre,box_pre)
  search = croppad(im_cur,box_aug)
  label = bbox_transform(size(im_cur),box_aug,box_cur)

  target_rs = imresize(target,(resiz,resiz))
  search_rs = imresize(search,(resiz,resiz))
  label_rs = resize_label(label,size(search_rs),size(search))
  return target_rs, search_rs, label_rs
end



function resize_label(label,size_new,size_old)
  y_old,x_old,_ = size_old
  y_new,x_new,_ = size_new

  label_new = [1+x_new/x_old*(label[1]-1) ; 1+y_new/y_old*(label[2]-1) ; 1+x_new/x_old*(label[3]-1) ; 1+y_new/y_old*(label[4]-1)]
  #label_new = convert(Array{Int},round(label_new))
end

function croploc(imsiz,box)
  # provides cropping information
  crop_factor = 2 # crop 2 times the area around ground-truth box
  box = convert(Array{Int},round(box))
  box_w = ((box[3] - box[1])) # width
  box_h = ((box[4] - box[2])) # height
  box_cx = Int(round(box[1] + box_w/2)) # center
  box_cy = Int(round(box[2] + box_h/2))
  crop_w = Int(round(box_w*crop_factor)) # crop width
  crop_h = Int(round(box_h*crop_factor)) # crop height
  crop_x1 = Int(round(max(1,box_cx - crop_w/2))) # crop coordinates
  crop_x2 = Int(round(min(imsiz[2],box_cx + crop_w/2)))
  crop_y1 = Int(round(max(1,box_cy - crop_h/2)))
  crop_y2 = Int(round(min(imsiz[1],box_cy + crop_h/2)))
  # if crop coordinates area outside image, record the value of how much it is
  edge_space_x1 = Int(round(crop_x1 - (box_cx - crop_w/2)))
  edge_space_x2 = Int(round((box_cx + crop_w/2) - crop_x2))
  edge_space_y1 = Int(round(crop_y1 - (box_cy - crop_h/2)))
  edge_space_y2 = Int(round((box_cy + crop_h/2) - crop_y2))

  crop_w = crop_w + 1 # true width and height values are 1 larger than the difference
  crop_h = crop_h + 1

  crop_loc = (crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h)
  edge_space = (edge_space_x1,edge_space_y1,edge_space_x2,edge_space_y2)

  return crop_loc,edge_space
end

function croppad(im,box)
  # crops and (if necessary) pads the image
  crop_loc,edge_space = croploc(size(im),box)
  (crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h) = crop_loc
  (edge_space_x1,edge_space_y1,edge_space_x2,edge_space_y2) = edge_space
  im = im[crop_y1:crop_y2,crop_x1:crop_x2,:]
  im_padded = zeros(Int64,crop_h,crop_w,3)
  im_padded[edge_space_y1+1:end-edge_space_y2,edge_space_x1+1:end-edge_space_x2,:] = im

  return im_padded
end

function bbox_transform(imsize,box,gt)
  # transform bbox to get coordinates relative to the cropped region
  crop_loc,edge_space = croploc(imsize,box)
  (crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h) = crop_loc
  (edge_space_x1,edge_space_y1,edge_space_x2,edge_space_y2) = edge_space
  x1_recentered = gt[1] - crop_x1 + edge_space_x1
  x2_recentered = gt[3] - crop_x1 + edge_space_x1
  y1_recentered = gt[2] - crop_y1 + edge_space_y1
  y2_recentered = gt[4] - crop_y1 + edge_space_y1
  box_recentered = [x1_recentered; y1_recentered; x2_recentered; y2_recentered]
  return box_recentered
end

function inv_bbox_transform(imsize,box,pred)
  # inverse opereation
  crop_loc,edge_space = croploc(imsize,box)
  (crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h) = crop_loc
  (edge_space_x1,edge_space_y1,edge_space_x2,edge_space_y2) = edge_space
  x1_trans = pred[1] + crop_x1 - edge_space_x1
  x2_trans = pred[3] + crop_x1 - edge_space_x1
  y1_trans = pred[2] + crop_y1 - edge_space_y1
  y2_trans = pred[4] + crop_y1 - edge_space_y1
  trans = [x1_trans; y1_trans; x2_trans; y2_trans]
  return trans
end

function shiftscale(box)
  # probabilistically shifts and scale the bounding box to create augmented samples
  # get width, height and center of previous frame's bbox
  w = box[3] - box[1] + 1
  h = box[4] - box[2] + 1
  xc = (box[1] + box[3]) / 2
  yc = (box[2] + box[4]) / 2
  # sample factors for scale and position change from a Laplace distribution
  shift_x = laplacian(5) # mean 0 lambda 5
  shift_y = laplacian(5)
  scale_w = 1 + laplacian(15) # mean 1 lambda 15
  scale_h = 1 + laplacian(15)

  # shift and scale to find new values
  xcn = xc + w * shift_x
  ycn = yc + h * shift_y
  wn = w * scale_w
  hn = h * scale_h
  # find new coordinates
  x1n = (xcn-(wn-1)/2)
  x2n = (xcn+(wn-1)/2)
  y1n = (ycn-(hn-1)/2)
  y2n = (ycn+(hn-1)/2)
  box_aug = Float32[x1n;y1n;x2n;y2n]
end

function laplacian(lambda) # two sided laplacian probability function
  return log(rand()) / lambda * rand([-1 1]);
end

function alignbox(rotbox)
  #= VOT 2014 dataset uses rotated rectangular boxes as annotations
  this function aligns the boxes to unrotated rectangles
  =#
  x1 = minimum(rotbox[1:2:end-1])
  x2 = maximum(rotbox[1:2:end-1])
  y1 = minimum(rotbox[2:2:end])
  y2 = maximum(rotbox[2:2:end])
  return [x1;y1;x2;y2]
end

function changelr(prms,lr) # changes learning rate
  for k=1:length(prms)
    prms[k].lr = lr
  end
  return prms
end

function initparams(weights,o) # optimization parameters
  prms = Array(Any,length(weights))
  for k = 1:length(weights)
    if o[:optim] == "Adagrad"
      prms[k] = Adagrad(lr=o[:lr])
    elseif o[:optim] == "Adagrad"
      prms[k] = Adagrad(lr=o[:lr])
    elseif o[:optim] == "Adam"
      prms[k] = Adam(lr=o[:lr])
    end
  end
  return prms
end

function draw_box(im,box) # draws a box on the image, specified by coordinates
  # for debugging purposes
  im2=im
  box = convert(Array{Int},round(box))
  im2[box[2],box[1]:box[3],:]=0
  im2[box[4],box[1]:box[3],:]=0
  im2[box[2]:box[4],box[1],:]=0
  im2[box[2]:box[4],box[3],:]=0
  myimshow(im2)
end

function drop_out(x,p)
  if p > 0
    x .* (rand!(similar(x)) .> p) ./ (1-p)
  else
    x
  end
end

function get_fcinput(w,x_target,x_search)
  # put each patch to CNN separately
  y_target = forward_conv(w,x_target)
  y_search = forward_conv(w,x_search)
  # cat function malfunctions with knet somehow so I convert it to Array{Float32}
# concatanete outputs to obtain final input for FC layers
  x_fc = cat(3,convert(Array{Float32},y_target),convert(Array{Float32},y_search))
  x_fc = convert(atype,x_fc)
end

function loss(w_fc,x_fc,y_gold,drop)
  # L1 loss
  y_pred = forward_fc(w_fc,x_fc,drop)
  sumabsdiff = sum(abs(y_pred - y_gold),1)
  loss = sum(sumabsdiff)/size(sumabsdiff,2)
end

function trnaccuracy(w_fc,x_fc,box_pre,box_cur,imsize,augmented)
# to calculate accuracy, inverse transform is performed:
# coordinates relevant to the patch are turned into coordinates relative
# to the entire image (true coordinates)

  y_pred = forward_fc(w_fc,x_fc,0)
  acc = 0
  count = 0
  for i = 1:size(y_pred,2)
      if !augmented[i]
        # accuracy is only calculated on true samples
        # since on augmented images are changed at patch level, not on
        # the entire image level
        count = count + 1
        (crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h),edge_space = croploc(imsize[:,i],box_pre[:,i])
        box_res = resize_label(y_pred[:,i],(crop_h,crop_w,3),(resiz,resiz,3))
        box_pred = inv_bbox_transform(imsize[:,i],box_pre[:,i],box_res)
        acc += iou(box_pred,box_cur[:,i])
      end
  end

  return acc/count
end

function iou(a,b)
  # intersection-over-union value between two boxes

  x1 = max(a[1], b[1]);
  y1 = max(a[2], b[2]);
  x2 = min(a[3], b[3]);
  y2 = min(a[4], b[4]);

  w = x2-x1+1;
  h = y2-y1+1;
  inter = w.*h;
  aarea = (a[3]-a[1]+1) .* (a[4]-a[2]+1);
  barea = (b[3]-b[1]+1) * (b[4]-b[2]+1);
  # intersection over union overlap
  o = inter ./ (aarea+barea-inter);
  # set invalid entries to 0 overlap
  if w<(0) || h<(0)
    o=0
  end
  return o
end

function train(w_fc,x_fc,y_gold,prms,drop)
  # conv layers are fixed, only FC layers are trained
  g = lossgradient(w_fc,x_fc,y_gold,drop)

  for i=1:length(w_fc)
    #w[i] -= lr * g[i]
    #axpy!(-lr, g[i], w_fc[i])
    update!(w_fc[i],g[i],prms[i])
  end
  return w_fc
end

lossgradient = grad(loss)

function forward_conv(w,x) # forward convolution
  x = relu(conv4(w[1],x;padding=0,stride=4) .+ w[2])
  x = pool(x;window=3,padding=0,stride=2)
  x = relu(conv4(w[3],x;padding=2,stride=1) .+ w[4])
  x = pool(x;window=3,padding=0,stride=2)
  x = relu(conv4(w[5],x;padding=1,stride=1) .+ w[6])
  x = relu(conv4(w[7],x;padding=1,stride=1) .+ w[8])
  x = relu(conv4(w[9],x;padding=1,stride=1) .+ w[10])
  x = pool(x;window=3,padding=0,stride=2)
end

function forward_fc(w,x,drop) # forward FC layers pass
  x = mat(x)
  for i=1:2:length(w)-3
    x = relu(w[i]*x.+w[i+1])
    if drop>0
      x = dropout(x,drop)
    end
  end
  return w[end-1]*x.+w[end]
end

function weights_fc() # initialize FC layers randomly
  w = Array(Any,8)
  w[1] = 0.005*randn(4096,512*6*6)
  w[2] = ones(4096,1)
  w[3] = 0.005*randn(4096,4096)
  w[4] = ones(4096,1)
  w[5] = 0.005*randn(4096,4096)
  w[6] = ones(4096,1)
  w[7] = 0.01*randn(4,4096)
  w[8] = zeros(4,1)
  for i=1:length(w)
    w[i] = convert(atype,w[i])
  end
  return w
end

function weights_conv()
  # get FIXED convolutional weights from ImageNet
  f = matopen("w.mat")
  w = read(f,"w")
  w_any = Array(Any,length(w))
  for i=2:2:length(w)
    w[i] = reshape(w[i],(1,1,size(w[i],1),size(w[i],2)))
  end
  for i=1:length(w)
    w_any[i] = convert(atype,w[i])
  end
  return w_any
end

function rand_weights()
  # (for trial purposes) initialize convolutional weights randomly
  w=Array(Any,10)
  w[1] = 0.01*randn(11,11,3,64)
w[2] = xavier(1,1,64,1)
w[3] = 0.01*randn(5,5,64,256)
w[4] = xavier(1,1,256,1)
w[5] = 0.01*randn(3,3,256,256)
w[6] = xavier(1,1,256,1)
w[7] = 0.01*randn(3,3,256,256)
w[8] = xavier(1,1,256,1)
w[9] = 0.01*randn(3,3,256,256)
w[10] = xavier(1,1,256,1)
  for i=1:length(w)
    w[i] = convert(atype,w[i])
  end
  return w
end

function discardsample(label)
  #= if the bbox is close to the edge of the image, when it is augmented,
  resulting box can be outside the image. these samples are discarded =#
  return maximum(label)>resiz || minimum(label)<0
end

function myimshow(im)
  # (for debugging) display image
  im_float = convert(Array{Float64},im)
  im_permuted = permutedims(im_float,[3 1 2])
  im_normed = im_permuted./255
  im_color = ColorView{RGB}(im_normed)
  imshow(im_color)
end

function myimread(filename)
  # reads image (this is used while previously converting to .jld)
  im = load(filename)
  im = convert(Array{Int64},rawview(channelview(im)))
  if ndims(im)==2
    tmp = zeros(3,size(im,1),size(im,2))
    tmp[1,:,:] = im
    tmp[2,:,:] = im
    tmp[3,:,:] = im
    im = tmp
  end
  im = permutedims(im,[2 3 1])
end

function display_boxed(im,box)
  # (for debugging) displays patch cropped from image
  box = convert(Array{Int},round(box))
  myimshow(im[box[2]:box[4],box[1]:box[3],:])
end

function tstaccuracy(w_c,w_fc,n,t)
  # written to be compatible with VOT standards

#= accuracy is measured by taking intersection-over-union ratio
between ground truth and predicted bounding boxes

robustness is measured by (1-number of resets)/number of iterations

tracker is reset when accuracy becomes zero (target lost),
 and the ground-truth is provided just like in the first frame

to reduce bias, the tracker is reset 5 frames after target is lost
and first 10 frames after reset are not included in the computation =#

    videos = readdir(string("dataset/tst/")) # videos

    if n==0
      n = length(videos)
    end
    acc_vid_tot = 0 # cumulative accuracy
    rob_vid_tot = 0 # cumulative robustness
    #println("Testing. Remaining videos: ")
    for j=1:n # for every video

      frames = readdir(string("dataset/tst/",videos[j])) # frames
      frames = frames[1:end-1] # exclude last file (groundtruth.txt)
      f = open(string("dataset/tst/",videos[j],"/groundtruth.txt"))
      gt = readlines(f) # ground truth
      acc_fr_tot = 0 # cumulative accuracy of frames
      count_tst = 0 # counter for number of tested frames
      count_reset = 0 # counter for tracker reset
      box_pre = Any[]
      reset = true
      k = -4 # reset will add 5 anyway so it'll start from frame 1
      cdown = 10 # countdown to start performance calculations (don't include next 10 frames)

      while k<length(frames) # for every frame

        if reset
          reset = false
          if k<length(frames)-5
            k = k + 5 # skip next 5 frames
          end
          # if the tracker resets, provide ground-truth info
          l_pre = gt[k][1:end-1] # get line (box coordinates)
          l_pre = split(l_pre,",")
          box_rot_pre = [parse(Float32,l_pre[1]);parse(Float32,l_pre[2]);parse(Float32,l_pre[3]);parse(Float32,l_pre[4]);parse(Float32,l_pre[5]);parse(Float32,l_pre[6]);parse(Float32,l_pre[7]);parse(Float32,l_pre[8])]
          box_pre = alignbox(box_rot_pre) # bbox for previous frame
          if cdown<=0
            count_reset = count_reset + 1
          end
          cdown = 10
        end

        im_pre = load(string("dataset/tst/",videos[j],"/",frames[k]))
        im_pre = im_pre["im"] # previous frame

        l_cur = gt[k+1][1:end-1]
        l_cur = split(l_cur,",")
        box_rot_cur = [parse(Float32,l_cur[1]);parse(Float32,l_cur[2]);parse(Float32,l_cur[3]);parse(Float32,l_cur[4]);parse(Float32,l_cur[5]);parse(Float32,l_cur[6]);parse(Float32,l_cur[7]);parse(Float32,l_cur[8])]
        box_cur = alignbox(box_rot_cur)

        im_cur = load(string("dataset/tst/",videos[j],"/",frames[k+1]))
        im_cur = im_cur["im"] # current frame

        target = croppad(im_pre,box_pre) # crop target region
        search = croppad(im_cur,box_pre) # crop search region
        target_rs = imresize(target,(resiz,resiz)) # resize to a fixed size
        search_rs = imresize(search,(resiz,resiz))
        target_rs = convert(atype,target_rs)
        search_rs = convert(atype,search_rs)
        target_rs = reshape(target_rs,(size(target_rs)...,1)) # adjust dimensions
        search_rs = reshape(search_rs,(size(search_rs)...,1))
        y_target = forward_conv(w_c,target_rs) # put through conv layers
        y_search = forward_conv(w_c,search_rs)

        # concatanate outputs
        x_fc = cat(3,convert(Array{Float32},y_target),convert(Array{Float32},y_search))
        x_fc = convert(atype,x_fc)

        y_pred = forward_fc(w_fc,x_fc,0) # put through FC layers to get prediction
        # the predicted coordinates are relative to the search region
        # we need a transform (inverse) from relative coordinates to true coordinates
        # get cropping information necessary to inverse transform
        (crop_x1,crop_y1,crop_x2,crop_y2,crop_w,crop_h),edge_space = croploc(size(im_pre),box_pre)
        # inverting the resize operation
        box_res = resize_label(y_pred,(crop_h,crop_w,3),(resiz,resiz,3))
        # inverting the bbox transform (cropping operation)
        box_pred = inv_bbox_transform(size(im_pre),box_pre,box_res)
        acc_fr = iou(box_pred,box_cur) # intersection over union gives accuracy

        box_pre = box_pred # prediction provides the target region for next frame
        cdown = cdown - 1
        k = k+1
        if cdown<=0
          reset = acc_fr <= t # if intersection is less than threshold, reset the tracker
            count_tst = count_tst + 1
            acc_fr_tot += acc_fr
        end
      end
      rob_vid_tot += (count_tst-count_reset)/count_tst # average robustness over frames
      acc_vid_tot += acc_fr_tot/count_tst # average accuracy over frames
    end

    rob = rob_vid_tot/n # average robustness over videos
    acc = acc_vid_tot/n # average accuracy over videos

    return acc,rob
end



function main(args=ARGS)
  s = ArgParseSettings()
  s.description="GOTURN"
  s.exc_handler=ArgParse.debug_handler
  @add_arg_table s begin

    ("--lr"; arg_type=Float64; default=1e-5; help="learning rate")
    ("--lrdecay"; arg_type=Float64; default=1.0; help="learning rate decay rate")
    ("--stepsize"; arg_type=Int; default=1; help="number of epochs until lr is decayed")
    ("--dropout"; arg_type=Float64; default=0.5; help="dropout rate")
    ("--treset"; arg_type=Float64; default=0.4; help="accuracy threshold to reset tracker")
    ("--epochs"; arg_type=Int; default=15; help="number of epochs for training")
    ("--optim"; default="Adam"; help="optimization method (Sgd, Momentum, Adam, Adagrad, Adadelta, Rmsprop)")
    ("--nsamples"; arg_type=Int; default=0; help="number of samples to train, 0 to use all")
    ("--ntst"; arg_type=Int; default=0; help="number of samples to test, 0 to use all")  
    ("--batchsize"; arg_type=Int; default=50; help="minibatch size")   
    ("--naug"; arg_type=Int; default=0; help="number of augmented samples per real sample")
    ("--randw"; arg_type=Bool; default=true; help="use random weights for FC layers")
    ("--savew"; arg_type=Bool; default=true; help="save weights for FC layers")
    ("--tstonly"; arg_type=Bool; default=false; help="only test, no training")
    ("--tstepoch"; arg_type=Int; default=1; help="epoch to test")
    ("--seed"; arg_type=Int; default=1; help="random number seed: use a nonnegative int for repeatable results")
  end

  println(s.description)
  isa(args, AbstractString) && (args=split(args))
  o = parse_args(args, s; as_symbols=true)

  o[:seed] > 0 && srand(o[:seed])
  if o[:tstonly]
    epochs = 1
  else
    epochs=o[:epochs]
  end

  global resiz=227 # resize patches to 227x227

  if gpu()>=0
    global atype = KnetArray{Float32}
  else
    global atype = Array{Float32}
  end

  w_c = weights_conv() # get imagenet weights.
  if o[:randw] && !o[:tstonly] # randomly initialize FC layer weights
    w_fc = weights_fc()
    prms = initparams(w_fc,o)
  else # continue from previously trained FC layer weights
    w_fc = load(string("w_prms/w_fc.jld"))
    w_fc = w_fc["w_fc"]
    prms = load(string("w_prms/prms.jld"))
    prms = prms["prms"]
    prms = changelr(prms,o[:lr])
  end

  trn_data_list = get_data("trn") # training
  val_data_list = get_data("val") # validation

  # smaller number of samples can be used if wished
  if o[:nsamples] == 0
    n_samp = length(trn_data_list)
  else
    n_samp = o[:nsamples]
  end
  trn_data_list = trn_data_list[1:n_samp]

  if o[:batchsize]>(o[:naug]+1)*o[:nsamples] && o[:nsamples]!=0
    bs = (o[:naug]+1)*o[:nsamples]
  else
    bs=o[:batchsize]
  end

  lr = o[:lr]
  drop=o[:dropout]
  all_trnacc = Float32[]
  all_trnlss = Float32[]
  all_tstacc = Float32[]
  all_tstrob = Float32[]

    for epoch=1:epochs
      #tic()
      if !o[:tstonly]

        shuffle!(trn_data_list)
        # Training loop

        i = 1
        while i<length(trn_data_list)

          (x_target,x_search,y_gold),(box_pre,box_cur,imsize,augmented),i = get_minibatch(trn_data_list,bs,o[:naug],i)
          x_fc = get_fcinput(w_c,x_target,x_search)

          w_fc = train(w_fc,x_fc,y_gold,prms,drop)
        end
   
        #run over dataset 2nd time to get loss and accuracy
        i=1
        trnlss = 0
        trnacc = 0
        trn_nbatch = 0
        while i<length(trn_data_list)

          (x_target,x_search,y_gold),(box_pre,box_cur,imsize,augmented),i = get_minibatch(trn_data_list,bs,0,i)
          x_fc = get_fcinput(w_c,x_target,x_search)

          trnlss += loss(w_fc,x_fc,y_gold,0)
          #trnacc+=trnaccuracy(w_fc,x_fc,box_pre,box_cur,imsize,augmented)
          trn_nbatch = trn_nbatch + 1
        end

        j=1
        vallss = 0
        valacc = 0
        val_nbatch = 0
        while j<length(val_data_list)

          # if mod(nbatch,5) == 0
          #   @printf "%d " length(trn_data_list)-i
          # end


          (x_target,x_search,y_gold),(box_pre,box_cur,imsize,augmented),j = get_minibatch(val_data_list,bs,0,j)

          # put each patch through CNN seperately
          y_target = forward_conv(w_c,x_target)
          y_search = forward_conv(w_c,x_search)
          # concatanete outputs to obtain final input for FC layers
          x_fc = cat(3,convert(Array{Float32},y_target),convert(Array{Float32},y_search))
          x_fc = convert(atype,x_fc)

          vallss += loss(w_fc,x_fc,y_gold,0)
          #valacc+=trnaccuracy(w_fc,x_fc,box_pre,box_cur,imsize,augmented)
          val_nbatch = val_nbatch + 1
        end

        if o[:savew]
            save(string("w_prms/w_fc.jld"),"w_fc",w_fc)
            save(string("w_prms/prms.jld"),"prms",prms)
        end

        trnlss_norm = trnlss/trn_nbatch
        vallss_norm = vallss/val_nbatch

        @printf "\nepoch: %d, " epoch
        @printf "trn_loss: %f val_loss: %f " trnlss_norm vallss_norm

        # update learning rate
        if mod(epoch,o[:stepsize])==0 && o[:lrdecay]!=1.0
          lr = lr*o[:lrdecay]
          prms = changelr(prms,lr)
          println("New learning rate: $lr")
        end
      end

        tstrob = NaN
        tstacc = NaN
        # get accuracy and robustness on test set
        tstacc,tstrob = tstaccuracy(w_c,w_fc,o[:ntst],o[:treset])
        tstoverall = (tstacc+tstrob)/2
        @printf "tst_accuracy: %f tst_robustness: %f tst_overall: %f" tstacc tstrob tstoverall
    #  toc()
    end
end

if VERSION >= v"0.5.0-dev+7720"
    PROGRAM_FILE == "goturn.jl" && main(ARGS)
else
    !isinteractive() && !isdefined(Core.Main,:load_only) && main(ARGS)
end

