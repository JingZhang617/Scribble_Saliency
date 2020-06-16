%% generate mask and new scribble
scribble_dir = '/home/jing-zhang/jing_file/RGB_sal_dataset/scribbled/scribble/';
mask_dir = '/home/jing-zhang/jing_file/RGB_sal_dataset/scribbled/mask/';
new_gt_dir = '/home/jing-zhang/jing_file/RGB_sal_dataset/scribbled/gt/';

%% back: 2; fore: 1
img_list = dir([scribble_dir '*' '.png']);
new_scribble = {};
mask = {};

for i =1:length(img_list)
    i
    img_cur = imread([scribble_dir img_list(i).name]);
    [h,w] = size(img_cur);
    
    mask_cur = zeros(h,w);
    d = img_cur==1;
    mask_cur(d>0) = 255;
    e = img_cur==2;
    mask_cur(e>0) = 255;
    
    new_scri = zeros(h,w);
    new_scri(d>0) = 255;
    imwrite(mask_cur,[mask_dir img_list(i).name]);
    imwrite(new_scri,[new_gt_dir img_list(i).name]);
    
%     imshow(img_cur,[])
%     figure,imshow(mask_cur)
%     figure,imshow(new_scri)
%     close all
end



