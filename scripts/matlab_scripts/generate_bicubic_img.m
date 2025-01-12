function generate_bicubic_img()
%% 该 MATLAB 函数用于生成修改后的图像（mod images），
%% 双线性下采样的图像（bicubic-downsampled images），以及
%% 双线性上采样的图像（bicubic_upsampled images）

%% 设置相关配置
% 根据需要注释掉不必要的代码行
root_folder = '../../datasets';
dataset_folder = 'AID';  % 设置数据集的文件夹名（例如：Set5）

% 使用 root_folder 和 dataset_folder 构建不同文件夹的路径
input_folder = fullfile(root_folder, dataset_folder, 'original');  % 原始图像文件夹路径
% save_bic_folder = '';  % 保存上采样图像的文件夹路径（可选）

mod_scale = 12;  % 用于modcrop操作的模数（即裁剪图像时的缩放比例）
up_scale = 2;    % 上采样比例，表示图像放大的倍数
save_mod_folder = fullfile(root_folder, dataset_folder, sprintf('GTmod%d', mod_scale));  % 保存修改后图像的文件夹路径
save_lr_folder = fullfile(root_folder, dataset_folder, 'LRbic%d', up_scale);  % 保存低分辨率图像的文件夹路径

%% 检查并创建保存文件夹
if exist('save_mod_folder', 'var')  % 检查是否定义了保存修改后图像的文件夹路径
    if exist(save_mod_folder, 'dir')  % 如果文件夹已经存在
        disp(['It will cover ', save_mod_folder]);  % 输出提示，表示会覆盖现有文件夹
    else
        mkdir(save_mod_folder);  % 如果文件夹不存在，则创建文件夹
    end
end

if exist('save_lr_folder', 'var')  % 检查是否定义了保存低分辨率图像的文件夹路径
    if exist(save_lr_folder, 'dir')  % 如果文件夹已经存在
        disp(['It will cover ', save_lr_folder]);  % 输出提示，表示会覆盖现有文件夹
    else
        mkdir(save_lr_folder);  % 如果文件夹不存在，则创建文件夹
    end
end

if exist('save_bic_folder', 'var')  % 检查是否定义了保存上采样图像的文件夹路径
    if exist(save_bic_folder, 'dir')  % 如果文件夹已经存在
        disp(['It will cover ', save_bic_folder]);  % 输出提示，表示会覆盖现有文件夹
    else
        mkdir(save_bic_folder);  % 如果文件夹不存在，则创建文件夹
    end
end

%% 遍历输入文件夹中的所有图像
idx = 0;  % 初始化计数器，用于计数处理的图像
filepaths = dir(fullfile(input_folder,'*.*'));  % 获取输入文件夹中所有文件的路径信息
for i = 1 : length(filepaths)  % 遍历文件夹中的每个文件
    [paths, img_name, ext] = fileparts(filepaths(i).name);  % 获取文件的路径、文件名和扩展名
    if isempty(img_name)  % 如果文件名为空，则跳过（忽略 '.' 文件夹）
        disp('Ignore . folder.');
    elseif strcmp(img_name, '.')  % 如果文件名是 '.'，则跳过（忽略 '..' 文件夹）
        disp('Ignore .. folder.');
    else  % 处理实际图像文件
        idx = idx + 1;  % 更新图像计数器
        str_result = sprintf('%d\t%s.\n', idx, img_name);  % 格式化输出当前处理的图像信息
        fprintf(str_result);

        % 读取图像文件
        img = imread(fullfile(input_folder, [img_name, ext]));  % 读取图像
        img = im2double(img);  % 将图像数据类型转换为双精度

        % 对图像进行 modcrop 操作
        img = modcrop(img, mod_scale);  % 根据给定的 mod_scale 值裁剪图像
        if exist('save_mod_folder', 'var')  % 如果定义了保存修改后图像的文件夹
            imwrite(img, fullfile(save_mod_folder, [img_name, '.png']));  % 保存裁剪后的图像
        end

        % 对图像进行低分辨率处理（下采样）
        im_lr = imresize(img, 1/up_scale, 'bicubic');  % 使用双三次插值进行下采样
        if exist('save_lr_folder', 'var')  % 如果定义了保存低分辨率图像的文件夹
            imwrite(im_lr, fullfile(save_lr_folder, [img_name, '.png']));  % 保存低分辨率图像
        end

        % 对图像进行双线性上采样处理
        if exist('save_bic_folder', 'var')  % 如果定义了保存上采样图像的文件夹
            im_bicubic = imresize(im_lr, up_scale, 'bicubic');  % 使用双三次插值进行上采样
            imwrite(im_bicubic, fullfile(save_bic_folder, [img_name, '.png']));  % 保存上采样图像
        end
    end
end
end

%% modcrop 函数：对图像进行裁剪，使其尺寸能够被指定的模数整除
%% 该函数根据给定的模数（modulo），裁剪输入图像，使得图像的尺寸能够被该模数整除。这个操作在处理图像时常常用于保持图像尺寸一致，尤其在图像缩放时避免出现尺寸不匹配的情况。
function img = modcrop(img, modulo)
if size(img,3) == 1  % 如果图像是灰度图（单通道）
    sz = size(img);  % 获取图像的尺寸
    sz = sz - mod(sz, modulo);  % 调整图像尺寸，使其能够被模数整除
    img = img(1:sz(1), 1:sz(2));  % 裁剪图像
else  % 如果图像是彩色图（多通道）
    tmpsz = size(img);  % 获取图像的尺寸
    sz = tmpsz(1:2);  % 获取图像的高和宽
    sz = sz - mod(sz, modulo);  % 调整图像尺寸，使其能够被模数整除
    img = img(1:sz(1), 1:sz(2), :);  % 裁剪图像
end
end
