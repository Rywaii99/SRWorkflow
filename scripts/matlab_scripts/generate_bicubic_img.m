function generate_bicubic_img()
%% 该 MATLAB 函数用于生成修改后的图像（mod images），
%% 双线性下采样的图像（bicubic-downsampled images）

%% 设置相关配置
% dataset = 'AID';
dataset = 'DOTA';
% dataset = 'DIOR';
root_folder = sprintf('../../datasets/processed/%s', dataset);  % 根目录路径
input_folder = fullfile(root_folder, 'test_HR');  % 输入文件夹路径
% input_folder = fullfile(root_folder, 'train_HR');  % 输入文件夹路径
% input_folder = fullfile(root_folder, 'AID_train_HR');  % 输入文件夹路径

mod_scale = 12;  % 用于modcrop操作的模数（即裁剪图像时的缩放比例）
up_scale = 2;    % 下采样和上采样的比例，表示图像缩小或放大的倍数

% save_mod_folder = fullfile(root_folder, sprintf('%s_train_HR', dataset));  % 输出修改后图像的文件夹路径
% save_lr_folder = fullfile(root_folder, sprintf('%s_train_LR_bicubic', dataset), sprintf('X%d', up_scale));  % 输出低分辨率图像的文件夹路径
save_mod_folder = fullfile(root_folder, sprintf('%s_test_HR', dataset));  % 输出修改后图像的文件夹路径
save_lr_folder = fullfile(root_folder, sprintf('%s_test_LR_bicubic', dataset), sprintf('X%d', up_scale));  % 输出低分辨率图像的文件夹路径

%% 检查并创建保存文件夹
% 创建保存修改后图像的文件夹
if exist(save_mod_folder, 'dir')
    disp(['It will cover ', save_mod_folder]);
else
    mkdir(save_mod_folder);
end

% 创建保存低分辨率图像的文件夹
if exist(save_lr_folder, 'dir')
    disp(['It will cover ', save_lr_folder]);
else
    mkdir(save_lr_folder);
end

%% 遍历输入文件夹中的所有图像
idx = 0;  % 初始化计数器，用于计数处理的图像
filepaths = dir(fullfile(input_folder, '*.*'));  % 获取输入文件夹中所有文件的路径信息
for i = 1:length(filepaths)  % 遍历文件夹中的每个文件
    [paths, img_name, ext] = fileparts(filepaths(i).name);  % 获取文件的路径、文件名和扩展名
    if isempty(img_name)  % 如果文件名为空，则跳过
        disp('Ignore empty file.');
    elseif strcmp(img_name, '.') || strcmp(img_name, '..')  % 如果文件名是 '.' 或 '..'，则跳过
        disp(['Ignore folder: ', img_name]);
    else  % 处理实际图像文件
        idx = idx + 1;  % 更新图像计数器
        str_result = sprintf('%d\t%s.\n', idx, img_name);  % 格式化输出当前处理的图像信息
        fprintf(str_result);

        % 读取图像文件
        img = imread(fullfile(input_folder, [img_name, ext]));  % 读取图像
        img = im2double(img);  % 将图像数据类型转换为双精度

        % 对图像进行 modcrop 操作
        % img = modcrop(img, mod_scale);  % 根据给定的 mod_scale 值裁剪图像
        % imwrite(img, fullfile(save_mod_folder, [img_name, '.png']));  % 保存裁剪后的图像

        % 对图像进行低分辨率处理（下采样）
        im_lr = imresize(img, 1/up_scale, 'bicubic');  % 使用双三次插值进行下采样
        imwrite(im_lr, fullfile(save_lr_folder, [img_name, '.png']));  % 保存低分辨率图像
    end
end

disp('处理完成！');

end

%% modcrop 函数：对图像进行裁剪，使其尺寸能够被指定的模数整除
function img = modcrop(img, modulo)
if size(img, 3) == 1  % 如果图像是灰度图（单通道）
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
