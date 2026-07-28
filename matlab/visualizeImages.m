function visualizeImages(img, mask, ground, stringMethod)
%VISUALIZEIMAGES Displays the results of a segmentation algorithm.
%
%   SYNTAX:
%   visualizeImages(img, mask, ground, stringMethod)
%
%   DESCRIPTION:
%   This function creates two figures to visualize the segmentation results.
%   The first figure compares the final computed mask with the ground truth.
%   The second figure shows the computed mask overlaid with transparency
%   on the original grayscale image.
%
%   INPUTS:
%   img             - The original input grayscale image.
%   mask            - The computed binary segmentation mask.
%   ground          - The ground truth image (can be RGB or binary).
%   stringMethod    - A string containing the name of the segmentation
%                     method used, for use in figure titles.
%
%   See also: visualizeImages_for_land, main.

%% Figure 1: Comparison of Ground Truth and Final Mask

% Create a blank background and overlay the computed mask in cyan.
background = zeros(size(img, 1), size(img, 2), 'double');
coloredMask = imoverlay(background, mask, 'cyan');

% Create a new maximized figure.
figure('Name', 'Segmentation Result vs. Ground Truth', 'WindowState', 'maximized');

% Display Ground Truth in the left subplot.
subplot(1, 2, 1);
imshow(ground);
title('Ground Truth');

% Display the computed mask in the right subplot.
subplot(1, 2, 2);
imshow(coloredMask);
title(sprintf('%s: MASK', stringMethod));

%% Figure 2: Original Image with Overlapped Mask

% Create a new maximized figure.
figure('Name', 'Original Image with Overlaid Mask', 'WindowState', 'maximized');

% Display the original grayscale image.
imshow(img);
title(sprintf('%s: Original image with overlapped mask', stringMethod));
hold on;

% Overlay the colored mask with transparency.
% The AlphaData property controls the transparency level (0 to 1).
h = imshow(coloredMask);
set(h, 'AlphaData', 0.4);

hold off;

end
