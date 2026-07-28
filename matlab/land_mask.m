function [oilThresh, BWautomatic] = land_mask(img, ground, areaToExplore, landThreshold, medFilt)
%LAND_MASK Segments SAR images containing both land and sea.
%
%   SYNTAX:
%   [oilThresh, BWautomatic] = land_mask(img, ground, areaToExplore, landThreshold, medFilt)
%
%   DESCRIPTION:
%   This function orchestrates the segmentation of SAR images that include
%   land masses. It first enhances the image, then creates a mask for the
%   land based on a brightness threshold. It then calls a separate function
%   to detect oil spills and combines the results for visualization and
%   evaluation.
%
%   INPUTS:
%   img             - The input grayscale SAR image (double format).
%   ground          - The ground truth image for visualization.
%   areaToExplore   - The minimum area (in pixels) for a blob to be
%                     considered an oil spill.
%   landThreshold   - The brightness threshold (0-1) to identify land.
%   medFilt         - The window size for the median/Wiener filter.
%
%   OUTPUTS:
%   oilThresh       - A colored (cyan) overlay mask of the detected spills.
%   BWautomatic     - The final combined binary mask of both land and spills.
%
%   See also: main, automatic_threshold_for_land, visualizeImages_for_land.

% Start timer
tic;

%% 1. Image Enhancement
% Apply a Wiener filter for adaptive noise reduction, which is often
% effective for speckle noise in SAR images.
noiseRemoved = wiener2(img, [medFilt medFilt]);

% Sharpen the image to enhance edges, which can help differentiate
% land from water.
sharpenedImg = imsharpen(noiseRemoved, 'Radius', 1.5, 'Amount', 1.5, 'Threshold', 0.5);

% Apply a Gaussian filter to smooth the image after sharpening.
Iblur = imgaussfilt(sharpenedImg, 'FilterSize', 5);

%% 2. Land Mask Creation
% Create a binary mask for the land. Land typically appears brighter than
% the sea in SAR images, so a simple threshold is used.
landMask = Iblur > landThreshold;

% Perform morphological closing and opening to refine the land mask.
% This helps to fill small holes in the land masses and remove isolated pixels.
se = strel('diamond', 2); % Use a diamond structuring element
landMask = imclose(landMask, se);
landMask = imopen(landMask, se);

%% 3. Oil Spill Segmentation
% Call a specialized function to perform automatic thresholding for oil spills.
% This function is similar to the standard automatic_threshold but is
% intended for use in this land+sea context.
[oilThresh, BW_oil] = automatic_threshold_for_land(img, ground, areaToExplore, medFilt);

%% 4. Visualization
% Call a specialized function to display the original image with both the
% land mask (green) and the oil spill mask (cyan) overlaid.
visualizeImages_for_land(img, landMask, ground, oilThresh, 'AUTOMATIC THRESH. OIL + LAND');

%% 5. Combine Masks for Evaluation
% Combine the binary land mask and the binary oil spill mask into a single
% mask for quantitative evaluation against the ground truth.
BWautomatic = or(landMask, BW_oil);

% Stop timer and display execution time
fprintf('Land Masking execution time: %.4f seconds.\n', toc);

end
