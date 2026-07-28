function [maschera,BW] = automatic_threshold(img, ground, maxArea, medFilter)
%AUTOMATIC_THRESHOLD Segments an image to find oil spills using automatic thresholding.
%
%   SYNTAX:
%   [maschera, BW] = automatic_threshold(img, ground, maxArea, medFilter)
%
%   DESCRIPTION:
%   This function performs image segmentation on a grayscale SAR image to
%   detect oil spills. It enhances the image, automatically determines
%   thresholds based on the image histogram, creates a binary mask, refines
%   it with morphological operations, and filters blobs by area.
%
%   INPUTS:
%   img             - The input grayscale SAR image (double format).
%   ground          - The ground truth image for visualization.
%   maxArea         - The minimum area (in pixels) for a blob to be
%                     considered an oil spill.
%   medFilter       - The window size for the median filter (e.g., 3 for 3x3).
%
%   OUTPUTS:
%   maschera        - A colored (cyan) overlay mask of the detected spills.
%   BW              - The final binary segmentation mask.
%
%   See also: main, manual_thresholding, land_mask.

% Start timer
tic

%% 1. Image Enhancement
% Apply a median filter to reduce speckle noise.
grayImg = medfilt2(img, [medFilter medFilter]);

% Enhance contrast using histogram equalization.
eqImg = histeq(grayImg, 50);

%% 2. Automatic Thresholding
% Get the histogram of the equalized image.
[counts, binLocations] = imhist(eqImg);
A = [counts, binLocations];

% Find intensity values corresponding to the highest and lowest pixel counts.
% This method assumes that the most frequent pixel value (background) and
% least frequent (potential spill) can help define the threshold range.
valueMax = max(A(:,1));
valueMin = min(A(:,1));

[rowMax, ~] = find(A == valueMax, 1, 'first');
Tmax = A(rowMax, 2); % Threshold corresponding to the highest frequency

[rowMin, ~] = find(A == valueMin, 1, 'first');
Tmin = A(rowMin, 2); % Threshold corresponding to the lowest frequency

% Create a binary mask based on the computed thresholds.
% If Tmin is 0, it's likely not a useful lower bound, so only Tmax is used.
if Tmin == 0
    mask = imbinarize(eqImg, Tmax);
else
    % Use the range between Tmin and Tmax. The order is corrected if needed.
    mask = imbinarize(eqImg, 'adaptive', 'Sensitivity', 0.5);
    %mask = (eqImg >= min(Tmin, Tmax)) & (eqImg <= max(Tmin, Tmax));
end

% Invert the mask because oil spills are dark regions.
mask = ~mask;

%% 3. Morphological Operations
% Refine the mask to remove noise and fill holes.
% Use imopen to remove small, isolated white pixels (noise).
se = strel('disk', 2);
openedImg = imopen(mask, se);

% Use imfill to fill holes within the detected regions.
filledMask = imfill(openedImg, 'holes');

%% 4. Spot Feature Extraction (Blob Analysis)
% Label connected components (blobs) in the binary image.
cc = bwconncomp(filledMask);
labeledImage = labelmatrix(cc);

% Get properties of all blobs, specifically the area.
stats = regionprops(labeledImage, 'Area');

% Filter blobs based on the minimum area criterion.
% This helps remove small, noisy detections that are unlikely to be oil spills.
idx = find([stats.Area] > maxArea);
BW = ismember(labeledImage, idx);

%% 5. Visualization
% Display the final results using a helper function.
visualizeImages(img, BW, ground, 'AUTOMATIC THRESHOLDING');

% Create a colored overlay for potential reuse (e.g., in land_mask).
background = zeros(size(img), 'double');
maschera = imoverlay(background, BW, 'cyan');

% Stop timer and display execution time
fprintf('Automatic Thresholding execution time: %.4f seconds.\n', toc);

end
