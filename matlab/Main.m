% main.m
%
% DESCRIPTION:
% This is the main script for the Oil Spill Detection and Segmentation project.
% It provides a command-line interface for users to:
%   1. Select the type of SAR image to analyze (with or without land).
%   2. Choose a specific image from the dataset.
%   3. Select a segmentation algorithm to apply.
%   4. Interactively fine-tune algorithm parameters to improve results.
%
% AUTHOR:
% Aaron Sequeira (Original)
%
% LAST REVISED:
% August 3, 2025
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function main()
    % Clear workspace, command window, and close all figures
    close all;
    clear;
    clc;

    fprintf('====================================================\n');
    fprintf('    Oil Spill Detection & Segmentation in SAR Images \n');
    fprintf('====================================================\n');

    % --- Dataset Path Setup ---
    % Use uigetdir to allow the user to select the dataset directory.
    % This avoids hardcoded paths and makes the script portable.
    datasetBaseDir = uigetdir('', 'Please select the main dataset directory');
    if datasetBaseDir == 0
        fprintf('\nNo directory selected. Exiting program.\n');
        return;
    end

    % Define subdirectories based on the selected base directory
    paths.imagesDir = fullfile(datasetBaseDir, 'train', 'images');
    paths.landImagesDir = fullfile(datasetBaseDir, 'train', 'images_with_land');
    paths.labelsDir = fullfile(datasetBaseDir, 'train', 'labels');
    paths.landLabelsDir = fullfile(datasetBaseDir, 'train', 'labels_with_land');
    
    % Verify that all necessary directories exist
    if ~checkPaths(paths)
        fprintf('\nError: One or more dataset directories not found. Please check the directory structure.\n');
        return;
    end

    % --- Main Program Loop ---
    while true
        % Main menu to choose image type or exit
        choice = displayMainMenu();

        if choice == 3 % Exit program
            fprintf('\nThanks for using this segmentation program!\n');
            break;
        end

        % Process user's choice for image type
        processImageSelection(choice, paths);
    end
end

function status = checkPaths(paths)
    % Checks if all required directories exist.
    status = true;
    fields = fieldnames(paths);
    for i = 1:length(fields)
        if ~isfolder(paths.(fields{i}))
            fprintf('Directory not found: %s\n', paths.(fields{i}));
            status = false;
        end
    end
end

function choice = displayMainMenu()
    % Displays the main menu and gets the user's choice.
    fprintf('\n----------------------------------------------------------------\n');
    fprintf('Select the type of image to analyze:\n');
    fprintf('  1 - ONLY SEA images\n');
    fprintf('  2 - LAND + SEA images\n');
    fprintf('  3 - EXIT the program\n\n');
    
    choice = input('Make your choice! ---> ');
    while ~ismember(choice, 1:3)
        fprintf('Invalid input. Please choose 1, 2, or 3.\n');
        choice = input('Make your choice! ---> ');
    end
end

function processImageSelection(choice, paths)
    % Handles the selection and processing of a specific image.
    
    isLandAndSea = (choice == 2);
    
    if isLandAndSea
        imageDir = paths.landImagesDir;
        labelDir = paths.landLabelsDir;
        imageTypeStr = 'LAND + SEA';
    else
        imageDir = paths.imagesDir;
        labelDir = paths.labelsDir;
        imageTypeStr = 'ONLY SEA';
    end

    imageFiles = dir(fullfile(imageDir, '*.jpg'));
    labelFiles = dir(fullfile(labelDir, '*.png'));

    numImages = numel(imageFiles);
    if numImages == 0
        fprintf('No images found in the specified directory: %s\n', imageDir);
        return;
    end

    % --- Image Index Selection ---
    fprintf('\nThere are %d images of type "%s".\n', numImages, imageTypeStr);
    fileIdx = input(sprintf('Enter an image number (1-%d) to analyze: ', numImages));
    while isempty(fileIdx) || ~isnumeric(fileIdx) || fileIdx < 1 || fileIdx > numImages
        fprintf('Invalid number. Please enter a value between 1 and %d.\n', numImages);
        fileIdx = input(sprintf('Enter an image number (1-%d) to analyze: ', numImages));
    end
    
    close all;
    tic; % Start timer for processing

    % --- Load Image and Ground Truth ---
    try
        originalImage = im2double(imread(fullfile(imageDir, imageFiles(fileIdx).name)));
        groundTruthRGB = im2double(imread(fullfile(labelDir, labelFiles(fileIdx).name)));
    catch ME
        fprintf('Error loading image or label file: %s\n', ME.message);
        return;
    end
    
    grayImage = im2gray(originalImage);
    
    % Prepare ground truth mask
    groundTruthGray = im2gray(groundTruthRGB);
    if isLandAndSea
        % For land+sea images, combine land (green) and oil (cyan) masks
        oilMaskGT = imbinarize(groundTruthGray, 0.7);
        landMaskGT = imbinarize(groundTruthGray, 0.35);
        groundTruth = or(oilMaskGT, landMaskGT);
    else
        % For sea-only images, just get the oil (cyan) mask
        groundTruth = imbinarize(groundTruthGray, 0.7);
    end

    % Display original image and ground truth
    figure('Name', 'Input Data', 'WindowState', 'maximized');
    subplot(1, 2, 1), imshow(originalImage), title('Original Image');
    subplot(1, 2, 2), imshow(groundTruthRGB), title('Ground Truth');

    % --- Segmentation Method Selection ---
    if isLandAndSea
        runLandSeaSegmentation(grayImage, groundTruth, groundTruthRGB);
    else
        runSeaOnlySegmentation(grayImage, groundTruth, groundTruthRGB);
    end
    
    fprintf('\nTotal processing time for this image: %.4f seconds.\n', toc);
end

function runSeaOnlySegmentation(grayImg, groundTruth, groundTruthRGB)
    % Handles the menu and logic for sea-only image segmentation.
    while true
        fprintf('\n--- Segmentation Methods for ONLY SEA Images ---\n');
        fprintf('1 - Manual Thresholding\n');
        fprintf('2 - Automatic Thresholding\n');
        fprintf('3 - Local Adaptive Thresholding\n');
        fprintf('4 - Superpixel + Otsu Thresholding\n');
        fprintf('5 - Fuzzy Logic Edge Detection\n');
        fprintf('6 - K-Means Clustering\n');
        fprintf('7 - Choose another image\n');
        
        choice3 = input('\nMake your choice! ---> ');
        while ~ismember(choice3, 1:7)
            fprintf('Invalid input. Please choose a number between 1 and 7.\n');
            choice3 = input('Make your choice! ---> ');
        end

        if choice3 == 7, return; end % Go back to the main menu
        
        % Execute the chosen segmentation method
        executeSegmentation(choice3, grayImg, groundTruth, groundTruthRGB, false);
    end
end

function runLandSeaSegmentation(grayImg, groundTruth, groundTruthRGB)
    % Handles the menu and logic for land+sea image segmentation.
     while true
        fprintf('\n--- Segmentation Methods for LAND + SEA Images ---\n');
        fprintf('1 - Automatic Thresholding\n');
        fprintf('2 - K-Means Clustering\n');
        fprintf('3 - Choose another image\n');
        
        choice3 = input('\nMake your choice! ---> ');
        while ~ismember(choice3, 1:3)
            fprintf('Invalid input. Please choose a number between 1 and 3.\n');
            choice3 = input('Make your choice! ---> ');
        end
        
        if choice3 == 3, return; end % Go back to the main menu
        
        % Map choice to the correct case in executeSegmentation
        methodChoice = (choice3 == 1) * 2; % Automatic Thresholding is case 2
        if choice3 == 2, methodChoice = 6; end % K-Means is case 6
        
        % Execute the chosen segmentation method
        executeSegmentation(methodChoice, grayImg, groundTruth, groundTruthRGB, true);
     end
end

function executeSegmentation(method, grayImg, groundTruth, groundTruthRGB, isLandAndSea)
    % Executes the selected segmentation algorithm and handles parameter tuning.
    
    % Keep running the same method until the user decides not to improve
    keepImproving = true;
    while keepImproving
        switch method
            case 1 % Manual Thresholding
                params.addThreshval = 0.06;
                params.areaToConsider = 45;
                params.medianFilter = 3;
                
                [BW, max_val] = manual_thresholding(grayImg, groundTruthRGB, params.addThreshval, params.areaToConsider, params.medianFilter);
                segmentation_evaluation(groundTruth, BW);
                
                [keepImproving, params] = askToImproveManual(max_val);
                if keepImproving
                    [BW, ~] = manual_thresholding(grayImg, groundTruthRGB, params.addThreshval, params.areaToConsider, params.medianFilter);
                    segmentation_evaluation(groundTruth, BW);
                end

            case 2 % Automatic Thresholding
                params.areaToExplore = 45;
                params.medFilter = 3;
                params.landThreshold = 0.5;

                if isLandAndSea
                    [~, BW] = land_mask(grayImg, groundTruthRGB, params.areaToExplore, params.landThreshold, params.medFilter);
                else
                    [~, BW] = automatic_threshold(grayImg, groundTruthRGB, params.areaToExplore, params.medFilter);
                end
                segmentation_evaluation(groundTruth, BW);
                
                [keepImproving, params] = askToImproveAuto(isLandAndSea);
                 if keepImproving
                    if isLandAndSea
                        [~, BW] = land_mask(grayImg, groundTruthRGB, params.areaToExplore, params.landThreshold, params.medFilter);
                    else
                        [~, BW] = automatic_threshold(grayImg, groundTruthRGB, params.areaToExplore, params.medFilter);
                    end
                    segmentation_evaluation(groundTruth, BW);
                end

            case 3 % Local Adaptive Thresholding
                params.noiseFilter = 5;
                params.SharpThresh = 0.5;
                params.GaussianFilter = 5;

                BW = local_threshold(grayImg, groundTruthRGB, params.noiseFilter, params.SharpThresh, params.GaussianFilter);
                segmentation_evaluation(groundTruth, BW);
                
                [keepImproving, params] = askToImproveLocal();
                if keepImproving
                    BW = local_threshold(grayImg, groundTruthRGB, params.noiseFilter, params.SharpThresh, params.GaussianFilter);
                    segmentation_evaluation(groundTruth, BW);
                end

            case 4 % Superpixel
                params.spNum = 25000;
                params.minDist = 450;
                
                BW = superpixel(grayImg, groundTruthRGB, params.spNum, params.minDist);
                segmentation_evaluation(groundTruth, BW);

                [keepImproving, params] = askToImproveSuperpixel();
                if keepImproving
                    BW = superpixel(grayImg, groundTruthRGB, params.spNum, params.minDist);
                    segmentation_evaluation(groundTruth, BW);
                end

            case 5 % Fuzzy Logic
                params.filterSize = 5;
                
                BW = fuzzy_edgeDetect(grayImg, groundTruthRGB, params.filterSize);
                segmentation_evaluation(groundTruth, BW);

                [keepImproving, params] = askToImproveFuzzy();
                if keepImproving
                     BW = fuzzy_edgeDetect(grayImg, groundTruthRGB, params.filterSize);
                     segmentation_evaluation(groundTruth, BW);
                end

            case 6 % K-Means Clustering
                params.filterSize = 3;
                params.numClusters = 5;
                params.numBlobs = 1;
                
                if isLandAndSea
                    BW = kmeansSegment_for_land(grayImg, groundTruthRGB, params.filterSize, params.numClusters, params.numBlobs);
                else
                    [~, BW] = kmeansSegment(grayImg, groundTruthRGB, params.filterSize, params.numClusters, params.numBlobs);
                end
                segmentation_evaluation(groundTruth, BW);
                
                [keepImproving, params] = askToImproveKmeans();
                if keepImproving
                    if isLandAndSea
                        BW = kmeansSegment_for_land(grayImg, groundTruthRGB, params.filterSize, params.numClusters, params.numBlobs);
                    else
                        [~, BW] = kmeansSegment(grayImg, groundTruthRGB, params.filterSize, params.numClusters, params.numBlobs);
                    end
                    segmentation_evaluation(groundTruth, BW);
                end
        end
        
        if ~keepImproving
            close all;
        end
    end
end

% --- Helper functions for parameter tuning ---

function [improve, params] = askToImproveManual(max_value)
    % Handles parameter tuning for Manual Thresholding.
    improve = false;
    params = struct();
    
    resp = input('\nWould you like to improve segmentation results? [y/n] ', 's');
    if ~strcmpi(resp, 'y')
        return;
    end
    improve = true;
    close all;

    fprintf('\n-- Parameter Tuning for Manual Thresholding --\n');
    
    % Median Filter
    params.medianFilter = getNumericInput('Enter Median Filter size (odd, >=3, default: 3): ', 3, @(x) x >= 3 && rem(x,2)==1);
    
    % Threshold Value
    fprintf('\nCurrent max threshold value: %f\n', max_value);
    params.addThreshval = getNumericInput('Enter value to add to threshold (e.g., 0.05 or -0.05, default: 0): ', 0, @(x) x > -1 && x < 1) + max_value;
    
    % Area to Consider
    params.areaToConsider = getNumericInput('Enter min blob area to consider (pixels, default: 45): ', 45, @(x) x > 0);
end

function [improve, params] = askToImproveAuto(isLandAndSea)
    % Handles parameter tuning for Automatic Thresholding.
    improve = false;
    params = struct();
    
    resp = input('\nWould you like to improve segmentation results? [y/n] ', 's');
    if ~strcmpi(resp, 'y')
        return;
    end
    improve = true;
    close all;

    fprintf('\n-- Parameter Tuning for Automatic Thresholding --\n');
    
    % Median Filter
    params.medFilter = getNumericInput('Enter Median Filter size (odd, >=3, default: 3): ', 3, @(x) x >= 3 && rem(x,2)==1);
    
    % Area to Explore
    params.areaToExplore = getNumericInput('Enter min blob area to consider (pixels, default: 45): ', 45, @(x) x > 0);

    % Land Threshold (only if applicable)
    if isLandAndSea
        params.landThreshold = getNumericInput('Enter land threshold (0-1, default: 0.5): ', 0.5, @(x) x > 0 && x < 1);
    else
        params.landThreshold = 0.5; % Default value not used
    end
end

function [improve, params] = askToImproveLocal()
    % Handles parameter tuning for Local Adaptive Thresholding.
    improve = false;
    params = struct();
    
    resp = input('\nWould you like to improve segmentation results? [y/n] ', 's');
    if ~strcmpi(resp, 'y')
        return;
    end
    improve = true;
    close all;

    fprintf('\n-- Parameter Tuning for Local Adaptive Thresholding --\n');
    
    params.noiseFilter = getNumericInput('Enter Wiener Filter size (odd, >=3, default: 5): ', 5, @(x) x >= 3 && rem(x,2)==1);
    params.SharpThresh = getNumericInput('Enter Unsharping Mask threshold (0-1, default: 0.5): ', 0.5, @(x) x > 0 && x < 1);
    params.GaussianFilter = getNumericInput('Enter Gaussian Filter size (odd, >=3, default: 5): ', 5, @(x) x >= 3 && rem(x,2)==1);
end

function [improve, params] = askToImproveSuperpixel()
    % Handles parameter tuning for Superpixel method.
    improve = false;
    params = struct();
    
    resp = input('\nWould you like to improve segmentation results? [y/n] ', 's');
    if ~strcmpi(resp, 'y')
        return;
    end
    improve = true;
    close all;

    fprintf('\n-- Parameter Tuning for Superpixel Segmentation --\n');
    
    params.spNum = getNumericInput('Enter number of superpixels (>=500, default: 25000): ', 25000, @(x) x >= 500);
    params.minDist = getNumericInput('Enter min distance from main blob (pixels, default: 450): ', 450, @(x) x > 0);
end

function [improve, params] = askToImproveFuzzy()
    % Handles parameter tuning for Fuzzy Logic method.
    improve = false;
    params = struct();
    
    resp = input('\nWould you like to improve segmentation results? [y/n] ', 's');
    if ~strcmpi(resp, 'y')
        return;
    end
    improve = true;
    close all;

    fprintf('\n-- Parameter Tuning for Fuzzy Logic Segmentation --\n');
    
    params.filterSize = getNumericInput('Enter Lee Filter size (odd, >=3, default: 5): ', 5, @(x) x >= 3 && rem(x,2)==1);
end

function [improve, params] = askToImproveKmeans()
    % Handles parameter tuning for K-Means method.
    improve = false;
    params = struct();
    
    resp = input('\nWould you like to improve segmentation results? [y/n] ', 's');
    if ~strcmpi(resp, 'y')
        return;
    end
    improve = true;
    close all;

    fprintf('\n-- Parameter Tuning for K-Means Clustering --\n');
    
    params.filterSize = getNumericInput('Enter Gaussian Filter size (odd, >=3, default: 3): ', 3, @(x) x >= 3 && rem(x,2)==1);
    params.numClusters = getNumericInput('Enter number of clusters (>=2, default: 5): ', 5, @(x) x >= 2);
    params.numBlobs = getNumericInput('Enter number of oil spill blobs to extract (>=1, default: 1): ', 1, @(x) x >= 1);
end

function value = getNumericInput(prompt, defaultValue, validationFcn)
    % Generic function to get a validated numeric input from the user.
    value = input(prompt);
    if isempty(value)
        value = defaultValue;
        return;
    end
    
    while ~isnumeric(value) || ~validationFcn(value)
        fprintf('Invalid input. Please try again.\n');
        value = input(prompt);
        if isempty(value)
            value = defaultValue;
            return;
        end
    end
end
