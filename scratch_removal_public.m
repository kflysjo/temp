clc
clear all

patch_right_top = imread("film_right_top.jpg");
full_grid = imread("film1_big.jpg");
y1 = 1;
y2 = 288;
x1 = 385;
x2 = 765;
patch_mid_top = full_grid(y1:y2, x1:x2, :);

median_h = 5;
median_v = 15;

filterSize_r = [median_h median_v]; % storlek av median kernel
filterSize_g = [median_h median_v]; % storlek av median kernel
filterSize_b = [median_h median_v]; % storlek av median kernel

textureWindow_r = 17; % window för std
textureWindow_g = 17; % window för std
textureWindow_b = 17; % window för std


%% Välj grön kanal och beräkna residual och lokal variation

channel_r = im2double(patch_mid_top(:,:,1));
channel_g = im2double(patch_mid_top(:,:,2));
channel_b = im2double(patch_mid_top(:,:,3));

background_r = medfilt2(channel_r, filterSize_r, "symmetric");
background_g = medfilt2(channel_g, filterSize_g, "symmetric");
background_b = medfilt2(channel_b, filterSize_b, "symmetric");

residual_r = abs(channel_r - background_r);
residual_g = abs(channel_g - background_g);
residual_b = abs(channel_b - background_b);

texture_r = stdfilt(background_r, true(textureWindow_r));
texture_g = stdfilt(background_g, true(textureWindow_g));
texture_b = stdfilt(background_b, true(textureWindow_b));


%% Visa bild, bakgrund och residual
figure;
tiledlayout(2,4);

nexttile;
imshow(patch_mid_top);
title("Original");

nexttile;
imshow(channel_r, []);
title("Röd kanal");

nexttile;
imshow(channel_g, []);
title("Grön kanal");

nexttile;
imshow(channel_b, []);
title("Blå kanal");

nexttile;
imshow(background_g, []);
title("Medianbild");

nexttile;
imshow(residual_r, []);
title("Röd residual");

nexttile;
imshow(residual_g, []);
title("Grön residual");

nexttile;
imshow(residual_b, []);
title("Blå residual");




%% Jämför base


baseThresholds = [0.01 0.02 0.03 0.04 0.05 0.06];
textureWeight = 1.0;
figure;
tiledlayout(2,3);

for i = 1:numel(baseThresholds)
    baseThreshold = baseThresholds(i);


    localThreshold = baseThreshold + textureWeight * texture_r;
    scratchMask = abs(residual_r) > localThreshold;

    nexttile;
    imshow(scratchMask);
    title(sprintf("Bas %.2f, vikt %.1f", baseThreshold, textureWeight));
end

%% Jämför vikt
baseThreshold = 0.05;
textureWeights = [0 1 2 3 4 5];
figure;
tiledlayout(2,3);

for i = 1:numel(textureWeights)
    textureWeight = textureWeights(i);


    localThreshold = baseThreshold + textureWeight * texture_r;
    scratchMask = abs(residual_r) > localThreshold;

    nexttile;
    imshow(scratchMask);
    title(sprintf("Bas %.2f, vikt %.1f", baseThreshold, textureWeight));
end

%% Välj en vikt och visa vad som händer lokalt för röd
baseThreshold = 0.001;
chosenTextureWeight = 3.5;

localThreshold = baseThreshold + chosenTextureWeight * texture_r;
scratchMask_r = abs(residual_r) > localThreshold;

figure;
tiledlayout(2,2);

nexttile;
imshow(texture_r, []);
colorbar;
title("Lokal variation - röd");

nexttile;
imshow(localThreshold, []);
colorbar;
title("Lokal tröskel - röd");

nexttile;
imshow(patch_mid_top, []);
colorbar;
title("Original");

nexttile;
imshow(scratchMask_r);
title(sprintf("Slutlig mask - röd, vikt %.1f", chosenTextureWeight));

%% Välj en vikt och visa vad som händer lokalt för grön
baseThreshold = 0.015;
chosenTextureWeight = 3.5;

localThreshold = baseThreshold + chosenTextureWeight * texture_g;
scratchMask_g = abs(residual_g) > localThreshold;

figure;
tiledlayout(2,2);

nexttile;
imshow(texture_g, []);
colorbar;
title("Lokal variation - grön");

nexttile;
imshow(localThreshold, []);
colorbar;
title("Lokal tröskel - grön");

nexttile;
imshow(patch_mid_top, []);
colorbar;
title("Original");

nexttile;
imshow(scratchMask_g);
title(sprintf("Slutlig mask - grön, vikt %.1f", chosenTextureWeight));

%% Välj en vikt och visa vad som händer lokalt för blå
baseThreshold = 0.03;
chosenTextureWeight = 3.5;

localThreshold = baseThreshold + chosenTextureWeight * texture_b;
scratchMask_b = abs(residual_b) > localThreshold;

figure;
tiledlayout(2,2);

nexttile;
imshow(texture_b, []);
colorbar;
title("Lokal variation - blå");

nexttile;
imshow(localThreshold, []);
colorbar;
title("Lokal tröskel - blå");

nexttile;
imshow(patch_mid_top, []);
colorbar;
title("Original");

nexttile;
imshow(scratchMask_b);
title(sprintf("Slutlig mask - blå, vikt %.1f", chosenTextureWeight));

%% Slutlig mask och restaurering

scratchMask = scratchMask_r | scratchMask_g | scratchMask_b;

% Utvidga masken något om gröna kanter finns kvar.
% Sätt edgeMargin = 0 för att enbart ersätta detekterade pixlar.


edgeMargin = 0;
repairMask = scratchMask;

if edgeMargin > 0
    repairMask = imdilate(scratchMask, strel("disk", edgeMargin));
end

% Återskapa samtliga färgkanaler på samma markerade positioner
original = im2double(patch_mid_top);
restored = original;

for colorIndex = 1:3
    restored(:,:,colorIndex) = regionfill(original(:,:,colorIndex), repairMask);
end

% Visa masken som vita pixlar ovanpå originalbilden
originalWithMask = original;

for colorIndex = 1:3
    channel = originalWithMask(:,:,colorIndex);
    channel(repairMask) = 1;
    originalWithMask(:,:,colorIndex) = channel;
end


figure;
tiledlayout(1,5);

nexttile;
imshow(original);
title("Original");

nexttile;
imshow(repairMask);
title("Slutlig mask");

nexttile;
imshow(originalWithMask);
title("Täckt original");

nexttile;
imshow(restored);
title("Försök till restaurering");

nexttile;
imshow(patch_right_top);
title("Deras restaurering");