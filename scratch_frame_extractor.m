%% extract top frames
clc
clear all

full_grid = imread("film1_big.jpg");
imshow(full_grid,[])


%% extract top frames

y1 = 1; 
y2 = 288;
x1 = 1; 
x2 = 1152;
patch_top = full_grid(y1:y2, x1:x2, :);

figure;

image([x1 x2],[y1 y2], patch_top);   
axis on
xlim([x1 x2])
ylim([y1 y2])                    
axis image

imwrite(patch_top, "film_top.jpg");

%% extract mid top



I1_y1 = 3; 
I1_y2 = 287;
I1_x1 = 385; 
I1_x2 = 765;
patch_mid_top = full_grid(I1_y1:I1_y2, I1_x1:I1_x2, :);


figure;
image([I1_x1 I1_x2],[I1_y1 I1_y2], patch_mid_top);   
axis on
xlim([I1_x1 I1_x2])
ylim([I1_y1 I1_y2])                    
axis image

imwrite(patch_mid_top, "film_mid_top.jpg");

%% extract mid second


I2_y1 = 291; 
I2_y2 = 575;
I2_x1 = 385; 
I2_x2 = 765;
patch_mid_second = full_grid(I2_y1:I2_y2, I2_x1:I2_x2, :);


figure;

image([I2_x1 I2_x2],[I2_y1 I2_y2], patch_mid_second);  
axis on
xlim([I2_x1 I2_x2])
ylim([I2_y1 I2_y2])
axis image

imwrite(patch_mid_second, "film_mid_second.jpg");


%% extract mid third


I3_y1 = 579; 
I3_y2 = 863;
I3_x1 = 385; 
I3_x2 = 765;
patch_mid_third = full_grid(I3_y1:I3_y2, I3_x1:I3_x2, :);


figure;

image([I3_x1 I3_x2],[I3_y1 I3_y2], patch_mid_third);   
axis on
xlim([I3_x1 I3_x2])
ylim([I3_y1 I3_y2])                   
axis image

imwrite(patch_mid_third, "film_mid_third.jpg");

%% extract mid fourth



I4_y1 = 867; 
I4_y2 = 1151;
I4_x1 = 385; 
I4_x2 = 765;

patch_mid_fourth = full_grid(I4_y1:I4_y2, I4_x1:I4_x2, :);


figure;
image([I4_x1 I4_x2],[I4_y1 I4_y2], patch_mid_fourth);  
axis on
xlim([I4_x1 I4_x2])
ylim([I4_y1 I4_y2])                  
axis image

imwrite(patch_mid_fourth, "film_mid_fourth.jpg");

%% extract mid fifth


I5_y1 = 1155; 
I5_y2 = 1439;
I5_x1 = 385; 
I5_x2 = 765;

patch_mid_fifth = full_grid(I5_y1:I5_y2, I5_x1:I5_x2, :);


figure;

image([I5_x1 I5_x2],[I5_y1 I5_y2], patch_mid_fifth);  
axis on
xlim([I5_x1 I5_x2])
ylim([I5_y1 I5_y2])                   
axis image

imwrite(patch_mid_fifth, "film_mid_fifth.jpg");


%% extract right top

clc
clear all

full_grid = imread("film1_big.jpg");


y1 = 1; 
y2 = 288;
x1 = 768; 
x2 = 1152;

patch_top = full_grid(y1:y2, x1:x2, :);


figure;

image([x1 x2],[y1 y2], patch_top);   
axis on
xlim([x1 x2])
ylim([y1 y2])                   
axis image

imwrite(patch_top, "film_right_top.jpg");


%% extract small scratch

clc
clear all

full_grid = imread("film1_big.jpg");


% Clip to image bounds
y1 = 37; 
y2 = 45;
x1 = 736; 
x2 = 746;

patch_top = full_grid(y1:y2, x1:x2, :);


figure;

image([x1 x2],[y1 y2], patch_top);  
axis on
xlim([x1 x2])
ylim([y1 y2])                 
axis image

imwrite(patch_top, "film_characteristic_scratch.jpg");

%% Grön kanal, med heltalsvärden 0–255
green = patch_top(:,:,2);
[nRows, nCols] = size(green);

figure;
imagesc(green, [0 255]);
colormap(gray(256));
colorbar;
axis image;


xticks(1:nCols);
yticks(1:nRows);
xticklabels(string(x1:x2));
yticklabels(string(y1:y2));
grid off;

xlabel("x-position i originalbilden");
ylabel("y-position i originalbilden");
title("Grön intensitet i varje pixel");

% Skriv intensiteten i respektive ruta
hold on;
for row = 1:nRows
    for col = 1:nCols
        value = double(green(row, col));

        % Kontrastfärg så att talet syns på både mörka och ljusa rutor
        if value < 128
            textColor = "white";
        else
            textColor = "black";
        end

        text(col, row, sprintf("%d", value), ...
            "HorizontalAlignment", "center", ...
            "VerticalAlignment", "middle", ...
            "Color", textColor, ...
            "FontSize", 9);
    end
end

exportgraphics(gcf, "film_scratch_green_values.png", "Resolution", 300);
%% Orkade inte skriva detta, OpenAIs verk
clc
clear

full_grid = imread("film1_big.jpg");

% Vanliga gränser för mid_top
mid_y1 = 1;   mid_y2 = 288;
mid_x1 = 385; mid_x2 = 765;

% Gränser för den karakteristiska repan
scratch_y1 = 37;  scratch_y2 = 45;
scratch_x1 = 736; scratch_x2 = 746;

patch_mid_top = full_grid(mid_y1:mid_y2, mid_x1:mid_x2, :);

figure;
highlight = image([mid_x1 mid_x2], [mid_y1 mid_y2], patch_mid_top);
axis image
axis off
hold on

% Halvpixelsmarginal gör att ramen omsluter hela de valda pixlarna
rectangle( ...
    "Position", [scratch_x1-0.5, scratch_y1-0.5, ...
                 scratch_x2-scratch_x1+1, scratch_y2-scratch_y1+1], ...
    "EdgeColor", "r", ...
    "LineWidth", 2);

xlim([mid_x1-0.5, mid_x2+0.5])
ylim([mid_y1-0.5, mid_y2+0.5])
title("Scratch location in mid\_top")

exportgraphics(gca, "film_scratch_highlight.jpg", "Resolution", 300);

%%

imshow(patch_mid_fourth, []);


%%


r = 148; c = 155;
rect = [c, r, 10, 10];      % [x y width height], x=column, y=row

patch1 = imcrop(patch_mid_top, [c-14, r, 20, 20]);
patch2 = imcrop(patch_mid_second, [c-8, r+5, 20, 20]);
patch3 = imcrop(patch_mid_third, [c-3, r+5, 20, 20]);
patch4 = imcrop(patch_mid_fourth, [c+3, r+5, 20, 20]);
patch5 = imcrop(patch_mid_fifth, [c+11, r+5, 20, 20]);

figure;
montage({patch1, patch2, patch3, patch4, patch5}, "Size", [1 5]);
title("Corresponding 9-by-9 regions across mid patches");
exportgraphics(gcf, "film_mid_region_comparison.jpg", "Resolution", 300);