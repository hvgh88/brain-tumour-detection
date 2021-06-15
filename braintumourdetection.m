function braintumour()
%read image
img=imread("C:\Users\Anusha V\Documents\MATLAB\BrainTumour\dataset\IM_00014q.TIF");
figure,imshow(img);
title('Original image');
pause(3);
imagesc(img);
colormap(gray);
title('ColorMap of Array Values');
pause(3);
%displays the size of image
disp('Size of image:');
disp(size(img));

%Check if the image is an RGB image
if(size(img,3) > 2)
img=rgb2gray(img);
imagesc(img);
colormap(gray);
pause(3);
end

%Converting image to double
im_dbl=im2double(img);
%removes noise by applying median filter
im_median=medfilt2(img);
imshow(im_dbl);
title('Image in double precision');
imshow(im_median);
title('After Applying Median Filter to remove noise');


%OTSU'S THRESHOLDING
%performs the morphological filtering of the top hat in the grayscale or binary image, 
%returning the filtered image. Top hat filtering calculates the morphological aperture of the 
%image (using structuring element), and then subtracts the result from the original image. 
%strel -> structuring element used in the top hat filtering
im_thresh=imtophat(im_median,strel('disk',40));
imshow(im_thresh);
title('Image after top hat filtering');
pause(3);
%improves the contrast of the image
im_adjust=imadjust(im_thresh);
imshow(im_adjust);
title('Improved contrast');
pause(3);
%determines the threshold value to perform segmentation
threshold_level=graythresh(im_adjust);
%segments the image into two classes: 0 if less than threshold_level and 1 if greater
%or equal to threshold_level
BW=imbinarize(im_adjust,threshold_level);
imshow(BW);
title('Binarized image');
%creating structural element
strel_erode=strel('disk',3);
%Performing morphological erosion
im_erode=imerode(BW,strel_erode);
imshow(im_erode);
title("Otsu's thresholding");
pause(5);

%performs normalized cross correlation 
norm_corr=normxcorr2(im_erode,img);
disp('Normalised Cross Correlation Value for Otsu Thresholding');
disp(max(norm_corr(:)));
pause(5);


%LOCAL THRESHOLDING

%setting lower threshold value
t0=50;
%upper threshold
th= t0+((max(im_median(:))+min(im_median(:)))./2);
seg_img=zeros(size(im_median,1),size(im_median,2));
%setting values to zero or one based on the threshold value obtained
for i= 1:1:size(im_median,1)
    for j=1:1:size(im_median,2)
        if im_median (i,j)>th
            %sets to 1 if greater than threshold
            seg_img(i,j)=1;
        else
            %sets to 0 if lesser than threshold
            seg_img(i,j)=0;
        end
    end
end

figure,imshow(seg_img);
title('Image after setting values to 0 and 1 after thresholding')
pause(3);

%Creating a structuring element
strel_erode=strel('disk',3);
%performs morphological erosion
im_erode=imerode(seg_img,strel_erode);
imshow(im_erode);
title('After first morphological erosion');
pause(3);

%performing morphological dilation with disk as the structuring element
strel_dilate=strel('disk',3);
im_dilate=imdilate(im_erode,strel_dilate);
imshow(im_dilate);
title('After morphological dilation');
pause(3);

%performs morphological erosion with disk as the structuring element
im_erode1=imerode(im_dilate,strel_erode);
imshow(im_erode1);
title('Local Thresholding');
pause(5);


%performs normalized cross correlation 
norm_corr=normxcorr2(im_erode1,img);
disp('Normalised Cross Correlation Value for Local Thresholding')
disp(max(norm_corr(:)));
pause(5);


%WATERSHED SEGMENTATION

%morphological operation
im_thresh=imtophat(im_median,strel('disk',40));
figure,imshow(im_thresh);
title('Top-hat filter with disk-40 structuring element')
pause(3);
%improves the contrast of the image
im_adjust=imadjust(im_thresh);
imshow(im_adjust);
title('Improving contrast')
pause(3);
%determines the threshold value
threshold_level=graythresh(im_adjust);
%segments the image using Otsu's threshold
BW=imbinarize(im_adjust,threshold_level);
imshow(BW);
title('Binarized image');
pause(3);
%performs morphological erosion with disk as a structuring element
strel_erode=strel('disk',3);
%performs erosion with a structuring element of shape disk
im_erode=imerode(BW,strel_erode);
imshow(im_erode);
title('Eroded image')
pause(3);
%take the compliment of the result
comp=~im_erode;
imshow(comp);
title('Compliment of eroded image');
pause(3);
%compute distance between every pixel to every non-zero pixel
dist=-bwdist(comp);
dist(comp)=-Inf;
%apply watershed segmentation to get the labelled image
label=watershed(dist);
%convert the image to rgb
img_final=label2rgb(label,'gray','w');
imshow(img_final);
title('Watershed segmentation');
pause(5);

%performs normalized cross correlation 
norm_corr=normxcorr2(rgb2gray(img_final),img);
disp('Normalised Cross Correlation Value for Watershed Segmentation')
disp(max(norm_corr(:)));
pause(3);


%K-MEANS CLUSTERING

%converts image to linear shape
img_reshape=reshape(im_median,[],1);
%apply k-means with k value as 4
[imgVecQ,~]=kmeans(double(img_reshape),4); 
%arranging back into image
img_res=reshape(imgVecQ,size(im_median)); 
figure,imagesc(img_res);
pause(3);
subplot(3,2,1),imshow(img_res==1,[]);
title('First Subplot');
subplot(3,2,2),imshow(img_res==2,[]);
title('Second Subplot');
subplot(3,2,3),imshow(img_res==3,[]);
title('Third Subplot');
subplot(3,2,4),imshow(img_res==4,[]);
title('Fourth Subplot');
pause(3);

%perform normalized cross-correlation for each cluster
norm_corr=normxcorr2(img_res==1,img);
disp('K-means cluster 1 Nomarlized Cross COrrelation')
disp(max(norm_corr(:)));
pause(3);

norm_corr=normxcorr2(img_res==2,img);
disp('K-means cluster 2 Nomarlized Cross COrrelation')
disp(max(norm_corr(:)));
pause(3);

norm_corr=normxcorr2(img_res==3,img);
disp('K-means cluster 3 Nomarlized Cross COrrelation')
disp(max(norm_corr(:)));
pause(3);

norm_corr=normxcorr2(img_res==4,img);
disp('K-means cluster 4 Nomarlized Cross COrrelation')
disp(max(norm_corr(:)));
pause(3);

% smoothing filter matrix
filter_matrix=[1 1 1 1 1 1 1 ;
       	          0 0 0 0 0 0 0];
   
%texture filter is applied to determine the texture image
img_texture=rangefilt(im_dbl);
%applying smoothing filter on the texture image
img_texture=imfilter(img_texture,filter_matrix);
figure,imshow(img_texture);
  
%takes coordinates of tumor region
[row,col]=ginput();
tumor_region=[row,col];
%determine the texture values of the tumor region
val_tumor=impixel(img_texture,tumor_region(:,1),tumor_region(:,2));

figure,imshow(img_texture);
%take coordinates of the skull region
[rols,cols]=ginput();
skull_region=[rols,cols];

%determine the texture values of the skull region
val_skull=impixel(im_dbl,skull_region(:,1),skull_region(:,2));
disp(val_skull(:,1));

%target variable is a vector which divides into two classes :0 represents
%skull region and 1 represents tumor region
target_variable=[zeros(numel(val_skull(:,1)),1); ones(numel(val_tumor(:,1)),1)];

%making the dimensions of the target variable and tumor region same
val_tumor=[val_tumor(:,1);zeros(length(target_variable) - length(val_tumor),1)];

%compute cross correlation
correlation=xcorr2(target_variable,val_tumor);
disp(max(correlation(:)));

end