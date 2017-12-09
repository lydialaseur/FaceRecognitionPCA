function [] = faceRecPCA(N,M,training,testing)
% Performs facial recognition using prinipal component analysis
%
% Input:    N = the dim of the testing and training faces
%           M = the number of training faces
%           training = folder name of the training faces
%           testing = folder name of the testing faces
% Note: ALL training and testing images must be square, grayscale images of
%       equal size
%
% Output:   no formal output but the results are stored in folder named
%           'results'

    %get the file name of the training faces
    img_names = textscan(ls(training),'%s')';
    cd(training)
    
    %initialize A
    A = zeros(N^2,M);

    %for each training face
    for i = 1:M
        %read in the image as a matrix
        img_name = img_names{1}{i};
        img = imread(img_name);
        %reshpe into a vector
        img_vec = reshape(img,N^2,1);
        %add vector to matrix A
        A(:,i) = img_vec;
    end

    cd('../')
    %compute the average face
    avg_face = mean(A,2);
    
    %normalize A by subtracting the average face from each column of A
    for i = 1:M
        A(:,i) = A(:,i) - avg_face;
    end
    
    %get eigenvectors of A'* A, to be used to find the M best eigenvector
    %of A*A' (the covariance matrix of A)
    [v,mu] = eig(A'*A);
    u = zeros(N^2,M);
    
    
    for i = 1:M
        %calc eigenvector of A*A' using v
        u(:,i) = A*v(:,i);
        %normalize the eigenvectors
        u(:,i) = u(:,i)/norm(u(:,i));
    end

    y = zeros(M);
    
    %find the coefficients of linear combination
    for i = 1:M
        y(:,i) = u'*A(:,i);
    end

    %test images
    %get the file names of the test faces
    test_names = textscan(ls(testing),'%s')';
    cd(testing)
    num_test_imgs = 4;
    mkdir('../results');
    
    %for each test face
    for j = 1:num_test_imgs
        %read in the images and reshape to vector
        test = imread(test_names{1}{j});
        test_vec = reshape(test,N^2,1);
        %subtract the avergae face
        test_vec = double(test_vec) - avg_face;
        %find coefficients
        test_y = u'*test_vec;

        err = Inf;
        %check the error using the euclidean distance for each training
        %face
        for i = 1:M

            new_err = norm(abs(test_y - y(:,i)));
            
            if new_err < err
                err = new_err;
                match = i;
            end
        end
        %show the test face and the recognized training face, save figure
        subplot(1,2,1)
        imshow(test_names{1}{j})
        title(sprintf('Test face %s',test_names{1}{j}))
        
        subplot(1,2,2)
        cd(sprintf('../%s',training))
        imshow(img_names{1}{match})
        title(sprintf('Test face recognized as %s',img_names{1}{match}))
        

        cd('../results')
        saveas(gcf,sprintf('result_%d.png',j))
        hold off
        cd(sprintf('../%s',testing))
    end
    cd('../')
    
end

