function plot_prcurve(res_folder, gt_folder)

% folders must end with /

close all

% parameters
%d = 32.573;         % a detection within d pixels of the ground truth is considered correct
d = 20.358;
%d = 10;
threshold = 0.1:0.02:1;
precision = zeros(1,size(threshold,2));
recall = zeros(1,size(threshold,2));
f1 = zeros(1,size(threshold,2));

res_listing = dir([res_folder '*.csv']);

g = waitbar(0);
for h=1:size(threshold,2)
    true_pos = 0;
    false_neg = 0;
    total_pos = 0;
    for i=1:size(res_listing)
        results = csvread([res_folder res_listing(i).name]);
        % cut results off at the threshold
        for j=1:size(results,1)
            if results(j,3) < threshold(h)
                results = results(1:j-1,:);
                break
            end
        end
        total_pos = total_pos + size(results,1);

        [slide, remain] = strtok(res_listing(i).name, '_');
        subfolder = [gt_folder slide '_v2/'];
        ground_truth = csvread([subfolder res_listing(i).name]);

        % find the centroids
        len_gt = size(ground_truth,1);
        tmp_gt = zeros(len_gt,2);
        for j=1:len_gt;
            tmp = ground_truth(j,:);
            tmp = tmp(tmp~=0);
            len_tmp = size(tmp,2);
            sum_X = 0;
            sum_Y = 0;
            for k=1:len_tmp/2
                sum_X = sum_X + tmp(2*k-1);
                sum_Y = sum_Y + tmp(2*k);
            end
            mean_X = 2*sum_X/len_tmp;
            mean_Y = 2*sum_Y/len_tmp;
            tmp_gt(j,:) = [mean_Y mean_X];
        end
        ground_truth = round(tmp_gt);

        for j=1:len_gt
            false_neg = false_neg + 1;
            for k=1:size(results,1)
                if (ground_truth(j,1) - results(k,1))^2 + (ground_truth(j,2) - results(k,2))^2 < d^2
                    true_pos = true_pos + 1;
                    false_neg = false_neg - 1;
                    break
                end
            end
        end
    end

    precision(h) = true_pos/total_pos;
    recall(h) = true_pos/(true_pos + false_neg);
    f1(h) = 2*precision(h)*recall(h)/(precision(h) + recall(h));
    
   waitbar(h/size(threshold,2));
end
close(g)

[M,I] = max(f1);
disp(['The F1 score reaches a maximum of ' num2str(M) ' when the threshold is ' num2str(threshold(I))])

figure
plot(precision, recall)
title('precision-recall curve')
xlabel('precision')
ylabel('recall')
xlim([0 1])
ylim([0 1])

figure
plot(threshold, f1)
title('f1 score')
xlabel('threshold')
ylabel('f1 score')
xlim([0 1])
ylim([0 1])

figure
plot(threshold, precision)
title('precision')
xlabel('threshold')
ylabel('precision')
xlim([0 1])
ylim([0 1])

figure
plot(threshold, recall)
title('recall')
xlabel('threshold')
ylabel('recall')
xlim([0 1])
ylim([0 1])
