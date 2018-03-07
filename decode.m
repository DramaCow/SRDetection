%% load and preprocess data
% load necessary mat files
spk = load('Tmaze_spiking_data.mat');
loc = load('Tmaze_location_data.mat');
rst = load('rippspin-times-FGHIJ.mat');

% gets start and end times of each epoch
pre_epoch  = cell2mat(spk.epochs(5,2:3));
maze_epoch = cell2mat(spk.epochs(5,4:5));
post_epoch = cell2mat(spk.epochs(5,6:7));

% looks at hippocampal neurons only
hc = spk.Jcells(strcmp({spk.Jcells.area}, 'hc'));

% ignores 0,0 positions (erroneous)
pos = loc.Jpositiondata(~all(loc.Jpositiondata(:,2:3)==0,2),:); pos(:,1) = pos(:,1)/1e6;
% maze_epoch = [max(maze_epoch(1),min(pos(:,1))), min(maze_epoch(2),max(pos(:,1)))];
maze_epoch = [max(maze_epoch(1),min(pos(:,1))), min(min(pos(:,1))+200,max(pos(:,1)))];

n_bins = 32; spatial_bin_size = max(pos(:,2:3))/n_bins;

% approximate position recording frequency (prf)
% dif = diff(pos(:,1)); prf = mean(dif(abs(dif-mean(dif)) < std(dif))); clear dif;

% get ripple periods (+/- 100ms around detected SPW-R peak times)
rip = merge_intervals([rst.Jrip-0.1, rst.Jrip+0.1]);

% free memory
clear spk loc rst;

% calculate (approximate) occupancy (total time spent in location bins)
occ = occupancy(pos(5:10,:),spatial_bin_size,n_bins,diff(maze_epoch));
imshow(10*(1/max(max(occ)))*occ);
pause;

% approximate position of neuron firing
f = cell(length(hc),1);
for i = 1:length(hc)
    tspk = hc(i).tspk(maze_epoch(1)<=hc(i).tspk & hc(i).tspk<=maze_epoch(2));
    f{i} = occupancy(pos_at_time(pos, tspk'),spatial_bin_size,n_bins,1);
%     imshow(10*fi);
end

%% functions
function result = merge_intervals(intervals)
result = [];
startj = intervals(1,1);
endj   = intervals(1,2);
for i = 2:length(intervals)
    startc = intervals(i,1);
    endc   = intervals(i,2);
    if startc <= endj
        endj = max(endc, endj);
    else
        result = [result ; startj, endj];
        startj = startc;
        endj   = endc;
    end
end
result = [result ; startj, endj];
end

function closest = pos_at_time(pos, times)
[~,inds] = min(abs(times-pos(:,1)));
closest = [times', pos(inds,2:3)];
end

function occ = occupancy(pos,spatial_bin_size,n,a)
bin_pos = round(pos(:,2:3)./spatial_bin_size);
occ = zeros(n+1,n+1);
for y = 0:n+1
    for x = 0:n+1
        occ(y+1,x+1) = sum(all(bin_pos == [x,y],2));
    end
end
occ = (a/sum(sum(occ)))*occ;
occ(isnan(occ)) = 0;
end