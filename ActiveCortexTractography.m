function ActiveCortexTractography(fodf,fvert,fnorm,fseed,outPrefix,options)
    %{
    ░█▀▀█ ░█▀▀▀█ ░█▀▄▀█ ░█▀▀▀ ░█▀▀▄ ▀█▀
    ░█─── ░█──░█ ░█░█░█ ░█▀▀▀ ░█─░█ ░█─
    ░█▄▄█ ░█▄▄▄█ ░█──░█ ░█▄▄▄ ░█▄▄▀ ▄█▄

    Active Cortex Tractography (ACT) without GPU


        Created by Ye Wu, PhD (dr.yewu@outlook.com)

        - Nanjing University of Science and Technology, China
        - University of North Carolina at Chapel Hill, USA
        
    %}
    
    arguments
        fodf                    (1,:)    {mustBeFile}
        fvert                   (1,:)    {mustBeNonzeroLengthText}
        fnorm                   (1,:)    {mustBeNonzeroLengthText}
        fseed                   (1,:)    {mustBeFile}
        outPrefix               string   {mustBeNonzeroLengthText}

        options.fmask           string   {mustBeNonzeroLengthText} = 'auto'
        options.fscale          string   {mustBeNonzeroLengthText} = 'auto'
        
        options.nthreads        (1,1)    {mustBeInteger,mustBeNonnegative} = 4;
        options.step            (1,1)    {mustBeNumeric,mustBeNonnegative} = 1;
        options.angle           (1,1)    {mustBeNumeric,mustBeNonnegative} = 0.45*pi;
        options.minlen          (1,1)    {mustBeNumeric,mustBeNonnegative} = 5;
        options.scale           (1,1)    {mustBeNumeric,mustBeNonnegative} = 0.01;
        options.cutoff          (1,1)    {mustBeNumeric,mustBeNonnegative} = 0.001;
        options.sample          (1,1)    {mustBeInteger,mustBeNonnegative} = 2;
        options.size            (1,1)    {mustBeNumeric,mustBeNonnegative} = 10000;
        options.maxnum          (1,1)    {mustBeInteger,mustBeNonnegative} = 1000;
        options.sh              (1,1)    {mustBeNumericOrLogical} = false;
    end

    %%
    addpath("third/scheme");
    addpath("third/csd");
    addpath("third/gifti");
    fscheme = 'sphere_362_vertices.txt';
    scheme = gen_scheme(fscheme,2);

    %%
    fodf_info = niftiinfo(fodf);    fodf_data = niftiread(fodf_info);  
    fseed_info = niftiinfo(fseed);  fseed_data = niftiread(fseed_info); 
    fvert_info = cellfun(@(x)gifti(x),fvert,'UniformOutput',false); 
    fnorm_info = cellfun(@(x)gifti(x),fnorm,'UniformOutput',false);

    switch options.fmask
        case 'auto'
            fmask_data = sum(fodf_data,4) > options.scale;
        otherwise
            fmask_info = niftiinfo(options.fmask);    fmask_data = niftiread(fmask_info);  
    end

    switch options.fscale
        case 'auto'
            fscale_data = single(fmask_data);
        otherwise
            fscale_info = niftiinfo(options.fmask);    fscale_data = niftiread(fscale_info);  
    end

    %% grid infomation
    dirNums = size(scheme.vert,1);
    dimNums = size(fmask_data);
    voxNums = numel(fmask_data);

    if options.sh
        fodf_data = cell2mat(arrayfun(@(x,y,z)SH2amp(fodf_data(x,y,z),scheme),...
            1:dimNums(1),1:dimNums(2),1:dimNums(3),'UniformOutput',false));
    end
   
    fodf_data   = reshape(fodf_data,voxNums,size(fodf_data,4));
    fmask_data  = reshape(fmask_data,voxNums,1);
    fscale_data = reshape(fscale_data,voxNums,1);
    transform   = fodf_info.Transform.T';
   
    %% vertices and normal vector
    fvert_data = [fvert_info{1}.vertices; fvert_info{2}.vertices];
    fvert_data = ras2ijk(fvert_data,transform);
    fnorm_data = [fnorm_info{1}.cdata; fnorm_info{2}.cdata];

    %% seeding
    [seeds(:,1),seeds(:,2),seeds(:,3)] = ind2sub(dimNums,find(fseed_data>options.scale));
    
    seeds = repmat(seeds, options.sample,1) + rand(size(seeds,1)*options.sample,3)-0.5;  
    seeds(any(round(seeds)<=0),:) = [];
    seeds(any(seeds>=dimNums,2) | any(seeds<=[1 1 1],2),:) = []; 
    seeds(mask(sub2ind(dimNums,round(seeds(:,1)),round(seeds(:,2)),round(seeds(:,3))))<0.5,:) = [];
    seedNums = size(seeds,1);
    
    seedBlock = mat2cell(seeds,[ones(1,(seedNums-mod(seedNums,options.size))/p_size)*p_size mod(seedNums,options.size)],3);
    seedBlockNums = length(seedBlock);

    %% temporary
    startparpool(options.nthreads);
    tmpFolder = outPrefix;
    if ~exist(tmpFolder,'dir')
        mkdir(tmpFolder);
    else
        delete(fullfile(tmpFolder,'*.mat'))
    end
    
    opt = [];
    opt.dimNums     = dimNums;
    opt.dirNums     = dirNums;
    opt.tmpFolder   = tmpFolder;
    opt.p_step      = options.step;
    opt.p_angle     = options.angle;
    opt.p_minlen    = options.minlen;
    opt.p_scale     = options.scale;
    opt.p_cutoff    = options.cutoff;
    opt.p_maxnum    = options.maxnum;
    opt.fscheme     = fscheme;
    opt.transform   = transform;
    opt.fvert       = fvert_data;
    opt.fnorm       = fnorm_data;
    opt.voxsize     = norm(fodf_info.pixdim(2:4));
 
    %% block tracking
    parfor idx = 1:seedBlockNums
        tracking(fodf_data,fmask_data,fscale_data,seedBlock{idx},opt,idx);
    end
    
    savetract(tmpFolder,t_filename,seedBlockNums);
    
    if exist(t_filename,'file')
        delete(fullfile(tmpFolder,'*.mat'))
    end

end

function tracking(fod,mask,scale,seedBlock,opt,idx)

    strname = fullfile(opt.tmpFolder,strcat('block_',num2str(idx),'.mat'));
    nsizeidx = size(seedBlock,1);

    % dimension space
    scheme = gen_scheme(opt.fscheme,2);
    nv = length(scheme.vert);

    % start tracking
    Tracts = single(zeros(nsizeidx,3,2500));

    vox_seeds = round(seedBlock);
    ind_seeds = sub2ind(opt.dimNums,vox_seeds(:,1),vox_seeds(:,2),vox_seeds(:,3));
    Tracts(:,:,1) = seedBlock;

    % initize fod and direction
    c_fod = fod(ind_seeds,:);

    ind = 1:nsizeidx;
    v_cen = seedBlock - vox_seeds;
    p_cen = 1 - pdist2(double(v_cen(ind,:)),scheme.vert,'cosine');

    temp = c_fod(ind,:) .* (p_cen>=0);
    [~,ind_cen] = max(temp,[],2);

    f_dir = c_fod(sub2ind([length(ind),nv],ind',ind_cen));

    % cut from ind
    ind2 = (~any(isnan(c_fod),2) & ...
            ~any(isinf(c_fod),2) & ...
            max(c_fod,[],2) >= opt.p_cutoff & ...
            f_dir >= opt.p_cutoff);

    ind = ind(ind2);
    f_dir = f_dir(ind,:);
    ind_cen = ind_cen(ind,:);

    init_dir{1} = scheme.vert(ind_cen,:); 
    init_dir{2} = -scheme.vert(ind_cen,:); 

    % start tracking along the initized seed/direction
    init_ind = ind;
    init_f_dir = f_dir;
    
    num = 1;
    flag = 0;
    for idr = 1:length(init_dir)
        
        ind = init_ind;
        
        while 1
            if num == 1 
                Tracts(ind,:,num+1) = Tracts(ind,:,num) + opt.p_step * init_f_dir .* init_dir{idr}; 
                num = num + 1;
                Tracts(ind,:,num+1) = Tracts(ind,:,num) + opt.p_step * init_f_dir .* init_dir{idr}; 
            elseif flag == 1
                Tracts = cat(3,Tracts,single(zeros(nsizeidx,3,2500)));
                Tracts(ind,:,num+1) = Tracts(ind,:,num) + opt.p_step * init_f_dir .* init_dir{idr};
                num = num + 1;
                Tracts(ind,:,num+1) = Tracts(ind,:,num) + opt.p_step * init_f_dir .* init_dir{idr}; 
                flag = flag + 1;
            else
                cur_dir = last_dir + opt.p_step * f_dir .* next_dir;
                Tracts(ind,:,num) = Tracts(ind,:,num-1) + vecnorm(last_dir,2,2) .* cur_dir./vecnorm(cur_dir,2,2);
                Tracts(ind,:,num+1) = Tracts(ind,:,num) + opt.p_step * f_dir .* next_dir;
            end
            
            if num >= opt.p_maxnum
                break
            end
            
            num = num + 1;
            vox = round(squeeze(Tracts(ind,:,num)));
            
            ind2 = find(all(vox <= opt.dimNums,2) & all(vox > [1,1,1],2));
            if isempty(ind2)
                Tracts(ind,:,num) = 0;
                num = num - 1;
                break;
            end
            
            ind = ind(ind2);
            vox = vox(ind2,:);
            
            % check whether the streamline is out of bounds
            ind_vox = sub2ind(opt.dimNums,vox(:,1),vox(:,2),vox(:,3));

            ind2 = find(mask(ind_vox)>0 & scale(ind_vox) >= opt.p_scale);

            if isempty(ind2)
                Tracts(ind,:,num) = 0;
                num = num - 1;
                break;
            end

            ind = ind(ind2);
            vox = vox(ind2,:);
            ind_vox = ind_vox(ind2);

            c_fod = fod(ind_vox,:);

            % centering hemisphere
            v_cen = squeeze(Tracts(ind,:,num)) - vox;
            p_cen = 1-pdist2(double(v_cen),scheme.vert,'cosine');

            % position-depence direction
            v_pos = squeeze(Tracts(ind,:,num) - Tracts(ind,:,num-1));
            ind2 = find(vecnorm(v_pos,2,2) > 1e-3);

            if isempty(ind2)
                Tracts(ind,:,num) = 0;
                num = num - 1;
                break;
            end

            ind = ind(ind2); 
            v_pos = v_pos(ind2,:);
            v_cen = v_cen(ind2,:);
            c_fod = c_fod(ind2,:);
            p_cen = p_cen(ind2,:);
            last_dir = v_pos;

            temp_flag = zeros(length(ind),1);
            ind2 = atan2d(vecnorm(cross(v_pos,v_cen),2,2),dot(v_pos,v_cen,2)) >= 90;

            v_pos(ind2,:) = -v_pos(ind2,:);
            temp_flag(ind2) = 1;

            p_pos = 1-pdist2(double(v_pos),scheme.vert,'cosine');

            % forward direction                    
            temp = c_fod .* (p_cen >=0 & p_pos >= 0 & abs(p_pos) >= cos(opt.p_angle));
            [~,ind_cen] = max(temp,[],2);
            f_dir = c_fod(sub2ind([length(ind),nv],(1:length(ind))',ind_cen));

            ind2 = f_dir >= opt.p_scale;
            ind = ind(ind2);

            if isempty(ind2)
                Tracts(ind,:,num) = 0;
                num = num - 1;
                break;
            end

            last_dir = last_dir(ind2,:);
            f_dir = f_dir(ind2,:);
            ind_cen = ind_cen(ind2,:);
            temp_flag = temp_flag(ind2,:);

            next_dir = scheme.vert(ind_cen,:);

            % find nearest vertices
            [idx_vert,D] = knnsearch(opt.fvert,squeeze(Tracts(ind,:,num)));
            idx_flag = D < opt.p_step * opt.voxsize * opt.p_minlen;
            idx_norm = opt.fnorm(idx_vert,:);
            idx_norm(~idx_flag,:) = next_dir(~idx_flag,:);

            % cortical position-depence direction
            idx_flag = sign(1-pdist2(double(idx_norm),double(next_dir),'cosine'));
            next_dir = next_dir + idx_flag .* idx_norm;
            next_dir(temp_flag>0,:) = -next_dir(temp_flag>0,:);
        end

        if idr == 1
            Tracts = flip(Tracts(:,:,~all(Tracts==0,[1,2])),3);
            flag = 1;
        else
            Tracts = Tracts(:,:,~all(Tracts==0,[1,2]));
        end
    end

    Tracts = fiber2cell(Tracts);
    Tracts(cellfun(@length,Tracts)<=3) = [];
    Tracts(fiber2length(Tracts)<=opt.p_minlen) = [];
    Tracts = fiber2ras(Tracts,opt.transform);
    Tracts = fibersmooth(Tracts);

    save(strname,'Tracts','-v7.3');
    clear Tracts fod mask;
end

function tract = fibersmooth(data)
    tract = cellfun(@(x)round(single(smoothdata(x,1,'sgolay','SmoothingFactor',0.5)),3),data,'UniformOutput',false);
end

function datastr = fiber2cell(data)
    [m,n,k] = size(data);
    datastr = mat2cell(data,ones(m,1),n,k);
    datastr = cellfun(@(x)transpose(squeeze(x)),datastr,'UniformOutput',false);
    datastr = cellfun(@(x)(x(~all(x==0,2),:)),datastr,'UniformOutput',false);
end

function lengths = fiber2length(data)
    lengths = cellfun(@(x)sum(vecnorm(diff(x),2,2)),data);
end

function tract = fiber2ras(data,affine)
    tract = cellfun(@(x) ijk2ras(x,affine), data,'UniformOutput',false);
end

function tract = fiber2ijk(data,affine)
    tract = cellfun(@(x) ras2ijk(x,affine), data,'UniformOutput',false);
end

function ras = ijk2ras(ijk,affine)
    flag = 0;
    if ~isequal(size(ijk,1),3)
        ijk = ijk';
        flag = 1;
    end
    ras = affine(1:3,1:3) * (ijk-1) + affine(1:3,4);

    if flag
        ras = ras';
    end
end 

function ijk = ras2ijk(ras,affine)
    flag = 0;
    if ~isequal(size(ras,1),3)
        ras = ras';
        flag = 1;
    end
    ijk = affine(1:3,1:3)\(ras-affine(1:3,4)) + 1;

    if flag
        ijk = ijk';
    end
end 

function savetract(tmpFolder,t_filename,seedBlockNums)

    data = cell(1,seedBlockNums);
    for idx = 1:seedBlockNums
        strname = fullfile(tmpFolder,strcat('block_',num2str(idx),'.mat'));
        mf = matfile(strname);
        data{idx} = mf.Tracts;
    end
    
    tck = [];
    tck.data = cat(1,data{:});
    tck.count = length(tck.data);
    clear data;
    
    write_mrtrix_tracks(tck,t_filename)
end
