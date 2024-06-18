% FAA_Cone_DB
% Isaac T. Leventon
% August 31, 2023 [Pittsburgh Airport; 12 hour layover as part of a new 3-leg trip after a cancelled/reooked flight]
% This script is designed to read in the .xlsx files contained in the FAA
% cone calorimeter database, extract key metadata and HRR measurements, and
% calculate flux-normalized HRR and 'fire growth' parameters

clear all

%% Read in your data
% ----------------- Specify where all your data is saved ----------------- 
Data_dir=[pwd,'/../DATA'];
Script_Figs_dir=[pwd,'/../SCRIPT_FIGURES/'];

% ----------------- Convert .xls files to .txt files, to help MATLAB ----------------- 
fileList = dir(fullfile(Data_dir,'*.xls')); 
% Loop through each .out file, copy it and give new extension: .txt
for i = 1:numel(fileList)
    file = fullfile(Data_dir, fileList(i).name);
    [tempDir, tempFile] = fileparts(file); 
    status = copyfile(file, fullfile(tempDir, [tempFile, '.txt']));
    % Delete the .out file; but don't do this until you've backed up your data!!
    % delete(file)  
end

clear fileList file tempDir tempFile

% ----------------- Read in / Process .txt files ----------------- 
FAA_Repo = dir(fullfile(Data_dir,'*.txt'));
FAA_Repo = FAA_Repo(~[FAA_Repo.isdir]);      %remove folders from list
%Get the filenames and folders of all files and folders inside your Root Directory
filenames={FAA_Repo(:).name}';
filefolders={FAA_Repo(:).folder}';

%Make a cell array of strings containing the full file locations of the files0
filepaths=fullfile(filefolders,filenames);
N_files=size(filepaths,1);
% files=fullfile(csvfolders,csvfiles);

% Set up the Import Options and import the data
opts = delimitedTextImportOptions("NumVariables", 2); % only take the first two columns of data (HRR + metadata)

% Specify range and delimiter
opts.DataLines = [1, inf]; 
opts.Delimiter = ["\t", ":"];

% Specify file level properties
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";

% % Specify variable properties
% opts = setvaropts(opts, ["Var1", "Scans1Secsscan"], "WhitespaceRule", "preserve");
% opts = setvaropts(opts, ["Var1", "Scans1Secsscan"], "EmptyFieldRule", "auto");

% Import the data
Tab_Data=NaN*ones(N_files,7);
for i=1:N_files
    temp  = readmatrix(filepaths{i}, opts);
    % Determine Sample Name
    matl_names{i,1}=char(temp(find(contains(temp,'Sample Material'))+1,1));
    matl_names{i,1}=erase(matl_names{i,1}, "  ");
    % matl_names{i,1}=strtrim(matl_names{i,1});
    matl_names{i,1}=strip(matl_names{i,1});

    % Find Index where HRR data begins, convert all HRR data to 'Double'
    % add time, HRR data to variable 'HRR{} = [ time (s) | HRR (kW/m2) | THR (MJ/m2) | t*q"ext (MJ/m2) | d(HRR)/dt (kW/s)]]'
    HRR_idx=find(contains(temp(:,1),'Scan'));
    HRR_idx=HRR_idx(end)+2;
    HRR{i,1}(1,3)=0;
    for j=HRR_idx:length(temp)
        HRR{i,1}(j-(HRR_idx-1),1)=str2double(temp(j,1));
        HRR{i,1}(j-(HRR_idx-1),2)=str2double(temp(j,2));
        if j-(HRR_idx-1) > 1 %&& j<length(temp)
            HRR{i,1}(j-(HRR_idx-1),3) = HRR{i,1}(j-(HRR_idx),3) + HRR{i,1}(j-(HRR_idx),2);
        end
    end
    %Calculate d(HRR)/dt
    for j=1:length(HRR{i,1})-1
        HRR{i,1}(j,5)=(HRR{i,1}(j+1,2)-HRR{i,1}(j,2))/(HRR{i,1}(j+1,1)-HRR{i,1}(j,1));
    end
    
    %Remove all rows of HRR data when d(HRR)/dt > 250 kW/s (jump when conditions change as test ends)
    HRR_idx_end=min(find(HRR{i,1}(:,5)>250));     % Find index of jump
    HRR{i,1}(HRR_idx_end:end,:)=[];
    clear HRR_idx_end

    % Determine Tabulated values: Tab_Data = [q_ext (kW/m2) | t_ign (m2) | t_peak (s) | PHRR (kW/m2) | THR (MJ/m2) | m0 (g) | mf (g) ]
    
    % Define [1] q_ext(kW/mw): External heat flux applied during test
    q_ext_temp=split(temp(find(contains(temp,'Radiant Heat Flux')),2), " ");
    Tab_Data(i,1)=str2double(q_ext_temp{1});
    % q_ext(i,1)=str2double(q_ext_temp{1});
    clear q_ext_temp

    % Define flux time product in HRR_array (q_ext*t)
    HRR{i,1}(:,4)=HRR{i,1}(:,1)*Tab_Data(i,1);


    % Define [2] t_ign(s): Ignition time
    t_ign_temp=split(temp(find(contains(temp,'Time to Sustained Ignition')),2), " ");
    Tab_Data(i,2)=str2double(t_ign_temp{1});
    % t_ign(i,1)=str2double(t_ign_temp{1});
    clear t_ign_temp

    % Determine [4] Peak HRR, PHRR (kW/m2), and [3]the time at which it occurs, t_peak (s)
    PHRR_temp=split(temp(find(contains(temp,'Peak Heat Release Rate')),2)," ");
    Tab_Data(i,4)=str2double(PHRR_temp{1});
    % PHRR(i,1)=str2double(PHRR_temp{1});
    
    PHRR_idx=find(contains(temp(:,2),PHRR_temp{1}));
    Tab_Data(i,3)=str2double(temp(PHRR_idx(end),1));
    % t_peak(i,1)=str2double(temp(PHRR_idx(end),1));
    clear PHRR_temp PHRR_idx

    % Determine [5] Total Heat Release, THR (MJ/m2)
    THR_temp=split(temp(find(contains(temp,'Total Heat Released')),2), " ");
    Tab_Data(i,5)=str2double(THR_temp{1});
    % THR(i,1)=str2double(THR_temp{1});
    clear THR_temp

    % Determine [6] Initial Mass, m_0 (g)
    m0_temp=split(temp(find(contains(temp,'Entered Initial Specimen Mass')),2), " ");
    Tab_Data(i,6)=str2double(m0_temp{1});
    % m0(i,1)=str2double(m0_temp{1});
    clear m0_temp    

    % Determine [7] Final Mass, m_f (g)
    mf_temp=split(temp(find(contains(temp,'Measured Final Specimen Mass')),2), " ");
    Tab_Data(i,7)=str2double(mf_temp{1});
    % mf(i,1)=str2double(mf_temp{1});
    clear mf_temp    
end

%% Plot HRR vs Time
unique_matl_names=unique(matl_names)
        h=3;                                  % height of plot in inches
        w=5;                                  % width of plot in inches

for i=1:length(unique_matl_names)
    figure
    for j=1:N_files
        if contains(matl_names{j,1},unique_matl_names{i})
            hold on
            box on
            plot(HRR{j,1}(:,1),HRR{j,1}(:,2));
            axis([0 inf 0 inf]);
            xlabel('time [s]');
            ylabel('HRR [kW/m2]');
        end
    end
    title(char(unique_matl_names{i}(2:end-1)))
    set(gcf, 'PaperSize', [w h]);           % set size of PDF page
    set(gcf, 'PaperPosition', [0 0 w h]);   % put plot in lower-left corner
    fig_name=convertCharsToStrings(unique_matl_names{i}(2:end-1));
    fig_name=strrep(fig_name,'/','_');  %remove backslashes from your figure name so it can be saved
    fig_name=strrep(fig_name,'\','_');  %remove backslashes from your figure name so it can be saved
    fig_name=strrep(fig_name,'"','in.');  %remove " symbols from your figure name so it can be saved
    fig_filename=fullfile(char([Script_Figs_dir, fig_name,'_HRR']));
    print(fig_filename,'-dpdf')
    clear fig_name
end
%% Plot THR vs Flux Time
for i=1:length(unique_matl_names)
    figure
    for j=1:N_files
        if contains(matl_names{j,1},unique_matl_names{i})
            hold on
            box on
            plot(HRR{j,1}(:,4),HRR{j,1}(:,3));
            axis([0 inf 0 inf]);
            xlabel('HRR*q_{ext} (MJ/m2)');
            ylabel('THR [MJ/m2]');
        end
    end
    title(char(unique_matl_names{i}(2:end-1)))
    set(gcf, 'PaperSize', [w h]);           % set size of PDF page
    set(gcf, 'PaperPosition', [0 0 w h]);   % put plot in lower-left corner
        fig_name=convertCharsToStrings(unique_matl_names{i}(2:end-1));
    fig_name=strrep(fig_name,'/','_');  %remove backslashes from your figure name so it can be saved
    fig_name=strrep(fig_name,'\','_');  %remove backslashes from your figure name so it can be saved
    fig_name=strrep(fig_name,'"','in.');  %remove " symbols from your figure name so it can be saved
    fig_filename=fullfile(char([Script_Figs_dir, fig_name,'_THR']));
    print(fig_filename,'-dpdf')
end

% close all
%%

    for j=1:N_files
        if contains(matl_names{j,1},unique_matl_names{5})
            tiledlayout(1,2)
            
            % Left plot
            ax1 = nexttile;
            ax2 = nexttile;
            % hold(ax1,'on')
            hold([ax1 ax2],'on')
            plot(ax1,HRR{j,1}(:,1),HRR{j,1}(:,2))
            % hold(ax1,'off')
            % Right plot
            
            hold(ax2,'on')
            plot(ax2,HRR{j,1}(:,4),HRR{j,1}(:,3),'-k')
            % axis([0 inf 0 inf]);
            % xlabel('time [s]');
            % ylabel('HRR [kW/m2]');
            % hold(ax2,'off')
            % figure(2)
            % hold on
            % title(char(unique_matl_names{i}));     %title the figure based on the material, i
            % plot(ax2,HRR{j,1}(:,4),HRR{j,1}(:,3),'-k');
            % axis([0 inf 0 inf]);
            % xlabel('HRR*q_{ext} (MJ/m2)');
            % ylabel('THR [MJ/m2]');
            % title(unique_matl_names{i});     %title the figure based on the material, i
        end
    end