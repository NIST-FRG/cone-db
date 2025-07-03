% Analyze C-factor trends on FTT Cone
% Isaac T. Leventon
% Sept. 3, 2024

clear all
close all
clc

data_path=['\\firedata\FLAMMABILITY_DATA\DATA\Cone\FTT-White\Calib'];
% Script_Figs_dir=[pwd,'/../SCRIPT_FIGURES/'];
Colors={'DarkViolet' 'DarkGray' 'Red' 'Gray' 'Gold' 'Green' 'Blue' 'Magenta' 'DeepSkyBlue' ...
    'Indigo' 'Lime' 'Navy' 'DeepPink' 'Firebrick' 'Cyan' 'Khaki' 'DarkGreen' 'darkorange' 'tea' 'goldenrod'}';
Markers = '*+<>^dhopsvx*+<>^dhopsvx';


%% Read in Model Data
Data_dir = dir(fullfile(data_path,'*.csv'));

%Remove folders from list
Data_dir = Data_dir(~[Data_dir.isdir]);      

% Remove log file from list
% Find out which structures in the structure array have the log file
indexesToDelete = contains({Data_dir(:).name}, 'Log', 'IgnoreCase', true);
% Delete items with 'Log' in the field.
Data_dir(indexesToDelete) = [];

%Get the filenames and folders of all files and folders inside your Root Directory
filenames={Data_dir(:).name}';
Nfiles=length(filenames);
for i=1:Nfiles
    temp=strsplit(filenames{i},["_", "."]);
    logfile{i,1}=temp{1};
    clear temp
end


%Find data header/column names (i.e., what column holds what data?)
temp=readtable([data_path,'\',filenames{2,1}],'PreserveVariableNames',true);
idx_V_CH4=find(contains(temp.Properties.VariableNames,'Methane MFM'))-2;  %[SLPM]
idx_X_O2=find(contains(temp.Properties.VariableNames,'O2'),1)-2;          %[Vol. %]
idx_Te=find(contains(temp.Properties.VariableNames,'Stack TC'))-2;      %[K]
idx_dP=find(contains(temp.Properties.VariableNames,'DPT'))-2;           %[Pa]

metadata=table2array(temp(:,1:2));
    idx_t_burner_on=find(contains(metadata,'Burner on'));
    idx_t_burner_off=find(contains(metadata,'Burner off'));
    idx_t_delay_O2=find(contains(metadata,'O2 delay'),1);
    idx_OD_correction=find(contains(metadata,'OD correction'));
    idx_C_fact_mean=find(contains(metadata,'Mean C-factor'));
clear temp metadata
% 
%% Read in test data, metadata, calculate C factor & associated input values
%SLPM CH4 to mass flow rate CH4 
SLPM_to_mdot=(1/60)*(1/22.4)*(16);   %[SLPM] to [g/s]
HOC_CH4=50.0;   %[kJ/g]
for i=1:Nfiles
    temp=readtable([data_path,'\',filenames{i,1}],'PreserveVariableNames',true);
    % Find needed test metadata
    metadata=table2array(temp(:,1:2));
    t_burner_on(i,1)=str2double(table2array(temp(idx_t_burner_on,2)));
    t_burner_off(i,1)=str2double(table2array(temp(idx_t_burner_off,2)));
    t_delay_O2(i,1)=str2double(table2array(temp(idx_t_delay_O2,2)));
    OD_correction(i,1)=str2double(table2array(temp(idx_OD_correction,2)));
    C_fact_mean(i,1)=str2double(table2array(temp(idx_C_fact_mean,2)));
    clear metadata

    % restructure 'temp' to report only numeric data as 'doubles'
    data=table2array(temp(:,3:end));
    Te(i,1)=mean(data(t_burner_off(i,1)-75:t_burner_off(i,1)-15,idx_Te));
    
    X_O2(i,1)=0.01*mean(data(t_burner_off(i,1)-75:t_burner_off(i,1)-15,idx_X_O2));
    X_O2_init(i,1)=0.01*mean(data(t_delay_O2(i,1)+1:t_delay_O2(i,1)+31,idx_X_O2));
    dP(i,1)=mean(data(t_burner_off(i,1)-75:t_burner_off(i,1)-15,idx_dP));
    V_CH4(i,1)=mean(data(t_burner_off(i,1)+t_delay_O2(i,1)-75:t_burner_off(i,1)+t_delay_O2(i,1)-15,idx_V_CH4));
    Q_CH4(i,1)= V_CH4(i)*SLPM_to_mdot*HOC_CH4;
    

    C(i,1)=(Q_CH4(i))/(1.10*12.54*10^3) * sqrt(Te(i)/dP(i)) * (1.105-1.5*X_O2(i))/(X_O2_init(i,1)-X_O2(i));
%     clear temp 
end

%% Plot some stuff
fig = figure 
set(fig,'defaultAxesColorOrder',[0 0 0]);
yyaxis left
plot(C_fact_mean, 'k','LineWidth',2)
xlabel('Calibration number(oldest first)')
ylabel('Calculated C-Factor')
axis([0 inf 0.02 0.06])
hold on

yyaxis right % Plot either dP, Q_CH4, XO2, or T_duct
plot(dP,'m','LineWidth',2)
ylabel('Average Pressure Drop, DPT (Pa)', 'Color', 'm')

% plot(Q_CH4,'b','LineWidth',2)
% ylabel('Reported Methane HRR (kW)', 'Color', 'b')
% 
% plot(X_O2,'c','LineWidth',2)
% ylabel('X_O_2 (vol. %)', 'Color', 'c')
% 
% plot(Te,'r','LineWidth',2)
% ylabel('Temperature in Exhaust Duct (K)', 'Color', 'r')


%%
% %% Read in Calibration Data, Calculate C-Factor, and plot vs. time

% % Import the data
% legend_counter=1;
% figure
% hold on
% line_style={"--","-", ":", "-."};
% %Plot all cases where parameter is at value 2 out of 4 (solid line type, helps with legends)
% for i=1:Nfiles
%     if contains(devc_HRR{i},'hrr')==1
%         FDS_temp=readtable(files{i}, opts);
%         FDS_HRR{i,1}=table2array(FDS_temp);
%         if contains(parameter{i},'Average')==1
%             plot(FDS_HRR{i}(:,1), FDS_HRR{i}(:,2),'-k', 'LineWidth', 3);
%             legend_HRR{legend_counter,1}=[parameter{i},'-',delta{i}];
%             legend_counter=legend_counter+1;
%         elseif contains(delta{i},'2')
%             % legend_HRR{legend_counter,1}=[parameter{i},'-',delta{i}];
%             legend_HRR{legend_counter,1}=parameter{i};
%             legend_counter=legend_counter+1;
%             color_idx=find(strcmp(unique_params,parameter{i}));
%             plot(FDS_HRR{i}(:,1), FDS_HRR{i}(:,2),'LineStyle',line_style{2}, 'LineWidth', 2, 'Color', rgb(Colors{color_idx}));
%         end
%     end
% end
