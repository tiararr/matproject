function skripsiehe
%rng(1)

%Membuat tempat untuk MSE dan Matrix
vektor_mse=[];
vektor_mse_test=[];
vektor_matrix=[];
vektor_akurasi=[];
vektor_akurasi_test=[];
recall_tr=[];

aIW1=[];
ab1=[];
aLW2=[];
ab2=[];
aLW3=[];
ab3=[];

IW1=[];
b1=[];
LW2=[];
b2=[];
LW3=[];
b3=[];

mean_bobotawal_input2hidden=[];
mean_bobotawal_bias2hidden=[];
mean_bobotawal_hidden2hidden=[];
mean_bobotawal_biashid2hidden=[];
mean_bobotawal_hidden2output=[];
mean_bobotawal_bias2output=[];

mean_bobotakhir_input2hidden=[];
mean_bobotakhir_bias2hidden=[];
mean_bobotakhir_hidden2hidden=[];
mean_bobotakhir_biashid2hidden=[];
mean_bobotakhir_hidden2output=[];
mean_bobotakhir_bias2output=[];
meantr=[];
hasilts=[];

%inisialisasi
akurasi_hit=0;
x_data=[];

%input data
data=table2array(readtable('darah.csv'));
%data=filloutliers(data, 'clip', 'percentiles',[15,85]);
%melakukan preprocessing data menggunakan normalisasi
%data_gender=data(:,10);
data_tgt=data(:,11);
data_tertinggi=max(data(:,1:10));
data_terendah=min(data(:,1:10));
[m,n]=size(data(:,1:10));
prep_data=zeros(m,n);
for x=1:m
    for y=1:n
        prep_data(x,y)=((0.8*(x-data_terendah))/(data_tertinggi-data_terendah))+0.1;
    end
end
data=[prep_data data_tgt];


%mengambil index data training
%perbandingan data 90:10
index_tr=randsample(1:size(data),4191,true);    %random sampling index 4191 data training
index_tr(1:20)

boot=100;
[bootstat,bootsam]=bootstrp(boot,@mean,index_tr); %bootstraping x kali, 4191x

for i=1:boot
id_boot_tr=bootsam(:,i);                        %mengambil index data training untuk setiap baris
train_id=data(id_boot_tr,:);                    %mengambil seluruh kolom data dari index bootstrap
                                                %berisi 4191x11 kolom
data_tr=train_id';                              %11x4191
ind_tr=(train_id(:,1:10))';                     %mengambil data variabel independen 10x3971
tgt_tr=(train_id(:,11))';                       %mengambil target 1x4191


%membangun jaringan syaraf tiruan
jaringan=newff(minmax(ind_tr),[16,2,1],{'purelin','tansig','purelin'},'traincgb');

%Inisialisasi bobot dan bias awal
bobotawal_input_ke_hiddenlayer=jaringan.IW{1,1};
bobotawal_bias_ke_hiddenlayer=jaringan.b{1,1};
bobotawal_hidden_ke_hidden=jaringan.LW{2,1};
bobotawal_biashid_ke_hidden=jaringan.b{2,1};
bobotawal_hiddenlayer_ke_output=jaringan.LW{3,2};
bobotawal_bias_ke_output=jaringan.b{3,1};

%Mencari rata-rata nilai bobot awal
input2hidden1=mean(bobotawal_input_ke_hiddenlayer);
bias2hidden1=mean(bobotawal_bias_ke_hiddenlayer);
hidden2hidden1=mean(bobotawal_hidden_ke_hidden);
biashid2hidden1=mean(bobotawal_biashid_ke_hidden);
hidden2output1=mean(bobotawal_hiddenlayer_ke_output);
bias2output1=mean(bobotawal_bias_ke_output);

mean_bobotawal_input2hidden=[mean_bobotawal_input2hidden, input2hidden1];
mean_bobotawal_bias2hidden=[mean_bobotawal_bias2hidden, bias2hidden1];
mean_bobotawal_hidden2hidden=[mean_bobotawal_hidden2hidden, hidden2hidden1];
mean_bobotawal_biashid2hidden=[mean_bobotawal_biashid2hidden, biashid2hidden1];
mean_bobotawal_hidden2output=[mean_bobotawal_hidden2output, hidden2output1];
mean_bobotawal_bias2output=[mean_bobotawal_bias2output, bias2output1];

%Training jaringan
jaringan.trainParam.epochs=1000;
jaringan.trainParam.lr=0.01;
jaringan.performParam.regularization = 0.6;
%jaringan.trainParam.showCommandLine=false;
%jaringan.trainParam.showWindow=true;
jaringan.trainParam.goal=8e-4;
%jaringan.trainParam.time=inf;
jaringan.trainParam.min_grad=8e-4;
jaringan.trainParam.max_fail=2;
jaringan.trainParam.searchFcn='srchcha';


%Inisialisasi bobot dan bias akhir
bobotakhir_input_ke_hiddenlayer=jaringan.IW{1,1};
bobotakhir_bias_ke_hiddenlayer=jaringan.b{1,1};
bobotakhir_hidden_ke_hidden=jaringan.LW{2,1};
bobotakhir_biashid_ke_hidden=jaringan.b{2,1};
bobotakhir_hiddenlayer_ke_output=jaringan.LW{3,2};
bobotakhir_bias_ke_output=jaringan.b{3,1};

%Mencari rata-rata nilai bobot akhir
input2hidden=mean(bobotakhir_input_ke_hiddenlayer);
bias2hidden=mean(bobotakhir_bias_ke_hiddenlayer);
hidden2hidden=mean(bobotakhir_hidden_ke_hidden);
biashid2hidden=mean(bobotakhir_biashid_ke_hidden);
hidden2output=mean(bobotakhir_hiddenlayer_ke_output);
bias2output=mean(bobotakhir_bias_ke_output);

mean_bobotakhir_input2hidden=[mean_bobotakhir_input2hidden, input2hidden];
mean_bobotakhir_bias2hidden=[mean_bobotakhir_bias2hidden, bias2hidden];
mean_bobotakhir_hidden2hidden=[mean_bobotakhir_hidden2hidden, hidden2hidden];
mean_bobotakhir_biashid2hidden=[mean_bobotakhir_biashid2hidden, biashid2hidden];
mean_bobotakhir_hidden2output=[mean_bobotakhir_hidden2output, hidden2output];
mean_bobotakhir_bias2output=[mean_bobotakhir_bias2output, bias2output];

%simulasi data training
jaringan=init(jaringan);
[jaringan_output, tr]=train(jaringan,ind_tr,tgt_tr);
sim_output=sim(jaringan_output, ind_tr);

%Output data training
hasil_train=[];
for j=1:4191
    if sim_output(j)<0.5
        hasil_train(j)=0;
    else
        hasil_train(j)=1;
    end
end

hasil_train;
sim_output(1:20)

%MSE data training
error_training=tgt_tr-hasil_train;
mse_training=mse(error_training);

%confusion matrix training
matriks_konfusi=(confusionmat(tgt_tr',hasil_train'));
mk_trans=matriks_konfusi';
recall_train=diag(mk_trans)./sum(mk_trans,1)';
recall_tr=[recall_tr,recall_train];

%Akurasi data training
jumlah_training=size(tgt_tr',1);
hasil_training_benar=trace(matriks_konfusi);            %menghitung hasil benar pada matriks konfusi
hasil_training_salah=jumlah_training-hasil_training_benar;
akurasi_training=hasil_training_benar/jumlah_training;

%Menampilkan hasil dalam vektor
vektor_mse=[vektor_mse,mse_training];
vektor_matrix=[matriks_konfusi];
vektor_akurasi=[vektor_akurasi,akurasi_training];

if akurasi_hit==0
    akurasi_hit=vektor_akurasi(:,i);
    x_data=bootsam(:,i);
else if akurasi_hit<vektor_akurasi(:,i)
        akurasi_hit=vektor_akurasi(:,i);
        x_data=bootsam(:,i);
        
        aIW1=bobotawal_input_ke_hiddenlayer;
        ab1=bobotawal_bias_ke_hiddenlayer;
        aLW2=bobotawal_hidden_ke_hidden;
        ab2=bobotawal_biashid_ke_hidden;
        aLW3=bobotawal_hiddenlayer_ke_output;
        ab3=bobotawal_bias_ke_output;
    
        IW1=bobotakhir_input_ke_hiddenlayer;
        b1=bobotakhir_bias_ke_hiddenlayer;
        LW2=bobotakhir_hidden_ke_hidden;
        b2=bobotakhir_biashid_ke_hidden;
        LW3=bobotakhir_hiddenlayer_ke_output;
        b3=bobotakhir_bias_ke_output;
        
    end
end
end


disp('Interval Konfidensi Bobot Awal Training');
[amuHat_input2hidden, asigmaHat_input2hidden, amuCI_input2hidden, asigmaCI_input2hidden]=normfit(mean_bobotawal_input2hidden);
[amuHat_bias2hidden, asigmaHat_bias2hidden, amuCI_bias2hidden, asigmaCI_bias2hidden]=normfit(mean_bobotawal_bias2hidden, 0.05);
[amuHat_hidden2hidden, asigmaHat_hidden2hidden, amuCI_hidden2hidden, asigmaCI_hidden2hidden]=normfit(mean_bobotawal_hidden2hidden, 0.05);
[amuHat_biashid2hidden, asigmaHat_biashid2hidden, amuCI_biashid2hidden, asigmaCI_biashid2hidden]=normfit(mean_bobotawal_biashid2hidden, 0.05);
[amuHat_hidden2output, asigmaHat_hidden2output, amuCI_hidden2output, asigmaCI_hidden2output]=normfit(mean_bobotawal_hidden2output, 0.05);
[amuHat_bias2output, asigmaHat_bias2output, amuCI_bias2output, asigmaCI_bias2output]=normfit(mean_bobotawal_bias2output, 0.05);

disp('Interval Konfidensi Bobot Akhir Training');
[muHat_input2hidden, sigmaHat_input2hidden, muCI_input2hidden, sigmaCI_input2hidden]=normfit(mean_bobotakhir_input2hidden, 0.05);
[muHat_bias2hidden, sigmaHat_bias2hidden, muCI_bias2hidden, sigmaCI_bias2hidden]=normfit(mean_bobotakhir_bias2hidden, 0.05);
[muHat_hidden2hidden, sigmaHat_hidden2hidden, muCI_hidden2hidden, sigmaCI_hidden2hidden]=normfit(mean_bobotakhir_hidden2hidden, 0.05);
[muHat_biashid2hidden, sigmaHat_biashid2hidden, muCI_biashid2hidden, sigmaCI_biashid2hidden]=normfit(mean_bobotakhir_biashid2hidden, 0.05);
[muHat_hidden2output, sigmaHat_hidden2output, muCI_hidden2output, sigmaCI_hidden2output]=normfit(mean_bobotakhir_hidden2output, 0.05);
[muHat_bias2output, sigmaHat_bias2output, muCI_bias2output, sigmaCI_bias2output]=normfit(mean_bobotakhir_bias2output, 0.05);

disp('Tabel Hasil MSE dan Akurasi Training');
tabel_training=[vektor_mse; vektor_akurasi].';

disp('Interval Konfidensi MSE dan Akurasi Training');
[muHat_msetr, sigmaHat_msetr, muCI_msetr, sigmaCI_msetr]=normfit(vektor_mse, 0.05);
[muHat_akurasitr, sigmaHat_akurasitr, muCI_akurasitr, sigmaCI_akurasitr]=normfit(vektor_akurasi, 0.05);

disp('Recall Training');
hasil_recall_tr=recall_tr(1,:);
overall_recall_tr=mean(hasil_recall_tr);
[murc_tr, sigmarc_tr, muCIrc_tr, sigmaCIrc_tr]=normfit(hasil_recall_tr, 0.05);

%Penerapan jaringan baru
data_baru=(data(x_data,:))';
train_baru=data(x_data,:); 
ind_baru=(train_baru(:,1:10))';        
tgt_baru=(train_baru(:,11))';

jaringan_baru=train(jaringan,ind_baru,tgt_baru);
%weights=getwb(jaringan_baru);
%Iw=cell2mat(jaringan_baru.IW)
%b1=cell2mat(jaringan_baru.b(1))
%LW1=cell2mat(jaringan_baru.LW(2))

%mengambil index data testing
index_ts=randsample(1:size(data),221,true);             %random sampling index 221 data testing

data_ts=(data(index_ts,:));                                                         
ind_ts=(data_ts(:,1:10))';                              %mengambil data variabel independen 10x441
tgt_ts=(data_ts(:,11))';                                %mengambil target 1x221


%Simulasi data testing
sim_testing=sim(jaringan_baru, ind_ts);

%Output data testing
hasil_testing=[];
for j=1:221
    if sim_testing<0.5
        hasil_testing(j)=0;
    else
        hasil_testing(j)=1;
    end
end

hasil_testing;

%MSE data testing
error_testing=tgt_ts-hasil_testing;
mse_testing=mse(error_testing);

%Confussion Matrix Testing
matriks_konfusi_testing=(confusionmat(tgt_ts',hasil_testing'));
recall_test=diag(matriks_konfusi_testing)./sum(matriks_konfusi_testing,2);

%Akurasi data testing
jumlah_testing=size(tgt_ts',1);
hasil_testing_benar=trace(matriks_konfusi_testing);
hasil_testing_salah=jumlah_testing-hasil_testing_benar;
akurasi_testing=hasil_testing_benar/jumlah_testing;

%Menampilkan hasil dalam vektor
vektor_mse_test=[vektor_mse_test,mse_testing];
vektor_matrix_test=[matriks_konfusi_testing];
vektor_akurasi_test=[vektor_akurasi_test,akurasi_testing];

disp('Tabel Hasil MSE dan Akurasi Testing');
tabel_testing=[vektor_mse_test; vektor_akurasi_test].';

%input data baru
data_baru_bgt=[43.1, 14.7, 5.14, 7.7, 393, 28.6, 34.1, 83.9, 20, 2]';
%data_baru_bgt=[38.2,12.3,4.29,5,225,28.7,32.2,89,19,1]';
prep_data_baru=((0.8*(data_baru_bgt-data_terendah))/(data_tertinggi-data_terendah))+0.1
r_data_baru=sim(jaringan_baru, prep_data_baru)
prediksi=[];
if r_data_baru<0.5
    prediksi=0;
else
    prediksi=1;
end
table(r_data_baru, prediksi)

meantr=[meantr, muHat_akurasitr];
hasilts=[hasilts,akurasi_testing];


meantr';
hasilts';

