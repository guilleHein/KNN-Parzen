% OPCION dibujos de TRANSF. COSENO
% EJEMPLO SENCILO

vector=0:255
vector=floor(abs(sin(pi/2*vector)));
vector=reshape(vector,16,16);
data=vector;
figure('name','Pruebas')
subplot(2,2,3)
imagesc(data)
colorbar
title('Imagen')
subplot(2,2,1)
dataC=dct2(data);
imagesc(dataC);
colorbar
title('DCT')

subplot(2,2,2)
A=hadamard(16);
dataH=A*(data)*A';

imagesc(dataH);
colorbar
title('Hadamard')

